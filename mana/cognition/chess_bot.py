"""
mana.cognition.chess_bot — MANA playing on Lichess, and saying why.

The shape of it
---------------
`lichess.py` is transport and decides nothing. `chess_arena.SearchPlayer`
is the brain and knows nothing about the network. This module is the only
place the two meet, and it is deliberately small: a loop that reads what
the world says, asks the player for a move, sends it, and writes down what
happened.

    события Lichess ──> принять вызов ──> поток партии
                                            │
                                       наш ход? ──> SearchPlayer.choose
                                            │            │
                                            │        Thought (что рассматривала,
                                            │                 с каким отрывом)
                                            ▼
                                        отправить ──> Judge (сколько стоило)
                                            │
                                            └──> events.emit ──> живая доска

Why the judge runs after the move is sent
------------------------------------------
It costs a tenth of a second and the clock is running. Sent first, judged
second: the cost lands on the opponent's time, and the judgement is about
a move that has already been made anyway. Judging before sending would
also put an engine in the loop between the position and the move, which is
the one thing the whole programme is built to avoid -- even with nobody
reading its answer, a reviewer could not tell from the code that it was
not consulted.

What is recorded, and why both halves
--------------------------------------
    судья    сколько стоил ход           — была ли это ошибка
    трасса   что рассматривала, отрыв    — было ли это вообще решение

Stage 0 measured 77% of moves chosen with no margin at all: the search saw
a tie and the shuffle picked. A record that keeps only the losses cannot
tell those apart from moves the search actually decided, and the repair
for the two is not the same one.

What this module does not do
-----------------------------
It does not learn. Nothing here adapts between games, nothing is adopted,
and no parameter moves in response to a result. That is Stage 0 on
purpose: a baseline collected by a player that changes while it is being
measured is not a baseline. The games are written down so the strategy
question can be asked later, against splits declared before the answer is
known -- the discipline `cognition/trials.py` already enforces.
"""
from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .. import events
from ..net import lichess as api

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Where finished games are kept, beside MANA's other state.
GAMES_DIRNAME = "lichess"

#: Which world a game came from. A game against itself and a game against
#: a stranger are different populations: the first has an opponent that
#: shares its evaluation and its blind spots, the second does not. Pooled,
#: an average over them is a number about nothing, so the source travels
#: with every record and the summary keeps them apart.
LOCAL = "local"
LIVE = "lichess"

#: Search depth for play. Two is what Stage 0 was measured at; raising it
#: silently would make the next measurement a comparison against nothing.
PLAY_DEPTH = 2

#: Judge depth during a game. Zero: the analysis reads the result of the
#: game, not a stronger player's opinion of each move, so paying an engine
#: per move buys a number nothing reads -- and it was most of the cost of
#: a self-play game, which is the thing outcome-based analysis needs more
#: of. The judge is not deleted: it remains an external check for a
#: separate question, and `--chess-rejudge` still calls it on demand.
LIVE_JUDGE_DEPTH = 0

#: A game nobody moves in. Lichess ends abandoned games on its own; this
#: is only so a dead stream cannot hold a thread forever.
GAME_IDLE_LIMIT = 600.0


def games_path() -> Path:
    from ..paths import resolve_data_path

    root = Path(resolve_data_path(GAMES_DIRNAME))
    root.mkdir(parents=True, exist_ok=True)
    return root / "games.jsonl"


@dataclass
class Policy:
    """What MANA agrees to play, decided before anyone asks.

    Stated as a policy rather than as an if-chain inside the loop because
    it is the part a person will want to change, and because "what did it
    agree to" is a question about a run that its record should answer.

    Rated games are off by default: a rating is a number that moves for
    reasons having nothing to do with what MANA learned, and a bot that
    collects one while nothing about it is being measured has produced a
    fact nobody can use.
    """
    variants: tuple = ("standard",)
    rated: bool = False
    max_games: int = 1
    #: Refuse anything faster than this. A search that thinks for a second
    #: is not slow, but a bullet clock turns every measurement into one
    #: about latency.
    slowest_first_move: float = 0.0
    speeds: tuple = ("classical", "correspondence", "rapid", "blitz")

    def verdict(self, challenge: api.Challenge) -> tuple:
        """(accept, reason). The reason travels either way, because a
        refusal nobody can read is indistinguishable from a bug."""
        if challenge.variant not in self.variants:
            return (False, f"вариант {challenge.variant} не играю")
        if challenge.rated and not self.rated:
            return (False, "рейтинговые партии выключены")
        if challenge.speed and challenge.speed not in self.speeds:
            return (False, f"контроль {challenge.speed} слишком быстрый")
        return (True, "подходит")


@dataclass
class Seat:
    """One game in progress, from MANA's side of the board."""
    game_id: str
    us: bool = True                      # True = white, chess's own convention
    opponent: str = ""
    initial_fen: str = "startpos"
    moves: List[str] = field(default_factory=list)
    thoughts: List[Dict[str, Any]] = field(default_factory=list)
    judged: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "started"
    winner: str = ""
    source: str = LIVE
    #: The shuffle's seed, for a self-play game. Written down so a game
    #: can be replayed exactly, which is the only thing that made the
    #: repeated-game bug findable after the fact.
    seed: int = 0
    #: Stockfish's level when the opponent is Lichess's own engine, 0 for
    #: a human. A condition of every measurement taken from the game, not
    #: a label on it.
    level: int = 0
    started: float = field(default_factory=time.time)

    @property
    def url(self) -> str:
        return ("" if self.source == LOCAL
                else f"https://lichess.org/{self.game_id}")

    def board(self) -> Any:
        """The current position, rebuilt from the move list.

        Rebuilt rather than kept, because the stream is the authority: a
        board held in this process and a board Lichess believes in can
        drift after a reconnect, and the one that decides the game is
        theirs.
        """
        import chess

        board = (chess.Board() if self.initial_fen in ("", "startpos")
                 else chess.Board(self.initial_fen))
        for uci in self.moves:
            try:
                board.push_uci(uci)
            except ValueError:
                break
        return board

    def as_dict(self) -> Dict[str, Any]:
        return {"game": self.game_id, "url": self.url, "source": self.source,
                "level": self.level,
                "us": "white" if self.us else "black",
                "opponent": self.opponent, "initial_fen": self.initial_fen,
                "moves": list(self.moves), "status": self.status,
                "seed": self.seed,
                "winner": self.winner, "thoughts": self.thoughts,
                "judged": self.judged, "started": self.started}


class Bot:
    """Reads the world, asks the player, sends the move, writes it down.

    Constructed with a client and a factory rather than reaching for
    either: a bot that builds its own connection can only be tested by
    connecting, and one that builds its own player cannot be handed a
    different one to compare against.
    """

    def __init__(self, client: Optional[api.Lichess] = None,
                 player: Optional[Callable[[], Any]] = None,
                 policy: Optional[Policy] = None,
                 judge_depth: int = LIVE_JUDGE_DEPTH,
                 record: bool = True) -> None:
        self.client = client or api.Lichess()
        self.policy = policy or Policy()
        self.judge_depth = int(judge_depth)
        self.record = bool(record)
        self.username = ""
        self.seats: Dict[str, Seat] = {}
        self.finished: List[Seat] = []
        self._judge: Any = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._player = player or (lambda: _default_player())

    # ---------- the outer loop ----------

    def run(self, games: int = 0) -> List[Seat]:
        """Play until asked to stop, or until `games` have finished.

        Blocks. The caller is either the CLI or a thread the desktop app
        owns; nothing here starts a thread the caller did not ask for
        except one per game, which is what the protocol requires.
        """
        me = self.client.account()
        self.username = str(me.get("username", ""))
        if str(me.get("title", "")).upper() != "BOT":
            raise api.LichessError(
                f"аккаунт {self.username} — не бот, играть через Bot API нельзя. "
                f"Это делается один раз и необратимо: {api.upgrade_command()}")
        events.emit(events.STATUS,
                    f"Lichess: играю как {self.username}; {judge_note(self.judge_depth)}",
                    chess={"kind": "ready", "user": self.username,
                           "judge": judge_note(self.judge_depth),
                           "policy": self.policy.__dict__})
        try:
            for event in self.client.stream_events():
                if self._stop.is_set():
                    break
                self._handle(event)
                if games and len(self.finished) >= games:
                    break
        finally:
            self.close()
        return list(self.finished)

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        if self._judge is not None:
            self._judge.close()
            self._judge = None

    def _handle(self, event: Dict[str, Any]) -> None:
        kind = str(event.get("type", ""))
        if kind == "challenge":
            self._on_challenge(api._challenge(event.get("challenge") or {}))
        elif kind == "gameStart":
            start = api.game_start(event)
            if start.id and start.id not in self.seats:
                threading.Thread(target=self._play, args=(start,),
                                 name=f"MANA-chess-{start.id}",
                                 daemon=True).start()
        elif kind == "gameFinish":
            pass                    # the game thread sees the final state

    def _on_challenge(self, challenge: api.Challenge) -> None:
        if challenge.by == self.username:
            return                  # our own challenge coming back to us
        accept, why = self.policy.verdict(challenge)
        with self._lock:
            busy = len(self.seats) >= self.policy.max_games
        if accept and busy:
            accept, why = False, "уже играю"
        events.emit(events.STATUS,
                    f"Вызов от {challenge.by}: {'принят' if accept else why}",
                    chess={"kind": "challenge", "id": challenge.id,
                           "by": challenge.by, "accepted": accept,
                           "reason": why})
        try:
            if accept:
                self.client.accept(challenge.id)
            else:
                self.client.decline(challenge.id)
        except api.LichessError as exc:
            events.emit(events.WARNING, f"вызов {challenge.id}: {exc}")

    # ---------- one game ----------

    def _play(self, start: api.GameStart) -> None:
        seat = Seat(game_id=start.id, opponent=start.opponent)
        with self._lock:
            self.seats[seat.game_id] = seat
        player = self._player()
        rng = random.Random(hash(seat.game_id) & 0xFFFFFFFF)
        try:
            for frame in self.client.stream_game(seat.game_id):
                if self._stop.is_set():
                    break
                if not self._advance(seat, frame, player, rng):
                    break
        except api.LichessError as exc:
            events.emit(events.ERROR, f"партия {seat.game_id}: {exc}")
        finally:
            with self._lock:
                self.seats.pop(seat.game_id, None)
                self.finished.append(seat)
            if self.record:
                self._write(seat)
            events.emit(events.STATUS,
                        f"партия {seat.game_id} окончена: {seat.status} "
                        f"{seat.winner}".strip(),
                        chess=self._frame(seat, kind="over"))

    def _advance(self, seat: Seat, frame: Dict[str, Any], player: Any,
                 rng: random.Random) -> bool:
        """One message from the game stream. False means the game is over."""
        kind = str(frame.get("type", ""))
        if kind == "gameFull":
            seat.initial_fen = str(frame.get("initialFen", "startpos"))
            white = str(((frame.get("white") or {}).get("id") or "")).lower()
            seat.us = white == self.username.lower()
            side = frame.get("black") if seat.us else frame.get("white")
            seat.opponent = str((side or {}).get("name", seat.opponent))
            seat.level = _level_of(seat.opponent)
            state = frame.get("state") or {}
        elif kind == "gameState":
            state = frame
        elif kind == "chatLine":
            return True
        else:
            return True

        seat.moves = [m for m in str(state.get("moves", "")).split(" ") if m]
        seat.status = str(state.get("status", seat.status))
        seat.winner = str(state.get("winner", "") or "")
        if seat.status not in ("created", "started"):
            return False

        board = seat.board()
        if board.is_game_over():
            return False
        events.emit(events.STATUS, "", chess=self._frame(seat, kind="position"))
        if board.turn != seat.us:
            return True                       # their move; the stream will say
        self._move(seat, board, player, rng)
        return True

    def _move(self, seat: Seat, board: Any, player: Any,
              rng: random.Random) -> None:
        """Choose, send, judge, report -- in that order, and it matters.

        Sent before it is judged so the judge's time is spent on the
        opponent's clock, and so no engine sits between the position and
        the move even by accident.
        """
        chosen = player.choose(board, rng)
        san = board.san(chosen)
        ply = board.ply() + 1
        try:
            self.client.move(seat.game_id, chosen.uci())
        except api.LichessError as exc:
            events.emit(events.WARNING, f"ход {san} не принят: {exc}")
            return
        seat.moves.append(chosen.uci())

        thought = (player.thoughts[-1].as_dict()
                   if getattr(player, "thoughts", None) else {})
        if thought:
            seat.thoughts.append(thought)
        judged = self._judge_move(board, chosen, ply)
        if judged:
            seat.judged.append(judged)

        events.emit(events.STATUS,
                    f"{seat.game_id}: {san}"
                    + (f" (потеря {judged['loss']:.0f})" if judged else ""),
                    chess=self._frame(seat, kind="move", san=san,
                                      thought=thought, judged=judged))

    def _judge_move(self, board: Any, move: Any, ply: int) -> Dict[str, Any]:
        if not self.judge_depth:
            return {}
        from . import chess_judge

        try:
            if self._judge is None:
                self._judge = chess_judge.Judge(depth=self.judge_depth)
            return self._judge.judge_move(board, move, ply).as_dict()
        except Exception as exc:                       # an absent or dead engine
            events.emit(events.WARNING, f"судья молчит: {type(exc).__name__}: {exc}")
            self.judge_depth = 0
            return {}

    # ---------- what the watcher sees ----------

    def _frame(self, seat: Seat, kind: str, san: str = "",
               thought: Optional[Dict[str, Any]] = None,
               judged: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return frame(seat, kind, san, thought, judged)

    def _write(self, seat: Seat) -> None:
        write(seat)


def _level_of(opponent: str) -> int:
    """Stockfish's level from the name Lichess gives it.

    Read from the name rather than handed in by whoever started the game,
    so a game begun from the website carries the same condition as one the
    bench began.
    """
    import re

    found = re.search(r"level\s*(\d+)", str(opponent), re.IGNORECASE)
    return int(found.group(1)) if found else 0


def judge_note(depth: int) -> str:
    """Which judge is about to work, said before it does.

    Three real games were judged by the fallback and nothing said so out
    loud. Every row carried `judged_by: material` -- which is why that
    field exists -- but a person watching a board saw "потеря 300" with no
    author, and a number whose source cannot be told apart is a number
    nobody can compare.
    """
    from . import chess_judge

    if not depth:
        # Not a degraded state any more: the analysis reads the result of
        # the game, so an engine's opinion per move is a number nothing
        # reads. Worded as the normal state it now is -- a warning that
        # fires when nothing is wrong is one nobody reads.
        return ("анализ по исходу партии — судья не вызывается "
                "(включить: --chess-rejudge)")
    if chess_judge.engine_path():
        return f"судья: Stockfish, глубина {depth}"
    where = "\n".join(f"    {row['where']} — {row['why']}"
                      for row in chess_judge.diagnose())
    return ("судья: материальный запасной — Stockfish не найден, "
            "потери сравнимы только между собой.\n"
            f"  искала:\n{where}\n"
            f"  можно указать прямо: задайте {chess_judge.ENGINE_ENV} "
            f"с полным путём к stockfish.exe")


def frame(seat: Seat, kind: str, san: str = "",
          thought: Optional[Dict[str, Any]] = None,
          judged: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One picture of the game, and everything a watcher needs to read it.

    A function rather than a method, because the local runner and the
    Lichess bot must produce identical frames: a window that shows one
    world differently from the other is a window you cannot compare two
    runs in.

    `last_uci` travels beside `last` because SAN says "Qxd6+" and a board
    needs to know which two squares to light.
    """
    board = seat.board()
    return {"kind": kind, "game": seat.game_id, "url": seat.url,
            "source": seat.source, "level": seat.level,
            "fen": board.fen(), "us": "white" if seat.us else "black",
            "opponent": seat.opponent, "ply": board.ply(),
            "turn": "white" if board.turn else "black",
            "status": seat.status, "winner": seat.winner,
            "last": san, "last_uci": seat.moves[-1] if seat.moves else "",
            "thought": thought or {}, "judged": judged or {},
            "summary": summarise(seat)}


def write(seat: Seat) -> None:
    """Append one finished game to the record."""
    try:
        with games_path().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(seat.as_dict(), ensure_ascii=False) + "\n")
    except OSError as exc:
        events.emit(events.WARNING, f"партия не записана: {exc}")


def summarise(seat: Seat) -> Dict[str, Any]:
    """MANA's own moves in this game, in the two terms that differ.

    The denominator travels with every count, because "три зевка" means
    nothing without how many moves were looked at -- the rule
    `invariants.summarise` states for turns and `chess_judge.summarise`
    for moves.
    """
    losses = [row.get("loss", 0.0) for row in seat.judged]
    # Only MANA's own moves: a self-play record now holds both sides'
    # traces, and counting them together would double every number in a
    # line that says "мои ходы".
    mine = [row for row in seat.thoughts if row.get("ours", True)]
    close = [row for row in mine if row.get("close_call")]
    # None, not zero, where nothing judged. "зевков 0" for a game no judge
    # looked at is unmeasured dressed as measured -- the one thing this
    # project refuses everywhere else, and it went out in a status line
    # the moment the judge was switched off.
    counted = bool(seat.judged)
    return {"moves": len(mine),
            "judged": len(seat.judged),
            "mistakes": (sum(1 for row in seat.judged if row.get("mistake"))
                         if counted else None),
            "blunders": (sum(1 for row in seat.judged if row.get("blunder"))
                         if counted else None),
            "mean_loss": round(sum(losses) / len(losses), 1) if losses else None,
            "close_calls": len(close),
            "close_share": (round(len(close) / len(mine), 2)
                            if mine else 0.0)}


def _default_player() -> Any:
    from .chess_arena import SearchPlayer

    return SearchPlayer(depth=PLAY_DEPTH, trace=True)


def play_locally(games: int = 1, depth: int = PLAY_DEPTH,
                 judge_depth: int = LIVE_JUDGE_DEPTH, pause: float = 0.35,
                 record: bool = True, stop: Any = None,
                 quiet: bool = False, seed: Optional[int] = None,
                 max_plies: int = 200) -> List[Seat]:
    """MANA against itself, through the same frames the live bot emits.

    Here so that the board, the reasoning panel and the judge can be
    looked at without an account, an upgrade or an opponent -- and so
    that what a person watches before playing on Lichess is the same
    thing they will watch during it, rather than a demo that resembles it.

    `pause` exists only for the watcher: without it a game finishes faster
    than a person can read one panel. It is not a think time and it does
    not touch the search.
    """
    import chess

    from .chess_arena import SearchPlayer

    judge = None
    if judge_depth:
        from . import chess_judge

        judge = chess_judge.Judge(depth=judge_depth)
    note = judge_note(judge_depth)
    # Only the fallback is worth repeating: a judge asked for and quietly
    # replaced is a defect, a judge deliberately not asked for is not.
    degraded = "запасной" in note
    # In quiet mode a working judge says nothing: this runs once per game
    # in the gaps between challenges, and the same sentence every minute
    # is noise. A degraded judge still speaks, every time -- that one is
    # worth repeating.
    events.emit(events.WARNING if degraded else events.STATUS,
                "" if (quiet and not degraded) else note,
                chess={"kind": "ready", "judge": note})
    played: List[Seat] = []
    try:
        for number in range(games):
            if stop is not None and stop.is_set():
                break
            # The seed used to be the index inside the batch, so a batch
            # of forty gave forty different games and a batch of one gave
            # game zero every single time. Three cooldowns in a row wrote
            # three byte-identical games into a record whose reader counts
            # games as independent observations.
            this_seed = (number if seed is None and games > 1
                         else (seed + number if seed is not None
                               else random.SystemRandom().randrange(2 ** 31)))
            seat = Seat(game_id=f"local-{int(time.time())}-{number + 1}",
                        source=LOCAL, opponent="сама с собой", seed=this_seed)
            white = SearchPlayer(depth=depth, trace=True)
            black = SearchPlayer(depth=depth, trace=True)
            rng = random.Random(this_seed)
            board = chess.Board()
            events.emit(events.STATUS,
                        "" if quiet else f"партия {number + 1}/{games}",
                        chess=frame(seat, kind="position"))
            while not board.is_game_over() and board.ply() < max_plies:
                if stop is not None and stop.is_set():
                    break
                player = white if board.turn else black
                move = player.choose(board, rng)
                san = board.san(move)
                ply = board.ply() + 1
                # White is MANA's judged side, the same convention the
                # baseline was measured under. Both sides are the same
                # player; judging both would double-count one search.
                # Both sides. Two searches run and only one was written
                # down, so every property about the search had zero
                # measurable games in the one world where both players
                # are MANA and the comparison means something.
                thought, judged = {}, {}
                trace = player.thoughts[-1].as_dict() if player.thoughts else {}
                if trace:
                    trace["ours"] = bool(board.turn)
                    seat.thoughts.append(trace)
                if board.turn:
                    thought = trace
                    if judge is not None:
                        judged = judge.judge_move(board, move, ply).as_dict()
                        seat.judged.append(judged)
                judged_side = bool(thought)
                board.push(move)
                seat.moves.append(move.uci())
                # Only the judged side is reported as a move: the log is
                # a list of decisions with a margin and a cost beside
                # each, and an entry with neither is a blank row that
                # reads as a defect. The board still advances on the
                # other side's reply, which is what `position` is for --
                # and it makes the window identical in both worlds,
                # where the live bot never sees the opponent think.
                # `quiet` keeps the board updating in the window while the
                # console stays readable: during a cooldown these games are
                # background work, and eighty move lines per game would bury
                # the one line that matters.
                said = "" if quiet else (
                    f"{san}" + (f" (потеря {judged['loss']:.0f})" if judged else ""))
                events.emit(events.STATUS, said,
                            chess=frame(seat,
                                        kind="move" if judged_side else "position",
                                        san=san, thought=thought, judged=judged))
                if pause:
                    time.sleep(pause)
            seat.status = ("mate" if board.is_checkmate()
                           else "draw" if board.is_game_over() else "stopped")
            seat.winner = ("white" if board.is_checkmate() and not board.turn
                           else "black" if board.is_checkmate() else "")
            played.append(seat)
            if record:
                write(seat)
            events.emit(events.STATUS,
                        "" if quiet else
                        f"партия {number + 1}: {seat.status} {seat.winner}".strip(),
                        chess=frame(seat, kind="over"))
    finally:
        if judge is not None:
            judge.close()
    return played


def recorded(limit: int = 0) -> List[Dict[str, Any]]:
    """Every game written down, newest last. Missing file means none."""
    path = games_path()
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue                  # a half-written line; the rest stand
    return rows[-limit:] if limit else rows


def stats(rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """What the record says, per world and never pooled.

    Every count carries its denominator. "23 зевка" is not a fact until
    it says out of how many moves -- the rule `invariants.summarise`
    states for turns and `chess_judge.summarise` for moves.
    """
    rows = recorded() if rows is None else rows
    out: Dict[str, Any] = {"games": len(rows), "by_source": {}}
    for source in (LOCAL, LIVE):
        mine = [row for row in rows if row.get("source", LIVE) == source]
        if not mine:
            continue
        judged = [move for row in mine for move in row.get("judged", [])]
        thoughts = [move for row in mine for move in row.get("thoughts", [])]
        losses = [move.get("loss", 0.0) for move in judged]
        close = [move for move in thoughts if move.get("close_call")]
        results: Dict[str, int] = {}
        for row in mine:
            key = f"{row.get('status', '?')} {row.get('winner', '')}".strip()
            results[key] = results.get(key, 0) + 1
        out["by_source"][source] = {
            "games": len(mine),
            "moves": len(thoughts),
            "judged": len(judged),
            "mistakes": sum(1 for move in judged if move.get("mistake")),
            "blunders": sum(1 for move in judged if move.get("blunder")),
            "mean_loss": round(sum(losses) / len(losses), 1) if losses else 0.0,
            "close_calls": len(close),
            "close_share": (round(len(close) / len(thoughts), 3)
                            if thoughts else 0.0),
            "results": results}
    return out


def rejudge(depth: int = 0, only: str = "", limit: int = 0,
            stale_only: bool = True, both_sides: bool = False,
            on_game: Optional[Callable[[str, str, str], None]] = None
            ) -> Dict[str, Any]:
    """Judge every recorded game again, without playing anything.

    `chess_judge` promised this and did not provide it: twice the record
    has needed it -- once because the engine was in the other data root,
    once because the process holding the old code had been waiting for
    challenges since before the fix -- and both times it took a script
    nobody else had.

    Nothing is replayed and nothing is re-decided. The moves are the
    moves that were played; only the verdict changes, and the record says
    at what depth it was redone.

    A backup is written first, and a game that finished while this ran is
    kept rather than overwritten -- the bot appends to the same file.
    """
    from . import chess_judge

    path = games_path()
    rows = recorded()
    if not rows:
        return {"games": 0, "judged": 0, "path": str(path)}
    depth = int(depth) or chess_judge.JUDGE_DEPTH
    engine = chess_judge.engine_path()
    if not engine:
        return {"games": len(rows), "judged": 0, "path": str(path),
                "error": "Stockfish не найден — пересуживать нечем"}

    wanted = [row for row in rows
              if (not only or str(row.get("source", LIVE)) == only)
              and (row.get("judged_both_sides") is not True if both_sides
                   else (not stale_only
                         or any(j.get("judged_by") == chess_judge.BY_MATERIAL
                                for j in row.get("judged", []))))]
    # Said before the record is touched. A rewrite that leaves no line
    # anywhere is one nobody can attribute afterwards -- which is the
    # position this was written from, after a full re-judge appeared in
    # the record with no caller anybody could name.
    events.emit(events.STATUS,
                f"пересуживаю {min(len(wanted), limit) if limit else len(wanted)}"
                f" из {len(rows)} партий на глубине {depth}: {path}",
                chess={"kind": "rejudge", "path": str(path),
                       "candidates": len(wanted), "limit": limit,
                       "depth": depth, "stale_only": stale_only})
    if not wanted:
        return {"games": len(rows), "judged": 0, "depth": depth,
                "kept_arrivals": 0, "path": str(path), "backup": "",
                "engine": str(engine)}

    backup = path.with_suffix(f".before-depth{depth}.jsonl")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    import chess

    done = 0
    with chess_judge.Judge(depth=depth) as judge:
        for row in rows:
            if only and str(row.get("source", LIVE)) != only:
                continue
            was = sorted({str(j.get("judged_by", "?"))
                          for j in row.get("judged", [])}) or ["-"]
            # `limit` and `stale_only` exist for the bench, which does this
            # in the gaps between games: a whole-record pass would hold a
            # cooldown open long after it ended, and re-judging a game the
            # engine already judged at this depth spends the time for
            # nothing.
            # Two different backlogs, and a run picks one. `both_sides`
            # is a back-fill: it wants games nobody has judged on both
            # sides yet, and skipping the ones already done makes it
            # resumable over an evening instead of all-or-nothing.
            if both_sides:
                if row.get("judged_both_sides"):
                    continue
            elif stale_only and chess_judge.BY_MATERIAL not in was:
                continue
            if limit and done >= limit:
                continue
            us = row.get("us") == "white"
            start = row.get("initial_fen", "startpos")
            board = (chess.Board() if start in ("", "startpos")
                     else chess.Board(start))
            fresh: List[Dict[str, Any]] = []
            for ply, uci in enumerate(row.get("moves", []), start=1):
                try:
                    move = chess.Move.from_uci(uci)
                except ValueError:
                    break
                # `both_sides` is how a stored game becomes symmetric. Live
                # judging stays one-sided because it runs against a clock
                # and the opponent's move would be judged on MANA's own
                # time; here nothing is running, so both cost nothing.
                if both_sides or board.turn == us:
                    scored = judge.judge_move(board, move, ply).as_dict()
                    scored["ours"] = bool(board.turn == us)
                    fresh.append(scored)
                board.push(move)
            row["judged"] = fresh
            row["judged_again_at_depth"] = depth
            row["judged_both_sides"] = bool(both_sides)
            done += 1
            if on_game is not None:
                on_game(str(row.get("game", "")), ", ".join(was), "stockfish")

    # A game may have finished while this ran; the bot appends to the same
    # file, and a whole-file rewrite would drop it.
    seen = {str(row.get("game", "")) for row in rows}
    arrived = [row for row in recorded() if str(row.get("game", "")) not in seen]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows + arrived:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"games": len(rows), "judged": done, "depth": depth,
            "kept_arrivals": len(arrived), "path": str(path),
            "backup": str(backup), "engine": str(engine)}
