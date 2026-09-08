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

#: Search depth for play. Two is what Stage 0 was measured at; raising it
#: silently would make the next measurement a comparison against nothing.
PLAY_DEPTH = 2

#: Judge depth during a game. Lower than `chess_judge.JUDGE_DEPTH`
#: because this one runs while a clock is going; the games are stored, so
#: the same moves can be judged again deeper afterwards without being
#: played again.
LIVE_JUDGE_DEPTH = 8

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
    started: float = field(default_factory=time.time)

    @property
    def url(self) -> str:
        return f"https://lichess.org/{self.game_id}"

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
        return {"game": self.game_id, "url": self.url,
                "us": "white" if self.us else "black",
                "opponent": self.opponent, "initial_fen": self.initial_fen,
                "moves": list(self.moves), "status": self.status,
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
        events.emit(events.STATUS, f"Lichess: играю как {self.username}",
                    chess={"kind": "ready", "user": self.username,
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
        """One picture of the game, and everything a watcher needs to
        read it. `last_uci` travels beside `last` because SAN says
        "Qxd6+" and a board needs to know which two squares to light."""
        board = seat.board()
        return {"kind": kind, "game": seat.game_id, "url": seat.url,
                "fen": board.fen(), "us": "white" if seat.us else "black",
                "opponent": seat.opponent, "ply": board.ply(),
                "turn": "white" if board.turn else "black",
                "status": seat.status, "winner": seat.winner,
                "last": san, "last_uci": seat.moves[-1] if seat.moves else "",
                "thought": thought or {}, "judged": judged or {},
                "summary": summarise(seat)}

    def _write(self, seat: Seat) -> None:
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
    close = [row for row in seat.thoughts if row.get("close_call")]
    return {"moves": len(seat.thoughts),
            "judged": len(seat.judged),
            "mistakes": sum(1 for row in seat.judged if row.get("mistake")),
            "blunders": sum(1 for row in seat.judged if row.get("blunder")),
            "mean_loss": round(sum(losses) / len(losses), 1) if losses else 0.0,
            "close_calls": len(close),
            "close_share": (round(len(close) / len(seat.thoughts), 2)
                            if seat.thoughts else 0.0)}


def _default_player() -> Any:
    from .chess_arena import SearchPlayer

    return SearchPlayer(depth=PLAY_DEPTH, trace=True)
