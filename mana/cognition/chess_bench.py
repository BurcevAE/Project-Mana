"""
mana.cognition.chess_bench — the ladder: play Stockfish until it stops you.

The rule, as asked for
-----------------------
    начать с уровня 1
    10 побед подряд      → следующий уровень
    30 побед подряд на 8 → конец
    Ctrl+C               → конец в любой момент

"Подряд" is taken literally: a loss **or a draw** resets the streak,
because a streak of consecutive wins is what was asked for and a draw is
not a win. Stated here rather than buried in a comparison, since it is
the rule most likely to be argued with.

What this is expected to do, said in advance
----------------------------------------------
Not finish. MANA plays a two-ply search over material and has won one
game in eight against level 3; ten in a row there is an event of about
one in a billion. The ladder will almost certainly stall low.

That is the measurement, not a disappointment. The level it stalls at is
a number about this player that no single game gives, it is comparable
across changes to the player, and it is the one number this bench exists
to produce. Saying so beforehand also removes the temptation to loosen
the rule later so the ladder climbs.

The by-product is the other reason to run it. `chess_findings` needs
thirty games in a world before it will say anything, and the Lichess
world has twelve. A bench that plays unattended produces exactly the
corpus the reader is waiting for.

Levels are conditions, not noise
---------------------------------
A mistake against level 1 and the same mistake against level 8 do not
cost the same, so the level goes into every game record and into the
conditions of every finding drawn from it. Pooling levels would be the
same error as pooling worlds, one layer down.

What it does not do
--------------------
It does not adapt. Nothing about the player changes between games, at any
level. The ladder is a measuring instrument; changing the thing being
measured while it measures is how a bench stops being one.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import events
from ..net import lichess as api

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

FIRST_LEVEL = 1
LAST_LEVEL = 8

#: Consecutive wins that promote to the next level.
WINS_TO_ADVANCE = 10

#: Consecutive wins at the top that end the run.
WINS_TO_FINISH = 30

#: Clock for the games the bench starts. Fast, because a two-ply search
#: answers in milliseconds and the judge runs on the opponent's time --
#: but not bullet, where a measurement becomes one about latency.
CLOCK_LIMIT = 180
CLOCK_INCREMENT = 2

#: A pause between games. Not a rate limit -- Lichess states its own --
#: but a bench that hammers an endpoint unattended for hours is a bad
#: neighbour, and the cost of three seconds against a five-minute game
#: is nothing.
GAP_SECONDS = 3.0

#: How long one game may take before the bench stops waiting for it. A
#: stuck stream would otherwise hold the ladder forever with no sign.
GAME_LIMIT = 1200.0

#: How long to wait for a challenge to turn into a game.
START_LIMIT = 90.0

#: How long to wait before trying the event stream again after it fails,
#: and how many failures in a row before the bench stops asking. The
#: local half never stops for either: a network that is unavailable is a
#: reason to play by yourself, not a reason to sit still.
LISTEN_RETRY = 20.0
LISTEN_GIVE_UP = 20


#: What to do when Lichess refuses to start another game. It limits how
#: often a client may create one, and each attempt costs two requests to
#: the endpoint that just refused -- so the wait grows instead of being a
#: fixed ten seconds, which was answering rate limiting by producing more
#: of what caused it.
REFUSED_WAIT = 60.0
REFUSED_WAIT_MAX = 900.0

#: How often the run wakes during a cooldown: to report the time left and
#: to do one unit of local work. Short enough that a cooldown ending is
#: noticed promptly, long enough not to be a spin loop.
TICK = 5.0

#: Games re-judged in one unit of local work. Bounded so a cooldown that
#: ends is not held open by work that could have waited.
REJUDGE_PER_TICK = 2

#: A self-play game is only started when at least this much cooldown is
#: left. Starting one with ten seconds to go would hold the next
#: challenge back for the sake of work that had no deadline.
SELF_PLAY_FLOOR = 25.0

#: What happened on the last game, in the ladder's own terms.
ADVANCED = "advanced"
FINISHED = "finished"
BROKEN = "broken"
CLIMBING = "climbing"


def state_path() -> Path:
    from ..paths import resolve_data_path

    root = Path(resolve_data_path("lichess"))
    root.mkdir(parents=True, exist_ok=True)
    return root / "bench.json"


@dataclass
class Ladder:
    """Where the run stands, and the whole of the promotion rule.

    Kept as data rather than as control flow inside the loop, because
    "what did it need and what did it get" is a question about a run that
    its record should answer without replaying it.
    """
    level: int = FIRST_LEVEL
    streak: int = 0
    games: int = 0
    wins: int = 0
    #: Per level: games played and games won, so a stall is legible.
    by_level: Dict[str, Dict[str, int]] = field(default_factory=dict)
    best_streak: Dict[str, int] = field(default_factory=dict)
    started: float = field(default_factory=time.time)
    done: bool = False
    #: When Lichess may next be asked for a game, on the local clock, and
    #: how many refusals in a row led here. Saved because a cooldown that
    #: lives only in a process is not a cooldown: restarting reset both,
    #: so every restart asked again at once and began the escalation
    #: afresh -- the very behaviour the escalation exists to prevent.
    next_request_at: float = 0.0
    refused: int = 0

    def cooldown_left(self) -> float:
        return max(0.0, self.next_request_at - time.time())

    def needed(self) -> int:
        return WINS_TO_FINISH if self.level >= LAST_LEVEL else WINS_TO_ADVANCE

    def note(self, won: bool, drawn: bool = False) -> str:
        """Fold one result in and say what it changed.

        A draw is not a win: the streak asked for is of consecutive wins,
        and treating a draw as neutral would let a ladder climb without
        ever winning ten of anything.
        """
        key = str(self.level)
        row = self.by_level.setdefault(key, {"games": 0, "wins": 0})
        row["games"] += 1
        self.games += 1
        if won:
            row["wins"] += 1
            self.wins += 1
            self.streak += 1
            self.best_streak[key] = max(self.best_streak.get(key, 0), self.streak)
            if self.streak >= self.needed():
                if self.level >= LAST_LEVEL:
                    self.done = True
                    return FINISHED
                self.level += 1
                self.streak = 0
                return ADVANCED
            return CLIMBING
        self.streak = 0
        return BROKEN

    def describe(self) -> str:
        row = self.by_level.get(str(self.level), {"games": 0, "wins": 0})
        return (f"уровень {self.level}, подряд {self.streak}/{self.needed()}"
                f" — на этом уровне {row['wins']} из {row['games']}")

    def as_dict(self) -> Dict[str, Any]:
        return {"level": self.level, "streak": self.streak, "games": self.games,
                "wins": self.wins, "by_level": self.by_level,
                "best_streak": self.best_streak, "started": self.started,
                "done": self.done, "needed": self.needed(),
                "next_request_at": self.next_request_at,
                "refused": self.refused,
                "cooldown_left": round(self.cooldown_left(), 1)}

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "Ladder":
        return cls(level=int(row.get("level", FIRST_LEVEL)),
                   streak=int(row.get("streak", 0)),
                   games=int(row.get("games", 0)),
                   wins=int(row.get("wins", 0)),
                   by_level=dict(row.get("by_level", {})),
                   best_streak=dict(row.get("best_streak", {})),
                   started=float(row.get("started", time.time())),
                   done=bool(row.get("done", False)),
                   next_request_at=float(row.get("next_request_at", 0.0)),
                   refused=int(row.get("refused", 0)))

    def save(self) -> None:
        try:
            state_path().write_text(
                json.dumps(self.as_dict(), ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            events.emit(events.WARNING, f"состояние стенда не записано: {exc}")

    @classmethod
    def load(cls) -> "Ladder":
        """Resume where the last run stopped.

        A bench interrupted at level 3 and restarted at level 1 would
        spend its first hour re-measuring what it already measured, and
        the numbers from that hour would be indistinguishable from new
        ones.
        """
        path = state_path()
        if not path.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return cls()


class Bench:
    """Challenge, wait, judge the result, decide the next challenge.

    Built on the ordinary `Bot`: the bench decides only who to play, and
    the playing, the trace, the judge and the record are the same ones a
    human-started game goes through. A bench with its own game loop would
    be measuring a code path nobody uses.
    """

    def __init__(self, client: Optional[api.Lichess] = None,
                 bot: Optional[Any] = None, ladder: Optional[Ladder] = None,
                 gap: float = GAP_SECONDS) -> None:
        from .chess_bot import Bot, Policy

        self.client = client or api.Lichess()
        self.ladder = ladder if ladder is not None else Ladder.load()
        self.gap = float(gap)
        #: Consecutive refusals, so the wait can grow with them. Restored
        #: from the saved state, so a restart continues the escalation
        #: rather than beginning it again at sixty seconds.
        self._refused = self.ladder.refused
        self.bot = bot or Bot(client=self.client,
                              policy=Policy(rated=False, max_games=1,
                                            speeds=("blitz", "rapid",
                                                    "classical",
                                                    "correspondence")))
        self._stop = threading.Event()
        #: A wait carried over from a previous run is applied before
        #: anything is asked of Lichess.
        left = self.ladder.cooldown_left()
        if left > 0:
            self._hold(left)
        #: What `think()` said last time, so an unchanged answer is not
        #: repeated every tick. Twelve identical lines is noise pretending
        #: to be progress.
        self._said = ""
        #: Games already reduced to two rows, by id. Reducing costs a board
        #: replay; measuring the reduced rows is arithmetic, so a
        #: conclusion after every game is affordable only with this.
        self._sides: Dict[str, Any] = {}
        #: The verdicts as of the last pass, so only what moved is said.
        self._verdicts: Dict[str, str] = {}

    def stop(self) -> None:
        self._stop.set()
        self.bot.stop()

    # ---------- the run ----------

    def run(self, limit: int = 0) -> Ladder:
        """Climb until the rule ends it, or until stopped.

        `limit` caps the number of games, for a test or a short run. Zero
        means the rule decides, which is what was asked for.
        """
        played = 0
        self._announce("старт")
        listening = threading.Thread(target=self._listen, name="MANA-bench",
                                     daemon=True)
        listening.start()
        try:
            while not self._stop.is_set() and not self.ladder.done:
                if limit and played >= limit:
                    break
                left = self._cooldown_left()
                if left > 0:
                    # The network is the one thing that may not be touched
                    # now, and it is exactly when there is most to do
                    # locally. Lichess is a slow external sensor, not a
                    # lock on MANA.
                    self._while_waiting(left)
                    continue
                seat = self._one_game()
                played += 1
                if seat is None:
                    continue
                self._fold(seat)
                if self.gap and not self._stop.is_set():
                    self._sleep(self.gap)
        finally:
            self.bot.stop()
            self.ladder.save()
            self._announce("итог")
        return self.ladder

    def _listen(self) -> None:
        """Keep the event stream up, and never stop the run by failing.

        `Bot.run()` opens with an account check, and a cooldown set by a
        refused challenge refuses that too. Treating it as fatal stopped
        the bench, which then could not even play by itself -- the
        network half silencing the half that needs no network.
        """
        failures = 0
        while not self._stop.is_set() and failures < LISTEN_GIVE_UP:
            left = self._cooldown_left()
            if left > 0:
                # Not a failure: a state with a known end. Wait it out
                # without counting it against the retries.
                self._stop.wait(min(left, LISTEN_RETRY))
                continue
            try:
                self.bot.run()
                failures = 0
            except api.RateLimited as exc:
                self._stop.wait(min(max(1.0, exc.retry_after), LISTEN_RETRY))
            except Exception as exc:
                failures += 1
                events.emit(events.WARNING,
                            f"поток событий оборвался ({exc}); попытка "
                            f"{failures} из {LISTEN_GIVE_UP} через "
                            f"{LISTEN_RETRY:.0f}с")
                self._stop.wait(LISTEN_RETRY)
        if failures >= LISTEN_GIVE_UP and not self._stop.is_set():
            events.emit(events.ERROR,
                        "поток событий не поднимается; партий на lichess не "
                        "будет, играю с собой")

    def _one_game(self) -> Optional[Any]:
        """Challenge this level and wait for the game to be over."""
        before = len(self.bot.finished)
        try:
            self.client.challenge_ai(level=self.ladder.level,
                                     clock_limit=CLOCK_LIMIT,
                                     clock_increment=CLOCK_INCREMENT)
        except api.RateLimited:
            # The client has already set its own cooldown; the bench adds
            # its escalation on top. Lichess warns that the limits are
            # composite, so a second refusal after a full minute is not an
            # error to retry through -- it is a longer wait.
            self._refused += 1
            wait = min(REFUSED_WAIT * (2 ** (self._refused - 1)), REFUSED_WAIT_MAX)
            left = self._hold(wait)
            events.emit(events.WARNING,
                        f"Lichess не даёт начать партию. Отказ подряд "
                        f"{self._refused}, сеть не трогаю {left:.0f}с")
            return None
        except api.LichessError as exc:
            self._refused += 1
            left = self._hold(REFUSED_WAIT)
            events.emit(events.WARNING,
                        f"вызов не принят ({exc}); сеть не трогаю {left:.0f}с")
            return None
        self._refused = 0
        self.ladder.next_request_at = 0.0
        self.ladder.refused = 0
        self.ladder.save()
        events.emit(events.STATUS,
                    f"вызвала Stockfish уровня {self.ladder.level} "
                    f"({self.ladder.describe()})",
                    chess={"kind": "bench", "ladder": self.ladder.as_dict()})

        waited = 0.0
        while waited < GAME_LIMIT and not self._stop.is_set():
            if len(self.bot.finished) > before:
                return self.bot.finished[-1]
            time.sleep(0.5)
            waited += 0.5
            if waited > START_LIMIT and not self.bot.seats and waited % 30 < 0.5:
                events.emit(events.WARNING,
                            f"партия не началась за {waited:.0f}с — жду дальше")
        # Once more before giving up. Ctrl+C arriving while this loop was
        # asleep used to discard a game that had already finished: the
        # record kept it and the ladder did not, so the two disagreed
        # about what had been played. Found by the test that stops a run
        # mid-game.
        if len(self.bot.finished) > before:
            return self.bot.finished[-1]
        if not self._stop.is_set():
            events.emit(events.WARNING,
                        f"партия не кончилась за {GAME_LIMIT:.0f}с — бросаю ждать")
        return None

    def _sleep(self, seconds: float) -> None:
        """Wait, but wake at once if the run is stopped.

        A fifteen-minute backoff that ignores Ctrl+C is a bench nobody
        can stop, which is one of the two ways this was asked to end.
        """
        self._stop.wait(seconds)

    def _cooldown_left(self) -> float:
        """Seconds before the network may be touched. Local arithmetic."""
        left = getattr(self.client, "cooldown_left", None)
        return float(left()) if callable(left) else 0.0

    def _hold(self, seconds: float, remember: bool = True) -> float:
        hold = getattr(self.client, "hold", None)
        left = float(hold(seconds)) if callable(hold) else float(seconds)
        if remember:
            self.ladder.next_request_at = time.time() + left
            self.ladder.refused = self._refused
            self.ladder.save()
        return left

    def _while_waiting(self, left: float) -> None:
        """Do one unit of local work, report it if it is new, wake early.

        Not a sleep: a cooldown spent idle is a network limit turned into
        a stop for everything MANA could be doing. And not a broadcast
        either -- the same sentence every five seconds is noise pretending
        to be progress, so an unchanged answer is emitted for the window
        and kept off the console.
        """
        # As much work as the tick allows, not one unit per tick. A
        # self-play game costs half a second now that no engine is asked,
        # and waiting five seconds between them would spend a
        # fifteen-minute cooldown on a fraction of the games it could hold
        # -- which is the whole reason the judge was switched off.
        until = time.time() + min(TICK, max(0.5, left))
        did = self.think(budget=left)
        while (time.time() < until and not self._stop.is_set()
               and self._cooldown_left() > 0):
            did = self.think(budget=self._cooldown_left())
        fresh = did != self._said
        self._said = did
        events.emit(events.STATUS,
                    f"Lichess: охлаждение, ещё {left:.0f}с. MANA: {did}"
                    if fresh else "",
                    chess={"kind": "bench", "ladder": self.ladder.as_dict(),
                           "cooldown": round(left, 1), "doing": did})
        self._stop.wait(max(0.0, until - time.time()))

    def think(self, budget: float = 0.0) -> str:
        """One bounded unit of work on the record. Never touches the network.

        Ordered by what the record needs, not by a schedule:

            судимые запасным   → пересудить (их потери в другой шкале)
            хватает времени    → сыграть партию с собой (новое наблюдение)
            иначе              → перечитать записи

        Re-reading an unchanged corpus is not work: the same sixty-three
        games cannot say anything on the second pass that they did not say
        on the first. What is short is games, and a self-play game adds
        one observation to the local world every time.

        The player is not touched. The ladder is a number about *this*
        player, and two copies of a player that changed mid-run measure
        something nobody can name.
        """
        from . import chess_bot, chess_findings, chess_judge

        try:
            rows = chess_bot.recorded()
        except Exception as exc:                       # a half-written file
            return f"записи не прочитались ({type(exc).__name__})"

        stale = [row for row in rows
                 if any(j.get("judged_by") == chess_judge.BY_MATERIAL
                        for j in row.get("judged", []))]
        if stale and chess_judge.engine_path():
            done = chess_bot.rejudge(depth=chess_judge.JUDGE_DEPTH,
                                     limit=REJUDGE_PER_TICK)
            if done.get("judged"):
                return (f"пересудила {done['judged']} "
                        f"(осталось {max(0, len(stale) - done['judged'])})")

        if budget >= SELF_PLAY_FLOOR:
            played = chess_bot.play_locally(games=1,
                                            judge_depth=chess_bot.LIVE_JUDGE_DEPTH,
                                            pause=0.0, quiet=True,
                                            stop=self._stop)
            if played:
                local = sum(1 for row in rows
                            if str(row.get("source", "")) == chess_bot.LOCAL) + 1
                self._report_conclusions()
                row = chess_bot.summarise(played[0])
                judged = ("" if row["blunders"] is None
                          else f", зевков {row['blunders']}")
                return (f"сыграла с собой: {row['moves']} ходов{judged}, "
                        f"жребием {row['close_share']:.0%}, исход "
                        f"{played[0].status} {played[0].winner}".rstrip()
                        + f" (партий с собой {local})")

        if not rows:
            return "записей пока нет"
        found = chess_findings.look(rows)
        settled = [f for f in found if f.verdict != "NOT_EVALUATED"]
        if settled:
            accepted = [f for f in settled if f.verdict == "ACCEPTED"]
            return (f"прочитала {len(rows)} партий: "
                    f"{len(accepted)} классов дороже остальных, "
                    f"{len(settled)} вопросов закрыто")
        short = chess_findings.MIN_PAIRED_TRIALS - max(
            (f.measurement["trials"] for f in found), default=0)
        return (f"прочитала {len(rows)} партий: до вывода не хватает "
                f"{max(0, short)}")

    def conclude(self) -> List[str]:
        """Re-measure what the record says, and report only what moved.

        Findings, not laws. Everything here is a regularity in what has
        already happened; a law is earned by changing something and
        measuring the change, and calling a correlation one would be the
        most expensive lie this system could tell itself.
        """
        from . import chess_bot, chess_outcome

        try:
            games = chess_bot.recorded()
        except Exception:
            return []
        fresh = 0
        for game in games:
            key = str(game.get("game", ""))
            if not key or key in self._sides:
                continue
            reduced = chess_outcome.reduce_game(game)
            if reduced is not None:
                self._sides[key] = reduced
                fresh += 1
        if not self._sides:
            return []
        out = chess_outcome.look(list(self._sides.values()))
        said = chess_outcome.changes(self._verdicts, out)
        self._verdicts = chess_outcome.state(out)
        return said

    def _report_conclusions(self) -> None:
        """Say what moved, and on the first pass say only what is settled.

        A fresh run has every verdict as "new", and fifty-two lines of
        mostly "не измерено" buries the two that carry a result. After
        that every line is a change, and a change is always worth saying.
        """
        first = not self._verdicts
        lines = self.conclude()
        if first:
            unmeasured = [line for line in lines if "NOT_EVALUATED" in line]
            lines = [line for line in lines if "NOT_EVALUATED" not in line]
            if unmeasured:
                lines.append(f"ещё не измерено: {len(unmeasured)} "
                             f"(наблюдение — это партия, а не ход)")
        for line in lines:
            events.emit(events.STATUS, f"вывод: {line}",
                        chess={"kind": "finding", "text": line})

    def _fold(self, seat: Any) -> None:
        # A game that never reached a result is not a drawn game -- it is
        # no evidence at all, and folding it in as a draw broke a real
        # streak. "Unmeasured is not zero" is the rule everywhere else in
        # this project; the ladder had been the exception.
        if seat.status in ("started", "created", ""):
            events.emit(events.WARNING,
                        f"партия {seat.game_id} без результата "
                        f"({seat.status or 'нет статуса'}) — в счёт не идёт")
            return
        won = bool(seat.winner) and seat.winner == ("white" if seat.us else "black")
        drawn = not seat.winner
        what = self.ladder.note(won, drawn)
        self.ladder.save()
        outcome = "победа" if won else ("ничья" if drawn else "поражение")
        said = {ADVANCED: f"→ уровень {self.ladder.level}",
                FINISHED: "→ лестница пройдена",
                BROKEN: "серия сброшена",
                CLIMBING: ""}[what]
        events.emit(events.STATUS,
                    f"{outcome}. {self.ladder.describe()} {said}".strip(),
                    chess={"kind": "bench", "ladder": self.ladder.as_dict(),
                           "outcome": outcome, "event": what})
        self._report_conclusions()

    def _announce(self, when: str) -> None:
        events.emit(events.STATUS,
                    f"стенд ({when}): {self.ladder.describe()}, "
                    f"всего {self.ladder.wins} из {self.ladder.games}",
                    chess={"kind": "bench", "ladder": self.ladder.as_dict()})


def summarise(ladder: Ladder) -> str:
    """The one number this bench exists to produce, with its denominator."""
    lines = [f"уровень, на котором остановилась: {ladder.level}",
             f"всего партий {ladder.games}, побед {ladder.wins}"
             + (f" ({ladder.wins / ladder.games:.0%})" if ladder.games else "")]
    for level in sorted(ladder.by_level, key=int):
        row = ladder.by_level[level]
        best = ladder.best_streak.get(level, 0)
        need = WINS_TO_FINISH if int(level) >= LAST_LEVEL else WINS_TO_ADVANCE
        lines.append(f"  уровень {level}: {row['wins']} из {row['games']}"
                     f", лучшая серия {best} из нужных {need}")
    left = ladder.cooldown_left()
    if left > 0:
        when = time.strftime("%H:%M:%S", time.localtime(ladder.next_request_at))
        lines.append(f"Lichess: следующий запрос не раньше {when} "
                     f"(через {left:.0f}с), отказов подряд {ladder.refused}")
        lines.append("  перезапуск не сбрасывает это ожидание")
    else:
        lines.append("Lichess: можно запускать")
    if ladder.done:
        lines.append("лестница пройдена целиком")
    return "\n".join(lines)
