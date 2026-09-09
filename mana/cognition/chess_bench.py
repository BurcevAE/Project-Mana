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
                "done": self.done, "needed": self.needed()}

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "Ladder":
        return cls(level=int(row.get("level", FIRST_LEVEL)),
                   streak=int(row.get("streak", 0)),
                   games=int(row.get("games", 0)),
                   wins=int(row.get("wins", 0)),
                   by_level=dict(row.get("by_level", {})),
                   best_streak=dict(row.get("best_streak", {})),
                   started=float(row.get("started", time.time())),
                   done=bool(row.get("done", False)))

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
        self.bot = bot or Bot(client=self.client,
                              policy=Policy(rated=False, max_games=1,
                                            speeds=("blitz", "rapid",
                                                    "classical",
                                                    "correspondence")))
        self._stop = threading.Event()

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
                seat = self._one_game()
                played += 1
                if seat is None:
                    continue
                self._fold(seat)
                if self.gap and not self._stop.is_set():
                    time.sleep(self.gap)
        finally:
            self.bot.stop()
            self.ladder.save()
            self._announce("итог")
        return self.ladder

    def _listen(self) -> None:
        try:
            self.bot.run()
        except Exception as exc:                       # the stream died
            events.emit(events.ERROR, f"поток событий оборвался: {exc}")
            self._stop.set()

    def _one_game(self) -> Optional[Any]:
        """Challenge this level and wait for the game to be over."""
        before = len(self.bot.finished)
        try:
            self.client.challenge_ai(level=self.ladder.level,
                                     clock_limit=CLOCK_LIMIT,
                                     clock_increment=CLOCK_INCREMENT)
        except api.LichessError as exc:
            events.emit(events.WARNING, f"вызов не принят: {exc}")
            time.sleep(max(self.gap, 10.0))
            return None
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

    def _fold(self, seat: Any) -> None:
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
    if ladder.done:
        lines.append("лестница пройдена целиком")
    return "\n".join(lines)
