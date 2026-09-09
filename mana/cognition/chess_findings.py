"""
mana.cognition.chess_findings — the games become findings, or say why not.

The gap this closes
--------------------
The chess apparatus recorded everything and concluded nothing. Every game
carried a trace and a judgement, and a person had to read the file to
learn anything from it -- which is this project's oldest failure with a
new subject: built, and connected to nothing. `findings.py` exists so that
"we tried this and here is what came of it" is a thing a machine can read,
and until now no chess result reached it.

What is asked
--------------
    вопрос:  какой класс собственных ходов дороже остальных
    подход:  один объявленный признак хода
    ответ:   ACCEPTED / REJECTED / NOT_EVALUATED

Not "is MANA good at chess". A rating moves for reasons that have nothing
to do with what MANA learned. "Which of my own moves cost more than my
other moves" is answerable from inside one game, and it points at a
repair.

Three disciplines, and without them this would be self-deception
-----------------------------------------------------------------
**An observation is a game, not a move.** Eighty-five moves from three
games are not eighty-five observations: they share an opponent, an opening
and every earlier mistake in that game. `chess_arena` already says this
about its corpus -- "независимых наблюдений здесь столько же, сколько
партий" -- and counting moves would produce confident intervals around
nothing. So each game contributes exactly one number.

**Paired inside the game.** For each game, the mean loss of the moves in
the class against the mean loss of that same game's other moves. A game
where MANA was simply outclassed moves both halves together and therefore
says nothing about the class, which is correct: it wasn't evidence about
the class.

**Every declared class is measured, every time.** The vocabulary below is
fixed and each entry is answered on every pass. Measuring ten and
reporting the one that came out is how a coincidence becomes a finding,
and the interval is widened for the number of questions asked (Bonferroni)
rather than left to be discounted by a reader who may not know how many
were asked.

Provenance of the vocabulary, stated because it matters
--------------------------------------------------------
`capture_moderate` is not an independent discovery. It came from looking
at three real games in one session -- captures at a margin near a pawn
cost 579 on average against 8 for a large margin -- so it enters here as a
hypothesis under test, on games it was not derived from. The others are
mechanical properties of the search that were declared before anything was
looked at. A class mined from the data it is then measured on is the one
mistake this file cannot make and still be worth having.

The classes are not independent questions
------------------------------------------
`no_margin` and `margin_large` cut the same axis, and each class is
measured against *everything else*, which includes the others. So an
ACCEPTED on one and a mirror-image result on another can be one fact seen
twice rather than two facts. The Bonferroni correction assumes independent
questions and is therefore conservative on the count and not a licence to
read nine separate discoveries off one table. Said here because the first
real run produced exactly that pair: coin-toss moves more expensive by 54,
large-margin moves cheaper by 111.

What a world does not carry over
---------------------------------
A class rejected in one world is not refuted in the other. Self-play has
an opponent that shares MANA's blind spot and will not punish a greedy
capture; Stockfish will. The first run rejected `capture_moderate` on
forty self-play games while the three games it was mined from were
against Stockfish -- which is a reason to measure it there, not a reason
to drop it. This is why the world is part of `conditions` and why
`look()` never pools.

What this does not do
----------------------
It does not change how MANA plays. A finding is a prior about where to
look, and turning one into a change is a separate experiment with its own
splits -- `cognition/trials.py` and the discipline already built there.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core.gates import ACCEPTED, MIN_PAIRED_TRIALS, NOT_EVALUATED, REJECTED
from . import findings as ledger_mod

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: The one question every finding here answers. Fixed text: the ledger
#: groups by it, and a reworded question is a different question.
QUESTION = "какой класс собственных ходов дороже остальных"

#: Two-sided confidence before correction.
ALPHA = 0.05

#: A game contributes a paired difference only if it has at least this
#: many moves on both sides. One move against one move is a difference
#: between two numbers, not a comparison of two behaviours.
MIN_MOVES_PER_SIDE = 2

#: Bands on the search's own margin, in centipawns. The middle one is
#: where a two-ply search believes it has won a pawn -- the belief it
#: cannot check, because it sees the capture and the recapture and not
#: the third ply.
SMALL_MARGIN = 25.0
LARGE_MARGIN = 300.0

#: Where the endgame starts, in plies. A round number, and stated rather
#: than tuned: a boundary chosen to make a result come out is not a
#: boundary, and this one is fixed before the measurement.
ENDGAME_PLY = 51


@dataclass(frozen=True)
class Partition:
    """One mechanical property of a move, and where it came from.

    `mined` is the honest half. A class taken out of the data and then
    measured on that same data is guaranteed to look real, so the record
    says which entries were declared in advance and which were not.
    """
    name: str
    what: str
    test: Callable[[Dict[str, Any], Dict[str, Any]], bool]
    mined: bool = False


def _margin(thought: Dict[str, Any]) -> float:
    return float(thought.get("margin", 0.0))


def _capture(judged: Dict[str, Any]) -> bool:
    # SAN marks every capture with 'x', en passant included, and no file
    # or piece letter is an x -- so the notation is enough and no board
    # has to be rebuilt to ask.
    return "x" in str(judged.get("move", ""))


def _check(judged: Dict[str, Any]) -> bool:
    move = str(judged.get("move", ""))
    return move.endswith("+") or move.endswith("#")


#: The vocabulary. Fixed, and answered in full on every pass.
PARTITIONS: Tuple[Partition, ...] = (
    Partition("forced", "ход был единственным легальным",
              lambda t, j: bool(t.get("forced"))),
    Partition("no_margin", "отрыва нет — выбрал жребий",
              lambda t, j: _margin(t) < SMALL_MARGIN and not t.get("forced")),
    Partition("margin_moderate", f"отрыв {SMALL_MARGIN:.0f}-{LARGE_MARGIN:.0f}",
              lambda t, j: SMALL_MARGIN <= _margin(t) < LARGE_MARGIN),
    Partition("margin_large", f"отрыв {LARGE_MARGIN:.0f}+",
              lambda t, j: _margin(t) >= LARGE_MARGIN),
    Partition("capture", "взятие", lambda t, j: _capture(j)),
    Partition("check", "шах", lambda t, j: _check(j)),
    Partition("endgame", f"полуход {ENDGAME_PLY}+",
              lambda t, j: int(t.get("ply", 0)) >= ENDGAME_PLY),
    Partition("many_tied", "десять и больше ходов делят лучшую оценку",
              lambda t, j: len([1 for _, score in t.get("considered", [])
                                if score == max((s for _, s in t["considered"]),
                                                default=0)]) >= 10),
    # Not declared in advance -- see the module docstring. Kept in the
    # vocabulary so it is measured under the same rules as everything
    # else, and marked so nobody reads it as an independent discovery.
    Partition("capture_moderate", "взятие с отрывом 25-300",
              lambda t, j: _capture(j) and SMALL_MARGIN <= _margin(t) < LARGE_MARGIN,
              mined=True),
)


def _z(alpha: float) -> float:
    """Two-sided normal quantile. Deterministic on purpose.

    A bootstrap would be a better fit for a clamped, heavy-tailed loss,
    and it would also mean a finding whose interval moves when the same
    record is read twice. A record that does not reproduce is not a
    record, so: a closed form, and the assumption said out loud rather
    than hidden in a seed.
    """
    from statistics import NormalDist

    return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def pairs(games: Sequence[Dict[str, Any]], part: Partition
          ) -> List[Tuple[str, float, int, int]]:
    """One difference per game: other moves minus this class's moves.

    Positive means the class was cheaper than the rest of that game, so
    the sign reads the way `findings.classify` expects -- an interval
    entirely below zero says this class is WORSE than the alternative.

    A game with too few moves on either side contributes nothing, and is
    absent rather than counted as zero. "Unmeasured is not zero" is the
    rule this project states everywhere else.
    """
    out: List[Tuple[str, float, int, int]] = []
    for game in games:
        by_ply = {row.get("ply"): row for row in game.get("judged", [])}
        inside: List[float] = []
        outside: List[float] = []
        for thought in game.get("thoughts", []):
            judged = by_ply.get(thought.get("ply"))
            if judged is None:
                continue                      # no verdict for this move
            loss = float(judged.get("loss", 0.0))
            (inside if part.test(thought, judged) else outside).append(loss)
        if len(inside) < MIN_MOVES_PER_SIDE or len(outside) < MIN_MOVES_PER_SIDE:
            continue
        out.append((str(game.get("game", "")),
                    statistics.mean(outside) - statistics.mean(inside),
                    len(inside), len(outside)))
    return out


def measure(games: Sequence[Dict[str, Any]], part: Partition,
            questions: int = 0) -> Dict[str, Any]:
    """The paired measurement for one class, with its denominator.

    `questions` widens the interval for the number of classes asked
    about, so the correction is applied where the number is known rather
    than left to a reader who may not know how many were asked.
    """
    rows = pairs(games, part)
    trials = len(rows)
    alpha = ALPHA / max(1, questions or len(PARTITIONS))
    if trials < 2:
        interval = (0.0, 0.0)
        effect = rows[0][1] if rows else 0.0
    else:
        values = [row[1] for row in rows]
        effect = statistics.mean(values)
        spread = statistics.stdev(values) / (trials ** 0.5)
        half = _z(alpha) * spread
        interval = (effect - half, effect + half)
    return ledger_mod.measurement_of(
        trials=trials, interval=interval, null=0.0,
        effect=round(effect, 1),
        alpha=round(alpha, 5),
        questions_asked=questions or len(PARTITIONS),
        moves_in_class=sum(row[2] for row in rows),
        moves_outside=sum(row[3] for row in rows),
        games_contributing=[row[0] for row in rows])


def verdict_for(measurement: Dict[str, Any]) -> str:
    """ACCEPTED means the class is measurably more expensive.

    Three states, because "we measured it and it is not so" and "we could
    not measure it" license different next moves: the first closes a
    question, the second says how many more games would open it.
    """
    if int(measurement["trials"]) < MIN_PAIRED_TRIALS:
        return NOT_EVALUATED
    low, high = measurement["interval"]
    return ACCEPTED if high < 0.0 else REJECTED


def conditions(games: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """What would make one of these findings stale.

    The judge is in here because a loss judged by the material fallback
    and a loss judged by Stockfish are not the same quantity, and three
    real games have already been judged by the wrong one without anything
    noticing.
    """
    from ..version import PRODUCT_VERSION

    judges = sorted({str(row.get("judged_by", "?"))
                     for game in games for row in game.get("judged", [])})
    sources = sorted({str(game.get("source", "?")) for game in games})
    depths = sorted({int(t.get("depth", 0))
                     for game in games for t in game.get("thoughts", [])})
    # The opponent's strength changes what a mistake costs, so a finding
    # drawn from a mixed-level run says which levels it was drawn from
    # rather than averaging over them silently.
    levels = sorted({int(game.get("level", 0) or 0) for game in games})
    return {"games": len(games), "judged_by": judges, "worlds": sources,
            "levels": levels,
            "search_depth": depths, "version": PRODUCT_VERSION,
            "questions_asked": len(PARTITIONS)}


def look(games: Optional[Sequence[Dict[str, Any]]] = None,
         ledger: Optional[Any] = None, record: bool = True
         ) -> List[ledger_mod.Finding]:
    """Read the record, answer every declared class, write the answers.

    Returns findings in the order of the vocabulary, not sorted by
    result: sorting by effect is the reading habit that turns a table of
    measurements into a headline.
    """
    from . import chess_bot

    games = list(chess_bot.recorded()) if games is None else list(games)
    book = ledger if ledger is not None else ledger_mod.Ledger()
    # One pass per world, never across them. A game against itself has an
    # opponent that shares MANA's evaluation and its blind spots, and a
    # game against a stranger does not; pooling them would answer a
    # question about neither. The world is in `conditions`, so the two
    # never collide in the ledger either.
    worlds = {}
    for game in games:
        worlds.setdefault(str(game.get("source", chess_bot.LIVE)), []).append(game)

    out = []
    for world in sorted(worlds):
        where = conditions(worlds[world])
        for part in PARTITIONS:
            approach = {"partition": part.name, "what": part.what,
                        "mined_from_data": part.mined}
            measurement = measure(worlds[world], part, questions=len(PARTITIONS))
            finding = ledger_mod.Finding(
                question=QUESTION,
                approach=approach,
                verdict=verdict_for(measurement),
                measurement=measurement,
                conditions=where,
                note=_note(part, measurement),
                version=where["version"])
            out.append(finding)
            if not record:
                continue
            # Asked before writing, not after. The same games measured
            # again produce the same numbers, and a second identical row
            # is a repeat, not a second observation.
            known = book.already_tried(QUESTION, approach, where)
            if known and known["match"] == "exact":
                continue
            book.record(finding)
    return out


def _note(part: Partition, measurement: Dict[str, Any]) -> str:
    trials = int(measurement["trials"])
    if trials < MIN_PAIRED_TRIALS:
        return (f"{part.what}: партий с обеими сторонами сравнения {trials}, "
                f"нужно {MIN_PAIRED_TRIALS}. Не измерено — это не ноль.")
    effect = measurement.get("effect", 0.0)
    low, high = measurement["interval"]
    side = ("дороже остальных ходов" if high < 0 else
            "не отличается от остальных" if low <= 0 <= high else
            "дешевле остальных ходов")
    return (f"{part.what}: {side} на {abs(effect):.0f} сантипешек "
            f"(интервал {low:.0f}…{high:.0f}, партий {trials})")


def describe(out: Sequence[ledger_mod.Finding]) -> str:
    """What MANA concluded, in the order it was asked."""
    lines = [f"вопрос: {QUESTION}"]
    world = None
    for finding in out:
        here = finding.conditions.get("worlds", ["?"])
        if here != world:
            world = here
            lines.append(
                "" + chr(10) + f"— мир: {', '.join(here)}, "
                f"партий {finding.conditions.get('games', 0)}, "
                f"судья {', '.join(finding.conditions.get('judged_by', []))}")
        approach = finding.approach
        mark = " (класс выведен из данных)" if approach.get("mined_from_data") else ""
        lines.append(f"  {finding.verdict:<14} {approach['partition']:<17}"
                     f"{finding.note}{mark}")
    return "\n".join(lines)
