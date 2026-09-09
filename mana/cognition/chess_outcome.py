"""
mana.cognition.chess_outcome — which properties go with losing, asked of
the result and nothing else.

The question
-------------
    вопрос:  у какой стороны свойства было больше — та и выиграла?
    подход:  одно объявленное наблюдаемое свойство
    ответ:   ACCEPTED / REJECTED / NOT_EVALUATED

No engine, no thresholds, no bands. `chess_findings` needed both: a
stronger player's opinion per move, and a person deciding where "moderate"
ended. Neither survives here.

Why the pairing removes the thresholds
---------------------------------------
A game has two sides. For any property, one side had more of it than the
other, and one of them won. So the question is a comparison inside a
single game and never a cutoff: "did the side with more captures win?" is
answerable without anybody deciding how many captures is a lot.

In self-play the opponent is the same player, so everything shared by the
two sides -- the evaluation, the depth, the opening book that does not
exist, the tie-breaking shuffle -- cancels exactly. What is left is what
actually differed between the side that won and the side that lost. That
is a natural experiment nobody had to design, and it is the reason
self-play is worth more here than it looks.

Two windows, because the property has to come first
-----------------------------------------------------
Measured over a whole game, "the side with more material won" is close to
a restatement of winning: a side that is already winning keeps its
material for the rest of the game, so the result causes the property. Six
of the first nine accepted results had that shape.

Deciding which of them are tautologies would be a person choosing the
answer again. Instead every property is measured twice -- over the whole
game, and over the first half of a side's moves, where a game is usually
still undecided. A connection that survives the early window is one where
the property came first; one that appears only over the whole game is the
result talking about itself. The pair says which without anybody ruling
on it.

Half is a declared fraction, not a tuned one. No other value was tried.

What is measured
-----------------
One game is one observation: the side with the higher value, and whether
it won. Draws are dropped -- they answer neither way -- and a game where
the two sides tie on the property is dropped too, for the same reason.
The share of games where the higher side won is tested against a half,
with the interval widened for the number of properties asked about.

An observation is a game. Not a move: sixty moves of one game share an
opponent, an opening and every earlier mistake, and counting them
separately would put a confident interval around nothing.

What this is not
-----------------
Not a law. A law in this project is a conditional statement earned by
changing something and measuring the change -- A against B. Everything
here is a regularity in what has already happened, and a regularity is a
reason to look, never a cause. Findings that survive become candidates
for an experiment; nothing here performs one.

And not an interpretation. The properties come from `chess_features`,
which records counts and stops. If one of them turns out to matter, what
it means is a further question that this file does not answer.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core.gates import ACCEPTED, MIN_PAIRED_TRIALS, NOT_EVALUATED, REJECTED
from . import chess_features as feat
from . import findings as ledger_mod

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

QUESTION = "у какой стороны свойства было больше — та и выиграла"

#: Two-sided confidence before the correction for how many properties are
#: asked about at once.
ALPHA = 0.05

#: The value that means "no connection": the higher side wins half the time.
NO_CONNECTION = 0.5


@dataclass(frozen=True)
class Property:
    """One number per side of one game, and where it can be measured.

    `needs_trace` is the half that keeps a spurious result out. The
    opponent on Lichess has no search trace, so "the side that considered
    more moves" would be "the side that was MANA" -- a measurement of
    MANA's win rate wearing a property's name. Properties like that are
    measured only where both sides have them, which in practice means
    self-play.
    """
    name: str
    what: str
    of: Callable[[Sequence[feat.Move]], Optional[float]]
    needs_trace: bool = False


def _mean(values: Sequence[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


#: The properties. Every one is an average or a share of what
#: `chess_features` counted -- no threshold appears here, because the
#: pairing makes thresholds unnecessary.
PROPERTIES: Tuple[Property, ...] = (
    Property("legal_moves", "сколько ходов было доступно",
             lambda rows: _mean([r.legal_moves for r in rows])),
    Property("material", "счёт материала",
             lambda rows: _mean([r.material for r in rows])),
    Property("pieces", "сколько своих фигур на доске",
             lambda rows: _mean([r.our_pieces for r in rows])),
    Property("captures", "доля ходов со взятием",
             lambda rows: _mean([1.0 if r.is_capture else 0.0 for r in rows])),
    Property("checks_given", "доля ходов с шахом",
             lambda rows: _mean([1.0 if r.gives_check else 0.0 for r in rows])),
    Property("checks_taken", "доля ходов из-под шаха",
             lambda rows: _mean([1.0 if r.in_check else 0.0 for r in rows])),
    Property("promotions", "доля превращений",
             lambda rows: _mean([1.0 if r.is_promotion else 0.0 for r in rows])),
    Property("pawn_moves", "доля ходов пешкой",
             lambda rows: _mean([1.0 if r.piece == "P" else 0.0 for r in rows])),
    Property("king_moves", "доля ходов королём",
             lambda rows: _mean([1.0 if r.piece == "K" else 0.0 for r in rows])),
    Property("material_gained_3", "как менялся материал через 3 своих хода",
             lambda rows: _mean([r.later[3] for r in rows
                                 if r.later.get(3) is not None])),
    Property("considered", "сколько ходов рассматривал поиск",
             lambda rows: _mean([r.considered for r in rows]), needs_trace=True),
    Property("tied_at_top", "сколько ходов делили лучшую оценку",
             lambda rows: _mean([r.tied_at_top for r in rows]), needs_trace=True),
    Property("margin", "отрыв выбранного хода от следующего",
             lambda rows: _mean([r.margin for r in rows]), needs_trace=True),
)


#: The two windows a property is measured over. `EARLY` is the first half
#: of a side's own moves -- a fraction rather than a number of plies, so a
#: short game and a long one are treated alike.
WHOLE, EARLY = "whole", "early"
WINDOWS = (WHOLE, EARLY)


def _window(rows: Sequence[feat.Move], window: str) -> Sequence[feat.Move]:
    if window == EARLY:
        return rows[:max(1, len(rows) // 2)]
    return rows


@dataclass
class Sides:
    """One game reduced to two rows: what each side did, and who won."""
    game: str
    source: str = ""
    level: int = 0
    #: {(property, window): value} for each side.
    ours: Dict[Tuple[str, str], Optional[float]] = field(default_factory=dict)
    theirs: Dict[Tuple[str, str], Optional[float]] = field(default_factory=dict)
    #: True where MANA won, False where it lost, None for a draw or an
    #: unfinished game -- which answer the question neither way.
    we_won: Optional[bool] = None
    both_traced: bool = False


def reduce_game(game: Dict[str, Any]) -> Optional[Sides]:
    """One game to one observation per property, for each side.

    Reduced rather than kept: a comparison inside a game needs two
    numbers, and carrying sixty moves per side into the measurement is
    what makes a reader count moves as though they were independent.
    """
    moves = feat.observe(game)
    if not moves:
        return None
    ours = [row for row in moves if row.ours]
    theirs = [row for row in moves if not row.ours]
    if not ours or not theirs:
        return None
    outcome = ours[0].outcome
    won = (True if outcome == feat.WON else
           False if outcome == feat.LOST else None)
    return Sides(
        game=str(game.get("game", "")), source=str(game.get("source", "")),
        level=int(game.get("level", 0) or 0),
        ours={(p.name, w): p.of(_window(ours, w))
              for p in PROPERTIES for w in WINDOWS},
        theirs={(p.name, w): p.of(_window(theirs, w))
                for p in PROPERTIES for w in WINDOWS},
        we_won=won,
        both_traced=any(row.considered for row in ours)
                    and any(row.considered for row in theirs))


def pairs(games: Sequence[Sides], prop: Property,
          window: str = WHOLE) -> List[Tuple[str, bool]]:
    """For each usable game: did the side with more of it win?

    A drawn game answers neither way and is absent. So is a game where
    the two sides tie on the property, and so is one where a property is
    not defined for both sides -- absent rather than counted as a half,
    because an unanswerable question is not half a yes.
    """
    out: List[Tuple[str, bool]] = []
    for row in games:
        if row.we_won is None:
            continue
        if prop.needs_trace and not row.both_traced:
            continue
        mine = row.ours.get((prop.name, window))
        yours = row.theirs.get((prop.name, window))
        if mine is None or yours is None or mine == yours:
            continue
        higher_was_ours = mine > yours
        out.append((row.game, higher_was_ours == row.we_won))
    return out


def measure(games: Sequence[Sides], prop: Property, window: str = WHOLE,
            questions: int = 0) -> Dict[str, Any]:
    """How often the side with more of it won, with the interval.

    A Wilson interval on a share: it behaves at the edges, where a
    normal approximation reports impossible bounds, and this measurement
    lives near the edges whenever a property is decisive.
    """
    from .self_model import wilson_interval

    rows = pairs(games, prop, window)
    trials = len(rows)
    wins = sum(1 for _, higher_won in rows if higher_won)
    alpha = ALPHA / max(1, questions or len(PROPERTIES) * len(WINDOWS))
    if trials:
        from statistics import NormalDist

        z = float(NormalDist().inv_cdf(1.0 - alpha / 2.0))
        low, high = wilson_interval(wins, trials, z=z)
    else:
        low, high = 0.0, 1.0
    return ledger_mod.measurement_of(
        trials=trials, interval=(low, high), null=NO_CONNECTION,
        effect=round(wins / trials, 3) if trials else None,
        higher_side_won=wins, alpha=round(alpha, 5), window=window,
        questions_asked=questions or len(PROPERTIES) * len(WINDOWS))


def verdict_for(measurement: Dict[str, Any]) -> str:
    """ACCEPTED means the property goes with the result, either way.

    Which way is in the numbers beside it. A property whose higher side
    loses is as much a finding as one whose higher side wins, and
    collapsing the two into "no effect" would throw away the half that
    points at a mistake.
    """
    if int(measurement["trials"]) < MIN_PAIRED_TRIALS:
        return NOT_EVALUATED
    low, high = measurement["interval"]
    return ACCEPTED if (low > NO_CONNECTION or high < NO_CONNECTION) else REJECTED


def conditions(games: Sequence[Sides]) -> Dict[str, Any]:
    from ..version import PRODUCT_VERSION

    return {"games": len(games),
            "decided": sum(1 for row in games if row.we_won is not None),
            "worlds": sorted({row.source for row in games}),
            "levels": sorted({row.level for row in games}),
            "version": PRODUCT_VERSION,
            "questions_asked": len(PROPERTIES) * len(WINDOWS)}


def look(games: Sequence[Sides], ledger: Optional[Any] = None,
         record: bool = True) -> List[ledger_mod.Finding]:
    """Answer every declared property, once per world, and write it down.

    Every property every pass: measuring thirteen and reporting the one
    that came out is how a coincidence becomes a finding.
    """
    book = ledger if ledger is not None else ledger_mod.Ledger()
    worlds: Dict[str, List[Sides]] = {}
    for row in games:
        worlds.setdefault(row.source or "?", []).append(row)

    out: List[ledger_mod.Finding] = []
    for world in sorted(worlds):
        where = conditions(worlds[world])
        for prop in PROPERTIES:
          for window in WINDOWS:
            measurement = measure(worlds[world], prop, window,
                                  questions=len(PROPERTIES) * len(WINDOWS))
            finding = ledger_mod.Finding(
                question=QUESTION,
                approach={"property": prop.name, "what": prop.what,
                          "window": window, "needs_trace": prop.needs_trace},
                verdict=verdict_for(measurement),
                measurement=measurement, conditions=where,
                note=_note(prop, measurement, window), version=where["version"])
            out.append(finding)
            if record:
                known = book.already_tried(QUESTION, finding.approach, where)
                if not (known and known["match"] == "exact"):
                    book.record(finding)
    return out


def _note(prop: Property, measurement: Dict[str, Any],
          window: str = WHOLE) -> str:
    when = "вся партия" if window == WHOLE else "первая половина ходов"
    trials = int(measurement["trials"])
    if trials < MIN_PAIRED_TRIALS:
        return (f"{prop.what} ({when}): партий с ответом {trials}, нужно "
                f"{MIN_PAIRED_TRIALS}. Не измерено — это не ноль.")
    share = measurement.get("effect") or 0.0
    low, high = measurement["interval"]
    if low > NO_CONNECTION:
        side = "у кого больше — тот чаще выигрывает"
    elif high < NO_CONNECTION:
        side = "у кого больше — тот чаще проигрывает"
    else:
        side = "с исходом не связано"
    return (f"{prop.what} ({when}): {side} ({share:.0%} из {trials} "
            f"партий, интервал {low:.0%}…{high:.0%})")


def state(out: Sequence[ledger_mod.Finding]) -> Dict[str, str]:
    """Verdict per property per world, for comparing two passes."""
    return {f"{','.join(f.conditions.get('worlds', ['?']))}/"
            f"{f.approach['property']}/{f.approach.get('window', WHOLE)}":
            f.verdict for f in out}


def changes(before: Dict[str, str], after: Sequence[ledger_mod.Finding]
            ) -> List[str]:
    """What moved since the last pass, and nothing that did not.

    A conclusion repeated every game is noise; a conclusion that changed
    is the only thing worth interrupting anybody for.
    """
    now = state(after)
    said: List[str] = []
    for key, verdict in now.items():
        was = before.get(key)
        if was == verdict:
            continue
        finding = next(f for f in after
                       if f"{','.join(f.conditions.get('worlds', ['?']))}/"
                          f"{f.approach['property']}/"
                          f"{f.approach.get('window', WHOLE)}" == key)
        moved = "новое" if was is None else f"{was} → {verdict}"
        said.append(f"[{verdict}] {key}: {finding.note}  ({moved})")
    return said


def describe(out: Sequence[ledger_mod.Finding]) -> str:
    lines = [f"вопрос: {QUESTION}"]
    world = None
    for finding in out:
        here = finding.conditions.get("worlds", ["?"])
        if here != world:
            world = here
            lines.append(f"\n— мир: {', '.join(here)}, партий "
                         f"{finding.conditions.get('games', 0)}, из них с "
                         f"исходом {finding.conditions.get('decided', 0)}")
        lines.append(f"  {finding.verdict:<14} "
                     f"{finding.approach['property']:<20}{finding.note}")
    return "\n".join(lines)
