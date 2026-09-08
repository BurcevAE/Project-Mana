"""
mana.cognition.lawgiver — turning an isolated flip into a law candidate.

The orphan this connects
------------------------
`laws.py` is well built and, until now, called by nothing: zero
references outside itself. It has conditions that decide by matching
rather than by argument, four statuses, and a demotion path as automatic
as promotion. What it never had was anything that produced a law.

A series does. An **isolated** flip -- one condition differed, the class
changed, and both sides used the same method -- is exactly a conditional
claim: under these held conditions, this intervention moves the outcome
from here to there.

Only isolated flips
-------------------
A confounded flip names several conditions and cannot say which mattered.
A flat axis says the class did not move, which is a real fact and not a
law: `laws.py` scores an intervention by its effect, so a null effect
would be recorded as `positive_share < 0.5` and the law refuted on
arrival. Recording "nothing happened" as a refuted law would put a
falsehood in the book. Flat axes belong in the series, where they are,
and this module leaves them there.

The claim is about the method, not about chess
-----------------------------------------------
"Extended features beat material counting" is a fact about chess. It is
not a claim about MANA's cognition and could never transfer anywhere.
What can be stated as a law is the shape of the finding:

    when a fitted evaluation is not better than a hand-written baseline,
    enriching the feature set moves it above the baseline and raises the
    cost

That is testable in another domain, which is what `transfer` in `laws.py`
is for and why VALIDATED requires a domain the law was not found in.

PROPOSED is the ceiling here, and that is correct
--------------------------------------------------
`_derive_status` needs at least one hidden-set confirmation before a law
rises above PROPOSED. The chess experiments had no hidden holdout, so
`hidden_confirmed` is never passed and the candidate stays PROPOSED with
240 trials behind it. Passing `True` to move it up would be inventing a
confirmation, which is precisely what the four statuses exist to prevent.

What the law cannot claim is written down as an exception
----------------------------------------------------------
The chess flip landed on `COSTS_MORE_THAN_IT_GAINS`: the interval sits
entirely above the baseline and the cost is 7.4x against a tolerance of
1.5x. So "better per unit of search" is established and "better on a
clock" is not. That gap is recorded as an exception on the law rather
than dropped, because a law whose limits are not written down is a law
that will be applied outside them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .findings import (BETTER, COSTS_MORE_THAN_IT_GAINS, Finding, Ledger,
                       NOT_BETTER, WORSE)
from .laws import Condition, CognitiveLaw, LawBook, PROPOSED
from .series import Comparison, Series

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.1"

#: Where the book lives. Beside the rest of the user's state, like the
#: journal and the ledger: it is evidence about the code, and evidence
#: kept inside the thing it describes disappears with it on reinstall.
BOOK_DIRNAME = "laws"


def book_path() -> Path:
    from ..paths import data_root
    return Path(data_root()) / BOOK_DIRNAME / "laws.json"


#: How an intervention names the axis it varies. A contract between this
#: module and `probes.py`, owned here because this module writes it: the
#: alternative is `probes` parsing prose, which is guessing about a format
#: nobody agreed to. `axis_of` is the only reader.
INTERVENTION_FORMAT = "{axis}: {was} -> {now}"

_AXIS_SEPARATOR = ": "


def axis_of(intervention: str) -> str:
    """The condition an intervention varies, or "" if it is not one of ours.

    Returns "" rather than guessing when the string was not written in
    the format above -- a law imported from elsewhere, or one written by
    hand, names its intervention however it likes.
    """
    head, separator, rest = str(intervention or "").partition(_AXIS_SEPARATOR)
    if not separator or " -> " not in rest:
        return ""
    return head.strip()


def axes_of(law: Any) -> List[str]:
    """Every condition a law's interventions vary."""
    found = [axis_of(entry) for entry in getattr(law, "intervention", ())]
    return [axis for axis in found if axis]


#: How a class change reads as a direction. Used to phrase the claim, not
#: to decide anything: the classes were already derived from the numbers.
_BETTER_SIDE = (BETTER, COSTS_MORE_THAN_IT_GAINS)

#: Failure classes a comparison may rest on. NOT_MEASURED on either side
#: means the outcome was never measured there, so a flip out of it is a
#: change in what was measured rather than in what happened.
#: `Comparison.usable` screens out UNCLASSIFIED and stops there, which
#: let NOT_MEASURED -> BETTER read as an improvement caused by the axis.
_MEASURED = (WORSE, NOT_BETTER, COSTS_MORE_THAN_IT_GAINS, BETTER)

#: Agreeing isolated flips on one axis before a claim may enter the book
#: at all. One isolated flip is one paired comparison: it says the outcome
#: moved while this axis differed, which is not the claim that the axis
#: moves the outcome. Two agreeing flips under otherwise different
#: conditions is the smallest thing that separates those, and it is the
#: same replication argument `core/gates.py` makes with paired trials.
#:
#: A convention, stated rather than derived -- like MIN_TRIALS_FOR_SUPPORT
#: next door. What matters is that it is written down and that a claim
#: below it is refused out loud instead of quietly not appearing.
MIN_AGREEING_FLIPS = 2


@dataclass(frozen=True)
class Refusal:
    """An axis the record cannot yet support a claim about, and why."""
    axis: str
    reason: str
    flips: int = 0
    source: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"axis": self.axis, "reason": self.reason,
                "flips": self.flips, "source": self.source}

    def describe(self) -> str:
        return f"«{self.axis}»: {self.reason}"


@dataclass(frozen=True)
class Assessment:
    """What a series supports, and what it does not support yet."""
    candidates: Tuple[Candidate, ...] = ()
    refusals: Tuple[Refusal, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {"candidates": [c.as_dict() for c in self.candidates],
                "refusals": [r.as_dict() for r in self.refusals]}


@dataclass(frozen=True)
class Candidate:
    """A law a series supports, with the evidence that would be folded in."""
    condition: Condition
    intervention: tuple
    claimed_effect: str
    exceptions: tuple
    trials: int
    effect: float
    discovered_in: str
    scope: Dict[str, Any]
    source: str

    def as_dict(self) -> Dict[str, Any]:
        return {"condition": self.condition.describe(),
                "intervention": list(self.intervention),
                "claimed_effect": self.claimed_effect,
                "exceptions": list(self.exceptions),
                "trials": self.trials, "effect": round(self.effect, 4),
                "discovered_in": self.discovered_in, "scope": self.scope,
                "source": self.source}


def _finding_for(ledger: Ledger, finding_id: str) -> Optional[Finding]:
    for finding in ledger.latest():
        if finding.finding_id == finding_id:
            return finding
    return None


def _effect_of(finding: Optional[Finding]) -> float:
    """Distance from the no-effect value, from the recorded interval.

    The midpoint of the interval, not the point estimate: the interval is
    what the class was derived from, and using a different number here
    would let the law and its class disagree about the same experiment.
    """
    if finding is None:
        return 0.0
    measurement = finding.measurement or {}
    try:
        low, high = (float(x) for x in measurement["interval"])
        return (low + high) / 2.0 - float(measurement["null"])
    except Exception:
        return 0.0


def _from_flip(flip: Comparison, ledger: Ledger
               ) -> Tuple[Optional[Candidate], Optional[Refusal]]:
    """One isolated flip, as a claim or as a stated reason it is not one."""
    axis, values = next(iter(flip.changed.items()))
    left = _finding_for(ledger, flip.left)
    right = _finding_for(ledger, flip.right)
    if right is None:
        return (None, Refusal(axis, "находка не найдена в реестре", 1,
                              f"{flip.left}->{flip.right}"))

    if flip.from_class not in _MEASURED or flip.to_class not in _MEASURED:
        # A side nobody measured. The class did change, and what changed
        # is what was measured rather than what happened -- reading it as
        # an effect of the axis is the "unmeasured is not zero" rule
        # broken by the module that most depends on it.
        return (None, Refusal(
            axis,
            f"одна из сторон не измерена ({flip.from_class} -> {flip.to_class})",
            1, f"{flip.left}->{flip.right}"))

    domain = str((right.approach or {}).get("domain") or "")
    improved = flip.to_class in _BETTER_SIDE and flip.from_class not in _BETTER_SIDE
    worsened = flip.from_class in _BETTER_SIDE and flip.to_class not in _BETTER_SIDE
    if not (improved or worsened):
        # A flip between two kinds of failure -- WORSE to NOT_BETTER,
        # say. Real, and not an intervention that helps or hurts, so
        # phrasing it as one would be putting a direction into the book
        # that the numbers do not carry.
        return (None, Refusal(
            axis, f"переворот между видами неудачи ({flip.from_class} -> "
                  f"{flip.to_class}): направления в числах нет",
            1, f"{flip.left}->{flip.right}"))

    direction = "поднимает" if improved else "опускает"
    effect = _effect_of(right) - _effect_of(left)

    exceptions: List[str] = []
    if flip.to_class == COSTS_MORE_THAN_IT_GAINS:
        ratio = (right.measurement or {}).get("cost_ratio")
        exceptions.append(
            f"лучше на единицу поиска, но цена {ratio}x — на равном "
            f"времени не проверено")
    if not (right.measurement or {}).get("hidden_confirmed"):
        exceptions.append("на скрытой выборке не подтверждено")

    held = {k: v for k, v in (right.conditions or {}).items() if k != axis}
    return (Candidate(
        condition=Condition(domain=domain),
        intervention=(INTERVENTION_FORMAT.format(
            axis=axis, was=values["was"], now=values["now"]),),
        claimed_effect=(
            f"{axis} со значения {values['was']!r} на {values['now']!r} "
            f"{direction} исход с {flip.from_class} на {flip.to_class}"),
        exceptions=tuple(exceptions),
        trials=int((right.measurement or {}).get("trials") or 0),
        effect=effect, discovered_in=domain, scope=held,
        source=f"{flip.left}->{flip.right}"), None)


def assess(series: Series, ledger: Optional[Ledger] = None) -> Assessment:
    """What a series supports, and what it refuses to support yet.

    A claim needs a series behind it, not one comparison. Flips are
    grouped by the axis they name, and an axis gets into the book only
    when at least MIN_AGREEING_FLIPS of them agree about the direction --
    with any disagreement stopping the axis outright, because an axis
    whose flips point both ways does not explain the outcome by itself
    and averaging them would put a claim in the book that neither
    comparison supports.
    """
    ledger = ledger if ledger is not None else Ledger()
    by_axis: Dict[str, List[Candidate]] = {}
    refusals: List[Refusal] = []
    for flip in series.isolated_flips:
        candidate, refusal = _from_flip(flip, ledger)
        if refusal is not None:
            refusals.append(refusal)
            continue
        if candidate is not None:
            by_axis.setdefault(axis_of(candidate.intervention[0]),
                               []).append(candidate)

    out: List[Candidate] = []
    for axis in sorted(by_axis):
        found = by_axis[axis]
        directions = {("поднимает" if c.effect > 0 else "опускает")
                      for c in found}
        if len(directions) > 1:
            refusals.append(Refusal(
                axis, "перевороты на этой оси противоречат друг другу — "
                      "сама по себе ось исход не объясняет", len(found)))
            continue
        if len(found) < MIN_AGREEING_FLIPS:
            refusals.append(Refusal(
                axis, f"согласных изолированных перестановок {len(found)} "
                      f"против {MIN_AGREEING_FLIPS}: одно сравнение — это "
                      f"наблюдение, а не серия", len(found)))
            continue
        out.extend(found)
    return Assessment(candidates=tuple(out), refusals=tuple(refusals))


def candidates(series: Series, ledger: Optional[Ledger] = None) -> List[Candidate]:
    """Law candidates a series supports. Empty is the usual answer.

    The claims half of `assess`. Confounded flips, flat axes and single
    comparisons produce none, and that is not a gap: a claim nobody can
    attribute is not a law, "nothing changed" is a fact the series
    already holds, and one flip is not a series.
    """
    return list(assess(series, ledger).candidates)


def propose(series: Series, book: Optional[LawBook] = None,
            ledger: Optional[Ledger] = None) -> List[CognitiveLaw]:
    """Put a series' candidates into the book, with their evidence.

    `hidden_confirmed` is never passed. The chess experiments had no
    hidden holdout, and a law cannot rise above PROPOSED without one --
    which is the correct ceiling for this evidence, not a limitation to
    work around. Passing True to move it up would be inventing a
    confirmation, exactly what the four statuses exist to prevent.
    """
    book = book if book is not None else LawBook()
    made: List[CognitiveLaw] = []
    for candidate in candidates(series, ledger):
        # The same claim reached twice is stronger evidence for one law,
        # not two laws each carrying half of it. Measured: the chess flip
        # is isolated twice -- once at each eval_version -- and proposing
        # it twice would split 480 trials into two lots of 240 and leave
        # both short of what a status needs.
        law = _existing(book, candidate)
        if law is None:
            law = book.propose(condition=candidate.condition,
                               intervention=candidate.intervention,
                               claimed_effect=candidate.claimed_effect,
                               discovered_in=candidate.discovered_in)
        if candidate.source in law.evidence.experiments:
            continue                    # already folded in on a prior run
        law.record_evidence(effect=candidate.effect, trials=candidate.trials,
                            experiment_id=candidate.source)
        for note in candidate.exceptions:
            if note not in law.exceptions:
                law.add_exception(note)
        if law not in made:
            made.append(law)
    return made


def _existing(book: LawBook, candidate: Candidate) -> Optional[CognitiveLaw]:
    """A law already in the book making this same claim.

    Matched on what the claim is -- condition, intervention, effect --
    rather than on an id, because two series arriving at the same law
    should meet in the book rather than pass each other in it.
    """
    for law in book.all():
        if (law.condition == candidate.condition
                and tuple(law.intervention) == tuple(candidate.intervention)
                and law.claimed_effect == candidate.claimed_effect):
            return law
    return None


def load_book(path: Optional[Path] = None) -> LawBook:
    return LawBook.load(Path(path) if path else book_path())


def save_book(book: LawBook, path: Optional[Path] = None) -> bool:
    try:
        book.save(Path(path) if path else book_path())
        return True
    except Exception:
        return False


def report(book: LawBook) -> Dict[str, Any]:
    """The book as it stands, with the ceiling stated.

    A summary that showed only counts would let PROPOSED read as "almost
    a law". It is not: it is one supported experiment and nothing more,
    which is what `laws.py` says of it.
    """
    summary = book.summary()
    summary["note"] = (
        "PROPOSED — один поддержанный эксперимент и ничего сверх того. "
        "Выше этого статуса нужна скрытая выборка, которой у шахматных "
        "опытов не было; для VALIDATED нужен ещё домен, в котором закон "
        "не находили.")
    return summary
