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
from typing import Any, Dict, List, Optional, Sequence

from .findings import (BETTER, COSTS_MORE_THAN_IT_GAINS, Finding, Ledger,
                       NOT_BETTER, WORSE)
from .laws import Condition, CognitiveLaw, LawBook, PROPOSED
from .series import Comparison, Series

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Where the book lives. Beside the rest of the user's state, like the
#: journal and the ledger: it is evidence about the code, and evidence
#: kept inside the thing it describes disappears with it on reinstall.
BOOK_DIRNAME = "laws"


def book_path() -> Path:
    from ..paths import data_root
    return Path(data_root()) / BOOK_DIRNAME / "laws.json"


#: How a class change reads as a direction. Used to phrase the claim, not
#: to decide anything: the classes were already derived from the numbers.
_BETTER_SIDE = (BETTER, COSTS_MORE_THAN_IT_GAINS)


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


def candidates(series: Series, ledger: Optional[Ledger] = None) -> List[Candidate]:
    """Law candidates a series supports. Empty is the usual answer.

    One per isolated flip. Confounded flips and flat axes produce none,
    and that is not a gap: a claim nobody can attribute is not a law, and
    "nothing changed" is a fact the series already holds.
    """
    ledger = ledger if ledger is not None else Ledger()
    out: List[Candidate] = []
    for flip in series.isolated_flips:
        axis, values = next(iter(flip.changed.items()))
        left = _finding_for(ledger, flip.left)
        right = _finding_for(ledger, flip.right)
        if right is None:
            continue

        domain = str((right.approach or {}).get("domain") or "")
        improved = flip.to_class in _BETTER_SIDE and flip.from_class not in _BETTER_SIDE
        worsened = flip.from_class in _BETTER_SIDE and flip.to_class not in _BETTER_SIDE
        if not (improved or worsened):
            # A flip between two kinds of failure -- WORSE to NOT_BETTER,
            # say. Real, and not an intervention that helps or hurts, so
            # phrasing it as one would be putting a direction into the
            # book that the numbers do not carry.
            continue

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
        out.append(Candidate(
            condition=Condition(domain=domain),
            intervention=(f"{axis}: {values['was']} -> {values['now']}",),
            claimed_effect=(
                f"{axis} со значения {values['was']!r} на {values['now']!r} "
                f"{direction} исход с {flip.from_class} на {flip.to_class}"),
            exceptions=tuple(exceptions),
            trials=int((right.measurement or {}).get("trials") or 0),
            effect=effect, discovered_in=domain, scope=held,
            source=f"{flip.left}->{flip.right}"))
    return out


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
