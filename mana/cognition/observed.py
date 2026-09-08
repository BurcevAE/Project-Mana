"""
mana.cognition.observed — a failure in real work, written down unasked.

The chain this closes
---------------------
    Episode  ->  Outcome  ->  Failure  ->  Finding

The first two links exist: `mana/journal.py` records what a turn did, and
`mana/outcome.py` makes a tool say whether the machine ended up in the
state it was asked for. The third exists too -- `invariants.py` names a
failure by comparing recorded fields. The fourth was reachable only by a
person typing `--findings` and then writing a finding by hand, which
means the record of what goes wrong in real use was as good as the
person's diligence in a week when nothing else was on fire.

So this runs at the end of a turn, reads the episode that just closed,
and appends a finding when something objectively went wrong. Nobody is
asked and no model is consulted.

Mechanical only, on purpose
----------------------------
`invariants.py` separates MECHANICAL violations -- decided by comparing
recorded fields -- from PATTERN ones, which match free text against a
word list and are useful guesses. Only the first kind is written here.
A guess belongs in a report a person reads, not in the ledger that later
work treats as established; putting one there would be the exact failure
this project keeps having, a claim resting on nothing, and it would be
self-inflicted this time.

The strongest of the mechanical ones needs no text at all:
`acted_but_state_disagrees` reads a verdict the tool reached by comparing
what it was asked for against what it found. There is nothing to
interpret in it and nothing for a word list to get wrong.

What the finding says, and what it does not
--------------------------------------------
A failure seen in production is not an experiment. It is recorded as
`NOT_EVALUATED` with no `trials`, `interval` or `null`, so `classify`
puts it in UNCLASSIFIED with the reason "в измерении нет trials,
interval, null" -- which is the truth. Inventing an interval to make it
land in a tidier class would be fabricating a measurement, and the whole
point of the ledger is that it does not hold those.

`measurement` carries the count of times the failure has been seen and
the episode it was last seen in. That is a real measurement -- of
occurrence, not of effect -- and it is what makes "this happens twice a
week" separable from "this happened once in March".

The question is deliberately not `candidates.question_for`. That one is
"какая настройка политики устраняет X?", the question an experiment
answers; this one is "наблюдается ли X в реальной работе?", which is a
different question with a different answer, and merging them would put
observations with no approach into the series reader beside real
experimental points.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..journal import Episode
from . import invariants
from .findings import Finding, Ledger
from ..core.gates import NOT_EVALUATED

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Marks a finding that came from watching real work rather than from
#: running an experiment. Also what keeps its identity clear of every
#: candidate approach: no proposed policy change will ever carry this.
OBSERVED_APPROACH: Dict[str, Any] = {"observed_in_live_work": True}

#: How much of the request is kept beside the count. Enough to recognise
#: the turn, not enough to make the ledger a second copy of the journal.
MAX_REQUEST = 160


def question_for(invariant: str) -> str:
    """How an observed failure is phrased in the ledger.

    Stable wording, because the ledger keys on it: rephrasing this later
    would make every past record invisible, which is the failure the
    ledger exists to prevent.
    """
    return f"Наблюдается ли «{invariant}» в реальной работе?"


def _clip(text: str, limit: int = MAX_REQUEST) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def seen_before(invariant: str, ledger: Ledger) -> Optional[Finding]:
    """The standing record for this failure, if there is one."""
    probe = Finding(question=question_for(invariant),
                    approach=dict(OBSERVED_APPROACH), verdict=NOT_EVALUATED)
    for finding in ledger.latest():
        if finding.finding_id == probe.finding_id:
            return finding
    return None


def finding_for(violation: invariants.Violation, episode: Episode,
                previous: Optional[Finding] = None) -> Finding:
    """One violation, as a record the ledger can hold.

    The count carries forward from the previous record rather than being
    recomputed from the journal: the journal rotates, and a count that
    silently resets when a file rolls over is worse than no count.
    """
    seen = 1
    if previous is not None:
        try:
            seen = int(previous.measurement.get("observed_failures", 0)) + 1
        except Exception:
            seen = 1
    return Finding(
        question=question_for(violation.invariant),
        approach=dict(OBSERVED_APPROACH),
        verdict=NOT_EVALUATED,
        measurement={"observed_failures": seen,
                     "last_episode": episode.episode_id,
                     "last_request": _clip(episode.request)},
        note=f"{violation.reason} (замечено без запроса, в обычной работе)")


def record(episode: Episode, earlier: Sequence[Episode] = (),
           ledger: Optional[Ledger] = None,
           targets: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Write down what went objectively wrong on this turn.

    Returns what was written, so a caller can show it and a test can
    check it. Never raises: this runs at the end of every turn and
    evidence-keeping that can break the thing it observes is not worth
    keeping.
    """
    written: List[Dict[str, Any]] = []
    if episode is None:
        return written
    try:
        ledger = ledger if ledger is not None else Ledger()
        found = invariants.check(episode, earlier, targets)
        for violation in found:
            if violation.kind != invariants.MECHANICAL:
                continue
            previous = seen_before(violation.invariant, ledger)
            finding = finding_for(violation, episode, previous)
            if ledger.record(finding):
                written.append({"invariant": violation.invariant,
                                "finding_id": finding.finding_id,
                                "seen": finding.measurement["observed_failures"],
                                "reason": violation.reason})
    except Exception:
        return written
    return written
