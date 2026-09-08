"""
mana.cognition.lessons — what we learned, as opposed to what we tried.

The distinction this exists to make
------------------------------------
The ledger answers one question today: "has exactly this been tried?"
`Ledger.already_tried` matches an approach and returns the finding, and
`candidates.rank` reads one bit out of it -- REJECTED with the conditions
still holding -- to sink a candidate to the bottom. So a finding acts as
a blacklist entry and nothing else.

That conflates two different things:

    мы это пробовали   an attempt exists in the record
    мы это узнали      the attempt was measured, and the number stands

Both of this project's adoptions are NOT_EVALUATED: the gates want thirty
paired trials and the record held seven turns, so they were adopted under
stated uncertainty. Under the old reading they are "tried", which is true
and useless -- nothing was learned, and re-running one is not a
repetition of a settled result but the first real measurement of it.
Under the new reading they are `tried` and not `learned`, and the two go
to different places in the plan.

What changes because of it
---------------------------
A lesson is a model of one failure, assembled from the record, and it
changes the space of next actions rather than filtering a list:

  * `observed` counts come from what live work actually recorded, across
    sessions, not from whatever happened to be in the window somebody
    scanned. A failure seen fourteen times outranks one seen once, even
    when the window shows one of each.
  * an axis every option of which has been measured, none of them better,
    is exhausted: more settings on that axis buy nothing, and the honest
    next step is a different kind of mechanism rather than another knob.
  * when nothing in the policy space aims at a failure at all, that is
    recorded as the plan's answer instead of being dropped. `propose`
    computed exactly this conclusion and threw it away every time, so
    "MANA has no proposal for X" and "MANA has no idea X exists" looked
    identical from outside.

Nothing here decides anything. It reads a record and states what follows
from it; `core/gates.py` still owns acceptance, and the ordering of
candidates is still a suggestion about what to measure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .findings import (BETTER, Finding, Ledger, NOT_MEASURED, UNCLASSIFIED)
from ..core.gates import ACCEPTED, NOT_EVALUATED, REJECTED

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Keep proposing settings: there is an axis nobody has measured.
KNOB_SEARCH = "knob_search"
#: Every option on every axis that aims at this has been measured and
#: none was better. Another setting is not the answer.
EXHAUSTED = "knob_space_exhausted"
#: No knob in the policy space aims at this failure at all.
NO_KNOB = "no_knob_addresses_it"

STRATEGIES = (KNOB_SEARCH, EXHAUSTED, NO_KNOB)

#: Failure classes that rest on a measurement. Everything else is a
#: record that something was attempted.
MEASURED_CLASSES = frozenset({"WORSE", "NOT_BETTER",
                              "COSTS_MORE_THAN_IT_GAINS", BETTER})


def _invariant_of(question: str) -> str:
    """Which invariant a ledger question is about, or "".

    Matched against the canonical wording rather than parsed out of it:
    the wording is declared stable and a regex over it would be a second
    place that has to stay in step.
    """
    from . import candidates, invariants, observed

    for fn in invariants.INVARIANTS:
        name = getattr(fn, "__name__", "")
        if question in (candidates.question_for(name),
                        observed.question_for(name)):
            return name
    return ""


@dataclass(frozen=True)
class Attempt:
    """One thing that was tried against one failure, and what came of it."""
    approach: Dict[str, Any]
    verdict: str
    failure: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    created: float = 0.0

    @property
    def measured(self) -> bool:
        """Did this teach anything, or only happen?

        A verdict without a measurement behind it is a decision somebody
        made under stated uncertainty. Recording it is right; reading it
        as knowledge is not.
        """
        return self.failure in MEASURED_CLASSES

    @property
    def helped(self) -> bool:
        return self.measured and self.failure == BETTER

    def as_dict(self) -> Dict[str, Any]:
        return {"approach": dict(self.approach), "verdict": self.verdict,
                "failure": self.failure, "conditions": dict(self.conditions),
                "created": self.created, "measured": self.measured,
                "helped": self.helped}


@dataclass(frozen=True)
class Lesson:
    """Everything the record says about one failure."""
    invariant: str
    #: Times this was seen in real work, across sessions. Written by
    #: `cognition/observed.py` at the end of a turn, so it survives the
    #: window somebody happens to scan.
    observed: int = 0
    last_episode: str = ""
    last_request: str = ""
    attempts: Tuple[Attempt, ...] = ()

    # ---------- tried, and learned ----------

    @property
    def tried(self) -> Tuple[Attempt, ...]:
        return self.attempts

    @property
    def learned(self) -> Tuple[Attempt, ...]:
        """Attempts a number stands behind. The other kind is not this."""
        return tuple(a for a in self.attempts if a.measured)

    @property
    def unmeasured(self) -> Tuple[Attempt, ...]:
        return tuple(a for a in self.attempts if not a.measured)

    # ---------- the axes ----------

    @property
    def knobs(self) -> Tuple[str, ...]:
        from .. import policy as policy_mod
        return tuple(k.name for k in policy_mod.knobs_for(self.invariant))

    @property
    def measured_settings(self) -> Tuple[Tuple[str, Any], ...]:
        """(knob, value) pairs a measurement has been taken on."""
        seen: List[Tuple[str, Any]] = []
        for attempt in self.learned:
            for name, value in sorted(attempt.approach.items()):
                if name in self.knobs and (name, value) not in seen:
                    seen.append((name, value))
        return tuple(seen)

    @property
    def open_settings(self) -> Tuple[Tuple[str, Any], ...]:
        """(knob, value) pairs nobody has measured. Where the information is."""
        from .. import policy as policy_mod

        measured = set(self.measured_settings)
        out: List[Tuple[str, Any]] = []
        for knob in policy_mod.knobs_for(self.invariant):
            for option in knob.options:
                if option == knob.default:
                    # Today's setting is not a change to propose; the
                    # comparison against it is the baseline.
                    continue
                if (knob.name, option) not in measured:
                    out.append((knob.name, option))
        return tuple(out)

    @property
    def exhausted(self) -> bool:
        """Every option measured, none of them better.

        Strict on purpose: `measured` means a number stands behind the
        verdict, so an axis is never called exhausted on the strength of
        adoptions made under stated uncertainty. On today's ledger this
        is false for every invariant, and saying so is better than
        loosening the rule until it fires.
        """
        if not self.knobs or not self.learned:
            return False
        return not self.open_settings and not any(a.helped for a in self.learned)

    # ---------- what to do next ----------

    def strategy(self) -> Tuple[str, str]:
        """The space of next actions, and why it is that one."""
        if not self.knobs:
            return (NO_KNOB,
                    f"в политике нет настройки, которая целит в «{self.invariant}»; "
                    f"это чинится кодом, а не подбором")
        if self.exhausted:
            return (EXHAUSTED,
                    f"все настройки, целящие в «{self.invariant}», измерены "
                    f"({len(self.learned)}), ни одна не помогла — следующий шаг "
                    f"не настройка, а другой механизм")
        return (KNOB_SEARCH,
                f"не измерено настроек: {len(self.open_settings)}")

    def describe(self) -> str:
        strategy, why = self.strategy()
        return (f"«{self.invariant}»: наблюдений {self.observed}, "
                f"пробовали {len(self.tried)}, узнали {len(self.learned)} → "
                f"{strategy} ({why})")

    def as_dict(self) -> Dict[str, Any]:
        strategy, why = self.strategy()
        return {"invariant": self.invariant, "observed": self.observed,
                "last_episode": self.last_episode,
                "last_request": self.last_request,
                "tried": len(self.tried), "learned": len(self.learned),
                "attempts": [a.as_dict() for a in self.attempts],
                "open_settings": [list(s) for s in self.open_settings],
                "exhausted": self.exhausted,
                "strategy": strategy, "why": why}


def _attempt_of(finding: Finding) -> Attempt:
    return Attempt(approach=dict(finding.approach), verdict=finding.verdict,
                   failure=finding.failure.failure,
                   conditions=dict(finding.conditions),
                   created=finding.created)


def read(ledger: Optional[Ledger] = None,
         invariants_seen: Sequence[str] = ()) -> Dict[str, Lesson]:
    """Assemble one lesson per failure the record knows about.

    `invariants_seen` adds failures observed right now that the ledger has
    not caught up with, so a first occurrence is not invisible until the
    ledger is next written.
    """
    from . import observed as observed_mod

    ledger = ledger if ledger is not None else Ledger()
    try:
        rows = ledger.latest()
    except Exception:
        rows = []

    counts: Dict[str, int] = {}
    last: Dict[str, Tuple[str, str]] = {}
    attempts: Dict[str, List[Attempt]] = {}

    for finding in rows:
        invariant = _invariant_of(finding.question)
        if not invariant:
            continue
        if finding.approach == observed_mod.OBSERVED_APPROACH:
            try:
                counts[invariant] = int(
                    finding.measurement.get("observed_failures", 0))
            except Exception:
                counts[invariant] = counts.get(invariant, 0)
            last[invariant] = (
                str(finding.measurement.get("last_episode", "")),
                str(finding.measurement.get("last_request", "")))
            continue
        attempts.setdefault(invariant, []).append(_attempt_of(finding))

    for name in invariants_seen:
        counts.setdefault(name, 0)

    out: Dict[str, Lesson] = {}
    for invariant in sorted(set(counts) | set(attempts)):
        episode, request = last.get(invariant, ("", ""))
        out[invariant] = Lesson(
            invariant=invariant,
            observed=counts.get(invariant, 0),
            last_episode=episode, last_request=request,
            attempts=tuple(sorted(attempts.get(invariant, []),
                                  key=lambda a: a.created)))
    return out
