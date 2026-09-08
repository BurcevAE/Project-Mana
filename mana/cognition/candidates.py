"""
mana.cognition.candidates — proposing the change, so a person does not have to.

The last piece of the loop
--------------------------
    journal.py        what a turn did                  evidence
    invariants.py     which turns contradicted          findings
    failure_domain.py findings -> a domain with oracle  measurable
    THIS MODULE       findings -> changes to try        candidates
    core/gates.py     the verdict                       already there

Until now every one of those changes was written by hand and adopted
because somebody was persuaded by an argument. The pieces below do not
remove the code -- the fix for "the imperative must start the message"
is still a line of code somewhere -- but they move two decisions out of
opinion and into measurement: **which** change to try, and **whether** it
is kept.

Aimed, never blind
------------------
A candidate is only proposed for an invariant that actually fired on the
record. No observed failure, no candidate. Blind mutation over a policy
space would spend the whole evaluation budget on changes addressing
nothing, and would churn a system that was working.

`policy.knobs_for(invariant)` is how the aim is taken: each decision point
declares which invariant it can affect, so a proposal is connected to a
measurement by construction rather than by the proposer's belief.

One knob at a time, plus the combination
-----------------------------------------
Coordinate-wise: each candidate differs from the current policy in one
setting, and one further candidate turns on everything that addresses the
failing invariant. Enumerating the full product would be 288 policies for
six knobs, which no evaluation budget here can pay for, and an accepted
combination would not say which part of it did the work.

What `choose_mechanism` decides
-------------------------------
Whether a knob is even the right kind of answer. `brain_factory` asks
"what is the minimum mechanism sufficient for this?" and for both
invariants found so far the answer is `algorithmic`: detecting that an
answer repeats an earlier one is exact, and matching a name against the
machine's own list of bases is exact. A domain whose answer is computable
does not need a model that approximates it.

If it ever answers `classical_ml` or `keep_model` for an invariant, this
module proposes nothing for it and says why -- a policy knob would be the
wrong shape of fix, and offering one anyway is how a search ends up
optimising the thing it can adjust rather than the thing that is wrong.

Dry evaluation, and what it may not claim
------------------------------------------
`dry_responder` answers a recorded situation under a candidate policy
without a language model and without performing anything: it asks the
intent matcher what it would do, asks the application layer whether that
action is feasible, and builds the reply the way the live path would --
from the outcome.

It assumes a feasible action succeeds. 1C can be installed, the base can
exist, and the launch can still fail. So a dry score is an **upper
bound**, it is labelled as one, and `judge_change` is never handed a dry
score as though it were a measurement of the real thing.

The echo knobs are not dry-evaluable at all and are refused rather than
guessed: when the guard fires the live path retries with the recalled
context suppressed, and what that retry answers cannot be known without
running the model. Reporting a number there would be inventing evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .. import policy as policy_mod
from ..journal import ToolCall
from ..policy import Policy, Knob
from .brain_factory import ALGORITHMIC, choose_mechanism
from .experiments import MIN_EXPERIMENT_VALUE, VALUE_WEIGHTS, select
from .failure_domain import Situation
from .invariants import Violation

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.1"

#: Invariants whose fix is exactly computable, stated rather than guessed.
#: `choose_mechanism` takes this as a fact about the domain -- a factory
#: that inferred it from the name would be asserting something it has not
#: checked.
EXACTLY_COMPUTABLE = {
    "repeats_earlier_answer": True,     # two strings and a threshold
    "actionable_request_not_acted_on": True,   # a name against a known list
    "claimed_action_without_acting": True,     # a call list, and whether it ran
}

#: How much measuring a setting would tell us. The same three numbers
#: `probes.py` uses for an experimental axis, with the same meanings:
#: nobody has measured this, somebody measured it under conditions that
#: have since moved, somebody measured it under conditions that still
#: hold. Stated here rather than imported so that changing one module's
#: scale does not silently reweight the other.
NEVER_MEASURED = 1.0
CONDITIONS_MOVED = 0.9
ALREADY_MEASURED = 0.3

#: What a candidate costs to measure properly, when nobody prices it.
#: Left at zero rather than guessed: an invented cost would order the
#: list by a number nobody measured.
COST_SCALE = 500.0

#: Knobs whose effect cannot be seen without running the model. Named
#: here rather than discovered, so a knob added without thinking about
#: this is treated as un-dry-evaluable by default.
NEEDS_A_MODEL = frozenset({
    "echo_lookback", "echo_same_answer", "echo_different_question",
})


@dataclass(frozen=True)
class Candidate:
    """One change to try, and why it is being tried."""
    policy: Policy
    addresses: str                 # the invariant it aims at
    rationale: str
    mechanism: str = ALGORITHMIC
    observed: int = 0              # how many recorded failures it aims at

    @property
    def candidate_id(self) -> str:
        return f"{self.addresses}:{self.policy.policy_id}"

    def as_dict(self) -> Dict[str, Any]:
        return {"candidate_id": self.candidate_id,
                "changes": self.policy.changes(),
                "addresses": self.addresses, "rationale": self.rationale,
                "mechanism": self.mechanism, "observed_failures": self.observed}


def _with(current: Policy, **changes: Any) -> Policy:
    settings = current.as_dict()
    settings.update(changes)
    trimmed = {k: v for k, v in settings.items()
               if v != policy_mod._BY_NAME[k].default}
    return Policy.of(**trimmed)


@dataclass(frozen=True)
class Plan:
    """What to try, and what nothing can be tried for.

    The second half used to be silence. An invariant with no knob aimed
    at it, or one whose knob space has been measured out, produced an
    empty list exactly like an invariant nobody had ever seen -- so a
    reader could not tell "we have no proposal" from "we have no idea".
    """
    candidates: Tuple[Candidate, ...] = ()
    refusals: Tuple[Dict[str, Any], ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {"candidates": [c.as_dict() for c in self.candidates],
                "refusals": [dict(r) for r in self.refusals]}


def plan(violations: Sequence[Violation],
         current: Optional[Policy] = None,
         lessons: Optional[Dict[str, Any]] = None) -> Plan:
    """Changes worth measuring, and the failures nothing here can address.

    With no lessons this aims at the violations it was handed, which is
    what it always did. With them it aims at the record: a failure seen
    fourteen times in real work outranks one seen once, even when the
    window shows one of each, and a failure the ledger knows about is
    addressed even when this particular window is clean.

    Returns nothing to try when nothing went wrong, which is the common
    and correct answer: a generator that produced candidates for a system
    with no observed failures would be changing a working system on
    speculation.
    """
    from .lessons import EXHAUSTED, NO_KNOB

    current = current or policy_mod.BASELINE
    counts: Dict[str, int] = {}
    for violation in violations:
        counts[violation.invariant] = counts.get(violation.invariant, 0) + 1
    # What the record knows, not only what this window holds.
    for name, lesson in (lessons or {}).items():
        counts[name] = max(counts.get(name, 0), int(getattr(lesson, "observed", 0)))
    counts = {name: n for name, n in counts.items() if n > 0}

    out: List[Candidate] = []
    refused: List[Dict[str, Any]] = []
    for invariant in sorted(counts, key=lambda k: (-counts[k], k)):
        observed = counts[invariant]
        lesson = (lessons or {}).get(invariant)

        knobs = policy_mod.knobs_for(invariant)
        if not knobs:
            refused.append({"addresses": invariant, "observed_failures": observed,
                            "strategy": NO_KNOB,
                            "why": (f"в политике нет настройки, которая целит в "
                                    f"«{invariant}»; это чинится кодом, а не "
                                    f"подбором")})
            continue

        choice = choose_mechanism(
            invariant, examples=observed,
            exactly_computable=EXACTLY_COMPUTABLE.get(invariant))
        if choice.mechanism != ALGORITHMIC:
            # A knob is the wrong shape of fix here. Saying so beats
            # offering one anyway, which is how a search ends up
            # optimising what it can adjust rather than what is wrong --
            # and until now this said it only to itself.
            refused.append({"addresses": invariant, "observed_failures": observed,
                            "strategy": choice.mechanism, "why": choice.reason})
            continue

        if lesson is not None and getattr(lesson, "exhausted", False):
            refused.append({"addresses": invariant, "observed_failures": observed,
                            "strategy": EXHAUSTED, "why": lesson.strategy()[1]})
            continue

        singles: List[Candidate] = []
        for knob in knobs:
            for option in knob.options:
                if option == current.get(knob.name):
                    continue
                singles.append(Candidate(
                    policy=_with(current, **{knob.name: option}),
                    addresses=invariant, observed=observed,
                    mechanism=choice.mechanism,
                    rationale=(f"{observed} нарушений «{invariant}» в записи; "
                               f"{knob.name}: {current.get(knob.name)!r} → "
                               f"{option!r}. {knob.why.split('.')[0]}.")))
        out.extend(singles)

        # Everything that addresses this invariant, at once. Worth trying
        # because single knobs can each be insufficient alone -- measured:
        # widening where the verb may sit does nothing for a request whose
        # verb form is not in the list either.
        combined = {k.name: k.options[1] for k in knobs
                    if len(k.options) > 1 and k.options[1] != current.get(k.name)}
        if len(combined) > 1:
            out.append(Candidate(
                policy=_with(current, **combined),
                addresses=invariant, observed=observed,
                mechanism=choice.mechanism,
                rationale=(f"{observed} нарушений «{invariant}»; всё сразу: "
                           + ", ".join(f"{k}={v!r}" for k, v in sorted(combined.items()))
                           + ". Отдельные настройки могут быть недостаточны "
                             "поодиночке.")))
    return Plan(candidates=tuple(out), refusals=tuple(refused))


def propose(violations: Sequence[Violation],
            current: Optional[Policy] = None,
            lessons: Optional[Dict[str, Any]] = None) -> List[Candidate]:
    """The changes half of `plan`. Kept because most callers want only it."""
    return list(plan(violations, current, lessons).candidates)


# --------------------------------------------------------------------------
# evaluating a candidate without running anything
# --------------------------------------------------------------------------

def dry_evaluable(policy: Policy) -> Tuple[bool, str]:
    """Whether this candidate can be scored without a language model."""
    blocked = sorted(set(policy.changes()) & NEEDS_A_MODEL)
    if blocked:
        return False, ("нельзя оценить всухую: " + ", ".join(blocked) +
                       " меняют то, что произойдёт после срабатывания "
                       "guard, а ответ повторной попытки без модели неизвестен")
    return True, "меняет только распознавание намерения"


def dry_responder(policy: Policy) -> Callable[[Situation], Tuple[str, List[ToolCall]]]:
    """Answer recorded situations under a policy, performing nothing.

    Where the policy changes nothing about a situation, the recorded
    answer is returned unchanged -- the honest statement that this
    candidate does not touch that turn.
    """
    from ..apps import intent as app_intent

    def respond(situation: Situation) -> Tuple[str, List[ToolCall]]:
        with policy_mod.use(policy):
            found = app_intent.match(situation.request)
            if found is None:
                return situation.recorded_answer, []
            feasible, note = app_intent.would_act(found)
            if not feasible:
                # A refusal reported with its reason is a real answer, and
                # a better one than a narration.
                return f"Не получилось: {note}", [ToolCall(found.tool, False, 0.0, note)]
            return (app_intent.describe_dry(found, note),
                    [ToolCall(found.tool, True, 0.0, "")])

    return respond


def dry_report(policy: Policy, situations: Sequence[Situation],
               current: Optional[Policy] = None) -> Dict[str, Any]:
    """Score a candidate against what MANA does now.

    The baseline is the **policy in force**, not the recorded answers.
    Found by looking at the panel after the 1C fix was adopted: turning
    the adopted setting back off scored 0.625 -> 0.75 and read as an
    improvement, because the recorded answers were produced under the
    older, worse policy. Scoring against the record means every candidate
    is compared with a version of MANA that no longer exists, and they
    all look good for free -- a bias that grows with every adoption.

    The recorded score is kept beside it: "what was actually said" is a
    real number, and losing it would hide how far behaviour has moved
    from the record the findings were drawn from.

    The `upper_bound` flag is not decoration. A feasible action can still
    fail, so this number is the best the candidate could do, never the
    number it would achieve.
    """
    from . import failure_domain as fd

    allowed, why = dry_evaluable(policy)
    if not allowed:
        return {"dry_evaluable": False, "reason": why}

    # The baseline is whatever the candidate is an alternative TO. That
    # is `current` when a caller supplied one -- `rank` generates
    # candidates from it, and comparing them against something else would
    # score them against a policy nobody proposed departing from.
    baseline = fd.evaluate(situations,
                           dry_responder(current or policy_mod.BASELINE))
    candidate = fd.evaluate(situations, dry_responder(policy))
    return {
        "dry_evaluable": True,
        "upper_bound": True,
        "note": ("сравнение с сегодняшним поведением; предполагается, что "
                 "выполнимое действие удаётся, а реальный запуск может "
                 "не удаться"),
        "policy": policy.changes(),
        "baseline_pass_rate": round(fd.pass_rate(baseline), 4),
        "candidate_pass_rate": round(fd.pass_rate(candidate), 4),
        "recorded_pass_rate": round(fd.pass_rate(fd.replay(situations)), 4),
        "counterexamples": fd.counterexamples(baseline, candidate),
        "changed": [j.situation_id for b, j in zip(baseline, candidate)
                    if b.passed != j.passed],
    }


def information_for(candidate: Candidate, lessons: Optional[Dict[str, Any]] = None,
                    conditions: Optional[Dict[str, Any]] = None
                    ) -> Tuple[float, str]:
    """How much a proper measurement of this candidate would tell us.

    Averaged over the settings it changes, because a candidate that turns
    two knobs is informative about both and reducing it to its best half
    would make every combination look as good as its most novel part.

    With no lesson to read, everything scores NEVER_MEASURED -- which is
    true of an empty record and keeps the ordering exactly as it was.
    """
    lesson = (lessons or {}).get(candidate.addresses)
    changed = sorted(candidate.policy.changes().items())
    if lesson is None or not changed:
        return (NEVER_MEASURED, "нет записи об измерениях этой оси")

    conditions = dict(conditions or {})
    scores: List[float] = []
    notes: List[str] = []
    for name, value in changed:
        attempts = lesson.measured_under(name, value)
        if not attempts:
            scores.append(NEVER_MEASURED)
            notes.append(f"{name}={value!r}: не измерялось")
            continue
        # The ledger's own rule, not a second copy of it: a result with
        # no conditions recorded is stale too, because nobody can say
        # whether they held.
        moved = all(a.stale_against(conditions)["stale"] for a in attempts)
        if moved:
            scores.append(CONDITIONS_MOVED)
            notes.append(f"{name}={value!r}: измерено в других условиях")
        else:
            scores.append(ALREADY_MEASURED)
            notes.append(f"{name}={value!r}: уже измерено в этих условиях")
    return (sum(scores) / len(scores), "; ".join(notes))


@dataclass(frozen=True)
class Scored:
    """A candidate priced the way every other experiment here is priced.

    `value` and `estimated_calls` are named to match `ExperimentPlan` and
    `Probe`, so `experiments.select` chooses among these without a third
    selector being written.
    """
    row: Dict[str, Any]
    information: float
    estimated_calls: int = 0
    priced: bool = False

    @property
    def value(self) -> float:
        """Information and cost. The dry bound is deliberately not here --
        it exists for some candidates and not others, and averaging it in
        would make "could not be evaluated" mean "no gain"."""
        cost_term = (VALUE_WEIGHTS["cost"]
                     * min(1.0, self.estimated_calls / COST_SCALE))
        return VALUE_WEIGHTS["information_gain"] * self.information + cost_term


def question_for(invariant: str) -> str:
    """How a candidate's aim is phrased in the findings ledger.

    Stable wording, because the ledger keys on it. Rephrasing this later
    would make every past result invisible, which is exactly the failure
    the ledger exists to prevent.
    """
    return f"Какая настройка политики устраняет «{invariant}»?"


def prior_for(candidate: Candidate, ledger: Any = None) -> Optional[Dict[str, Any]]:
    """What is already known about trying exactly this.

    Consulted before proposing again. Without it the ledger is a diary:
    it would hold "widening the verb list did not help" while the
    generator proposed widening the verb list next week.
    """
    from .findings import Ledger

    ledger = ledger if ledger is not None else Ledger()
    try:
        return ledger.already_tried(question_for(candidate.addresses),
                                    candidate.policy.changes(),
                                    {"observed_failures": candidate.observed})
    except Exception:
        return None


def rank(violations: Sequence[Violation], situations: Sequence[Situation],
         current: Optional[Policy] = None,
         ledger: Any = None,
         lessons: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Every candidate, with what is already known and what a dry run
    says, best first.

    Ordering is a suggestion of what to measure properly, not a verdict.
    Nothing here adopts anything: acceptance belongs to `core/gates.py`,
    on evidence, exactly as for a cognitive program -- a generator that
    could adopt its own output would be a second way to change the system.

    A candidate already measured and rejected under conditions that still
    hold sinks to the bottom rather than disappearing. Hiding it would
    make the generator silently unable to revisit a result, and the whole
    point of recording conditions is that a finding is a prior rather
    than a prohibition.
    """
    current = current or policy_mod.BASELINE
    out: List[Dict[str, Any]] = []
    for candidate in propose(violations, current, lessons):
        row = candidate.as_dict()
        row["dry"] = dry_report(candidate.policy, situations, current)
        conditions = {"observed_failures": candidate.observed}
        information, why = information_for(candidate, lessons, conditions)
        row["information"] = round(information, 4)
        row["information_why"] = why
        row["value"] = round(
            Scored(row=row, information=information).value, 4)
        lesson = (lessons or {}).get(candidate.addresses)
        if lesson is not None:
            # "Tried" and "learned" travel separately. Both of this
            # project's adoptions are NOT_EVALUATED, so a reader that
            # counted attempts would think two things were known here
            # when nothing is.
            row["lesson"] = {"observed": lesson.observed,
                             "tried": len(lesson.tried),
                             "learned": len(lesson.learned),
                             "strategy": lesson.strategy()[0]}
        prior = prior_for(candidate, ledger)
        if prior is not None:
            row["already_tried"] = {
                "verdict": prior["finding"]["verdict"],
                "when": prior["finding"]["created"],
                "measurement": prior["finding"]["measurement"],
                "conditions_moved": prior["staleness"]["stale"],
                "changed": prior["staleness"]["changed"]}
        out.append(row)

    def key(row: Dict[str, Any]) -> Tuple[int, float, float, float, int]:
        prior = row.get("already_tried") or {}
        settled = (prior.get("verdict") == "REJECTED"
                   and not prior.get("conditions_moved"))
        dry = row.get("dry") or {}
        # Value first: measure what nobody has measured before measuring a
        # predicted improvement on an axis already known. The dry bound is
        # the next key rather than part of the first, because it exists
        # for some candidates and not others.
        value = float(row.get("value", NEVER_MEASURED))
        if not dry.get("dry_evaluable"):
            return (0 if settled else 1, value, 0.0, 0.0, 0)
        gain = dry["candidate_pass_rate"] - dry["baseline_pass_rate"]
        broke = -float((dry.get("counterexamples") or {}).get("found", 0))
        return (0 if settled else 1, value, gain, broke, -len(row["changes"]))

    return sorted(out, key=key, reverse=True)


def choose(violations: Sequence[Violation], situations: Sequence[Situation],
           budget: int, current: Optional[Policy] = None, ledger: Any = None,
           lessons: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The candidate worth measuring next, or None.

    Delegates to `experiments.select`, the same selector that picks a
    pipeline experiment and a probe, so all three are held to one floor:
    running a worthless experiment because there is budget left is how a
    research loop converts compute into noise. None is a real answer.
    """
    rows = rank(violations, situations, current, ledger, lessons)
    scored = [Scored(row=row, information=float(row.get("information",
                                                       NEVER_MEASURED)),
                     estimated_calls=int(row.get("estimated_calls", 0) or 0))
              for row in rows
              if not ((row.get("already_tried") or {}).get("verdict") == "REJECTED"
                      and not (row.get("already_tried") or {}).get("conditions_moved"))]
    best = select(scored, budget)
    return best.row if best is not None else None
