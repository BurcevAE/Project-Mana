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
from .failure_domain import Situation
from .invariants import Violation

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Invariants whose fix is exactly computable, stated rather than guessed.
#: `choose_mechanism` takes this as a fact about the domain -- a factory
#: that inferred it from the name would be asserting something it has not
#: checked.
EXACTLY_COMPUTABLE = {
    "repeats_earlier_answer": True,     # two strings and a threshold
    "actionable_request_not_acted_on": True,   # a name against a known list
    "claimed_action_without_acting": True,     # a call list, and whether it ran
}

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


def propose(violations: Sequence[Violation],
            current: Optional[Policy] = None) -> List[Candidate]:
    """Changes worth measuring, given what actually went wrong.

    Returns an empty list when nothing went wrong, which is the common
    and correct answer: a system with no observed failures has nothing to
    propose, and a generator that produced candidates anyway would be
    changing a working system on speculation.
    """
    current = current or policy_mod.BASELINE
    counts: Dict[str, int] = {}
    for violation in violations:
        counts[violation.invariant] = counts.get(violation.invariant, 0) + 1

    out: List[Candidate] = []
    for invariant in sorted(counts, key=lambda k: (-counts[k], k)):
        observed = counts[invariant]
        choice = choose_mechanism(
            invariant, examples=observed,
            exactly_computable=EXACTLY_COMPUTABLE.get(invariant))
        if choice.mechanism != ALGORITHMIC:
            # A knob is the wrong shape of fix here. Saying so beats
            # offering one anyway, which is how a search ends up
            # optimising what it can adjust rather than what is wrong.
            continue

        knobs = policy_mod.knobs_for(invariant)
        if not knobs:
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
    return out


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


def dry_report(policy: Policy, situations: Sequence[Situation]) -> Dict[str, Any]:
    """Score a candidate on recorded situations, stating what it is worth.

    The `upper_bound` flag is not decoration. A feasible action can still
    fail, so this number is the best the candidate could do, never the
    number it would achieve.
    """
    from . import failure_domain as fd

    allowed, why = dry_evaluable(policy)
    if not allowed:
        return {"dry_evaluable": False, "reason": why}

    baseline = fd.replay(situations)
    candidate = fd.evaluate(situations, dry_responder(policy))
    return {
        "dry_evaluable": True,
        "upper_bound": True,
        "note": ("предполагается, что выполнимое действие удаётся; "
                 "реальный запуск может не удаться"),
        "policy": policy.changes(),
        "baseline_pass_rate": round(fd.pass_rate(baseline), 4),
        "candidate_pass_rate": round(fd.pass_rate(candidate), 4),
        "counterexamples": fd.counterexamples(baseline, candidate),
        "changed": [j.situation_id for b, j in zip(baseline, candidate)
                    if b.passed != j.passed],
    }


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
         ledger: Any = None) -> List[Dict[str, Any]]:
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
    out: List[Dict[str, Any]] = []
    for candidate in propose(violations, current):
        row = candidate.as_dict()
        row["dry"] = dry_report(candidate.policy, situations)
        prior = prior_for(candidate, ledger)
        if prior is not None:
            row["already_tried"] = {
                "verdict": prior["finding"]["verdict"],
                "when": prior["finding"]["created"],
                "measurement": prior["finding"]["measurement"],
                "conditions_moved": prior["staleness"]["stale"],
                "changed": prior["staleness"]["changed"]}
        out.append(row)

    def key(row: Dict[str, Any]) -> Tuple[int, float, float, int]:
        prior = row.get("already_tried") or {}
        settled = (prior.get("verdict") == "REJECTED"
                   and not prior.get("conditions_moved"))
        dry = row.get("dry") or {}
        if not dry.get("dry_evaluable"):
            return (0 if settled else 1, 0.0, 0.0, 0)
        gain = dry["candidate_pass_rate"] - dry["baseline_pass_rate"]
        broke = -float((dry.get("counterexamples") or {}).get("found", 0))
        return (0 if settled else 1, gain, broke, -len(row["changes"]))

    return sorted(out, key=key, reverse=True)
