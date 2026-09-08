"""
mana.cognition.failure_domain — turning found failures into a measurable domain.

Where this sits
---------------
    journal.py        what a turn actually did          (evidence)
    invariants.py     which turns contradicted themselves (findings)
    THIS MODULE       findings -> a domain with an oracle (measurable)
    core/gates.py     ACCEPTED / REJECTED / NOT_EVALUATED (existing)

The step this closes is the one that made the whole apparatus unusable on
real work. `ResearchCycle`, `choose_mechanism`, the genome and the
acceptance gates all speak one language: a set of tasks with an exact
oracle. Real failures do not arrive in that language, so none of that
machinery could ever be pointed at them, and every defect had to be
patched by hand.

The oracle is the invariant
---------------------------
`core/tasks.py` tasks carry a computed answer and a checker compares
against it. A recorded failure has no such answer -- nobody knows the
single right reply to "запусти конфигуратор". What is known exactly is
what the reply must NOT do: contradict itself, repeat an older answer,
claim an action nobody performed.

So the oracle here is not equality with a stored answer. It is the
invariant set re-applied to whatever a candidate produces. A situation is
passed when no invariant fires on it. That is decidable, cheap, and
involves no opinion about meaning -- which is exactly the property the
gates need and the reason this can be automated at all.

Gaming, and what actually stops it
-----------------------------------
The obvious cheat: answer "не знаю" to everything. Measured on the seven
recorded turns, it scored **0.8 against the baseline's 0.6** -- it fixed
the echo, left the healthy turns untouched and produced no
counterexamples. With thirty trials instead of five it would have passed
every gate and been ACCEPTED.

That is not a threshold set too low. Every invariant here is *negative*:
each says what an answer must not do, and silence does none of those
things. A purely negative oracle is maximised by saying nothing, and no
amount of tuning fixes that shape.

So a change gets a second way to break something, counted where
`gates.judge` already looks. `counterexamples` reports not only "this
situation used to pass and now fires" but also "the baseline answered
substantively here and the candidate answers with a stock refusal". A
change that degenerates into silence therefore fails the counterexample
gate rather than winning on the others.

That closes the obvious cheat, and it does not make the oracle positive.
Nothing here can tell a useful answer from a plausible wrong one, because
no invariant measures relevance -- the class this project has repeatedly
found to be outside mechanical reach.

Which forces an architectural rule rather than a patch: **a failure
domain is an additional gate, never a sufficient one.** A change may be
adopted on the strength of it only if it also holds on the synthetic
domains that do have computed answers. Measuring "stopped contradicting
itself" is not measuring "answers well", and this module must never be
read as though it were.

One trial per situation, not per invariant
-------------------------------------------
Three invariants on the same episode are not three independent
observations; McNemar over them would inflate the sample and buy
significance that is not there. So each situation contributes exactly one
paired trial -- passed iff nothing fired -- and the invariant that
originally flagged it becomes the trial's `domain`. The existing
`no_regression` gate then does the cross-invariant check for free: the
group of situations that used to fail on echoes must not get worse, and
neither must the group that was healthy.

The holdout is salted, like every other holdout here
-----------------------------------------------------
Split with `identity.salted_seed`, derived from the private key. Another
installation cannot regenerate this one's hidden situations, so agreement
between installations means something. Same reasoning as `core/splits.py`;
the mechanism is reused rather than reimplemented.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..journal import Episode, ToolCall
from . import invariants as inv
from ..core.gates import PairedOutcome

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Situations held back from anything that tunes. Chosen high because the
#: samples here are small and a hidden set of three tells nobody anything;
#: with a larger journal this is the number to revisit first.
HIDDEN_FRACTION = 0.3

#: The seed the split starts from, before salting. Fixed so the same
#: installation redraws the same hidden set every run and scores stay
#: comparable over time.
SPLIT_SEED = 20260907

#: The `domain` given to a situation nothing flagged. Healthy turns are in
#: the set on purpose: a change that fixes echoes by breaking everything
#: else must have somewhere to show up, and this is where.
CLEAN = "clean"


@dataclass(frozen=True)
class Situation:
    """A recorded turn, replayable as a question with a known context.

    Carries the session prefix because the invariants need it: whether an
    answer repeats an earlier one is not a property of the answer alone.
    """
    situation_id: str
    session: str
    request: str
    prefix: Tuple[Episode, ...] = ()
    recorded_answer: str = ""
    flagged_by: str = CLEAN

    def episode_for(self, answer: str,
                    calls: Sequence[ToolCall] = ()) -> Episode:
        """This situation answered differently, shaped for the oracle."""
        return Episode(episode_id=self.situation_id, session=self.session,
                       started=0.0, request=self.request, answer=answer,
                       calls=list(calls))


@dataclass(frozen=True)
class Judgement:
    """One situation, answered, and what the invariants said about it."""
    situation_id: str
    domain: str
    passed: bool
    fired: Tuple[str, ...] = ()
    #: What was actually said. Kept because passing the invariants is not
    #: the same as answering, and the degeneracy check needs the text.
    answer: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"situation_id": self.situation_id, "domain": self.domain,
                "passed": self.passed, "fired": list(self.fired),
                "answer": self.answer[:200]}


@dataclass(frozen=True)
class FailureDomain:
    """Situations drawn from the record, split into what may be tuned on
    and what may not."""
    name: str
    train: Tuple[Situation, ...] = ()
    hidden: Tuple[Situation, ...] = ()
    identity: str = ""

    def counts(self) -> Dict[str, Any]:
        flagged = [s for s in self.train + self.hidden if s.flagged_by != CLEAN]
        return {"name": self.name, "identity": self.identity,
                "train": len(self.train), "hidden": len(self.hidden),
                "flagged": len(flagged),
                "clean": len(self.train) + len(self.hidden) - len(flagged)}


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------

def situations_from(episodes: Sequence[Episode],
                    targets: Optional[Sequence[str]] = None) -> List[Situation]:
    """Every recorded turn as a replayable situation.

    Healthy turns included. A domain made only of failures can be scored
    perfectly by a change that ruins everything else, and would not notice.
    """
    if targets is None:
        targets = inv.known_targets()
    violations = inv.scan(episodes, targets)
    flagged: Dict[str, str] = {}
    for violation in violations:
        # First invariant to fire names the group. One situation, one
        # trial; the rest are still reported by the scan.
        flagged.setdefault(violation.episode_id, violation.invariant)

    out: List[Situation] = []
    for index, episode in enumerate(episodes):
        prefix = tuple(e for e in episodes[:index] if e.session == episode.session)
        out.append(Situation(
            situation_id=episode.episode_id, session=episode.session,
            request=episode.request, prefix=prefix,
            recorded_answer=episode.answer,
            flagged_by=flagged.get(episode.episode_id, CLEAN)))
    return out


def build(episodes: Sequence[Episode], name: str = "recorded_turns",
          hidden_fraction: float = HIDDEN_FRACTION,
          targets: Optional[Sequence[str]] = None,
          salted: bool = True) -> FailureDomain:
    """Split the record into a tuneable part and a hidden part.

    Stratified by group, so the hidden set is not accidentally all
    healthy turns -- with samples this small an unstratified draw does
    that often, and a hidden set with no failures in it confirms
    everything.
    """
    every = situations_from(episodes, targets)
    seed = SPLIT_SEED
    identity = f"{name}-v1"
    if salted:
        try:
            from ..core.identity import salted_seed, fingerprint
            seed = salted_seed(SPLIT_SEED)
            identity = f"{name}-v1@{fingerprint()[:12]}"
        except Exception:
            # An installation with no keypair still gets a split, just an
            # unsalted one -- and says so in `identity`, because a score
            # whose source cannot be told apart is a score that can be
            # quietly compared with somebody else's.
            identity = f"{name}-v1@unsalted"

    by_group: Dict[str, List[Situation]] = {}
    for situation in every:
        by_group.setdefault(situation.flagged_by, []).append(situation)

    rng = random.Random(seed)
    train: List[Situation] = []
    hidden: List[Situation] = []
    for group in sorted(by_group):
        members = sorted(by_group[group], key=lambda s: s.situation_id)
        rng.shuffle(members)
        cut = int(round(len(members) * hidden_fraction))
        hidden.extend(members[:cut])
        train.extend(members[cut:])
    return FailureDomain(name=name, train=tuple(train), hidden=tuple(hidden),
                         identity=identity)


# --------------------------------------------------------------------------
# the oracle
# --------------------------------------------------------------------------

#: What a candidate must provide: an answer and the tool calls it made.
#: The calls are not optional -- an answer graded without them cannot tell
#: "запустил" from "сказал, что запустил", which is the failure this whole
#: chain exists for.
Responder = Callable[[Situation], Tuple[str, Sequence[ToolCall]]]


def grade(situation: Situation, answer: str,
          calls: Sequence[ToolCall] = (),
          targets: Optional[Sequence[str]] = None) -> Judgement:
    """Run every invariant against this answer to this situation.

    Passed iff nothing fired. The invariants see the recorded prefix as
    history and the candidate as the current turn, which is what makes
    "repeats an earlier answer" answerable at all.
    """
    if targets is None:
        targets = inv.known_targets()
    episode = situation.episode_for(answer, calls)
    earlier = list(situation.prefix)
    fired: List[str] = []
    for invariant in inv.INVARIANTS:
        try:
            if invariant is inv.actionable_request_not_acted_on:
                violation = invariant(episode, earlier, targets)
            else:
                violation = invariant(episode, earlier)
        except Exception:
            continue
        if violation is not None:
            fired.append(violation.invariant)
    return Judgement(situation.situation_id, situation.flagged_by,
                     not fired, tuple(fired), answer or "")


def evaluate(situations: Sequence[Situation], respond: Responder,
             targets: Optional[Sequence[str]] = None) -> List[Judgement]:
    """Answer every situation and grade it.

    A responder that raises is graded as a failure rather than skipped:
    dropping it would quietly bias the sample toward the situations that
    happened to work, which is the same trap `curriculum.py` documents.
    """
    if targets is None:
        targets = inv.known_targets()
    out: List[Judgement] = []
    for situation in situations:
        try:
            answer, calls = respond(situation)
        except Exception:
            out.append(Judgement(situation.situation_id, situation.flagged_by,
                                 False, ("responder_error",), ""))
            continue
        out.append(grade(situation, answer, calls, targets))
    return out


def replay(situations: Sequence[Situation],
           targets: Optional[Sequence[str]] = None) -> List[Judgement]:
    """Grade what MANA actually said, from the record.

    The honest baseline for "did this change help?" is what a candidate
    is compared against, and re-running the old pipeline to get it costs
    calls for an answer already on disk.

    Its limit, stated rather than hidden: the recorded answer came from a
    particular state, and a candidate answering the same situations later
    is not a perfectly paired trial. Where the calls are affordable,
    running both arms live is the better comparison.
    """
    out: List[Judgement] = []
    for situation in situations:
        out.append(grade(situation, situation.recorded_answer, (), targets))
    return out


# --------------------------------------------------------------------------
# handing it to the gates that already exist
# --------------------------------------------------------------------------

def paired(baseline: Sequence[Judgement],
           candidate: Sequence[Judgement]) -> List[PairedOutcome]:
    """One trial per situation, in the shape `core/gates.py` judges.

    Only situations both arms answered. A trial missing from one arm is
    not a trial.
    """
    by_id = {j.situation_id: j for j in candidate}
    outcomes: List[PairedOutcome] = []
    for base in baseline:
        other = by_id.get(base.situation_id)
        if other is None:
            continue
        outcomes.append(PairedOutcome(
            task_id=base.situation_id, domain=base.domain,
            baseline_correct=base.passed, candidate_correct=other.passed))
    return outcomes


def pass_rate(judgements: Sequence[Judgement]) -> float:
    if not judgements:
        return 0.0
    return sum(1 for j in judgements if j.passed) / len(judgements)


#: Stock ways of saying nothing. A pattern, and therefore a guess -- but
#: a guess in the safe direction: a false positive here refuses a change,
#: and refusing a good change costs a retry while accepting a degenerate
#: one costs the ability to answer at all.
_REFUSALS = re.compile(
    r"^\s*(?:не\s+зна|не\s+могу|не\s+уверен|затрудня|"
    r"извини(?:те)?[,.\s]|к сожалению[,.\s]|нет данных|"
    r"i (?:don.t know|cannot|can.t))", re.IGNORECASE)

#: Below this an answer carries no content worth comparing.
SUBSTANTIVE_LENGTH = 40


def is_stock_refusal(text: str) -> bool:
    """A reply that declines rather than answers."""
    stripped = (text or "").strip()
    if not stripped:
        return True
    if _REFUSALS.match(stripped):
        return True
    return len(stripped) < SUBSTANTIVE_LENGTH


def counterexamples(baseline: Sequence[Judgement],
                    candidate: Sequence[Judgement]) -> Dict[str, Any]:
    """Situations the change broke that were fine before.

    Looking for counterexamples and finding none is evidence; not looking
    is not, and `gates.judge` refuses a claim nobody tried to break. For a
    change judged on failures, the natural attempt is the healthy turns:
    a fix for echoes that starts refusing ordinary questions shows up
    here and nowhere else.
    """
    by_id = {j.situation_id: j for j in candidate}
    broken: List[Dict[str, Any]] = []
    sought = 0
    degenerate = 0
    for base in baseline:
        other = by_id.get(base.situation_id)
        if other is None:
            continue

        # Degeneration is checked on every situation, not only the ones
        # that used to pass: a change that turns a substantive answer into
        # "не знаю" has broken it whether or not an invariant was firing
        # there before. Measured -- without this, a responder that
        # answered "Не знаю." to everything scored 0.8 against a baseline
        # of 0.6 with zero counterexamples found.
        sought += 1
        if (not is_stock_refusal(base.answer)
                and is_stock_refusal(other.answer)):
            degenerate += 1
            broken.append({"situation_id": base.situation_id,
                           "reason": "answer degenerated into a refusal",
                           "was": base.answer[:80], "now": other.answer[:80]})
            continue
        if base.passed and not other.passed:
            broken.append({"situation_id": base.situation_id,
                           "reason": "was passing, now fires",
                           "now_fires": list(other.fired)})
    return {"sought": sought, "found": len(broken), "broken": broken,
            "degenerated": degenerate}


def report(domain: FailureDomain,
           baseline: Sequence[Judgement],
           candidate: Sequence[Judgement]) -> Dict[str, Any]:
    """Everything measured, before anything is judged.

    Separate from the verdict on purpose: the numbers are worth reading
    even -- especially -- when the sample is too small for a verdict, and
    a caller that only ever sees ACCEPTED/NOT_EVALUATED learns nothing
    about how close it was.
    """
    outcomes = paired(baseline, candidate)
    by_group: Dict[str, Dict[str, Any]] = {}
    for outcome in outcomes:
        row = by_group.setdefault(outcome.domain,
                                  {"n": 0, "baseline": 0, "candidate": 0})
        row["n"] += 1
        row["baseline"] += int(outcome.baseline_correct)
        row["candidate"] += int(outcome.candidate_correct)
    return {
        "domain": domain.counts(),
        "trials": len(outcomes),
        "baseline_pass_rate": round(pass_rate(baseline), 4),
        "candidate_pass_rate": round(pass_rate(candidate), 4),
        "by_group": by_group,
        "counterexamples": counterexamples(baseline, candidate),
    }


def judge_change(description: str,
                 domain: FailureDomain,
                 baseline: Sequence[Judgement],
                 candidate: Sequence[Judgement],
                 hidden_baseline: Sequence[Judgement] = (),
                 hidden_candidate: Sequence[Judgement] = (),
                 claim_id: str = "") -> Dict[str, Any]:
    """Ask `core/gates.py` whether this change may be adopted.

    An adapter, not a second set of gates. Everything that decides --
    the sample-size floor, McNemar, the regression tolerance, the hidden
    confirmation, the three-state verdict -- lives in the module that
    already does this for the synthetic domains and is already tested.
    Writing a parallel judge here would let a failure domain quietly
    accept on easier terms than everything else, which is the loophole
    the "a claim states which gates apply to it" design exists to close.

    With today's record this returns NOT_EVALUATED, and that is the
    correct answer rather than a defect: three flagged turns is not
    evidence about anything.
    """
    from ..core.gates import Claim, Evidence, judge

    outcomes = paired(baseline, candidate)
    found = counterexamples(baseline, candidate)
    claim = Claim(
        claim_id=claim_id or f"failure-domain-{domain.name}",
        kind="program", description=description,
        asserts_domains=tuple(sorted({o.domain for o in outcomes})))
    evidence = Evidence(
        paired_dev=outcomes,
        counterexamples_sought=found["sought"],
        counterexamples_found=found["found"])
    if hidden_baseline and hidden_candidate:
        evidence.baseline_hidden = pass_rate(hidden_baseline)
        evidence.candidate_hidden = pass_rate(hidden_candidate)

    verdict = judge(claim, evidence)
    return {"verdict": verdict.as_dict(),
            "report": report(domain, baseline, candidate),
            "hidden_identity": domain.identity}
