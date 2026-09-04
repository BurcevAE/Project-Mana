"""
mana.cognition.research — the autonomous cycle, and the two things it adds
beyond calling the other modules in order.

Orchestration alone would not have earned a module. Wiring the self-model
to the curriculum to the lab is a script, and a script that reports
"cycle complete" is exactly the kind of code growth this project is
supposed to avoid. Two capabilities here do not exist anywhere else:

**1. Failure clustering.** The self-model records that a slice fails. It
does not say what the failing tasks have in common. Clustering finds
structure shared by failures and absent from successes -- "every failure
had nesting depth 'deep'" -- using the same cheap field extractors that
measure representation insufficiency. That is a different statement from
"hard arithmetic is 0.2", and it is the one a hypothesis can be written
against.

**2. Budget arbitration.** Measuring and experimenting compete for one
budget and answer different questions. A cycle that always measures never
invents anything; one that always experiments builds on numbers too soft
to support a conclusion. Each is scored on what it would resolve per call
spent, and the cycle spends where that is highest -- the same principle
the experiment lab applies within its own queue, raised one level.

The scale only works because both are priced in the same unit: total
width of the capability intervals. Examining the vocabulary was a third
competitor here and it was a design error -- its payoff is in collision
pairs, not interval width, so putting it on the same axis compared two
different quantities. Being free, it then won every round: a live run
spent three of five steps recommending fields, resolved nothing, and
tripped the diminishing-returns guard with 94% of its budget unspent.
It is now what it always was: a free by-product, recorded on every step,
competing for nothing.

Everything else here is bookkeeping: a transaction per cycle so an
interrupted run is findable, and state that survives a restart.

Not a memory cleanup
--------------------
The brief calls this a Dream Cycle and warns against making it a
housekeeping pass. It is not one: nothing here compacts memory or
summarises history. It looks for structure in what went wrong and spends
a budget trying to do something about it.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core import transaction
from ..core.cost import CostVector
from . import curriculum as cur
from . import experiments as lab
from .gaps import COMPETENCE, KNOWLEDGE, detect
from .representations import (FIELD_LIBRARY, insufficiency_gap,
                              measure_insufficiency, propose_fields)
from ..core import tasks as core_tasks
from . import self_model
from .genome import CognitiveGenome
from .self_model import Observation, SelfModel
from .synthesis import CapabilitySynthesizer

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"

#: Gates that can only pass if the caller supplied the evidence for
#: them. Without a hidden-set scorer and a counterexample search, every
#: experiment the cycle ran came back "REFUTED: failed hidden_confirms,
#: counterexamples" -- which reads like the mechanism did not work when
#: in fact nothing was measured. A missing measurement reported as a
#: negative finding is worse than no finding: it is a false one, and it
#: made a supported discovery impossible for the cycle to ever produce.
SUPPLIED_GATES = ("hidden_confirms", "counterexamples", "transfer")

MEASURE = "measure"          # run a lesson: resolve what is unknown
EXPERIMENT = "experiment"    # test an intervention: change what is known-bad
#: Both are priced in interval width, which is what makes them
#: comparable. Nothing else belongs on this list until it can be.
ACTIVITIES = (MEASURE, EXPERIMENT)

#: How much likelier failure must be inside the group than outside it.
#:
#: Lift was the obvious measure and it does not work: when 58% of
#: everything fails, the largest lift any group can reach is 1/0.58 =
#: 1.71, so a perfectly pure failure cluster scores below a threshold of
#: 2.0 and is never reported. A system worth researching is exactly one
#: that fails often, so lift goes blind precisely where this module is
#: needed. Excess -- the failure rate inside the group minus the rate
#: outside it -- stays meaningful at any base rate. Lift is still
#: computed and reported, because "three times likelier" reads better
#: than "0.31 higher"; it is just not what decides.
MIN_EXCESS = 0.25

#: Minimum failures sharing a value before it is a pattern rather than a
#: coincidence.
MIN_CLUSTER_SIZE = 3


@dataclass
class FailureCluster:
    """Structure shared by failures and absent from successes."""
    field_name: str
    value: Any
    failures: int
    successes: int
    excess: float          # failure rate inside the group minus outside
    lift: float            # reported, not decisive -- see MIN_EXCESS
    task_ids: List[str] = field(default_factory=list)

    @property
    def purity(self) -> float:
        total = self.failures + self.successes
        return self.failures / total if total else 0.0

    def describe(self) -> str:
        return (f"{self.field_name}={self.value}: {self.failures} отказов против "
                f"{self.successes} успехов — отказ здесь на {self.excess:.0%} "
                f"вероятнее, чем вне группы (в {self.lift:.1f}× чаще среднего)")

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["value"] = str(self.value)
        payload["purity"] = round(self.purity, 3)
        payload["excess"] = round(self.excess, 3)
        return payload


def cluster_failures(observations: Sequence[Observation], task_texts: Dict[str, str],
                     min_excess: float = MIN_EXCESS,
                     min_size: int = MIN_CLUSTER_SIZE) -> List[FailureCluster]:
    """What do the failures have in common that the successes do not?

    Comparison, not raw count: the commonest value in a task set is also
    the commonest value among its failures, and reporting that as a
    pattern would send every cycle chasing whatever the generator
    produces most. The question is whether failures are *concentrated*
    there -- measured against the rest of the corpus rather than against
    the average, for the reason MIN_EXCESS documents.

    Uses the structural field library, so this costs nothing and can run
    over the whole history. Nothing here consults a model -- a clustering
    judged by a model would put its opinion inside the definition of
    MANA's own weaknesses.
    """
    rows = [(o, task_texts.get(o.task_id)) for o in observations]
    rows = [(o, text) for o, text in rows if text is not None]
    if not rows:
        return []
    total_failures = sum(1 for o, _ in rows if not o.correct)
    if total_failures < min_size:
        return []
    base_rate = total_failures / len(rows)

    clusters: List[FailureCluster] = []
    for field_name, extractor in FIELD_LIBRARY.items():
        buckets: Dict[Any, List[Tuple[Observation, str]]] = {}
        for o, text in rows:
            buckets.setdefault(extractor(text), []).append((o, text))
        for value, members in buckets.items():
            failures = [o for o, _ in members if not o.correct]
            if len(failures) < min_size:
                continue
            outside = len(members) < len(rows)
            if not outside:
                continue        # a group holding everything explains nothing
            rate_in = len(failures) / len(members)
            rate_out = ((total_failures - len(failures)) /
                        (len(rows) - len(members)))
            excess = rate_in - rate_out
            if excess < min_excess:
                continue
            clusters.append(FailureCluster(
                field_name=field_name, value=value, failures=len(failures),
                successes=len(members) - len(failures), excess=excess,
                lift=rate_in / base_rate if base_rate else 0.0,
                task_ids=[o.task_id for o in failures][:10]))
    clusters.sort(key=lambda c: (-c.excess, -c.failures))
    return clusters


@dataclass
class ActivityOption:
    """One thing the cycle could do next, and what it would be worth."""
    activity: str
    description: str
    expected_resolution: float      # how much uncertainty it would remove
    estimated_calls: int
    payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> float:
        """Resolution per call. The comparison that lets a cheap
        measurement outrank an expensive experiment when the experiment
        would be built on numbers too soft to conclude from."""
        return self.expected_resolution / max(1, self.estimated_calls)

    def as_dict(self) -> Dict[str, Any]:
        return {"activity": self.activity, "description": self.description,
                "expected_resolution": round(self.expected_resolution, 4),
                "estimated_calls": self.estimated_calls, "value": round(self.value, 5)}


def options_for(model: SelfModel, task_texts: Dict[str, str],
                lesson_size: int = 12) -> List[ActivityOption]:
    """Everything worth spending a call on right now, priced.

    Measuring is offered for knowledge gaps, experimenting for competence
    gaps. Examining the vocabulary is not here: it costs nothing and
    narrows no interval, so it is recorded on every step instead of
    competing for one.

Every option here is priced in the same unit -- how much total
    interval width it would remove -- because that is the only thing
    that makes a lesson and an experiment comparable at all.
    """
    options: List[ActivityOption] = []
    caps = model.capabilities()

    # A slice nothing has ever attempted produces no gap, because a gap
    # needs observations to exist at all -- so a cycle starting from
    # nothing had nothing to do and stopped on its first move. An
    # untouched slice is not absent from the space of things worth
    # doing; it is the widest interval in it. Third time this project
    # has hit the same mistake: the curriculum valuing its first lesson
    # at zero, _uncertainty summing only measured slices, and this.
    for domain in core_tasks.DOMAINS:
        for band, _lo, _hi in self_model.BANDS:
            if f"{domain}/{band}" in caps:
                continue
            options.append(ActivityOption(
                activity=MEASURE,
                description=f"измерить {domain}/{band}: ни разу не пробовалась",
                expected_resolution=1.0, estimated_calls=lesson_size,
                payload={"domain": domain, "band": band, "gap": ""}))

    gaps = detect(model)

    for gap in gaps[:4]:
        if gap.kind == KNOWLEDGE:
            options.append(ActivityOption(
                activity=MEASURE,
                description=f"измерить {gap.capability_id}: {gap.description[:60]}",
                expected_resolution=gap.information_gain,
                estimated_calls=lesson_size,
                payload={"domain": gap.domain, "band": gap.band, "gap": gap.gap_id}))
        else:
            hypotheses = lab.hypotheses_from_gap(gap)
            for hypothesis in hypotheses[:1]:
                plan = lab.plan(hypothesis, model, trials=30)
                options.append(ActivityOption(
                    activity=EXPERIMENT,
                    description=f"проверить: {hypothesis.statement[:70]}",
                    expected_resolution=plan.information_gain + plan.capability_gain,
                    estimated_calls=plan.estimated_calls,
                    payload={"hypothesis": hypothesis, "plan": plan}))

    return options


def representation_findings(model: SelfModel, task_texts: Dict[str, str],
                            already_proposed: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Fields that would explain outcomes the current vocabulary cannot.

    Free: insufficiency and its proposals are computed from history
    without a brain call. Reported rather than acted on -- adopting a
    field means a genome mutation judged by core.gates, and a research
    cycle that could quietly widen its own representation would be
    changing the terms of its own measurement.
    """
    from .genome import CognitiveGenome
    representation = CognitiveGenome().representations["task_view"]
    insufficiency = measure_insufficiency(representation, model.observations, task_texts)
    if not insufficiency_gap(insufficiency):
        return []
    return [{"field_name": p.field_name, "separates_pairs": p.separates_pairs,
             "remaining_pairs": p.remaining_pairs, "reduction": round(p.reduction, 4),
             "collision_rate": round(insufficiency.rate, 3)}
            for p in propose_fields(insufficiency, model.observations, task_texts, limit=4)
            if p.field_name not in already_proposed]


def choose(options: Sequence[ActivityOption], budget_left: int) -> Optional[ActivityOption]:
    """The most resolution per call, among what is affordable.

    Returns None rather than the cheapest leftover when nothing is
    affordable: starting an experiment that the budget will cut off
    halfway spends the calls and produces no verdict.
    """
    affordable = [o for o in options if o.estimated_calls <= budget_left]
    return max(affordable, key=lambda o: o.value) if affordable else None


@dataclass
class CycleStep:
    step_id: str
    activity: str
    description: str
    calls_used: int
    resolution: float
    outcome: str
    at: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResearchCycle:
    """Runs itself until the budget or the evidence says stop."""

    def __init__(self, model: SelfModel, task_texts: Optional[Dict[str, str]] = None,
                 budget_calls: int = 600, max_steps: int = 12,
                 genome: Optional[CognitiveGenome] = None,
                 hidden_fn: Optional[Callable[[Sequence[str]], float]] = None,
                 counterexample_fn: Optional[Callable[[Any], Tuple[int, int]]] = None,
                 ) -> None:
        self.model = model
        #: Scores one operator chain against the hidden holdout. Optional
        #: because a cycle can legitimately run without one -- but then
        #: it cannot conclude anything, and says so rather than dressing
        #: the absence up as a refutation.
        #:
        #: Known gap: the calls this makes are NOT charged to
        #: `budget_calls`. It returns an accuracy and nothing else, so
        #: the cycle cannot see what it cost. The holdout has its own
        #: hard budget in `core.splits`, which is what protects the
        #: holdout -- but a cycle told it may spend 900 calls can spend
        #: more than 900. Charging it properly means widening the
        #: contract to return a cost, which is a change to how every
        #: caller supplies one.
        self.hidden_fn = hidden_fn
        #: Returns (probes sought, counterexamples found) for a hypothesis.
        self.counterexample_fn = counterexample_fn
        # The genome capabilities are installed into. Held here rather
        # than passed per call, because an adopted capability has to be
        # visible to the step after the one that adopted it -- otherwise
        # the cycle proves something and then compiles from the genome it
        # had before, which is the gap this whole layer exists to close.
        self.synthesizer = CapabilitySynthesizer(genome or CognitiveGenome())
        self.task_texts: Dict[str, str] = dict(task_texts or {})
        self.budget_calls = budget_calls
        self.max_steps = max_steps
        self.calls_used = 0
        self.proposed_fields: set = set()
        self.adopted: List[str] = []
        self.representation_findings: List[Dict[str, Any]] = []
        self._exhausted = False
        self.steps: List[CycleStep] = []
        self.clusters: List[FailureCluster] = []
        self.lab = lab.ExperimentLab(model, budget_calls=budget_calls)
        self.curriculum = cur.Curriculum(model, budget_calls=budget_calls)

    def budget_left(self) -> int:
        return max(0, self.budget_calls - self.calls_used)

    def stop_reason(self) -> str:
        if self._exhausted:
            return "нечего делать: не осталось работы, которую можно оплатить"
        if self.budget_left() <= 0:
            return f"бюджет исчерпан ({self.calls_used}/{self.budget_calls})"
        if len(self.steps) >= self.max_steps:
            return f"достигнут предел шагов ({self.max_steps})"
        recent = [s.resolution for s in self.steps[-3:]]
        if len(recent) >= 3 and max(recent) < 0.02:
            return "убывающая отдача: последние 3 шага ничего не разрешили"
        return ""

    def analyse(self) -> List[FailureCluster]:
        """Look at what went wrong before deciding what to do about it."""
        self.clusters = cluster_failures(self.model.observations, self.task_texts)
        return self.clusters

    def step(self, lesson_runner: cur.LessonRunner,
             trial_runner: Optional[lab.TrialRunner] = None,
             task_source: Optional[Callable[[str, int], Sequence[Any]]] = None) -> Optional[CycleStep]:
        """One decision and its consequence, inside one transaction.

        The transaction is per step rather than per cycle: a cycle
        interrupted after three useful steps should not look like three
        lost ones, and `unfinished()` should name the step that was
        actually in flight.
        """
        if self.stop_reason():
            return None
        # Re-read the failures before every decision, not once at the
        # start: analysing a model that has not run anything yet finds
        # nothing, and the clusters a hypothesis gets written against are
        # exactly the ones the last few steps produced. Costs no calls.
        self.analyse()
        # Free, so it happens every step rather than instead of one.
        for finding in representation_findings(self.model, self.task_texts,
                                               self.proposed_fields):
            self.proposed_fields.add(finding["field_name"])
            self.representation_findings.append(finding)
        options = options_for(self.model, self.task_texts)
        chosen = choose(options, self.budget_left())
        if chosen is None:
            # Nothing affordable left is a stop condition in its own
            # right, and a run that ends without naming one cannot be
            # told apart from a run that crashed.
            self._exhausted = True
            return None

        with transaction.TransactionScope(
                uuid.uuid4().hex[:12], "research", chosen.description[:120]) as txn:
            txn.step(transaction.MEASURED, activity=chosen.activity,
                     estimated_calls=chosen.estimated_calls)
            before = self._uncertainty()

            if chosen.activity == MEASURE:
                outcome = self._do_measure(chosen, lesson_runner)
            elif chosen.activity == EXPERIMENT and trial_runner and task_source:
                outcome = self._do_experiment(chosen, trial_runner, task_source)
            else:
                outcome = "пропущен: нет исполнителя для этого вида работы"

            resolution = max(0.0, before - self._uncertainty())
            record = CycleStep(step_id=uuid.uuid4().hex[:10], activity=chosen.activity,
                               description=chosen.description,
                               calls_used=chosen.estimated_calls, resolution=resolution,
                               outcome=outcome)
            self.steps.append(record)
            txn.commit(result=outcome, resolution=round(resolution, 4))
        return record

    def _uncertainty(self) -> float:
        """Total width of every interval over a *fixed* set of slices.

        The quantity a research cycle exists to reduce, and the one that
        makes "did this step help?" answerable without waiting for a
        capability to cross a threshold.

        Summing only the measured slices was the obvious version and it
        is wrong: measuring a slice for the first time adds its width to
        a total that did not contain it, so the sum goes *up* and the
        step that discovered a whole new capability is scored as a loss.
        An unmeasured slice is not absent from the total -- it is in it
        at width 1.0, because "nothing known" is the widest interval
        there is. Same mistake the curriculum made valuing its first
        lesson at zero, in a different place.
        """
        caps = self.model.capabilities()
        total = 0.0
        for domain in core_tasks.DOMAINS:
            for band, _lo, _hi in self_model.BANDS:
                cap = caps.get(f"{domain}/{band}")
                total += cap.uncertainty if cap else 1.0
        return total

    def _do_measure(self, option: ActivityOption, runner: cur.LessonRunner) -> str:
        # A capability recorded under band "all" has no stage to practise
        # at; medium is where a lesson tells the most about which way the
        # ceiling lies.
        band = option.payload.get("band")
        stage = band if band in cur.STAGES else cur.MEDIUM
        goal = cur.LearningGoal(
            goal_id=uuid.uuid4().hex[:10], domain=option.payload["domain"],
            stage=stage, reason=option.description, kind=KNOWLEDGE)
        lesson = self.curriculum.plan_lesson(goal)
        if lesson is None:
            return "урок не спланирован"
        self.curriculum.run_lesson(lesson, runner)
        self.calls_used += lesson.calls_used
        # Keep the prompts the lesson used, or clustering has nothing to
        # read. Depending on the caller's runner to record them made the
        # cycle's own analysis a function of how someone else wired it
        # up. Regenerated from the lesson's own seed, so these are the
        # same tasks and not a second draw.
        for task in core_tasks.generate(lesson.goal.domain, lesson.size, lesson.seed,
                                        difficulty_range=lesson.difficulty):
            self.task_texts.setdefault(task.task_id, task.prompt)
        return f"урок: {lesson.correct}/{lesson.observations}, уточнено {lesson.information:.3f}"

    def _do_experiment(self, option: ActivityOption, runner: lab.TrialRunner,
                       task_source: Callable[[str, int], Sequence[Any]]) -> str:
        plan = option.payload["plan"]
        tasks_for_trial = task_source(plan.hypothesis.domain, plan.trials)
        measurement = self.lab.run_experiment(plan, tasks_for_trial, runner)
        self.calls_used += measurement.calls_used
        discovery = self.lab.conclude(plan, measurement, **self._external_evidence(plan))
        outcome = self._describe(discovery)
        return outcome + self._maybe_synthesize(discovery, runner, task_source)

    def _external_evidence(self, plan: Any) -> Dict[str, Any]:
        """Whatever the caller can supply beyond the paired dev run."""
        evidence: Dict[str, Any] = {}
        if self.hidden_fn is not None:
            evidence["hidden"] = (self.hidden_fn(plan.hypothesis.baseline_steps),
                                  self.hidden_fn(plan.hypothesis.candidate_steps))
        if self.counterexample_fn is not None:
            evidence["counterexamples"] = self.counterexample_fn(plan.hypothesis)
        return evidence

    def _unmeasured_gates(self) -> Tuple[str, ...]:
        missing = []
        if self.hidden_fn is None:
            missing.append("hidden_confirms")
        if self.counterexample_fn is None:
            missing.append("counterexamples")
        return tuple(missing)

    def _describe(self, discovery: Any) -> str:
        """Report a refutation as one only when it rests on measurement.

        A verdict that failed nothing except gates whose evidence was
        never collected is not a result about the mechanism. Calling it
        REFUTED would put a false negative into the record and, worse,
        stop the mechanism being tried again.
        """
        failed = tuple(discovery.verdict.get("failed_gates") or ())
        unmeasured = self._unmeasured_gates()
        if failed and set(failed) <= set(unmeasured):
            return ("эксперимент: не удалось судить — нет данных для гейтов "
                    + ", ".join(failed))
        return f"эксперимент: {discovery.status} — {discovery.verdict['reason'][:70]}"

    def _maybe_synthesize(self, discovery: Any, runner: lab.TrialRunner,
                          task_source: Callable[[str, int], Sequence[Any]]) -> str:
        """Try to turn a supported discovery into a capability that persists.

        The confirming run is a second measurement on tasks the first did
        not use, and it is not optional: the discovery's own verdict
        belongs to the experiment's claim, so adopting on it would be the
        same evidence counted twice -- which `genome.apply` refuses
        outright. Skipped when the budget cannot cover it, because a
        confirmation cut off halfway spends the calls and settles nothing.
        """
        proposal = self.synthesizer.consider(discovery)
        if proposal is None:
            return ""
        if self._unmeasured_gates():
            # A confirmation that cannot clear the same gates the
            # discovery could not clear will spend the budget and refuse
            # the capability every time.
            return (f"; способность {proposal.name} нельзя подтвердить: "
                    f"нет данных для {', '.join(self._unmeasured_gates())}")
        confirm_plan = lab.plan(_confirming_hypothesis(discovery), self.model,
                                trials=lab.gates.MIN_PAIRED_TRIALS)
        if confirm_plan.estimated_calls > self.budget_left():
            return f"; способность {proposal.name} ждёт подтверждения (не хватает бюджета)"
        fresh = task_source(confirm_plan.hypothesis.domain, confirm_plan.trials)
        confirmation = self.lab.run_experiment(confirm_plan, fresh, runner)
        self.calls_used += confirmation.calls_used
        install_evidence = self._external_evidence(confirm_plan)
        installed = self.synthesizer.install(
            proposal, confirmation.outcomes,
            cost=CostVector(calls=confirmation.calls_used,
                            wall_seconds=confirmation.elapsed,
                            unmeasured_token_calls=confirmation.calls_used),
            **install_evidence)
        if installed:
            self.adopted.append(proposal.name)
            return f"; принята способность {proposal.name}"
        return (f"; способность {proposal.name} отклонена на подтверждении: "
                f"{proposal.confirmation.get('reason', '?')[:60]}")

    def run(self, lesson_runner: cur.LessonRunner,
            trial_runner: Optional[lab.TrialRunner] = None,
            task_source: Optional[Callable[[str, int], Sequence[Any]]] = None,
            max_steps: Optional[int] = None) -> Dict[str, Any]:
        limit = max_steps or self.max_steps
        while len(self.steps) < limit:
            if self.step(lesson_runner, trial_runner, task_source) is None:
                break
        self.analyse()          # so the report describes the end, not the start
        return self.report()

    def report(self) -> Dict[str, Any]:
        by_activity: Dict[str, int] = {}
        for s in self.steps:
            by_activity[s.activity] = by_activity.get(s.activity, 0) + 1
        return {
            "steps": len(self.steps),
            "by_activity": by_activity,
            "calls_used": self.calls_used,
            "budget_left": self.budget_left(),
            "stop_reason": self.stop_reason(),
            "total_resolution": round(sum(s.resolution for s in self.steps), 4),
            "failure_clusters": [c.as_dict() for c in self.clusters[:5]],
            "representation_findings": self.representation_findings,
            "adopted_capabilities": list(self.adopted),
            "genome": self.synthesizer.genome.signature(),
            "history": [s.as_dict() for s in self.steps],
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.report(), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(path)


def _confirming_hypothesis(discovery: Any) -> lab.Hypothesis:
    """The same comparison again, as its own hypothesis.

    A copy rather than the original: the original carries the status the
    first experiment left on it, and reusing it would let a second run
    overwrite the record of the first.
    """
    h = discovery.hypothesis
    return lab.Hypothesis(
        hypothesis_id=uuid.uuid4().hex[:12],
        statement="подтверждение: " + str(h.get("statement", ""))[:120],
        gap_id=str(h.get("gap_id", "")), domain=str(h.get("domain", "")),
        band=str(h.get("band", "")),
        baseline_steps=tuple(h.get("baseline_steps") or ()),
        candidate_steps=tuple(h.get("candidate_steps") or ()),
        source="confirmation")
