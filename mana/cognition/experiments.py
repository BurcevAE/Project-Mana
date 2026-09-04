"""
mana.cognition.experiments — turning a gap into a question that can be
answered, and choosing which question to answer next.

    gap -> hypothesis -> experiment -> measurement -> verdict -> discovery

Why hypotheses are generated mechanically first
-----------------------------------------------
The obvious design asks a model to invent hypotheses. That makes the whole
loop untestable without a network, unreproducible between runs, and
dependent on the one component whose judgement this project is not allowed
to trust. So hypotheses are derived from the gap's own failure pattern --
"most failures here are format violations, so try a program that ends with
a formatting step" -- and an LLM proposer is an optional accelerant added
on top, never the mechanism. A loop that only works when a model is
available is a loop that cannot be debugged.

Choosing what to run next
-------------------------
Not "the experiment with the best expected score". §12 asks for the
experiment that yields the most valuable new information, and those differ:
a candidate that will probably win by a little teaches less than one whose
outcome is genuinely unknown. `expected_value` combines the information a
result would resolve, the capability it would gain if it worked, and what
it costs -- with declared weights, because a ranking that decides what the
system studies should not be implicit in a sum.

Stopping
--------
Three conditions, and the third is the one that matters. A budget stops a
runaway loop; a maximum stops an endless one; but **diminishing returns**
stops a loop that is still spending usefully-looking effort on questions
that have stopped resolving anything. Without it, an autonomous researcher
keeps running experiments forever because each one is individually
affordable.
"""
from __future__ import annotations

import json
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core import gates, transaction
from ..core.cost import CostVector
from ..core.gates import Claim, Evidence, PairedOutcome
from .gaps import Gap, detect
from .programs import Budget, CognitiveProgram
from .self_model import SelfModel

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.3"

PROPOSED = "PROPOSED"
TESTING = "TESTING"
SUPPORTED = "SUPPORTED"
REFUTED = "REFUTED"
ABANDONED = "ABANDONED"

#: Declared, like PRIORITY_WEIGHTS. What a system chooses to investigate
#: shapes everything it later knows, so the trade-off is written down.
VALUE_WEIGHTS = {
    "information_gain": 1.0,    # how much the outcome would resolve
    "capability_gain": 0.8,     # how much it would help if it worked
    "cost": -0.5,               # calls it would spend
}

#: Below this marginal value, an experiment is not worth running even with
#: budget left over.
MIN_EXPERIMENT_VALUE = 0.05

#: How many consecutive uninformative experiments before the lab stops.
#: An autonomous researcher without this keeps going forever, because each
#: individual experiment remains affordable.
DIMINISHING_WINDOW = 4


@dataclass
class Hypothesis:
    """A claim that a specific intervention will improve a specific slice.

    Deliberately narrow. "Simulation helps planning" cannot be refuted;
    "adding CRITIQUE before ANSWER raises arithmetic/hard above its current
    interval" can, and a hypothesis that cannot be refuted is not one.
    """
    hypothesis_id: str
    statement: str
    gap_id: str
    domain: str
    band: str
    baseline_steps: Tuple[str, ...]
    candidate_steps: Tuple[str, ...]
    predicted_effect: str = ""
    status: str = PROPOSED
    source: str = "gap"          # gap | llm | manual
    created: float = field(default_factory=time.time)
    result: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentPlan:
    """What would be run, and what it would cost before anything is spent."""
    plan_id: str
    hypothesis: Hypothesis
    trials: int
    estimated_calls: int
    information_gain: float
    capability_gain: float
    value: float
    #: Whether an experiment this size could produce a verdict at all.
    #: Priced alongside cost and value, because an experiment that cannot
    #: conclude is not cheap -- it is worthless at any price.
    power: Optional[Any] = None

    def as_dict(self) -> Dict[str, Any]:
        return {"plan_id": self.plan_id, "hypothesis": self.hypothesis.as_dict(),
                "trials": self.trials, "estimated_calls": self.estimated_calls,
                "information_gain": round(self.information_gain, 4),
                "capability_gain": round(self.capability_gain, 4),
                "value": round(self.value, 4),
                "power": self.power.as_dict() if self.power else None}


@dataclass
class Measurement:
    """What actually happened. Paired, because the gate needs it paired."""
    plan_id: str
    outcomes: List[PairedOutcome] = field(default_factory=list)
    calls_used: int = 0
    elapsed: float = 0.0
    baseline_failures: int = 0
    candidate_failures: int = 0

    def summary(self) -> Dict[str, Any]:
        return {"plan_id": self.plan_id, "trials": len(self.outcomes),
                "baseline": round(gates.accuracy(self.outcomes, "baseline"), 4),
                "candidate": round(gates.accuracy(self.outcomes, "candidate"), 4),
                "calls_used": self.calls_used, "elapsed": round(self.elapsed, 2)}


@dataclass
class Discovery:
    """An accepted result, in the shape §37 asks for.

    Kept even when the verdict was a rejection: a refuted hypothesis is a
    result, and a lab that records only its successes cannot tell whether
    it has learned anything or merely been lucky.
    """
    discovery_id: str
    hypothesis: Dict[str, Any]
    measurement: Dict[str, Any]
    verdict: Dict[str, Any]
    status: str
    at: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# hypothesis generation
# ---------------------------------------------------------------------------

#: Interventions, keyed by the failure pattern they answer. Each is a
#: change to the program chain, expressed as steps rather than prose,
#: because an experiment has to be runnable and "be more careful" is not.
_INTERVENTIONS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "format": (("OBSERVE", "GENERATE", "CRITIQUE", "REPAIR", "ANSWER"),
               "критик заметит нарушение формата и REPAIR его исправит"),
    "wrong": (("OBSERVE", "GENERATE", "VERIFY", "CRITIQUE", "REPAIR", "ANSWER"),
              "проверка независимым оракулом до ответа поймает неверный результат"),
    "hard": (("OBSERVE", "DECOMPOSE", "GENERATE", "SYNTHESIZE", "ANSWER"),
             "разбиение снижает сложность каждого шага"),
    "unknown": (("OBSERVE", "RETRIEVE", "GENERATE", "ANSWER"),
                "контекст из памяти может дать недостающее"),
}

BASELINE_STEPS = ("OBSERVE", "GENERATE", "ANSWER")


@dataclass(frozen=True)
class Power:
    """Whether an experiment of this size could detect anything at all."""
    trials: int
    min_discordant: int
    mde: float                  # minimum detectable net improvement
    headroom: float             # how much the slice could improve at most
    ok: bool
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return {"trials": self.trials, "min_discordant": self.min_discordant,
                "mde": round(self.mde, 4), "headroom": round(self.headroom, 4),
                "ok": self.ok, "reason": self.reason}


def _min_discordant_pairs() -> int:
    """The fewest one-directional discordant pairs the GATE calls significant.

    Derived by asking the gate rather than by deriving it here. A number
    computed independently would be a second opinion about the same
    threshold, and the day the gate's correction changes, the two would
    disagree silently -- with this layer refusing experiments the gate
    would have accepted, or worse, allowing ones it cannot.
    """
    for count in range(2, 60):
        outcomes = ([PairedOutcome(f"d{i}", "x", False, True) for i in range(count)] +
                    [PairedOutcome(f"s{i}", "x", True, True) for i in range(count)])
        if gates.mcnemar(outcomes)["p_value"] < gates.ALPHA:
            return count
    return 60                                          # pragma: no cover


#: Cached: the gate's thresholds do not change at runtime.
MIN_DISCORDANT = _min_discordant_pairs()


def power(trials: int, baseline_score: float) -> Power:
    """Can an experiment this size produce a verdict at all?

    Two ways it cannot, both knowable before a single call is spent:

    **Too few trials.** Below `gates.MIN_PAIRED_TRIALS` the sample-size
    gate refuses regardless of the result.

    **No room to improve.** Significance needs at least
    `MIN_DISCORDANT` pairs where the candidate wins and the baseline
    does not, so the smallest detectable net improvement is
    `MIN_DISCORDANT / trials`. A slice already at 1.00 has no room for
    it, and one at 0.95 has less room than 30 trials can resolve.

    What this does NOT catch, said plainly: an experiment on a slice at
    0.00 passes here, because the candidate might improve it by
    anything. A live run spent 114 calls on exactly that and got a
    refutation. I earlier called that waste; it was not -- it was
    exploration whose outcome was unknowable in advance, and no power
    calculation could have said otherwise.
    """
    if trials < gates.MIN_PAIRED_TRIALS:
        return Power(trials, MIN_DISCORDANT, 1.0, 0.0, False,
                     f"{trials} пар против {gates.MIN_PAIRED_TRIALS}, "
                     f"требуемых гейтом sample_size")
    mde = MIN_DISCORDANT / float(trials)
    headroom = max(0.0, 1.0 - float(baseline_score))
    if headroom < mde:
        return Power(trials, MIN_DISCORDANT, mde, headroom, False,
                     f"срез на {baseline_score:.2f}: улучшить можно максимум на "
                     f"{headroom:.2f}, а различить гейт может только "
                     f"{mde:.2f} при {trials} парах")
    return Power(trials, MIN_DISCORDANT, mde, headroom, True,
                 f"различимо от {mde:.2f}, запас {headroom:.2f}")


def hypotheses_from_gap(gap: Gap) -> List[Hypothesis]:
    """Derive testable interventions from what the failures look like.

    One hypothesis per distinct failure pattern, not one per gap: a slice
    that fails half on format and half on arithmetic has two different
    problems, and an intervention aimed at their average would address
    neither.
    """
    if gap.kind != "competence":
        # A knowledge gap is closed by measuring, not by intervening.
        # Proposing a mechanism for something not yet known to be broken
        # is how a search wastes its budget on phantoms.
        return []
    failure_modes = (gap.evidence.get("failure_modes") or {})
    patterns = [p for p, count in sorted(failure_modes.items(), key=lambda kv: -kv[1])
                if count > 0][:2]
    if not patterns:
        patterns = ["unknown"]
    if gap.band == "hard" and "hard" not in patterns:
        patterns.append("hard")

    out: List[Hypothesis] = []
    for pattern in patterns:
        steps, why = _INTERVENTIONS.get(pattern, _INTERVENTIONS["unknown"])
        if steps == BASELINE_STEPS:
            continue
        out.append(Hypothesis(
            hypothesis_id=uuid.uuid4().hex[:12],
            statement=(f"На срезе {gap.capability_id} программа {' → '.join(steps)} "
                       f"даст более высокую точность, чем {' → '.join(BASELINE_STEPS)}"),
            gap_id=gap.gap_id, domain=gap.domain, band=gap.band,
            baseline_steps=BASELINE_STEPS, candidate_steps=steps,
            predicted_effect=why, source="gap"))
    return out


def hypotheses_from_model(model: SelfModel, limit: int = 4) -> List[Hypothesis]:
    out: List[Hypothesis] = []
    for gap in detect(model):
        out.extend(hypotheses_from_gap(gap))
        if len(out) >= limit:
            break
    return out[:limit]


# ---------------------------------------------------------------------------
# planning and selection
# ---------------------------------------------------------------------------

def plan(hypothesis: Hypothesis, model: SelfModel, trials: int = 30,
         calls_per_trial: int = 0) -> ExperimentPlan:
    """Cost and value an experiment before running it.

    `trials` defaults to the gate's minimum rather than to something
    generous: an experiment sized below what the gate will accept cannot
    produce a verdict, and one sized far above it spends budget buying
    resolution nobody asked for.
    """
    from .gaps import expected_information_gain
    caps = model.capabilities()
    slice_key = f"{hypothesis.domain}/{hypothesis.band}"
    cap = caps.get(slice_key)

    info = expected_information_gain(cap, probe=trials) if cap else 1.0
    headroom = max(0.0, 0.75 - (cap.score if cap else 0.0))
    if calls_per_trial <= 0:
        # Baseline plus candidate, each costing about one call per
        # generative step. Estimated, not guessed: the chains are known.
        calls_per_trial = _calls(hypothesis.baseline_steps) + _calls(hypothesis.candidate_steps)
    estimated = trials * calls_per_trial

    value = (VALUE_WEIGHTS["information_gain"] * info +
             VALUE_WEIGHTS["capability_gain"] * headroom +
             VALUE_WEIGHTS["cost"] * min(1.0, estimated / 500.0))
    return ExperimentPlan(
        plan_id=uuid.uuid4().hex[:12], hypothesis=hypothesis, trials=trials,
        estimated_calls=estimated, information_gain=info,
        capability_gain=headroom, value=value,
        power=power(trials, cap.score if cap else 0.0))


_GENERATIVE = {"GENERATE", "CRITIQUE", "REPAIR", "SYNTHESIZE", "DECOMPOSE",
               "ABSTRACT", "PREDICT", "COUNTEREXAMPLE", "COMPARE"}


def _calls(steps: Sequence[str]) -> int:
    return sum(1 for s in steps if s in _GENERATIVE)


def select(plans: Sequence[ExperimentPlan], budget_left: int) -> Optional[ExperimentPlan]:
    """The most valuable experiment that fits, or nothing.

    Returns None rather than the cheapest available when nothing clears
    MIN_EXPERIMENT_VALUE: running a worthless experiment because there is
    budget left is how a research loop converts compute into noise.
    """
    affordable = [p for p in plans if p.estimated_calls <= budget_left]
    if not affordable:
        return None
    best = max(affordable, key=lambda p: p.value)
    return best if best.value >= MIN_EXPERIMENT_VALUE else None


# ---------------------------------------------------------------------------
# the lab
# ---------------------------------------------------------------------------

#: Runs one program over one task and reports whether it was correct.
#: Injected so the lab is testable without brains, and so the same lab can
#: drive a live agent or a simulation.
TrialRunner = Callable[[Tuple[str, ...], Any], Tuple[bool, int]]


class UnderpoweredExperiment(RuntimeError):
    """An experiment that could not have concluded anything."""


class ExperimentLab:
    """Runs the loop, and knows when to stop running it."""

    def __init__(self, model: SelfModel, budget_calls: int = 2000,
                 max_experiments: int = 20) -> None:
        self.model = model
        self.budget_calls = budget_calls
        self.max_experiments = max_experiments
        self.calls_used = 0
        self.discoveries: List[Discovery] = []
        self._recent_values: List[float] = []

    # ---------- stopping ----------

    def budget_left(self) -> int:
        return max(0, self.budget_calls - self.calls_used)

    def stop_reason(self) -> str:
        if self.budget_left() <= 0:
            return f"бюджет исчерпан ({self.calls_used}/{self.budget_calls} вызовов)"
        if len(self.discoveries) >= self.max_experiments:
            return f"достигнут предел экспериментов ({self.max_experiments})"
        if self._diminishing():
            return (f"убывающая отдача: последние {DIMINISHING_WINDOW} экспериментов "
                    f"ничего не разрешили")
        return ""

    def _diminishing(self) -> bool:
        """Have the last few experiments stopped resolving anything?

        Measured on realised information, not on acceptance: a run of
        rejections that each narrowed an interval is productive, while a
        run that changed nothing is not, however affordable each one was.
        """
        if len(self._recent_values) < DIMINISHING_WINDOW:
            return False
        window = self._recent_values[-DIMINISHING_WINDOW:]
        return statistics.fmean(window) < MIN_EXPERIMENT_VALUE

    # ---------- running ----------

    def run_experiment(self, plan_: ExperimentPlan, tasks: Sequence[Any],
                       runner: TrialRunner, verifier: Any = None) -> Measurement:
        if plan_.power is not None and not plan_.power.ok:
            # Refused before a single call. An experiment that cannot
            # reach significance produces a refutation whatever happens,
            # and that refutation then goes into the record as evidence
            # against a mechanism that was never actually tested.
            raise UnderpoweredExperiment(plan_.power.reason)
        """Run baseline and candidate over the same tasks.

        Same tasks, both arms, so the comparison is paired -- the gate
        reads discordant pairs, and unpaired samples would throw away the
        biggest source of variance (task difficulty) and need several
        times the trials to see the same effect.
        """
        measurement = Measurement(plan_id=plan_.plan_id)
        started = time.perf_counter()
        for task in tasks[:plan_.trials]:
            base_ok, base_calls = runner(plan_.hypothesis.baseline_steps, task)
            cand_ok, cand_calls = runner(plan_.hypothesis.candidate_steps, task)
            measurement.outcomes.append(PairedOutcome(
                task_id=getattr(task, "task_id", "?"),
                domain=getattr(task, "domain", plan_.hypothesis.domain),
                baseline_correct=base_ok, candidate_correct=cand_ok))
            measurement.calls_used += base_calls + cand_calls
            measurement.baseline_failures += int(not base_ok)
            measurement.candidate_failures += int(not cand_ok)
        measurement.elapsed = time.perf_counter() - started
        self.calls_used += measurement.calls_used
        return measurement

    def conclude(self, plan_: ExperimentPlan, measurement: Measurement,
                 hidden: Optional[Tuple[float, float]] = None,
                 transfer: Optional[Tuple[float, float]] = None,
                 counterexamples: Tuple[int, int] = (0, 0)) -> Discovery:
        """Hand the evidence to the gate and record whatever it says.

        The lab does not decide. It assembles, asks, and writes down the
        answer -- including a refutation, because a lab that records only
        successes cannot tell learning from luck.
        """
        claim = Claim(
            claim_id=plan_.plan_id, kind="program",
            description=plan_.hypothesis.statement,
            asserts_transfer=transfer is not None)
        evidence = Evidence(
            paired_dev=measurement.outcomes,
            baseline_hidden=hidden[0] if hidden else None,
            candidate_hidden=hidden[1] if hidden else None,
            baseline_transfer=transfer[0] if transfer else None,
            candidate_transfer=transfer[1] if transfer else None,
            counterexamples_sought=counterexamples[0],
            counterexamples_found=counterexamples[1],
            cost=CostVector(calls=measurement.calls_used,
                            wall_seconds=measurement.elapsed,
                            unmeasured_token_calls=measurement.calls_used))

        with transaction.TransactionScope(
                plan_.plan_id, "experiment", plan_.hypothesis.statement[:120]) as txn:
            txn.step(transaction.MEASURED, **measurement.summary())
            verdict = gates.judge(claim, evidence)
            txn.step(transaction.DECIDED, accepted=verdict.accepted, reason=verdict.reason)
            plan_.hypothesis.status = SUPPORTED if verdict.accepted else REFUTED
            plan_.hypothesis.result = verdict.as_dict()
            discovery = Discovery(
                discovery_id=uuid.uuid4().hex[:12],
                hypothesis=plan_.hypothesis.as_dict(),
                measurement=measurement.summary(),
                verdict=verdict.as_dict(),
                status=plan_.hypothesis.status)
            self.discoveries.append(discovery)
            # Realised value, for the diminishing-returns check: what the
            # experiment actually resolved, not what it promised.
            self._recent_values.append(
                abs(gates.accuracy(measurement.outcomes, "candidate") -
                    gates.accuracy(measurement.outcomes, "baseline")))
            txn.commit(status=discovery.status)
        return discovery

    # ---------- reporting ----------

    def report(self) -> Dict[str, Any]:
        supported = [d for d in self.discoveries if d.status == SUPPORTED]
        return {
            "experiments": len(self.discoveries),
            "supported": len(supported),
            "refuted": sum(1 for d in self.discoveries if d.status == REFUTED),
            "calls_used": self.calls_used,
            "budget_left": self.budget_left(),
            "stop_reason": self.stop_reason(),
            "discoveries": [d.as_dict() for d in supported],
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"calls_used": self.calls_used,
                   "discoveries": [d.as_dict() for d in self.discoveries]}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
