"""
mana.cognition.meta — evolving the search itself, and the four things
that keep it from eating its own criteria.

Every layer below this one searches: the curriculum picks what to
practise, the lab picks what to test, the cycle splits a budget between
them. All of them are steered by weights someone declared -- how much a
gap's information gain counts against its cost, how much a novel
behaviour is worth, how fast a brain's reputation moves. Those weights
were written by hand and never measured. This layer measures them.

That is the whole of it, and the narrowness is deliberate. A meta layer
that could change anything would eventually change the thing that says
whether a change was good.

**1. It tunes how the search looks, never what counts as a result.** The
acceptance criteria -- gate thresholds, the oracle, the hidden holdout,
the task generator -- live in `mana.core` and are refused here by path,
not by a name list that a new file could slip past.

**2. It is judged by the same gates as everything else.** A meta-change
is a claim like any other, with paired evidence, McNemar, and a hidden
holdout. Nothing about being "meta" earns it an easier ruling.

**3. Yield is measured on what the search was for, not on how much it
accepted.** Counting accepted discoveries rewards a policy that proposes
safe trivia: twenty tiny true claims beat one real mechanism on that
scale. The measure here is uncertainty resolved and capability gained --
the thing accepted claims were supposed to produce.

**4. The bar is fixed before the episodes run.** A threshold chosen
after seeing the results is not a threshold, and this is the one place
where the temptation is strongest, because episodes are expensive and
there will never be many of them.

A trap this layer does not detect
---------------------------------
A meta-experiment on a parameter the episodes never consult will show no
effect, and that is not evidence the parameter does not matter. The
first live run hit exactly this: both policies scored an identical
1.398, because at the episode budget used the cycle spent every step
measuring untouched slices, where gap ranking is not applied at all. The
weight under test was never read. Nothing here checks that an episode
actually exercises the parameter it is varying -- a caller has to make
the episode large enough to reach the code the weight steers.

What this costs
---------------
An episode is a whole search run. A paired comparison needs
`gates.MIN_PAIRED_TRIALS` of them per arm. At a few hundred calls per
episode that is tens of thousands of calls for one meta-conclusion --
which on a free tier is weeks. This layer is built and correct and will
mostly not be affordable, and saying so is more useful than shipping it
with an encouraging default.
"""
from __future__ import annotations

import json
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core import gates, instrument, transaction
from ..core.cost import CostVector
from ..core.gates import Claim, Evidence, PairedOutcome
from . import genome as genome_mod
from .genome import CognitiveGenome

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.3"

#: Modules whose constants this layer may tune. Every one of them is a
#: *search* weight: it changes which experiment gets run first, never
#: which result is believed.
TUNABLE_MODULES = ("mana.cognition.gaps", "mana.cognition.experiments",
                   "mana.cognition.novelty", "mana.cognition.population")

#: Never tunable, whatever a proposal says. `mana.core` is refused by
#: path so a new acceptance rule added to the core is covered without
#: anyone remembering to list it; these names are the ones that live
#: outside core and would still amount to moving the goalposts.
FORBIDDEN_NAMES = frozenset({
    "ALPHA", "MIN_PAIRED_TRIALS", "MIN_ABSOLUTE_MARGIN", "MIN_EFFECT",
    "PASS_BAR", "RETIRE_MARGIN", "MIN_OBSERVATIONS", "MIN_LIFT", "MIN_EXCESS",
    "EPISODE_BAR", "META_PARAMETERS", "FORBIDDEN_NAMES", "TUNABLE_MODULES",
})

#: How much a single episode must resolve to count as a success. Declared
#: here, in the source, so that it is fixed before any episode runs --
#: see rule 4 in the module docstring. Chosen as "clearly more than
#: nothing": an episode that narrows total interval width by less than
#: this has not paid for itself at any plausible cost per call.
EPISODE_BAR = 0.5

#: The most consecutive changes to one parameter in the same direction.
#: A search that can raise exploration every round will raise it every
#: round, because each step looks locally justified.
MAX_CONSECUTIVE = 3


class MetaError(ValueError):
    """A proposal that this layer must not act on."""


@dataclass(frozen=True)
class MetaParameter:
    """One knob, its bounds, and what it is supposed to change."""
    name: str
    module: str
    key: str
    value: float
    minimum: float
    maximum: float
    description: str

    def clamped(self, value: float) -> float:
        """Bounds are not themselves tunable.

        A parameter that can widen its own range has no range, and the
        first thing an unbounded search does is find the edge.
        """
        return max(self.minimum, min(self.maximum, value))

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def baseline_parameters() -> Dict[str, MetaParameter]:
    """The hand-written weights, read out of the modules that use them.

    Read rather than restated: a copy here would drift from the value
    actually in force, and this layer would then be tuning a number
    nothing reads.
    """
    from . import experiments, gaps, novelty
    out: Dict[str, MetaParameter] = {}
    for key, value in gaps.PRIORITY_WEIGHTS.items():
        out[f"gap.{key}"] = MetaParameter(
            f"gap.{key}", "mana.cognition.gaps", key, float(value), -2.0, 2.0,
            f"Вес {key} при ранжировании разрывов.")
    for key, value in experiments.VALUE_WEIGHTS.items():
        out[f"experiment.{key}"] = MetaParameter(
            f"experiment.{key}", "mana.cognition.experiments", key, float(value),
            -2.0, 2.0, f"Вес {key} при выборе следующего эксперимента.")
    for key, value in novelty.CHANNEL_WEIGHTS.items():
        out[f"novelty.{key}"] = MetaParameter(
            f"novelty.{key}", "mana.cognition.novelty", key, float(value), 0.0, 1.0,
            f"Вес канала новизны {key}.")
    return out


def check_tunable(parameter: MetaParameter) -> None:
    """Refuse anything that would move the goalposts. Raises.

    Two independent checks, because either alone has a hole: the path
    check covers every acceptance rule in the core including ones not
    written yet, and the name check covers the thresholds that live
    outside it.
    """
    from ..core import is_immutable_path
    module_path = parameter.module.replace(".", "/") + ".py"
    if is_immutable_path(module_path) or parameter.module.startswith("mana.core"):
        raise MetaError(f"{parameter.name}: {parameter.module} принадлежит ядру")
    if parameter.key.upper() in FORBIDDEN_NAMES or parameter.name in FORBIDDEN_NAMES:
        raise MetaError(f"{parameter.name}: это критерий приёмки, а не параметр поиска")
    if parameter.module not in TUNABLE_MODULES:
        raise MetaError(f"{parameter.name}: {parameter.module} не в списке настраиваемых")


@dataclass
class EpisodeResult:
    """One seeded search run under one policy."""
    seed: int
    policy_id: str
    resolved: float          # total interval width removed
    capability_gain: float   # hidden-holdout accuracy gained, if measured
    calls_used: int
    accepted_claims: int = 0     # recorded, never optimised -- see rule 3
    #: What the episode actually chose to do, in order. Compared between
    #: the two arms to tell "the parameter changed nothing" from "the
    #: parameter changed no decision" -- see MetaProposal.decisions_differed.
    decisions: Tuple[str, ...] = ()

    @property
    def value(self) -> float:
        return self.resolved + self.capability_gain

    def met(self, bar: float = EPISODE_BAR) -> bool:
        return self.value >= bar

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["value"] = round(self.value, 4)
        return payload


#: Runs one search episode: (seed, policy) -> EpisodeResult.
EpisodeRunner = Callable[[int, Dict[str, float]], EpisodeResult]


@dataclass
class MetaProposal:
    """A candidate change to how the search behaves."""
    proposal_id: str
    parameter: MetaParameter
    new_value: float
    rationale: str
    status: str = "PROPOSED"
    #: How many times the parameter under test was actually read during
    #: the episodes. Zero means the experiment measured something else.
    parameter_reads: int = 0
    #: On how many seeds the two arms made different decisions. Zero
    #: means the parameter was consulted and its answer discarded.
    decisions_differed: int = 0
    verdict: Dict[str, Any] = field(default_factory=dict)
    baseline: List[EpisodeResult] = field(default_factory=list)
    candidate: List[EpisodeResult] = field(default_factory=list)
    created: float = field(default_factory=time.time)

    @property
    def direction(self) -> int:
        return 1 if self.new_value > self.parameter.value else -1

    def describe(self) -> str:
        return (f"{self.parameter.name}: {self.parameter.value:.3f} → "
                f"{self.new_value:.3f} ({self.rationale})")

    def as_dict(self) -> Dict[str, Any]:
        return {"proposal_id": self.proposal_id, "parameter": self.parameter.as_dict(),
                "new_value": self.new_value, "rationale": self.rationale,
                "status": self.status, "verdict": dict(self.verdict),
                "parameter_reads": self.parameter_reads,
                "decisions_differed": self.decisions_differed,
                "baseline": [e.as_dict() for e in self.baseline],
                "candidate": [e.as_dict() for e in self.candidate]}


def propose(parameter: MetaParameter, new_value: float, rationale: str = "",
            history: Sequence[MetaProposal] = ()) -> MetaProposal:
    """Build a candidate change. Raises on anything inadmissible.

    Refuses a run of changes in one direction: each step looks locally
    justified, and a search that can raise exploration every round will.
    """
    check_tunable(parameter)
    clamped = parameter.clamped(new_value)
    if abs(clamped - parameter.value) < 1e-9:
        raise MetaError(f"{parameter.name}: значение не меняется")

    direction = 1 if clamped > parameter.value else -1
    same = 0
    for past in reversed(list(history)):
        if past.parameter.name != parameter.name or past.status != "ACCEPTED":
            continue
        if past.direction != direction:
            break
        same += 1
    if same >= MAX_CONSECUTIVE:
        raise MetaError(
            f"{parameter.name}: {same} подряд принятых изменений в ту же сторону; "
            f"дальше это не поиск, а дрейф")

    return MetaProposal(proposal_id=uuid.uuid4().hex[:12], parameter=parameter,
                        new_value=clamped, rationale=rationale)


def run_episodes(proposal: MetaProposal, seeds: Sequence[int],
                 runner: EpisodeRunner) -> None:
    """Run both policies on the same seeds, and count whether the
    parameter was ever consulted.

    The same seeds, so the comparison is paired: episode-to-episode
    variance is enormous -- one lucky gap found early changes a whole
    run -- and unpaired samples would need far more episodes than anyone
    can afford to see an effect through it.

    The activation count is not decoration. Without it an episode that
    never reached the code reading the parameter is indistinguishable
    from an episode where the parameter made no difference: identical
    numbers, opposite meanings. See `judge`, which refuses the first.
    """
    base_policy = {proposal.parameter.name: proposal.parameter.value}
    new_policy = {proposal.parameter.name: proposal.new_value}
    with instrument.watching() as used:
        for seed in seeds:
            proposal.baseline.append(runner(seed, base_policy))
            proposal.candidate.append(runner(seed, new_policy))
    proposal.parameter_reads = used.get(proposal.parameter.name, 0)
    proposal.decisions_differed = sum(
        1 for b, c in zip(proposal.baseline, proposal.candidate)
        if tuple(b.decisions) != tuple(c.decisions))


def paired_outcomes(proposal: MetaProposal, bar: float = EPISODE_BAR
                    ) -> List[PairedOutcome]:
    """Episodes as paired binary outcomes the core gate can read.

    `bar` is EPISODE_BAR by default and is not a free parameter: it is
    declared in this module's source, before any episode runs. A caller
    passing something else is doing so knowingly and it goes into the
    claim.
    """
    return [PairedOutcome(task_id=f"seed-{b.seed}", domain="search",
                          baseline_correct=b.met(bar), candidate_correct=c.met(bar))
            for b, c in zip(proposal.baseline, proposal.candidate)]


def judge(proposal: MetaProposal, bar: float = EPISODE_BAR,
          hidden: Optional[Tuple[float, float]] = None,
          counterexamples: Tuple[int, int] = (0, 0)) -> Any:
    """Rule on a meta-change with the same gates as everything else.

    Nothing about being meta earns an easier ruling. In particular the
    sample-size gate applies to episodes, which is what makes this layer
    honestly expensive rather than quietly cheap.
    """
    refusal = _not_a_result(proposal)
    if refusal is not None:
        return refusal

    outcomes = paired_outcomes(proposal, bar)
    claim = Claim(claim_id=proposal.proposal_id, kind="genome",
                  description=f"мета: {proposal.describe()}")
    evidence = Evidence(
        paired_dev=outcomes,
        baseline_hidden=hidden[0] if hidden else None,
        candidate_hidden=hidden[1] if hidden else None,
        counterexamples_sought=counterexamples[0],
        counterexamples_found=counterexamples[1],
        cost=CostVector(
            calls=sum(e.calls_used for e in proposal.baseline + proposal.candidate),
            unmeasured_token_calls=sum(e.calls_used for e in
                                       proposal.baseline + proposal.candidate)))

    with transaction.TransactionScope(
            proposal.proposal_id, "meta", proposal.describe()[:120]) as txn:
        txn.step(transaction.MEASURED, episodes=len(outcomes), bar=bar,
                 parameter_reads=proposal.parameter_reads,
                 decisions_differed=proposal.decisions_differed,
                 baseline_value=round(_mean(proposal.baseline), 4),
                 candidate_value=round(_mean(proposal.candidate), 4))
        verdict = gates.judge(claim, evidence)
        txn.step(transaction.DECIDED, accepted=verdict.accepted, reason=verdict.reason)
        proposal.status = "ACCEPTED" if verdict.accepted else "REJECTED"
        proposal.verdict = verdict.as_dict()
        txn.commit(result=proposal.status)
    return verdict


def _not_a_result(proposal: MetaProposal) -> Optional[Any]:
    """Refuse to rule when the episodes cannot say anything about the
    parameter. Returns a Verdict to hand back, or None to proceed.

    Two distinct ways an experiment can produce clean-looking numbers
    about nothing, and the second was found by fixing the first.

    **Never read.** The code consulting the parameter did not run. The
    arms are identical because nothing varied.

    **Read and discarded.** The parameter was consulted -- a live run
    counted 48 reads -- and every decision came out the same anyway,
    because the options it ranks lost to a category of options that is
    not ranked by it at all. Identical decisions on the same seed give
    identical outcomes necessarily, so "no effect" is vacuous rather
    than measured.

    Neither is recorded as REFUTED. A refutation goes into the record as
    evidence about the parameter, and there is none; worse, it would stop
    the parameter being tried again on the strength of an experiment that
    never tested it.
    """
    if not proposal.baseline:
        return None
    episodes = len(proposal.baseline)

    if proposal.parameter_reads == 0:
        reason = (f"{proposal.parameter.name} ни разу не прочитан за "
                  f"{episodes} эпизодов — измерялось что-то другое")
        gate = "parameter_not_exercised"
    elif (proposal.decisions_differed == 0
          and any(e.decisions for e in proposal.baseline)):
        reason = (f"{proposal.parameter.name} прочитан "
                  f"{proposal.parameter_reads} раз, но ни на одном сиде не изменил "
                  f"ни одного решения — его ответ отбрасывается")
        gate = "parameter_had_no_influence"
    else:
        return None

    proposal.status = "NOT_EXERCISED"
    measurements = {"parameter_reads": proposal.parameter_reads,
                    "decisions_differed": proposal.decisions_differed}
    proposal.verdict = {"accepted": False, "failed_gates": [gate],
                        "reason": reason, "measurements": measurements}
    return gates.Verdict(accepted=False, reason=reason, failed_gates=(gate,),
                         measurements=measurements)


def _mean(episodes: Sequence[EpisodeResult]) -> float:
    return statistics.fmean([e.value for e in episodes]) if episodes else 0.0


def yield_report(proposal: MetaProposal) -> Dict[str, Any]:
    """What the two policies actually bought, including what is ignored.

    `accepted_claims` is reported and plays no part in the verdict. It is
    here precisely so that a reader can see a policy accepting more while
    resolving less -- the failure mode rule 3 exists to prevent.
    """
    return {
        "baseline_value": round(_mean(proposal.baseline), 4),
        "candidate_value": round(_mean(proposal.candidate), 4),
        "baseline_accepted": sum(e.accepted_claims for e in proposal.baseline),
        "candidate_accepted": sum(e.accepted_claims for e in proposal.candidate),
        "calls": sum(e.calls_used for e in proposal.baseline + proposal.candidate),
        "episodes": len(proposal.baseline),
        "parameter_reads": proposal.parameter_reads,
        "decisions_differed": proposal.decisions_differed,
    }


#: Where the modules keep the weights this layer tunes. The write path
#: has to know it, because until phase 16 there was none: an accepted
#: meta-change updated a field on the MetaEvolution object and nothing
#: else, so the loop was open at the far end and "accepted" changed no
#: behaviour at all. The live script papered over it by patching
#: PRIORITY_WEIGHTS itself, which is how an architectural gap turns into
#: "everything works".
_WEIGHT_TABLES = {
    "gap": ("mana.cognition.gaps", "PRIORITY_WEIGHTS"),
    "experiment": ("mana.cognition.experiments", "VALUE_WEIGHTS"),
    "novelty": ("mana.cognition.novelty", "CHANNEL_WEIGHTS"),
}


def in_force(parameter: MetaParameter) -> float:
    """The value the module is actually using right now."""
    import importlib
    prefix = parameter.name.split(".", 1)[0]
    module_name, table_name = _WEIGHT_TABLES[prefix]
    table = getattr(importlib.import_module(module_name), table_name)
    return float(table[parameter.key])


def put_in_force(parameter: MetaParameter, value: float) -> None:
    """Write an accepted value where the search will actually read it.

    Goes through `check_tunable` first, and that is not belt-and-braces:
    `propose` guards what may be *proposed*, and without the same guard
    here the write path would be a way around it. One entrance, one
    check.

    Mutates the module's dict in place rather than rebinding the name,
    because callers that did `from .gaps import PRIORITY_WEIGHTS` hold
    the old object and would silently keep the old policy.
    """
    check_tunable(parameter)
    import importlib
    prefix = parameter.name.split(".", 1)[0]
    module_name, table_name = _WEIGHT_TABLES[prefix]
    table = getattr(importlib.import_module(module_name), table_name)
    if parameter.key not in table:
        raise MetaError(f"{parameter.name}: в {table_name} нет ключа {parameter.key!r}")
    table[parameter.key] = parameter.clamped(value)


class MetaEvolution:
    """Holds the search policy in force, and the record of how it got there."""

    def __init__(self, genome: Optional[CognitiveGenome] = None) -> None:
        self.genome = genome or CognitiveGenome()
        self.parameters = baseline_parameters()
        self.history: List[MetaProposal] = []

    def policy(self) -> Dict[str, float]:
        return {name: p.value for name, p in self.parameters.items()}

    def propose(self, name: str, new_value: float, rationale: str = "") -> MetaProposal:
        parameter = self.parameters.get(name)
        if parameter is None:
            raise MetaError(f"неизвестный параметр {name!r}")
        return propose(parameter, new_value, rationale, self.history)

    def evaluate(self, proposal: MetaProposal, seeds: Sequence[int],
                 runner: EpisodeRunner, **evidence: Any) -> bool:
        """Run the episodes, rule on them, and adopt only if the gate says so."""
        run_episodes(proposal, seeds, runner)
        verdict = judge(proposal, **evidence)
        self.history.append(proposal)
        if not verdict.accepted:
            return False
        self.parameters = dict(self.parameters)
        adopted = MetaParameter(**{**proposal.parameter.as_dict(),
                                   "value": proposal.new_value})
        self.parameters[proposal.parameter.name] = adopted
        # The far end of the loop. Without this the object's opinion
        # changes and the search keeps reading the old number.
        put_in_force(adopted, proposal.new_value)
        return True

    def rollback(self, proposal: MetaProposal) -> None:
        """Put a parameter back where it was.

        Needs no gate, for the reason a capability's retirement needs
        none: requiring evidence to undo a change leaves it in force
        exactly when the evidence for it has gone.
        """
        self.parameters = dict(self.parameters)
        self.parameters[proposal.parameter.name] = proposal.parameter
        put_in_force(proposal.parameter, proposal.parameter.value)
        proposal.status = "ROLLED_BACK"

    # ---------- persistence ----------

    def save(self, path: Any) -> None:
        """Write the policy in force so it survives a restart.

        Only the values -- bounds, modules and keys are declared in code
        and must not be restorable from a file, or a hand-edited file
        could widen a bound or point a parameter at a table this layer is
        not allowed to write.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"policy": self.policy(),
                   "accepted": [p.parameter.name for p in self.history
                                if p.status == "ACCEPTED"]}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(path)

    def load(self, path: Any) -> Dict[str, float]:
        """Restore a saved policy and put it back in force.

        An unknown name is ignored rather than raising: a policy file
        written by a later version must not stop an earlier one from
        starting. A known name still goes through `put_in_force`, so a
        file cannot smuggle in a parameter this layer may not tune.
        """
        path = Path(path)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        restored: Dict[str, float] = {}
        for name, value in (payload.get("policy") or {}).items():
            parameter = self.parameters.get(name)
            if parameter is None:
                continue
            try:
                put_in_force(parameter, float(value))
            except (MetaError, KeyError, TypeError, ValueError):
                continue
            self.parameters = dict(self.parameters)
            self.parameters[name] = MetaParameter(
                **{**parameter.as_dict(), "value": parameter.clamped(float(value))})
            restored[name] = self.parameters[name].value
        return restored

    def verify_in_force(self) -> Dict[str, Any]:
        """Does the search actually read what this object believes?

        The check a restart has to pass. "Accepted" means nothing if the
        module is still using the old number, and that was the state of
        this layer from phase 13 until now.
        """
        mismatches = {}
        for name, parameter in self.parameters.items():
            try:
                actual = in_force(parameter)
            except Exception as exc:                   # pragma: no cover
                mismatches[name] = f"не прочитан: {exc}"
                continue
            if abs(actual - parameter.value) > 1e-9:
                mismatches[name] = {"объект": parameter.value, "модуль": actual}
        return {"ok": not mismatches, "mismatches": mismatches}

    def report(self) -> Dict[str, Any]:
        return {
            "policy": {k: round(v, 4) for k, v in self.policy().items()},
            "in_force": self.verify_in_force(),
            "changes": [p.as_dict() for p in self.history],
            "accepted": [p.parameter.name for p in self.history
                         if p.status == "ACCEPTED"],
        }
