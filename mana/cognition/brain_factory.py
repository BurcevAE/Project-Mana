"""
mana.cognition.brain_factory — deciding what machinery a gap needs, and
building a candidate for it.

The factory assembles candidates. It accepts nothing: acceptance belongs
to `core.gates`, on evidence, exactly as for a cognitive program. A
factory that could adopt its own output would be a second way to change
the system, and the second way is always the one with the hole in it.

The decision that matters is made before any building
-----------------------------------------------------
"What is the minimum computational mechanism sufficient for this
capability?" is answered from properties of the task, not from taste:

    exactly computable, verifiable by an oracle  -> algorithmic
    features known and thousands of examples    -> classical_ml
    narrow pattern, structure is sequential     -> small_neural
    needs open language, domain is narrow       -> adapter
    none of the above                           -> keep the model

A finding, not a limitation
---------------------------
On MANA's own five domains that rule answers `algorithmic` for four of
them: arithmetic evaluates, sequences fit a proved recurrence, ordering
logic topologically sorts, text operations count. Only `code` -- write a
function to a specification -- needs open generation. Measured, on 120
generated tasks: zero wrong answers, refusals where the rule could not
be proved.

So the classical-ML path here is built and unused, and that is the right
outcome rather than a gap. A slice whose answer is computable does not
need a model that approximates it, and building one to have built one
would be the clearest possible case of adding code without adding
capability.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core import gates, tasks as core_tasks, transaction
from ..core.cost import CostVector
from ..core.gates import Claim, Evidence, PairedOutcome
from . import genome as genome_mod
from .experiments import power
from .gaps import COMPETENCE, Gap
from .genome import BrainGene, CognitiveGenome
from .self_model import MIN_OBSERVATIONS, Observation, SelfModel

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"

ALGORITHMIC = "algorithmic"
CLASSICAL_ML = "classical_ml"
SMALL_NEURAL = "small_neural"
ADAPTER = "adapter"
KEEP_MODEL = "keep_model"

#: How many examples a classical-ML brain needs before fitting one is
#: anything but memorising. Below this the model has fewer facts than
#: parameters and its cross-validation score is a description of the
#: sample.
MIN_ML_EXAMPLES = 1000

#: How far below the model a brain may fall before it is withdrawn.
#: Mirrors `synthesis.RETIRE_MARGIN`: judged against what it beat, not
#: against an absolute bar, so a brain on a hard slice is not retired for
#: the difficulty of the slice.
RETIRE_MARGIN = 0.05


@dataclass(frozen=True)
class MechanismChoice:
    mechanism: str
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def choose_mechanism(domain: str, examples: int = 0,
                     exactly_computable: Optional[bool] = None) -> MechanismChoice:
    """The cheapest machinery that could close this gap.

    `exactly_computable` is passed in rather than guessed: whether a
    domain has an exact solver is a fact about the domain, and a factory
    that inferred it from the name would be asserting something it has
    not checked.
    """
    if exactly_computable:
        return MechanismChoice(
            ALGORITHMIC, f"{domain}: ответ вычислим точно, приближать нечего")
    if examples >= MIN_ML_EXAMPLES:
        return MechanismChoice(
            CLASSICAL_ML, f"{domain}: {examples} примеров, признаки известны",
            {"examples": examples})
    if examples > 0:
        return MechanismChoice(
            KEEP_MODEL,
            f"{domain}: {examples} примеров против {MIN_ML_EXAMPLES} — "
            f"обучение здесь запомнит выборку, а не выучит правило",
            {"examples": examples})
    return MechanismChoice(
        KEEP_MODEL, f"{domain}: нужен открытый язык, дешевле механизма нет")


@dataclass(frozen=True)
class BuildDecision:
    build: bool
    reason: str
    checks: Dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def needs_new_brain(gap: Gap, model: SelfModel,
                    existing_scores: Optional[Dict[str, float]] = None,
                    expected_saving: Optional[float] = None) -> BuildDecision:
    """Three conditions, all of them, or nothing is built.

    **The gap is settled.** A competence gap with enough observations,
    not a slice that happens to look bad today.

    **The brains available are indistinguishable on it.** If one of them
    is already clearly better, the answer is routing, not construction --
    and building instead would spend a training budget to rediscover
    something the router could have used for free.

    **The saving is worth the build.** Measured in real units, which is
    why this could not be stated at all before cost stopped being counted
    in "calls". `None` means not yet computed, and an unknown saving is
    not a justification.
    """
    checks: Dict[str, bool] = {}
    caps = model.capabilities()
    cap = caps.get(gap.capability_id)

    checks["gap_settled"] = bool(
        gap.kind == COMPETENCE and cap is not None
        and cap.observations >= MIN_OBSERVATIONS and cap.measured)
    scores = existing_scores or (cap.by_brain if cap else {})
    spread = (max(scores.values()) - min(scores.values())) if len(scores) > 1 else 0.0
    checks["brains_indistinguishable"] = spread <= 0.15
    checks["saving_justified"] = bool(expected_saving is not None and expected_saving > 0)

    if not checks["gap_settled"]:
        return BuildDecision(False, "разрыв не устоялся: слишком мало наблюдений "
                                    "или это разрыв знания, а не компетенции", checks)
    if not checks["brains_indistinguishable"]:
        return BuildDecision(False, f"мозги уже различаются на {spread:.2f} — "
                                    f"нужна маршрутизация, а не стройка", checks)
    if not checks["saving_justified"]:
        return BuildDecision(False, "экономия не посчитана в реальных единицах", checks)
    return BuildDecision(True, "разрыв устойчив, мозги неразличимы, "
                               "экономия оправдывает постройку", checks)


class HoldoutLeak(RuntimeError):
    """Something from the hidden set reached a training dataset."""


@dataclass
class TrainingSet:
    """What a candidate brain may learn from."""
    domain: str
    prompts: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)
    sources: Dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.prompts)

    def summary(self) -> Dict[str, Any]:
        return {"domain": self.domain, "examples": len(self), "sources": dict(self.sources)}


def assemble_dataset(model: SelfModel, domain: str, task_texts: Dict[str, str],
                     generated: int = 0, seed: int = 0) -> TrainingSet:
    """Real attempts, generated practice, and above all the failures.

    Failures are the valuable part: a brain trained only on what the
    system already gets right learns the easy half of the slice and is
    then measured on the hard half.

    Never the hidden holdout. It is not reachable from here -- there is
    no `hidden_tasks()` to import and there never will be -- but
    `assert_no_holdout_leak` checks the assembled set anyway, because
    "unreachable" is an argument and a test is a fact.
    """
    data = TrainingSet(domain=domain)
    for observation in model.observations:
        if observation.domain != domain:
            continue
        text = task_texts.get(observation.task_id)
        if not text:
            continue
        data.prompts.append(text)
        data.answers.append("")
        key = "real_correct" if observation.correct else "real_failure"
        data.sources[key] = data.sources.get(key, 0) + 1

    if generated > 0:
        for task in core_tasks.generate(domain, generated, seed=seed):
            data.prompts.append(task.prompt)
            data.answers.append(str(task.answer))
            data.sources["generated"] = data.sources.get("generated", 0) + 1
    return data


def assert_no_holdout_leak(data: TrainingSet, per_domain: int = 25) -> None:
    """Raise if any assembled prompt is a hidden-set task.

    Regenerates the holdout's prompts here, inside the check, and never
    returns them. The single way to destroy the meaning of every verdict
    this system has ever issued is to train on the set that judges it,
    and an argument that it cannot happen is worth less than a test that
    it did not.
    """
    from ..core import splits
    hidden_prompts = {t.prompt for t in splits.generate_mixed(
        per_domain, splits._SEED_HIDDEN, splits.DEVELOPMENT_DOMAINS)}
    leaked = [p for p in data.prompts if p in hidden_prompts]
    if leaked:
        raise HoldoutLeak(
            f"{len(leaked)} задач обучающего набора совпали со скрытой выборкой")


@dataclass
class BrainCandidate:
    """A brain built but not adopted."""
    brain_id: str
    substrate: str
    domain: str
    band: str
    mechanism: MechanismChoice
    dataset: Optional[TrainingSet] = None
    status: str = "BUILT"
    verdict: Dict[str, Any] = field(default_factory=dict)
    measured_cost: Dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.time)

    @property
    def slice_id(self) -> str:
        return f"{self.domain}/{self.band}"

    def as_dict(self) -> Dict[str, Any]:
        payload = {k: v for k, v in asdict(self).items() if k != "dataset"}
        payload["mechanism"] = self.mechanism.as_dict()
        payload["dataset"] = self.dataset.summary() if self.dataset else None
        return payload


def evaluate(candidate: BrainCandidate, outcomes: Sequence[PairedOutcome],
             baseline_hidden: Any = None, candidate_hidden: Any = None,
             counterexamples: Tuple[int, int] = (0, 0),
             cost: Optional[CostVector] = None) -> Any:
    """Hand the evidence to the gate and record whatever it says.

    The claim asserts exactly the domain the brain was built for, so the
    holdout scopes its confirmation there and separately refuses a brain
    that collapses a domain it does not claim.
    """
    claim = Claim(
        claim_id=candidate.brain_id, kind="program",
        description=f"мозг {candidate.brain_id} на {candidate.slice_id}",
        asserts_domains=(candidate.domain,))
    evidence = Evidence(
        paired_dev=list(outcomes),
        counterexamples_sought=counterexamples[0],
        counterexamples_found=counterexamples[1],
        cost=cost or CostVector())
    if baseline_hidden is not None and candidate_hidden is not None:
        evidence.with_hidden(baseline_hidden, candidate_hidden)

    with transaction.TransactionScope(
            candidate.brain_id, "brain", candidate.slice_id) as txn:
        txn.step(transaction.MEASURED, trials=len(outcomes),
                 mechanism=candidate.mechanism.mechanism)
        verdict = gates.judge(claim, evidence)
        txn.step(transaction.DECIDED, accepted=verdict.accepted, reason=verdict.reason)
        candidate.status = "ACCEPTED" if verdict.accepted else "REJECTED"
        candidate.verdict = verdict.as_dict()
        txn.commit(result=candidate.status)
    return verdict


def adopt(candidate: BrainCandidate, verdict: Any,
          current: CognitiveGenome) -> CognitiveGenome:
    """Install an accepted brain into the genome, and no wider than proven."""
    if not getattr(verdict, "accepted", False):
        raise genome_mod.NotAccepted(
            f"вердикт не принимает: {getattr(verdict, 'reason', '?')}")
    gene = BrainGene(
        brain_id=candidate.brain_id, substrate=candidate.substrate,
        applicability=(candidate.slice_id,),
        learning_policy="none" if candidate.substrate == ALGORITHMIC else "fit",
        measured_cost=dict(candidate.measured_cost),
        notes=candidate.mechanism.reason)
    proposal = genome_mod.propose(current, "create_brain",
                                  rationale=candidate.mechanism.reason, gene=gene)
    return genome_mod.apply(proposal, verdict, proposal.proposal_id)


def should_retire(gene: BrainGene, model: SelfModel,
                  baseline: float, margin: float = RETIRE_MARGIN) -> Tuple[bool, str]:
    """Has the brain stopped earning its place?

    Against the baseline it beat, never an absolute bar: a brain adopted
    on a slice where everything scores 0.3 is doing its job at 0.4.
    """
    used = [o for o in model.observations
            if getattr(o, "brain", "") == gene.brain_id]
    if len(used) < MIN_OBSERVATIONS:
        return False, f"слишком мало применений ({len(used)})"
    rate = sum(1 for o in used if o.correct) / len(used)
    if rate < baseline - margin:
        return True, (f"{rate:.2f} на {len(used)} применениях против базовой "
                      f"{baseline:.2f}, ради которой её приняли")
    return False, f"работает: {rate:.2f} на {len(used)} применениях"


def retire(gene: BrainGene, current: CognitiveGenome, reason: str) -> CognitiveGenome:
    """Take it back out. No gate: requiring proof to undo a change leaves
    it in force exactly when the proof for it has evaporated."""
    with transaction.TransactionScope(
            gene.brain_id, "retire_brain", reason[:120]) as txn:
        txn.step(transaction.DECIDED, reason=reason)
        proposal = genome_mod.propose(current, "retire_brain", rationale=reason,
                                      brain_id=gene.brain_id)
        txn.commit(result="RETIRED")
    return proposal.candidate
