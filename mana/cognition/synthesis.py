"""
mana.cognition.synthesis — the step that turns a proven mechanism into
something the system actually uses next time.

Until this module existed the loop did not close. The research cycle
found a weak slice, wrote a hypothesis against it, ran both arms on the
same tasks, and the gate said SUPPORTED -- and then the next task
compiled from exactly the genome as before. The finding was written into
a Discovery record and nothing read it. A system that proves
`critique_loop` beats `direct` on logic and then goes on answering logic
with `direct` has not learned anything; it has only measured something.

What closes it is `genome.apply`, which already refuses to adopt a
candidate without a verdict belonging to that specific proposal. So this
module's job is to build the proposal, get a verdict that legitimately
belongs to it, and only then install the capability into the genome the
compiler reads from.

Three rules the structure enforces, not the prose
-------------------------------------------------
**A capability is proven where it was measured, nowhere else.** The
applicability of a synthesised template is derived from the slice the
experiment ran on. Widening it is a separate claim with its own evidence
(`asserts_transfer`), because "helps arithmetic" and "helps thinking"
are different assertions and only one of them was tested.

**One result is not a capability.** The discovery that motivates a
proposal cannot also confirm it: its verdict belongs to the experiment's
claim id, and `genome.apply` checks that the id matches the proposal.
Confirmation therefore needs a second measurement, run for this
proposal, on tasks the first one did not use. This is not extra rigour
bolted on -- it is the only way the id check can be satisfied honestly.

**A capability that stops working is withdrawn.** Adoption is not
permanent. `should_retire` reads the self-model for what the capability
actually did after it was installed, and a template whose measured
success falls below the baseline it beat goes back out.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core import gates, transaction
from ..core.gates import Claim, Evidence, PairedOutcome
from . import genome as genome_mod
from .experiments import Discovery, SUPPORTED
from .genome import CognitiveGenome, MutationProposal, NotAccepted
from .self_model import MIN_OBSERVATIONS, SelfModel

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"

PROPOSED = "PROPOSED"
CONFIRMED = "CONFIRMED"
REJECTED = "REJECTED"
ADOPTED = "ADOPTED"
RETIRED = "RETIRED"

#: Task domains and compiler task kinds are two different vocabularies:
#: `core.tasks.DOMAINS` names what generates a task, `compiler.classify`
#: names what a task looks like. A synthesised template is scored against
#: the second, so a capability proven on a domain has to be translated or
#: it is installed and never fires. Written out rather than guessed,
#: because a silent mismatch looks exactly like a capability that does
#: not help.
DOMAIN_KIND = {
    "arithmetic": "math",
    "sequence": "sequence",
    "logic": "reasoning",
    "text_ops": "general",
    "code": "programming",
}

#: The only band words `compiler._score_template` understands. A capability
#: proven on the medium band gets no band term -- deliberately, because
#: inventing one the scorer ignores would read like a narrower claim than
#: was actually installed.
SCORED_BANDS = ("easy", "hard")

#: How far below the baseline it beat a capability may drift before it is
#: withdrawn. Zero would retire on noise; this is one clear step back.
RETIRE_MARGIN = 0.05


@dataclass
class CapabilityProposal:
    """A capability that has been proven once and is asking to be installed."""
    proposal_id: str
    name: str
    steps: Tuple[str, ...]
    applicability: Tuple[str, ...]
    domain: str
    band: str
    baseline_steps: Tuple[str, ...]
    source_discovery: str
    first_margin: float
    mutation: Optional[MutationProposal] = None
    status: str = PROPOSED
    confirmation: Dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.time)

    @property
    def slice_id(self) -> str:
        return f"{self.domain}/{self.band}"

    def as_dict(self) -> Dict[str, Any]:
        payload = {k: v for k, v in asdict(self).items() if k != "mutation"}
        payload["steps"] = list(self.steps)
        payload["applicability"] = list(self.applicability)
        payload["baseline_steps"] = list(self.baseline_steps)
        return payload

    def describe(self) -> str:
        return (f"{self.name}: {' → '.join(self.steps)} для "
                f"{', '.join(self.applicability) or 'без условий'} "
                f"(доказано на {self.slice_id}, перевес {self.first_margin:+.2f})")


def applicability_for(domain: str, band: str) -> Tuple[str, ...]:
    """The conditions a capability may claim, and no wider.

    Proven on arithmetic/hard, it claims math and hard. Not "reasoning",
    not "everything" -- the experiment did not run there. Widening is a
    transfer claim with its own gate.
    """
    terms: List[str] = []
    kind = DOMAIN_KIND.get(domain)
    if kind:
        terms.append(kind)
    if band in SCORED_BANDS:
        terms.append(band)
    return tuple(terms)


def name_for(discovery: Discovery, steps: Sequence[str]) -> str:
    """A name that says what the chain does and where it was proven.

    Descriptive rather than sequential ("capability_7" tells a reader
    nothing when the genome is being audited two hundred mutations
    later).
    """
    hypothesis = discovery.hypothesis
    slice_id = f"{hypothesis.get('domain', '?')}_{hypothesis.get('band', '?')}"
    body = "_".join(s.lower() for s in steps if s not in ("OBSERVE", "ANSWER"))
    return f"{body or 'chain'}__{slice_id}"


def propose_capability(discovery: Discovery, current: CognitiveGenome,
                       name: str = "") -> Optional[CapabilityProposal]:
    """Turn an accepted discovery into a candidate genome. Adopts nothing.

    Returns None rather than raising for the ordinary cases -- refuted,
    or a chain the genome already has -- because a research cycle asks
    this of every discovery it makes and most will not become
    capabilities.
    """
    if discovery.status != SUPPORTED:
        return None
    hypothesis = discovery.hypothesis
    steps = tuple(hypothesis.get("candidate_steps") or ())
    if len(steps) < 2:
        return None
    domain = str(hypothesis.get("domain") or "")
    band = str(hypothesis.get("band") or "")

    # Already present, under any name: installing a second template with
    # the same chain splits the evidence for one mechanism across two
    # records and makes both look weaker than the thing they measure.
    for template in current.program_templates.values():
        if tuple(template.steps) == steps:
            return None

    template_name = name or name_for(discovery, steps)
    if template_name in current.program_templates:
        return None

    applicability = applicability_for(domain, band)
    margin = float((discovery.verdict.get("measurements") or {}).get("dev_margin", 0.0))
    proposal = CapabilityProposal(
        proposal_id="", name=template_name, steps=steps, applicability=applicability,
        domain=domain, band=band,
        baseline_steps=tuple(hypothesis.get("baseline_steps") or ()),
        source_discovery=discovery.discovery_id, first_margin=margin)

    mutation = genome_mod.propose(
        current, "create_program_template",
        rationale=(f"подтверждено на {proposal.slice_id}: {' → '.join(steps)} "
                   f"обошла базовую цепочку на {margin:+.2f}"),
        name=template_name, steps=steps, applicability=applicability,
        description=(f"Синтезировано из открытия {discovery.discovery_id}; "
                     f"доказано на {proposal.slice_id}."))
    proposal.mutation = mutation
    # The genome's id, not a second one: `genome.apply` matches the
    # verdict's claim against the mutation proposal, so the confirming
    # experiment has to be run under exactly this id or it cannot admit
    # the change. One id, one claim, one capability.
    proposal.proposal_id = mutation.proposal_id
    return proposal


def confirm(proposal: CapabilityProposal, outcomes: Sequence[PairedOutcome],
            hidden: Optional[Tuple[float, float]] = None,
            transfer: Optional[Tuple[float, float]] = None,
            counterexamples: Tuple[int, int] = (0, 0),
            cost_calls: int = 0) -> Any:
    """Judge a *second* measurement, run for this proposal.

    The discovery that motivated the proposal cannot serve here: its
    verdict belongs to the experiment's claim, and adopting on it would
    be the same evidence counted twice. Nothing in this function checks
    that the tasks are fresh -- that is the caller's to arrange, and the
    reason `ResearchCycle` regenerates from a new seed.
    """
    claim = Claim(
        claim_id=proposal.proposal_id, kind="program",
        description=f"способность {proposal.name} на {proposal.slice_id}",
        asserts_transfer=transfer is not None,
        asserts_domains=(proposal.domain,))
    evidence = Evidence(
        paired_dev=list(outcomes),
        baseline_hidden=hidden[0] if hidden else None,
        candidate_hidden=hidden[1] if hidden else None,
        baseline_transfer=transfer[0] if transfer else None,
        candidate_transfer=transfer[1] if transfer else None,
        counterexamples_sought=counterexamples[0],
        counterexamples_found=counterexamples[1],
        cost_calls=cost_calls)

    with transaction.TransactionScope(
            proposal.proposal_id, "synthesis", proposal.describe()[:120]) as txn:
        txn.step(transaction.MEASURED, trials=len(outcomes),
                 first_margin=round(proposal.first_margin, 4))
        verdict = gates.judge(claim, evidence)
        txn.step(transaction.DECIDED, accepted=verdict.accepted, reason=verdict.reason)
        proposal.status = CONFIRMED if verdict.accepted else REJECTED
        proposal.confirmation = verdict.as_dict()
        txn.commit(result=proposal.status)
    return verdict


def adopt(proposal: CapabilityProposal, verdict: Any) -> CognitiveGenome:
    """Install the capability. Raises unless the verdict belongs to it.

    The check lives in `genome.apply` and is not duplicated here: two
    places deciding whether a change is admissible is how they come to
    disagree.
    """
    if proposal.mutation is None:
        raise NotAccepted("proposal carries no genome mutation")
    adopted = genome_mod.apply(proposal.mutation, verdict, proposal.proposal_id)
    proposal.status = ADOPTED
    return adopted


def measured_success(model: SelfModel, template_name: str) -> Optional[Tuple[float, int]]:
    """What the capability actually did after it was installed.

    Reads the self-model's per-program record rather than anything the
    capability reports about itself: a capability trusted on its own
    account is a capability that cannot be withdrawn.
    """
    successes = 0
    total = 0
    for observation in model.observations:
        if getattr(observation, "program", "") != template_name:
            continue
        total += 1
        successes += int(observation.correct)
    return (successes / total, total) if total else None


def should_retire(proposal: CapabilityProposal, model: SelfModel,
                  margin: float = RETIRE_MARGIN) -> Tuple[bool, str]:
    """Has the capability stopped earning its place?

    Compared against the baseline accuracy it beat, not against an
    absolute bar: a capability installed on a slice where everything
    scores 0.3 is doing its job at 0.4, and an absolute threshold would
    retire it for the difficulty of the slice.
    """
    if proposal.status != ADOPTED:
        return False, "не принята"
    measured = measured_success(model, proposal.name)
    if measured is None:
        return False, "ещё не применялась"
    rate, n = measured
    if n < MIN_OBSERVATIONS:
        return False, f"слишком мало применений ({n})"
    baseline = float((proposal.confirmation.get("measurements") or {})
                     .get("dev_baseline", 0.0))
    if rate < baseline - margin:
        return True, (f"измеренная точность {rate:.2f} на {n} применениях ниже "
                      f"базовой {baseline:.2f}, ради которой её приняли")
    return False, f"работает: {rate:.2f} на {n} применениях"


def retire(proposal: CapabilityProposal, current: CognitiveGenome,
           reason: str) -> CognitiveGenome:
    """Take the capability back out.

    A removal needs no gate: the gates exist to stop unproven things
    being *added*, and requiring the same evidence to undo a change would
    leave a capability installed precisely when the evidence for it has
    evaporated.
    """
    with transaction.TransactionScope(
            proposal.proposal_id, "retire", f"{proposal.name}: {reason}"[:120]) as txn:
        txn.step(transaction.DECIDED, reason=reason)
        templates = dict(current.program_templates)
        templates.pop(proposal.name, None)
        result = CognitiveGenome(
            operators=dict(current.operators),
            representations=dict(current.representations),
            program_templates=templates,
            learning_rules=dict(current.learning_rules),
            selection_policy=current.selection_policy,
            parent_id=current.genome_id, mutation="retire_program_template")
        proposal.status = RETIRED
        txn.commit(result=RETIRED)
    return result


class CapabilitySynthesizer:
    """Holds the genome capabilities are installed into, and their history."""

    def __init__(self, current: CognitiveGenome) -> None:
        self.genome = current
        self.proposals: List[CapabilityProposal] = []

    def consider(self, discovery: Discovery) -> Optional[CapabilityProposal]:
        proposal = propose_capability(discovery, self.genome)
        if proposal is not None:
            self.proposals.append(proposal)
        return proposal

    def install(self, proposal: CapabilityProposal,
                outcomes: Sequence[PairedOutcome], **evidence: Any) -> bool:
        """Confirm and, if the gate allows, adopt. Returns whether it went in."""
        verdict = confirm(proposal, outcomes, **evidence)
        if not verdict.accepted:
            return False
        self.genome = adopt(proposal, verdict)
        return True

    def review(self, model: SelfModel) -> List[Dict[str, Any]]:
        """Withdraw whatever has stopped working, and say what was kept."""
        report: List[Dict[str, Any]] = []
        for proposal in list(self.proposals):
            retire_it, reason = should_retire(proposal, model)
            if retire_it:
                self.genome = retire(proposal, self.genome, reason)
            report.append({"name": proposal.name, "status": proposal.status,
                           "reason": reason})
        return report

    def adopted(self) -> List[CapabilityProposal]:
        return [p for p in self.proposals if p.status == ADOPTED]

    def report(self) -> Dict[str, Any]:
        return {
            "genome": self.genome.signature(),
            "templates": sorted(self.genome.program_templates),
            "proposals": [p.as_dict() for p in self.proposals],
            "adopted": [p.name for p in self.adopted()],
        }
