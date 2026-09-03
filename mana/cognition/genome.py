"""
mana.cognition.genome — the space of what MANA can think with, and the
only sanctioned way to change it.

What a genome is here
---------------------
Not a configuration. `PipelineSpec` is a configuration: thirty fields with
fixed names and closed value sets, and evolution over it can only ever
retune what already exists. A genome describes a *space*: which operators
exist at all, which vocabularies tasks are described in, which programs
can be assembled, and how evidence is allowed to change any of that. A
mutation can therefore add a member the space did not previously contain
-- which is the difference between tuning and open-endedness, and the
reason this file exists rather than eight more fields on PipelineSpec.

Four levels, and this file is about the top two:

    Level 0  the answer to one task              (existing: solve_task)
    Level 1  the parameters of a strategy        (existing: PipelineSpec)
    Level 2  which operators and programs exist  (here)
    Level 3  which representations exist         (here)

Mutation is a proposal, not a change
------------------------------------
`mutate()` returns a `MutationProposal`. Nothing in this module can adopt
one. `apply()` demands a `core.gates.Verdict` that is accepted AND whose
claim id matches the proposal -- so a verdict earned by one candidate
cannot be reused to admit another, which is the cheapest imaginable way to
smuggle a change past the gate.

What a genome may never contain
-------------------------------
The forbidden list is not about danger, it is about circularity. An
operator called VERIFY that a genome could redefine would let MANA improve
its scores by changing what verification means. So names that belong to
the immutable core are rejected at construction, and `validate()` is run
on every proposal before it is even measured -- a malformed genome should
cost nothing to reject.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .ir import (COSTS, TYPES, CognitiveOperator, CompositionError,
                 check_chain, compose, primitive_operators)

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Names a genome may not define, because core already means something by
#: them. Not a safety list -- a circularity list: a genome that could
#: redefine these would be able to improve its score by changing the
#: meaning of the measurement.
RESERVED_NAMES = frozenset({
    "hidden_score", "transfer_score", "judge", "grade", "oracle",
    "acceptance", "holdout", "rollback", "budget", "audit",
})

#: The mutation vocabulary. Each is one structural change, so a rejected
#: proposal says which single thing failed -- the same "one mutation, one
#: measurable effect" discipline the existing GA already follows.
MUTATIONS = (
    "add_operator", "remove_operator", "compose_operators", "split_operator",
    "merge_operators", "create_representation", "modify_representation",
    "create_program_template", "modify_learning_rule", "modify_selection_policy",
)


class GenomeError(ValueError):
    """A genome or proposal that is malformed. Raised early and cheaply --
    measuring a nonsensical candidate wastes the budget that measuring
    real ones needs."""


# ---------------------------------------------------------------------------
# representations (Level 3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Representation:
    """A vocabulary for describing a task or a cognitive process.

    This is the Level-3 object. MANA currently describes a task as
    (task, strategy, result); it may turn out that (state, transition,
    constraint, invariant, counterexample) supports strategies the first
    vocabulary cannot express. Adding a field is a hypothesis about what
    needs to be represented, and like any other it has to earn its place
    through the gate.
    """
    name: str
    fields: Tuple[str, ...]
    description: str = ""
    derived_from: Optional[str] = None

    def signature(self) -> str:
        blob = json.dumps({"fields": sorted(self.fields)}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def baseline_representations() -> Dict[str, Representation]:
    """What MANA actually uses today, written down.

    Honest starting point: these are descriptions of the existing system,
    not aspirations. A genome whose representations nothing implements
    would make every early measurement meaningless.
    """
    return {
        "task_view": Representation(
            "task_view", ("task", "category", "difficulty"),
            "How a task is described before work starts (RoutingMixin._task_category)."),
        "attempt": Representation(
            "attempt", ("draft", "critique", "verification", "confidence"),
            "How one attempt at an answer is described (ExecutionMixin trace)."),
        "outcome": Representation(
            "outcome", ("answer", "quality", "latency", "brain"),
            "How a finished attempt is recorded (record_outcome, routing_stats)."),
    }


# ---------------------------------------------------------------------------
# programs and rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProgramTemplate:
    """A named operator chain, with the conditions it claims to suit."""
    name: str
    steps: Tuple[str, ...]
    applicability: Tuple[str, ...] = ()
    description: str = ""

    def signature(self) -> str:
        return hashlib.sha256("->".join(self.steps).encode()).hexdigest()[:16]


def baseline_templates() -> Dict[str, ProgramTemplate]:
    """The graphs `_graph_for_task` can already produce, as templates.

    Written from the existing dispatcher rather than invented, so the
    starting genome is a description of MANA and the first measurements
    have something real to compare against.
    """
    return {
        "direct": ProgramTemplate(
            "direct", ("OBSERVE", "GENERATE", "ANSWER"), ("easy",),
            "Ask once. What graph_nodes=('LLM','EVALUATE') already does."),
        "retrieve_then_answer": ProgramTemplate(
            "retrieve_then_answer", ("OBSERVE", "RETRIEVE", "GENERATE", "ANSWER"),
            ("current", "reasoning"), "Memory/web before the model."),
        "critique_loop": ProgramTemplate(
            "critique_loop", ("OBSERVE", "GENERATE", "CRITIQUE", "REPAIR", "ANSWER"),
            ("reasoning",), "The existing critic/repair pass."),
        "verified": ProgramTemplate(
            "verified", ("OBSERVE", "GENERATE", "VERIFY", "ANSWER"),
            ("math", "programming"), "Draft, then check against a real oracle."),
        "decomposed": ProgramTemplate(
            "decomposed", ("OBSERVE", "DECOMPOSE", "GENERATE", "SYNTHESIZE", "ANSWER"),
            ("hard",), "The existing decompose.solve path."),
    }


@dataclass(frozen=True)
class LearningRule:
    """How evidence is allowed to move a number.

    Evolvable because the right learning rate is an empirical question,
    and bounded because a rule that can set its own bounds is not a rule.
    """
    name: str
    parameter: str
    value: float
    minimum: float
    maximum: float
    description: str = ""

    def clamped(self, value: float) -> float:
        return max(self.minimum, min(self.maximum, value))


def baseline_learning_rules() -> Dict[str, LearningRule]:
    return {
        "brain_reputation": LearningRule(
            "brain_reputation", "ewma_alpha", 0.20, 0.01, 0.60,
            "How fast a brain's measured quality moves (BrainPool.record_outcome)."),
        "route_confidence": LearningRule(
            "route_confidence", "min_observations", 5.0, 2.0, 50.0,
            "How much evidence before routing history may override the rule."),
        "operator_credit": LearningRule(
            "operator_credit", "decay", 0.10, 0.01, 0.50,
            "How fast an operator's success rate forgets old runs."),
    }


# ---------------------------------------------------------------------------
# the genome
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CognitiveGenome:
    operators: Dict[str, CognitiveOperator] = field(default_factory=primitive_operators)
    representations: Dict[str, Representation] = field(default_factory=baseline_representations)
    program_templates: Dict[str, ProgramTemplate] = field(default_factory=baseline_templates)
    learning_rules: Dict[str, LearningRule] = field(default_factory=baseline_learning_rules)
    selection_policy: str = "capability_first"
    genome_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: Optional[str] = None
    born_at: float = field(default_factory=time.time)
    #: The single mutation that produced this genome from its parent.
    #: Lineage is not decoration: without it a discovery cannot be traced
    #: to the change that caused it, and novelty cannot tell a genuinely
    #: new line from a rediscovery of an ancestor.
    mutation: Optional[str] = None

    def __post_init__(self) -> None:
        problems = self.validate()
        if problems:
            raise GenomeError("; ".join(problems))

    # ---------- validation ----------

    def validate(self) -> List[str]:
        """Everything wrong with this genome, cheaply.

        Returns all problems rather than the first: a proposal rejected
        one reason at a time turns search into a guessing game, and the
        search is the expensive part.
        """
        problems: List[str] = []
        for name in list(self.operators) + list(self.representations) + list(self.program_templates):
            if name.lower() in RESERVED_NAMES:
                problems.append(f"{name!r} is reserved by the immutable core")
        for op in self.operators.values():
            for t in op.inputs + op.outputs:
                if t not in TYPES:
                    problems.append(f"operator {op.op_id}: unknown type {t!r}")
            if op.cost not in COSTS:
                problems.append(f"operator {op.op_id}: unknown cost {op.cost!r}")
            for component in op.components:
                if component not in self.operators:
                    problems.append(f"operator {op.op_id}: component {component!r} does not exist")
        for tmpl in self.program_templates.values():
            unknown = [s for s in tmpl.steps if s not in self.operators]
            if unknown:
                problems.append(f"template {tmpl.name}: unknown steps {unknown}")
            else:
                try:
                    check_chain([self.operators[s] for s in tmpl.steps])
                except CompositionError as exc:
                    problems.append(f"template {tmpl.name}: {exc}")
        for rule in self.learning_rules.values():
            if not (rule.minimum <= rule.value <= rule.maximum):
                problems.append(f"rule {rule.name}: value {rule.value} outside "
                                f"[{rule.minimum}, {rule.maximum}]")
        if not self.operators:
            problems.append("a genome with no operators cannot think")
        return problems

    # ---------- identity ----------

    def signature(self) -> str:
        """Structural identity: what this genome can do, not what it is
        called or when it was born. Two genomes reached by different search
        paths that describe the same space are the same genome."""
        payload = {
            "operators": sorted(op.signature() for op in self.operators.values()),
            "representations": sorted(r.signature() for r in self.representations.values()),
            "templates": sorted(t.signature() for t in self.program_templates.values()),
            "rules": sorted(f"{r.name}={r.value}" for r in self.learning_rules.values()),
            "selection": self.selection_policy,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def size(self) -> Dict[str, int]:
        """How large the described space is. The number that has to grow
        for "expanded cognitive space" to mean anything measurable."""
        return {"operators": len(self.operators),
                "composite_operators": sum(1 for o in self.operators.values()
                                           if o.implementation == "composite"),
                "representations": len(self.representations),
                "representation_fields": sum(len(r.fields) for r in self.representations.values()),
                "templates": len(self.program_templates),
                "rules": len(self.learning_rules)}

    # ---------- persistence ----------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome_id": self.genome_id, "parent_id": self.parent_id,
            "born_at": self.born_at, "mutation": self.mutation,
            "selection_policy": self.selection_policy,
            "operators": {k: v.as_dict() for k, v in self.operators.items()},
            "representations": {k: asdict(v) for k, v in self.representations.items()},
            "program_templates": {k: asdict(v) for k, v in self.program_templates.items()},
            "learning_rules": {k: asdict(v) for k, v in self.learning_rules.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CognitiveGenome":
        from .ir import OperatorEvidence

        def op_from(d: Dict[str, Any]) -> CognitiveOperator:
            d = dict(d)
            ev = d.pop("evidence", None) or {}
            for key in ("inputs", "outputs", "preconditions", "postconditions",
                        "failure_modes", "applicability", "components", "derived_from"):
                d[key] = tuple(d.get(key) or ())
            evidence = OperatorEvidence(
                runs=int(ev.get("runs", 0)), successes=int(ev.get("successes", 0)),
                mean_latency=float(ev.get("mean_latency", 0.0)),
                domains_seen=tuple(ev.get("domains_seen") or ()),
                domains_transferred=tuple(ev.get("domains_transferred") or ()))
            return CognitiveOperator(evidence=evidence, **d)

        return cls(
            operators={k: op_from(v) for k, v in (data.get("operators") or {}).items()},
            representations={k: Representation(name=v["name"], fields=tuple(v["fields"]),
                                               description=v.get("description", ""),
                                               derived_from=v.get("derived_from"))
                             for k, v in (data.get("representations") or {}).items()},
            program_templates={k: ProgramTemplate(name=v["name"], steps=tuple(v["steps"]),
                                                  applicability=tuple(v.get("applicability") or ()),
                                                  description=v.get("description", ""))
                               for k, v in (data.get("program_templates") or {}).items()},
            learning_rules={k: LearningRule(**v) for k, v in (data.get("learning_rules") or {}).items()},
            selection_policy=data.get("selection_policy", "capability_first"),
            genome_id=data.get("genome_id") or uuid.uuid4().hex[:12],
            parent_id=data.get("parent_id"),
            born_at=float(data.get("born_at") or time.time()),
            mutation=data.get("mutation"),
        )


# ---------------------------------------------------------------------------
# mutation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MutationProposal:
    """A candidate genome, the change that produced it, and the claim id
    the gate will rule on. Not a change: nothing here has been adopted."""
    proposal_id: str
    mutation: str
    rationale: str
    parent: CognitiveGenome
    candidate: CognitiveGenome
    expands_space: bool

    def summary(self) -> Dict[str, Any]:
        before, after = self.parent.size(), self.candidate.size()
        return {"proposal_id": self.proposal_id, "mutation": self.mutation,
                "rationale": self.rationale, "expands_space": self.expands_space,
                "size_before": before, "size_after": after,
                "delta": {k: after[k] - before[k] for k in after}}


def _child(parent: CognitiveGenome, mutation: str, **changes: Any) -> CognitiveGenome:
    base = dict(operators=dict(parent.operators),
                representations=dict(parent.representations),
                program_templates=dict(parent.program_templates),
                learning_rules=dict(parent.learning_rules),
                selection_policy=parent.selection_policy)
    base.update(changes)
    return CognitiveGenome(parent_id=parent.genome_id, mutation=mutation, **base)


def propose(parent: CognitiveGenome, mutation: str, rationale: str = "",
            **params: Any) -> MutationProposal:
    """Build one candidate genome. Raises on a malformed change.

    Deliberately raises rather than returning an invalid proposal: an
    unmeasurable candidate that reaches the experiment queue consumes the
    budget that a real one needed.
    """
    if mutation not in MUTATIONS:
        raise GenomeError(f"unknown mutation {mutation!r}")
    builder = _BUILDERS.get(mutation)
    if builder is None:                                    # pragma: no cover - guarded by MUTATIONS
        raise GenomeError(f"no builder for {mutation!r}")
    candidate, expands = builder(parent, params)
    return MutationProposal(
        proposal_id=uuid.uuid4().hex[:12], mutation=mutation,
        rationale=rationale, parent=parent, candidate=candidate, expands_space=expands)


def _m_compose(parent: CognitiveGenome, p: Dict[str, Any]):
    """The mutation that matters most: turn a chain into a new operator.

    If GENERATE -> SIMULATE -> COUNTEREXAMPLE -> CRITIQUE systematically
    beats the alternatives, this is how it becomes a step in its own right
    and takes part in further composition. Everything else in this module
    exists so that this one can be measured honestly.
    """
    steps: Sequence[str] = p.get("steps") or ()
    new_id = str(p.get("op_id") or "").strip().upper()
    if len(steps) < 2:
        raise GenomeError("compose_operators needs at least two steps")
    if not new_id:
        raise GenomeError("compose_operators needs an op_id")
    if new_id in parent.operators:
        raise GenomeError(f"operator {new_id} already exists")
    missing = [s for s in steps if s not in parent.operators]
    if missing:
        raise GenomeError(f"unknown steps: {missing}")
    chain = [parent.operators[s] for s in steps]
    try:
        new_op = compose(new_id, chain, description=p.get("description", ""),
                         applicability=p.get("applicability", ()))
    except CompositionError as exc:
        raise GenomeError(str(exc)) from exc
    ops = dict(parent.operators)
    ops[new_id] = new_op
    return _child(parent, "compose_operators", operators=ops), True


def _m_add(parent: CognitiveGenome, p: Dict[str, Any]):
    op = p.get("operator")
    if not isinstance(op, CognitiveOperator):
        raise GenomeError("add_operator needs an `operator`")
    if op.op_id in parent.operators:
        raise GenomeError(f"operator {op.op_id} already exists")
    ops = dict(parent.operators)
    ops[op.op_id] = op
    return _child(parent, "add_operator", operators=ops), True


def _m_remove(parent: CognitiveGenome, p: Dict[str, Any]):
    """Removal is a real mutation, not only an undo: an operator that
    never earns its cost makes every plan worse by being available."""
    op_id = p.get("op_id")
    if op_id not in parent.operators:
        raise GenomeError(f"unknown operator {op_id!r}")
    used_by = [t.name for t in parent.program_templates.values() if op_id in t.steps]
    if used_by:
        raise GenomeError(f"{op_id} is used by templates {used_by}")
    depends = [o.op_id for o in parent.operators.values() if op_id in o.components]
    if depends:
        raise GenomeError(f"{op_id} is a component of {depends}")
    ops = dict(parent.operators)
    ops.pop(op_id)
    return _child(parent, "remove_operator", operators=ops), False


def _m_split(parent: CognitiveGenome, p: Dict[str, Any]):
    """Break a composite back into its parts under new names.

    The inverse of composition, and useful for the same reason refactoring
    is: a composite that wins for only one of its steps is better replaced
    by that step.
    """
    op_id = p.get("op_id")
    op = parent.operators.get(op_id)
    if op is None:
        raise GenomeError(f"unknown operator {op_id!r}")
    if op.implementation != "composite" or len(op.components) < 2:
        raise GenomeError(f"{op_id} is not a composite of two or more operators")
    ops = dict(parent.operators)
    ops.pop(op_id)
    return _child(parent, "split_operator", operators=ops), False


def _m_merge(parent: CognitiveGenome, p: Dict[str, Any]):
    """Replace two operators with one that covers both.

    Distinct from compose: merging says "these were the same idea", and
    the evidence for that is behavioural, not structural.
    """
    ids: Sequence[str] = p.get("op_ids") or ()
    new_id = str(p.get("op_id") or "").strip().upper()
    if len(ids) != 2 or not new_id:
        raise GenomeError("merge_operators needs exactly two op_ids and a new op_id")
    missing = [i for i in ids if i not in parent.operators]
    if missing:
        raise GenomeError(f"unknown operators: {missing}")
    a, b = parent.operators[ids[0]], parent.operators[ids[1]]
    merged = CognitiveOperator(
        op_id=new_id,
        inputs=tuple(dict.fromkeys(a.inputs + b.inputs)),
        outputs=tuple(dict.fromkeys(a.outputs + b.outputs)),
        description=p.get("description") or f"merge of {a.op_id} and {b.op_id}",
        cost=max((a.cost, b.cost), key=lambda c: COSTS.index(c) if c in COSTS else 0),
        uncertainty=max(a.uncertainty, b.uncertainty),
        failure_modes=tuple(dict.fromkeys(a.failure_modes + b.failure_modes)),
        applicability=tuple(dict.fromkeys(a.applicability + b.applicability)),
        implementation="composite", components=(a.op_id, b.op_id),
        derived_from=(a.op_id, b.op_id))
    ops = dict(parent.operators)
    ops[new_id] = merged
    return _child(parent, "merge_operators", operators=ops), True


def _m_create_repr(parent: CognitiveGenome, p: Dict[str, Any]):
    """Level 3: a new vocabulary for describing tasks or processes."""
    name = str(p.get("name") or "").strip()
    fields_ = tuple(p.get("fields") or ())
    if not name or len(fields_) < 2:
        raise GenomeError("create_representation needs a name and at least two fields")
    if name in parent.representations:
        raise GenomeError(f"representation {name} already exists")
    reps = dict(parent.representations)
    reps[name] = Representation(name, fields_, p.get("description", ""), p.get("derived_from"))
    return _child(parent, "create_representation", representations=reps), True


def _m_modify_repr(parent: CognitiveGenome, p: Dict[str, Any]):
    name = p.get("name")
    current = parent.representations.get(name)
    if current is None:
        raise GenomeError(f"unknown representation {name!r}")
    fields_ = tuple(p.get("fields") or ())
    if len(fields_) < 2:
        raise GenomeError("a representation needs at least two fields")
    reps = dict(parent.representations)
    reps[name] = replace(current, fields=fields_, derived_from=current.name)
    return _child(parent, "modify_representation", representations=reps), len(fields_) > len(current.fields)


def _m_create_template(parent: CognitiveGenome, p: Dict[str, Any]):
    name = str(p.get("name") or "").strip()
    steps = tuple(p.get("steps") or ())
    if not name or len(steps) < 2:
        raise GenomeError("create_program_template needs a name and at least two steps")
    if name in parent.program_templates:
        raise GenomeError(f"template {name} already exists")
    tmpls = dict(parent.program_templates)
    tmpls[name] = ProgramTemplate(name, steps, tuple(p.get("applicability") or ()),
                                  p.get("description", ""))
    return _child(parent, "create_program_template", program_templates=tmpls), True


def _m_modify_rule(parent: CognitiveGenome, p: Dict[str, Any]):
    name = p.get("name")
    rule = parent.learning_rules.get(name)
    if rule is None:
        raise GenomeError(f"unknown learning rule {name!r}")
    value = float(p.get("value", rule.value))
    # Clamped, not rejected: a proposal slightly outside the bounds is a
    # search step, and the bounds exist to contain it rather than to make
    # the search fail.
    rules = dict(parent.learning_rules)
    rules[name] = replace(rule, value=rule.clamped(value))
    return _child(parent, "modify_learning_rule", learning_rules=rules), False


def _m_modify_policy(parent: CognitiveGenome, p: Dict[str, Any]):
    policy = str(p.get("policy") or "").strip()
    if not policy:
        raise GenomeError("modify_selection_policy needs a policy")
    return _child(parent, "modify_selection_policy", selection_policy=policy), False


_BUILDERS = {
    "add_operator": _m_add,
    "remove_operator": _m_remove,
    "compose_operators": _m_compose,
    "split_operator": _m_split,
    "merge_operators": _m_merge,
    "create_representation": _m_create_repr,
    "modify_representation": _m_modify_repr,
    "create_program_template": _m_create_template,
    "modify_learning_rule": _m_modify_rule,
    "modify_selection_policy": _m_modify_policy,
}


# ---------------------------------------------------------------------------
# adoption
# ---------------------------------------------------------------------------

class NotAccepted(RuntimeError):
    """A proposal reached `apply()` without a verdict that admits it."""


def apply(proposal: MutationProposal, verdict: Any, claim_id: str) -> CognitiveGenome:
    """Adopt a candidate, and only with a verdict that belongs to it.

    Three checks, and the third is the one that is easy to omit: a verdict
    is tied to a claim, and a claim to a proposal. Without matching them,
    a single accepted result could admit any number of unrelated changes --
    the cheapest way there is to get a mutation past the gate.
    """
    if not getattr(verdict, "accepted", False):
        raise NotAccepted(f"verdict is not accepted: {getattr(verdict, 'reason', '?')}")
    if claim_id != proposal.proposal_id:
        raise NotAccepted(
            f"verdict belongs to claim {claim_id!r}, not to proposal {proposal.proposal_id!r}")
    problems = proposal.candidate.validate()
    if problems:
        raise NotAccepted("candidate is malformed: " + "; ".join(problems))
    return proposal.candidate
