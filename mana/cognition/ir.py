"""
mana.cognition.ir — cognitive operators as data, independent of any model.

Why an IR at all
----------------
MANA already had something operator-shaped: `PipelineSpec.graph_nodes`, a
tuple of eight allowed strings ("LLM", "CRITIC", "EXECUTE", ...) executed
by a dispatcher in `execution.py`. That is a fixed vocabulary with no
types, no preconditions, no cost, and -- decisively -- no way to add a
ninth. Evolution could rearrange those eight; it could not invent a step.
An open-ended system has to be able to grow its own vocabulary, and a
tuple of magic strings cannot express a new one.

So an operator becomes a described object: what it consumes, what it
produces, what has to hold before it runs, what it costs, how it can fail,
and how it is implemented. The description is what makes composition
checkable -- OBSERVE feeding SIMULATE is well-formed or it is not, and
that is decidable from the types rather than discovered at runtime.

Independent of the LLM, on purpose
----------------------------------
An operator names a *capability*, not a prompt. `GENERATE` might be a
frontier model, a local 7B, or a template; `VERIFY` might be arithmetic in
`LocalVerifier`, a sandboxed test run, or a symbolic check. The brain pool
is a runtime resource the operator draws on, which is what stops the
cognitive layer from becoming prompt engineering with extra steps.

Composition is where new operators come from
--------------------------------------------
`compose()` builds one operator out of a sequence of others. This is the
mechanism the whole project rests on: if GENERATE -> SIMULATE ->
COUNTEREXAMPLE -> CRITIQUE beats the alternatives systematically, that
chain can become a named operator with its own identity, statistics and
applicability -- and then take part in further composition. Nothing here
decides that it *should*; that verdict belongs to `core.gates`. This module
only makes the object expressible.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"


# ---------------------------------------------------------------------------
# types flowing between operators
# ---------------------------------------------------------------------------
#
# Deliberately coarse. Fine-grained types would make composition provably
# correct and practically impossible: almost nothing would type-check, and
# the search space this exists to open would close again. These say enough
# to catch "you cannot critique something nobody has drafted yet".

TASK = "task"                  # the question, as posed
CONTEXT = "context"            # retrieved material
DRAFT = "draft"                # a candidate answer
CRITIQUE = "critique"          # a judgement about a draft
EVIDENCE = "evidence"          # something checked against a non-model oracle
PLAN = "plan"                  # a decomposition into sub-questions
PREDICTION = "prediction"      # what a model expects to happen
ABSTRACTION = "abstraction"    # a restated, generalised form of a task
ANSWER = "answer"              # the final output

TYPES = (TASK, CONTEXT, DRAFT, CRITIQUE, EVIDENCE, PLAN, PREDICTION, ABSTRACTION, ANSWER)

#: Rough cost classes. Not measured -- declared, and marked as declared.
#: `CognitiveOperator.evidence` is where measured numbers accumulate; these
#: are the prior a planner starts from before it has any.
COSTS = ("free", "low", "medium", "high")


@dataclass(frozen=True)
class OperatorEvidence:
    """What has actually been observed about an operator.

    Separate from the declared metadata above it, and empty until
    something measures it. The distinction matters: `cost="high"` is an
    opinion, `mean_latency=4.2 over 180 runs` is a fact, and a planner
    that cannot tell them apart will trust the opinion forever.
    """
    runs: int = 0
    successes: int = 0
    mean_latency: float = 0.0
    domains_seen: Tuple[str, ...] = ()
    domains_transferred: Tuple[str, ...] = ()

    @property
    def success_rate(self) -> Optional[float]:
        return self.successes / self.runs if self.runs else None

    @property
    def is_measured(self) -> bool:
        return self.runs > 0


@dataclass(frozen=True)
class CognitiveOperator:
    """One step of thinking, described well enough to compose and to cost.

    Frozen: an operator whose declaration can be edited after it has
    accumulated evidence is an operator whose evidence describes something
    else. Mutation produces a new operator with a new id.
    """
    op_id: str
    inputs: Tuple[str, ...]              # required to run at all
    outputs: Tuple[str, ...]
    #: Consumed when present, not required. The distinction was forced by
    #: the first composition test: GENERATE declared CONTEXT as required,
    #: so no chain could start with it -- yet the real system generates
    #: from a task with an empty context every time the memory lookup
    #: finds nothing. Modelling optionality explicitly is better than
    #: weakening the type check, which would stop catching the case it
    #: exists for (CRITIQUE with nothing drafted).
    optional_inputs: Tuple[str, ...] = ()
    description: str = ""
    preconditions: Tuple[str, ...] = ()
    postconditions: Tuple[str, ...] = ()
    cost: str = "medium"
    uncertainty: float = 0.5          # declared prior: how unreliable is this step
    failure_modes: Tuple[str, ...] = ()
    applicability: Tuple[str, ...] = ()   # task domains/kinds it claims to suit
    implementation: str = "brain"     # brain | tool | composite | builtin
    components: Tuple[str, ...] = ()  # for composite operators: the chain
    evidence: OperatorEvidence = field(default_factory=OperatorEvidence)
    derived_from: Tuple[str, ...] = ()

    def signature(self) -> str:
        """Structural identity, ignoring name and accumulated evidence.

        Two operators built by different search paths that consume and
        produce the same things through the same chain are the same idea;
        `novelty` will need to say so, and a name is not a distinguishing
        feature.
        """
        payload = {"inputs": sorted(self.inputs), "outputs": sorted(self.outputs),
                   "optional": sorted(self.optional_inputs),
                   "implementation": self.implementation,
                   "components": list(self.components)}
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def accepts(self, available: Sequence[str]) -> bool:
        """Can this run given what is on the table?"""
        return all(i in available for i in self.inputs)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def with_evidence(self, evidence: OperatorEvidence) -> "CognitiveOperator":
        return replace(self, evidence=evidence)


# ---------------------------------------------------------------------------
# the starting vocabulary
# ---------------------------------------------------------------------------
#
# A starting point, not a closed set -- that is the whole difference from
# graph_nodes. Each of these corresponds to something MANA can already do,
# so the initial genome describes the system as it is rather than as an
# aspiration; a genome whose operators are not backed by anything would
# make every early measurement meaningless.

def primitive_operators() -> Dict[str, CognitiveOperator]:
    ops = [
        CognitiveOperator(
            "OBSERVE", (TASK,), (CONTEXT,), "Read the task and note its surface features.",
            cost="free", uncertainty=0.1, implementation="builtin",
            postconditions=("context is non-empty",)),
        CognitiveOperator(
            "RETRIEVE", (TASK,), (CONTEXT,), "Fetch relevant material from memory or the web.",
            cost="low", uncertainty=0.4, implementation="tool",
            failure_modes=("retrieves plausible but irrelevant material",),
            derived_from=("graph_nodes:MEMORY", "graph_nodes:WEB")),
        CognitiveOperator(
            "GENERATE", (TASK,), (DRAFT,), optional_inputs=(CONTEXT,),
            description="Produce a candidate answer.",
            cost="high", uncertainty=0.5, implementation="brain",
            failure_modes=("fabricates specifics", "answers a nearby question"),
            derived_from=("graph_nodes:LLM",)),
        CognitiveOperator(
            "DECOMPOSE", (TASK,), (PLAN,), "Split the task into independent sub-questions.",
            cost="high", uncertainty=0.5, implementation="composite",
            preconditions=("task has separable parts",),
            failure_modes=("splits a single question into halves of nothing",),
            derived_from=("decompose.solve",)),
        CognitiveOperator(
            "CRITIQUE", (TASK, DRAFT), (CRITIQUE,), "Judge a draft against the task.",
            cost="high", uncertainty=0.6, implementation="brain",
            preconditions=("a draft exists",),
            failure_modes=("self-review when the critic is the author",),
            derived_from=("graph_nodes:CRITIC",)),
        CognitiveOperator(
            "REPAIR", (DRAFT, CRITIQUE), (DRAFT,), "Revise a draft in light of a critique.",
            cost="high", uncertainty=0.5, implementation="brain",
            preconditions=("a critique exists",),
            derived_from=("graph_nodes:REPAIR",)),
        CognitiveOperator(
            "VERIFY", (TASK, DRAFT), (EVIDENCE,), "Check a draft against a non-model oracle.",
            cost="low", uncertainty=0.1, implementation="tool",
            postconditions=("evidence carries a trust level",),
            failure_modes=("no oracle applies to this task",),
            derived_from=("graph_nodes:EXECUTE", "verifier.verify")),
        CognitiveOperator(
            "SYNTHESIZE", (DRAFT,), (ANSWER,), optional_inputs=(PLAN,),
            description="Assemble parts into one answer.",
            cost="high", uncertainty=0.4, implementation="brain",
            derived_from=("graph_nodes:SYNTHESIS", "decompose.synthesize")),
        CognitiveOperator(
            # One DRAFT type, several instances: the type system says what
            # kind of thing is needed, not how many, and inventing a
            # "pair of drafts" type to express arity would close far more
            # of the search space than it protects.
            "COMPARE", (DRAFT,), (CRITIQUE,), description="Set two candidates against each other.",
            cost="medium", uncertainty=0.4, implementation="brain",
            derived_from=("brains.ask_consensus",)),
        CognitiveOperator(
            "ABSTRACT", (TASK,), (ABSTRACTION,), "Restate the task in more general terms.",
            cost="high", uncertainty=0.7, implementation="brain",
            failure_modes=("generalises away the part that mattered",)),
        CognitiveOperator(
            "PREDICT", (DRAFT,), (PREDICTION,), optional_inputs=(CONTEXT,),
            description="State what should follow if the draft is right.",
            cost="high", uncertainty=0.7, implementation="brain"),
        CognitiveOperator(
            "COUNTEREXAMPLE", (DRAFT,), (CRITIQUE,), "Look for a case where the draft fails.",
            cost="high", uncertainty=0.6, implementation="brain",
            failure_modes=("produces a case the draft never claimed to cover",)),
        CognitiveOperator(
            "ANSWER", (DRAFT,), (ANSWER,), "Emit the draft as the final answer.",
            cost="free", uncertainty=0.0, implementation="builtin"),
    ]
    return {op.op_id: op for op in ops}


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

class CompositionError(ValueError):
    """A chain that cannot run. Raised rather than returned because a
    malformed operator must never reach the registry -- an operator that
    type-checks nowhere would be selected, fail at runtime, and be scored
    as a capability deficit."""


def check_chain(chain: Sequence[CognitiveOperator], available: Sequence[str] = (TASK,)) -> List[str]:
    """Type-check a sequence, returning what is on the table at the end.

    Types accumulate rather than being consumed: a draft is still a draft
    after being critiqued, and modelling it otherwise would forbid the
    critique-then-repair-then-critique shapes that are the most obviously
    useful thing to discover.
    """
    on_table = list(dict.fromkeys(available))
    for i, op in enumerate(chain):
        if not op.accepts(on_table):
            missing = [t for t in op.inputs if t not in on_table]
            raise CompositionError(
                f"step {i} ({op.op_id}) needs {missing} but only {on_table} is available")
        for out in op.outputs:
            if out not in on_table:
                on_table.append(out)
    return on_table


def compose(op_id: str, chain: Sequence[CognitiveOperator], description: str = "",
            applicability: Sequence[str] = ()) -> CognitiveOperator:
    """Fuse a chain into one named operator.

    This is where a new cognitive step can come from. The resulting
    operator inherits the chain's outer signature (what the whole thing
    needs, what it leaves behind), the union of the failure modes -- a
    composite can fail in every way its parts can -- and the maximum cost,
    since a chain is at least as expensive as its dearest step.

    Uncertainty is NOT averaged. Composing four steps at 0.5 does not give
    0.5; errors compound, and a planner told otherwise will keep choosing
    long chains. It is combined as 1 - prod(1 - u), which is what
    "any step can spoil it" actually implies.
    """
    if not chain:
        raise CompositionError("cannot compose an empty chain")
    check_chain(chain)
    inputs = tuple(dict.fromkeys(t for op in chain for t in op.inputs
                                 if t not in _produced_before(chain, op)))
    # Everything the chain leaves on the table, not just the last step's
    # output. Caught by composing SELF_CHECK (GENERATE->CRITIQUE) and then
    # feeding it to REPAIR: the draft is still there after being
    # critiqued, and a composite that forgets it cannot be built on.
    outputs = tuple(dict.fromkeys(t for op in chain for t in op.outputs))
    combined_uncertainty = 1.0
    for op in chain:
        combined_uncertainty *= (1.0 - max(0.0, min(1.0, op.uncertainty)))
    return CognitiveOperator(
        op_id=op_id,
        inputs=inputs or (TASK,),
        outputs=outputs,
        description=description or " -> ".join(op.op_id for op in chain),
        cost=max((op.cost for op in chain), key=lambda c: COSTS.index(c) if c in COSTS else 0),
        uncertainty=round(1.0 - combined_uncertainty, 4),
        failure_modes=tuple(dict.fromkeys(f for op in chain for f in op.failure_modes)),
        applicability=tuple(applicability),
        implementation="composite",
        components=tuple(op.op_id for op in chain),
        derived_from=tuple(op.op_id for op in chain),
    )


def _produced_before(chain: Sequence[CognitiveOperator], target: CognitiveOperator) -> List[str]:
    """Types already on the table by the time `target` runs -- so a
    composite does not demand as input something its own earlier step
    produces."""
    produced: List[str] = []
    for op in chain:
        if op is target:
            break
        produced.extend(op.outputs)
    return produced
