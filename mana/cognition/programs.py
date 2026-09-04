"""
mana.cognition.programs — a program of thought, and what happens while it
runs.

What this is not
----------------
Not a second implementation of `_answer_core`. The existing execution path
decides what to do next with a dispatcher over eight hardcoded node names;
a program here is *data* -- an ordered chain of operator ids drawn from the
genome -- and the runtime is an interpreter for it. The behaviour can be
identical; what changes is that the sequence became something that can be
generated, compared, mutated and measured, instead of something written
into a control flow.

Why a chain and not a general graph
-----------------------------------
The type system in `ir` accumulates rather than consuming: once a draft
exists it stays available. That makes a linear chain able to express every
shape the current system actually uses, including the branch-like ones --
"critique, and repair only if the critique is bad" is a chain plus a
runtime guard, not a graph edge. A general DAG would add expressiveness
the search cannot yet use and a novelty comparison that is much harder to
define. `steps` is a tuple, and `as_graph()` exists for when that stops
being enough.

Budget is part of the program's world, not a caller's courtesy
--------------------------------------------------------------
Every run carries a budget in calls and seconds, checked between steps.
A cognitive program that can spend without limit is one that a search will
learn to make longer, because more steps almost always score slightly
better and nothing pushes back.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .ir import ANSWER, TASK, CognitiveOperator, CompositionError, check_chain

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"


@dataclass(frozen=True)
class Budget:
    """What one run of a program may spend.

    `calls` counts brain invocations, which is the resource that is
    actually scarce: the free tiers this runs on allow a few thousand a
    day, and an experiment needs thousands per hypothesis.
    """
    calls: int = 8
    seconds: float = 120.0

    def exhausted(self, used_calls: int, elapsed: float) -> Optional[str]:
        if used_calls >= self.calls:
            return f"call budget spent ({used_calls}/{self.calls})"
        if elapsed >= self.seconds:
            return f"time budget spent ({elapsed:.1f}s/{self.seconds:.0f}s)"
        return None


@dataclass
class StepRecord:
    """One executed operator. Kept per step rather than aggregated because
    "which step failed" is the first question about any bad answer, and
    reconstructing it from totals is impossible."""
    index: int
    op_id: str
    ok: bool
    produced: Tuple[str, ...] = ()
    calls: int = 0
    latency: float = 0.0
    brain: str = ""
    error: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgramState:
    """The table the operators work on.

    Values are keyed by IR type, so an operator asking for `draft` gets the
    latest draft regardless of which step produced it. History is kept per
    type so a chain that revises a draft twice can still be inspected --
    aggregating would erase exactly the thing a critique loop is for.
    """
    task: str
    values: Dict[str, Any] = field(default_factory=dict)
    history: Dict[str, List[Any]] = field(default_factory=dict)
    trace: List[StepRecord] = field(default_factory=list)
    calls_used: int = 0
    started: float = field(default_factory=time.perf_counter)

    def __post_init__(self) -> None:
        if TASK not in self.values:
            self.values[TASK] = self.task
            self.history.setdefault(TASK, []).append(self.task)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def available(self) -> List[str]:
        return [t for t, v in self.values.items() if v is not None]

    def put(self, type_name: str, value: Any) -> None:
        self.values[type_name] = value
        self.history.setdefault(type_name, []).append(value)

    def get(self, type_name: str, default: Any = None) -> Any:
        return self.values.get(type_name, default)

    def answer(self) -> str:
        """The best thing on the table to hand back.

        Falls through ANSWER -> DRAFT rather than returning nothing when a
        program was cut short by its budget: a draft produced but never
        finalised is still the honest best answer, and returning empty
        would score a budget stop as a capability failure.
        """
        for key in (ANSWER, "draft"):
            value = self.values.get(key)
            if value:
                return str(value)
        return ""


@dataclass(frozen=True)
class CognitiveProgram:
    """An executable chain of operator ids drawn from a genome."""
    program_id: str
    steps: Tuple[str, ...]
    template: str = ""
    rationale: str = ""
    genome_signature: str = ""

    @classmethod
    def build(cls, steps: Sequence[str], template: str = "", rationale: str = "",
              genome_signature: str = "") -> "CognitiveProgram":
        return cls(program_id=uuid.uuid4().hex[:12], steps=tuple(steps),
                   template=template, rationale=rationale,
                   genome_signature=genome_signature)

    def signature(self) -> str:
        """Structural identity: the chain, not the id or the rationale.

        Novelty compares programs by what they do. Two programs generated
        by different searches with the same steps are the same program,
        and a novelty measure that counts them as two would report
        discovery where there was none.
        """
        return hashlib.sha256("->".join(self.steps).encode()).hexdigest()[:16]

    def validate(self, operators: Dict[str, CognitiveOperator]) -> List[str]:
        """Everything wrong with this program against a genome's operators."""
        problems: List[str] = []
        unknown = [s for s in self.steps if s not in operators]
        if unknown:
            return [f"unknown operators: {unknown}"]
        if not self.steps:
            return ["a program with no steps cannot run"]
        try:
            check_chain([operators[s] for s in self.steps])
        except CompositionError as exc:
            problems.append(str(exc))
        return problems

    def expand(self, operators: Dict[str, CognitiveOperator]) -> Tuple[str, ...]:
        """Flatten composite operators into the primitives that run.

        A composed operator has to actually execute as its chain, or
        composition would be a naming exercise: the search would "discover"
        COUNTERFACTUAL_REFINEMENT, the runtime would have no
        implementation for it, and the measurement would score the failure
        of a step that was never run.

        Depth-limited because a genome can, in principle, contain a
        composite whose component list cycles back -- validation forbids
        it, but an interpreter should not loop forever on a genome that
        got there another way.
        """
        out: List[str] = []

        def walk(op_id: str, depth: int) -> None:
            op = operators.get(op_id)
            if op is None or depth > 8:
                out.append(op_id)
                return
            if op.implementation == "composite" and op.components:
                for component in op.components:
                    walk(component, depth + 1)
            else:
                out.append(op_id)

        for step in self.steps:
            walk(step, 0)
        return tuple(out)

    def as_graph(self) -> Dict[str, Any]:
        """Node/edge form, for novelty comparison and for display.

        The chain is linear today; this is the shape a comparison should
        be written against so that widening to a real DAG later does not
        invalidate every novelty measurement taken before it.
        """
        nodes = [{"index": i, "op": s} for i, s in enumerate(self.steps)]
        edges = [{"from": i, "to": i + 1} for i in range(len(self.steps) - 1)]
        return {"nodes": nodes, "edges": edges}

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProgramResult:
    """What a run produced, and what it cost.

    `stopped_early` is separate from `ok` on purpose: a program halted by
    its budget with a usable draft is not a failure, and folding the two
    together would make every budget cut look like an inability to answer.
    """
    program: CognitiveProgram
    answer: str
    ok: bool
    stopped_early: bool = False
    stop_reason: str = ""
    calls_used: int = 0
    elapsed: float = 0.0
    trace: List[StepRecord] = field(default_factory=list)
    brains_used: Tuple[str, ...] = ()

    def summary(self) -> Dict[str, Any]:
        return {
            "program_id": self.program.program_id,
            "steps": list(self.program.steps),
            "signature": self.program.signature(),
            "ok": self.ok, "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "calls_used": self.calls_used, "elapsed": round(self.elapsed, 3),
            "brains_used": list(self.brains_used),
            "trace": [{"op": s.op_id, "ok": s.ok, "brain": s.brain,
                       "latency": round(s.latency, 3), "error": s.error}
                      for s in self.trace],
        }
