"""
mana.cognition.compiler — task in, candidate programs out.

The requirement that shapes this
--------------------------------
"MANA must be able to generate different cognitive programs for the same
task." That single sentence rules out the obvious design. A compiler that
returns *the* program for a task is a router, and a router cannot be
searched: there is nothing to compare, so nothing can be discovered. This
one returns a ranked list of candidates, and the caller decides whether it
wants the top one (ordinary work) or a comparison (an experiment).

How a program is chosen
-----------------------
Templates from the genome are scored against what is known about the task
and what is affordable right now:

  * **applicability** -- does the template claim to suit this kind of task
  * **capability**    -- can every step actually run here (a program with
                         VERIFY is worthless without a sandbox, and one
                         with GENERATE is worthless with no brain)
  * **cost**          -- does the chain fit the budget, in calls
  * **evidence**      -- how the template has actually performed, where
                         that has been measured

Evidence is weighted last and only where it exists. A compiler that leans
on measured performance from the first run locks in whatever the first few
samples happened to say, and open-ended search then never revisits it.

What this deliberately does not do
----------------------------------
It does not invent chains. Composing new operators and proposing new
templates is the genome's business (`genome.propose`), and admitting them
is the gate's. The compiler works with what the current genome contains --
so when the genome grows, the compiler's reach grows with it, without a
line changing here. That is the point of putting the vocabulary in data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .genome import CognitiveGenome, ProgramTemplate
from .ir import COSTS, CognitiveOperator
from .programs import Budget, CognitiveProgram

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Roughly how many brain calls each cost class implies. Declared, and
#: used only for fitting a program to a budget before it runs -- measured
#: cost replaces it as soon as `OperatorEvidence` has anything to say.
_CALL_COST = {"free": 0, "low": 0, "medium": 1, "high": 1}


@dataclass(frozen=True)
class Capabilities:
    """What is available to run a program right now.

    Passed in rather than discovered here, so the compiler stays testable
    without an agent and so a planner can ask "what would I choose if the
    sandbox were available?" -- which is a question the experiment layer
    will need.
    """
    brains: int = 0
    has_sandbox: bool = False
    has_web: bool = False
    has_memory: bool = True

    def supports(self, op: CognitiveOperator) -> bool:
        if op.implementation == "brain" and self.brains < 1:
            return False
        if op.op_id == "VERIFY" and not self.has_sandbox:
            # VERIFY without an oracle is not a weaker check, it is a
            # missing one -- and a program that includes it would score
            # its absence as a failure of the program.
            return False
        if op.op_id == "RETRIEVE" and not (self.has_web or self.has_memory):
            return False
        return True


@dataclass(frozen=True)
class Candidate:
    program: CognitiveProgram
    score: float
    estimated_calls: int
    reasons: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {"program": self.program.as_dict(), "score": round(self.score, 4),
                "estimated_calls": self.estimated_calls, "reasons": list(self.reasons)}


def classify(task: str) -> Tuple[str, float]:
    """A cheap (kind, difficulty) read of the task, with no model involved.

    Reuses the brain pool's difficulty heuristic rather than inventing a
    second one: two disagreeing estimates of the same quantity is worse
    than one imperfect estimate, and this one is already exercised.
    """
    from ..brains import BrainPool
    difficulty = BrainPool.estimate_difficulty(task)
    t = (task or "").lower()
    if any(m in t for m in ("вычисли", "посчитай", "сколько", "calculate")):
        kind = "math"
    elif any(m in t for m in ("функци", "код", "python", "напиши функцию")):
        kind = "programming"
    elif any(m in t for m in ("продолжи", "последовательность", "sequence")):
        kind = "sequence"
    elif any(m in t for m in ("почему", "объясни", "сравни", "обоснуй")):
        kind = "reasoning"
    else:
        kind = "general"
    return kind, difficulty


def estimate_calls(steps: Sequence[str], operators: Dict[str, CognitiveOperator]) -> int:
    total = 0
    for step in steps:
        op = operators.get(step)
        if op is None:
            continue
        if op.implementation == "composite" and op.components:
            total += estimate_calls(op.components, operators)
        else:
            total += _CALL_COST.get(op.cost, 1)
    return total


def _score_template(template: ProgramTemplate, genome: CognitiveGenome, kind: str,
                    difficulty: float, capabilities: Capabilities,
                    budget: Budget) -> Optional[Candidate]:
    steps = template.steps
    ops = [genome.operators.get(s) for s in steps]
    if any(op is None for op in ops):
        return None
    reasons: List[str] = []

    unsupported = [op.op_id for op in ops if not capabilities.supports(op)]
    if unsupported:
        return None                       # cannot run here at all; not a low score

    calls = estimate_calls(steps, genome.operators)
    if calls > budget.calls:
        return None                       # would be cut off mid-chain

    score = 0.0
    if kind in template.applicability:
        score += 1.0
        reasons.append(f"suits {kind}")
    if difficulty >= 0.55 and "hard" in template.applicability:
        score += 0.7
        reasons.append("suits hard tasks")
    if difficulty < 0.3 and "easy" in template.applicability:
        score += 0.7
        reasons.append("suits easy tasks")

    # Cheaper is better, all else equal: the budget is the scarce thing,
    # and a chain that spends four calls to gain what two would has made
    # the next experiment less affordable.
    score += 0.35 * (1.0 - min(1.0, calls / max(1, budget.calls)))

    # Longer chains are penalised beyond their call cost, because every
    # additional step is another way to fail (see how compose() compounds
    # uncertainty) and the search will otherwise drift toward length.
    score -= 0.08 * max(0, len(steps) - 3)

    measured = [genome.operators[s].evidence for s in steps
                if genome.operators[s].evidence.is_measured]
    if measured:
        mean_success = sum(e.success_rate or 0.0 for e in measured) / len(measured)
        score += 0.5 * (mean_success - 0.5)
        reasons.append(f"measured success {mean_success:.2f}")

    return Candidate(
        program=CognitiveProgram.build(steps, template=template.name,
                                       rationale="; ".join(reasons),
                                       genome_signature=genome.signature()),
        score=score, estimated_calls=calls, reasons=tuple(reasons))


def compile_candidates(task: str, genome: CognitiveGenome, capabilities: Capabilities,
                       budget: Optional[Budget] = None,
                       limit: int = 5) -> List[Candidate]:
    """Rank every template that could run this task here.

    Returns an empty list rather than a fallback when nothing fits: an
    empty result is information the caller can act on ("no brain, no
    sandbox, nothing to run"), while a silently substituted trivial
    program would be measured as if it had been chosen.
    """
    budget = budget or Budget()
    kind, difficulty = classify(task)
    candidates: List[Candidate] = []
    for template in genome.program_templates.values():
        scored = _score_template(template, genome, kind, difficulty, capabilities, budget)
        if scored is not None:
            candidates.append(scored)
    candidates.sort(key=lambda c: (-c.score, c.estimated_calls, c.program.template))
    return candidates[:limit]


def compile_program(task: str, genome: CognitiveGenome, capabilities: Capabilities,
                    budget: Optional[Budget] = None) -> Optional[CognitiveProgram]:
    """The single best candidate, for ordinary work."""
    candidates = compile_candidates(task, genome, capabilities, budget, limit=1)
    return candidates[0].program if candidates else None


def compile_alternatives(task: str, genome: CognitiveGenome, capabilities: Capabilities,
                         budget: Optional[Budget] = None, count: int = 2) -> List[CognitiveProgram]:
    """Structurally *different* programs for the same task.

    This is the function experiments need, and the reason the compiler
    returns a list at all. Deduplicated by signature rather than by
    template name: two templates that expand to the same chain are one
    alternative, and comparing them would spend a full experiment budget
    measuring the difference between a thing and itself.
    """
    seen: set = set()
    out: List[CognitiveProgram] = []
    for candidate in compile_candidates(task, genome, capabilities, budget, limit=20):
        signature = candidate.program.signature()
        if signature in seen:
            continue
        seen.add(signature)
        out.append(candidate.program)
        if len(out) >= count:
            break
    return out
