"""
mana.cognition.search — finding programs nobody wrote.

The audit found the mechanism for expanding the operator space built,
tested, and called from nowhere. `genome.compose_operators` turns a
proven chain into an operator that can take part in further composition
-- the mutation the whole cognitive layer was designed around -- and a
grep for it outside `genome.py` returned nothing. The compiler ranked
the five baseline templates and whatever synthesis had installed; it
never looked at the space those templates were drawn from.

So this module is a connection, not an invention. It enumerates chains
the type system admits, prices them with the estimator the compiler
already uses, and hands the survivors to the machinery that was waiting
for them.

Why enumeration rather than a model proposing chains
----------------------------------------------------
A model asked for "a good cognitive program" produces plausible ones,
and plausible is the failure mode this project exists to avoid: it
returns chains that read well and were never checked against the type
system, the budget, or each other. Enumeration is exhaustive within its
depth, costs no calls, and every chain it emits is executable by
construction. The expensive part -- finding out whether a chain is
*better* -- is left where it belongs, with the experiment lab and the
gates.

What makes the space open rather than fixed
-------------------------------------------
Composition. A chain accepted as an operator becomes a step other chains
can contain, so the reachable space after n adoptions is not the same
space enumerated n times -- it is larger, and grows with what has been
proven. That is the difference between searching inside a space and
extending one.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .compiler import estimate_calls
from .genome import CognitiveGenome, ProgramTemplate
from .ir import (ANSWER, TASK, CognitiveOperator, CompositionError,
                 check_chain)

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"

#: Chains longer than this are not enumerated. Not a statement about what
#: is possible -- composition reaches further by making a proven chain
#: one step -- but about what is affordable: the space grows as
#: operators^depth, and every candidate past here costs more to test than
#: the depth below it costs to exhaust.
MAX_DEPTH = 5

#: A chain must start by looking at the task and end by producing an
#: answer. Enumerating chains that do neither wastes the budget on
#: programs no runtime can execute.
REQUIRED_FIRST = "OBSERVE"
REQUIRED_LAST = "ANSWER"


@dataclass(frozen=True)
class Candidate:
    """One chain the type system admits, priced but not yet judged."""
    steps: Tuple[str, ...]
    estimated_calls: int
    novelty: float                  # how unlike the existing templates
    reasons: Tuple[str, ...] = ()

    @property
    def depth(self) -> int:
        return len(self.steps)

    def as_dict(self) -> Dict[str, Any]:
        return {"steps": list(self.steps), "estimated_calls": self.estimated_calls,
                "novelty": round(self.novelty, 3), "reasons": list(self.reasons)}

    def describe(self) -> str:
        return f"{' → '.join(self.steps)} ({self.estimated_calls} вызовов)"


def _middles(operators: Sequence[str], depth: int) -> Iterator[Tuple[str, ...]]:
    """Every ordering of `depth` steps between OBSERVE and ANSWER.

    Permutations rather than combinations: OBSERVE→GENERATE→CRITIQUE and
    OBSERVE→CRITIQUE→GENERATE are different programs, and one of them is
    nonsense -- which the type check is there to discover rather than
    this function to assume.
    """
    for combination in itertools.permutations(operators, depth):
        yield combination


def enumerate_chains(genome: CognitiveGenome, max_depth: int = MAX_DEPTH,
                     limit: int = 400) -> List[Tuple[str, ...]]:
    """Every chain the IR admits, up to `max_depth`, cheapest first.

    Type-checked by `ir.check_chain`, which is what makes this different
    from generating strings: a chain that reaches here can be executed,
    because every step's inputs are produced by something before it.
    """
    ops = genome.operators
    middle_pool = [op_id for op_id in sorted(ops)
                   if op_id not in (REQUIRED_FIRST, REQUIRED_LAST)]
    found: List[Tuple[str, ...]] = []
    for depth in range(1, max(1, max_depth - 1)):
        for middle in _middles(middle_pool, depth):
            steps = (REQUIRED_FIRST,) + middle + (REQUIRED_LAST,)
            chain = [ops[s] for s in steps if s in ops]
            if len(chain) != len(steps):
                continue
            try:
                check_chain(chain, available=(TASK,))
            except CompositionError:
                # A type error is a real rejection, not a low score: the
                # chain cannot run, and enumerating it further would spend
                # the experiment budget on a program no runtime accepts.
                continue
            found.append(steps)
            if len(found) >= limit:
                return found
    return found


def novelty_against(steps: Sequence[str], genome: CognitiveGenome) -> float:
    """How unlike everything already in the genome this chain is.

    Jaccard distance on the step sets, taking the closest existing
    template. Structural and cheap on purpose: behavioural novelty needs
    the chain to be run, and this runs before anything is spent.
    """
    candidate = set(steps)
    closest = 0.0
    for template in genome.program_templates.values():
        existing = set(template.steps)
        union = candidate | existing
        overlap = len(candidate & existing) / len(union) if union else 0.0
        closest = max(closest, overlap)
    return 1.0 - closest


def propose_candidates(genome: CognitiveGenome, budget_calls: int = 8,
                       max_depth: int = MAX_DEPTH,
                       limit: int = 12) -> List[Candidate]:
    """Chains worth the cost of finding out about, ranked.

    Ranked by novelty per call rather than by novelty: an unfamiliar
    chain that costs six calls per task has to be six times more
    interesting than one that costs one, and the experiment budget is
    what the whole layer is short of.

    Chains the genome already has are dropped, not scored low. A
    duplicate is not a weak candidate -- it is the same thing again, and
    testing it a second time splits the evidence for one mechanism.
    """
    existing = {tuple(t.steps) for t in genome.program_templates.values()}
    out: List[Candidate] = []
    for steps in enumerate_chains(genome, max_depth=max_depth):
        if steps in existing:
            continue
        calls = estimate_calls(steps, genome.operators)
        if calls > budget_calls:
            continue
        novelty = novelty_against(steps, genome)
        if novelty <= 0.0:
            continue
        reasons = [f"новизна {novelty:.2f}", f"{calls} вызовов"]
        out.append(Candidate(steps=steps, estimated_calls=calls,
                             novelty=novelty, reasons=tuple(reasons)))
    out.sort(key=lambda c: (-(c.novelty / max(1, c.estimated_calls)),
                            c.estimated_calls, c.steps))
    return out[:limit]


def reachable_space(genome: CognitiveGenome, max_depth: int = MAX_DEPTH) -> int:
    """How many executable chains this genome can express.

    The number that has to grow when an operator is adopted, or the
    search is exploring a fixed space rather than extending one. Reported
    rather than asserted anywhere: it is evidence about the claim this
    project rests on, and it should be readable at any moment.
    """
    return len(enumerate_chains(genome, max_depth=max_depth, limit=100000))


def as_template(candidate: Candidate, name: str,
                applicability: Sequence[str] = ()) -> ProgramTemplate:
    """The candidate as something the compiler can rank.

    Applicability is empty unless a caller has evidence for it. A chain
    discovered by enumeration has been proven nowhere, and a template
    that claims a domain it was never tested on is exactly the
    overgeneralisation the synthesis layer refuses.
    """
    return ProgramTemplate(
        name=name, steps=tuple(candidate.steps),
        applicability=tuple(applicability),
        description=(f"Найдена перебором: {candidate.describe()}, "
                     f"новизна {candidate.novelty:.2f}. Нигде не доказана."))
