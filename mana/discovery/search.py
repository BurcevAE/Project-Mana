"""
mana.discovery.search — changing a program, one syntactic edit at a time.

A beam of the shortest descriptions found so far; every program in it is
edited in every way the edits allow; the shortest distinct results form
the next beam. Stops when the best description has not shortened for
`PATIENCE` rounds, or the evaluation budget is spent.

The edits know nothing about meaning:

    replace a node by a leaf    (a variable or a constant from the data)
    swap an operator            add <-> sub, < <-> ==, arguments swapped
    remove a condition          if(c, a, b) -> a, or -> b
    add a condition             n -> if(c, n, leaf) or if(c, leaf, n)
    combine with a leaf         n -> n + leaf, n - leaf, leaf - n

A condition is a comparison between a variable and a constant or another
variable -- the smallest one the language can write. Anything larger is
reached by editing inside it. Constants come from the values seen in the
data, never from a list somebody wrote for this world.

Nothing here reads a world's truth: the search is handed columns and
outcomes and returns a program. A test asserts it does not import
`worlds`, because a learner that peeks scores perfectly and means nothing.
"""
from __future__ import annotations

import itertools
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import description
from .language import (ADD, CMP, EQUAL, IF, LESS, SUB, Evaluator, Primitive,
                       Program, add, cmp, const, free_variables, get, if_, nodes,
                       prim, replace, show, size, sub)

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.3"

BEAM = 4
MAX_SIZE = 15
PATIENCE = 2
MAX_ROUNDS = 30
BUDGET = 400000

#: Most constants taken from the data. A world with many distinct values
#: keeps the most frequent; the search never invents a number.
MAX_CONSTANTS = 16

#: Why a search stopped. LIMIT: the evaluation budget or the round limit
#: ran out while the neighbourhood of the beam was not yet exhausted --
#: more budget might have found more. EXHAUSTED: every neighbour of the
#: beam was tried for PATIENCE rounds and none shortened the description
#: -- a local end. Neither says anything about the rest of the space; that
#: is what `ceiling` is for.
SEARCH_LIMIT = "SEARCH_LIMIT"
SEARCH_EXHAUSTED = "SEARCH_EXHAUSTED"

#: What it found, by what it could see: exact on the training points, or
#: not.
FOUND_EXACT = "FOUND_EXACT"
FOUND_APPROX = "FOUND_APPROX"


@dataclass
class Found:
    program: Program
    bits: float
    program_bits: float
    error_bits: float
    evaluations: int
    rounds: int
    #: (round, bits, program) each time the best description shortened.
    history: List[Tuple[int, float, str]] = field(default_factory=list)
    #: Programs evaluated before the final best was first seen: the cost
    #: of finding it, as opposed to the cost of making sure.
    found_at: int = 0
    #: Why it stopped -- SEARCH_LIMIT or SEARCH_EXHAUSTED, see below.
    termination: str = ""
    seconds: float = 0.0
    #: Training points the program gets wrong.
    train_errors: int = 0

    @property
    def status(self) -> str:
        """FOUND_EXACT: no error on what it saw -- which is not the same
        claim as being the rule, and says nothing about unseen states."""
        return FOUND_EXACT if self.train_errors == 0 else FOUND_APPROX

    def describe(self) -> str:
        return (f"{show(self.program)}  — {self.bits:.1f} бит "
                f"(программа {self.program_bits:.1f}, ошибки {self.error_bits:.1f}); "
                f"перебрано {self.evaluations}, раундов {self.rounds}")


def vocabulary(columns: Dict[str, Sequence[int]],
               outcomes: Sequence[int],
               library: Optional[Dict[str, Primitive]] = None
               ) -> Tuple[List[Program], List[Program]]:
    """Leaves and conditions, from nothing but the data -- and, once the
    language has grown, every primitive called on the variables. Only on
    variables: constants inside a call are reached by editing it."""
    variables = sorted(columns)
    counts: Counter = Counter()
    for values in list(columns.values()) + [outcomes]:
        counts.update(int(v) for v in values)
    counts.update({0: 0, 1: 0})
    values = sorted(v for v, _ in counts.most_common(MAX_CONSTANTS))
    for fixed in (0, 1):
        if fixed not in values:
            values.append(fixed)
    values = sorted(set(values))
    leaves = [get(v) for v in variables] + [const(c) for c in values]
    for name, word in sorted((library or {}).items()):
        # A word that names a variable this world does not have means
        # nothing here. One learned as |#0 - y| crashed the search on a
        # world of p, q and r before this was checked.
        if not free_variables(word.template) <= set(variables):
            continue
        for args in itertools.product([get(v) for v in variables], repeat=word.arity):
            leaves.append(prim(name, *args))
    conditions: List[Program] = []
    for v in variables:
        for c in values:
            conditions.append(cmp(LESS, get(v), const(c)))
            conditions.append(cmp(LESS, const(c), get(v)))
            conditions.append(cmp(EQUAL, get(v), const(c)))
    for v in variables:
        for w in variables:
            if v != w:
                conditions.append(cmp(LESS, get(v), get(w)))
                if v < w:
                    conditions.append(cmp(EQUAL, get(v), get(w)))
    return leaves, conditions


def neighbours(p: Program, leaves: Sequence[Program],
               conditions: Sequence[Program], max_size: int) -> Iterator[Program]:
    """Every program one syntactic edit away, no larger than `max_size`."""
    whole = size(p)
    for path, node in nodes(p):
        grown = whole - size(node)                 # the rest of the program
        kind = node[0]
        for leaf in leaves:
            if leaf != node:
                yield replace(p, path, leaf)
        if kind in (ADD, SUB):
            yield replace(p, path, (SUB if kind == ADD else ADD, node[1], node[2]))
            yield replace(p, path, (kind, node[2], node[1]))
        elif kind == CMP:
            other = EQUAL if node[1] == LESS else LESS
            yield replace(p, path, (CMP, other, node[2], node[3]))
            yield replace(p, path, (CMP, node[1], node[3], node[2]))
        elif kind == IF:
            yield replace(p, path, node[2])
            yield replace(p, path, node[3])
            yield replace(p, path, (IF, node[1], node[3], node[2]))
        if grown + size(node) + 4 <= max_size:        # if(cmp(a, b), n, leaf)
            for condition in conditions:
                for leaf in leaves:
                    yield replace(p, path, if_(condition, node, leaf))
                    yield replace(p, path, if_(condition, leaf, node))
        if grown + size(node) + 2 <= max_size:
            for leaf in leaves:
                yield replace(p, path, add(node, leaf))
                yield replace(p, path, sub(node, leaf))
                yield replace(p, path, sub(leaf, node))


def search(columns: Dict[str, Sequence[int]], outcomes: Sequence[int],
           beam_width: int = BEAM, max_size: int = MAX_SIZE,
           patience: int = PATIENCE, max_rounds: int = MAX_ROUNDS,
           budget: int = BUDGET,
           library: Optional[Dict[str, Primitive]] = None) -> Found:
    """The shortest description of the outcomes this search can reach."""
    started = time.time()
    actual = np.asarray(outcomes, dtype=np.int64)
    library = dict(library or {})
    evaluator = Evaluator(columns, library)
    variables = len(columns)
    alphabet = description.alphabet_of(actual)
    known = description.membership(actual)
    leaves, conditions = vocabulary(columns, actual, library)
    seen: Dict[Program, Tuple[float, float, float]] = {}
    first_seen: Dict[Program, int] = {}

    def score(p: Program) -> Tuple[float, float, float]:
        first_seen.setdefault(p, len(first_seen) + 1)
        program_part = description.program_bits(p, variables, len(library))
        error_part = description.error_bits(evaluator(p), actual, alphabet, known)
        return (program_part + error_part, program_part, error_part)

    def rank(p: Program) -> Tuple[float, int, str]:
        # The text last: equal descriptions of equal size are the same rule
        # written two ways, and without it the one kept depended on set
        # order -- the same seed gave if(x < 6, z, y) in one process and
        # if(5 < x, y, z) in another.
        return (seen[p][0], size(p), show(p))

    for leaf in leaves:
        seen[leaf] = score(leaf)
    beam = sorted(seen, key=rank)[:beam_width]
    best = beam[0]
    history = [(0, seen[best][0], show(best))]
    stalled = 0
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        fresh: List[Program] = []
        for p in beam:
            for q in neighbours(p, leaves, conditions, max_size):
                if q in seen:
                    continue
                if len(seen) >= budget:
                    break
                seen[q] = score(q)
                fresh.append(q)
        beam = sorted(set(fresh) | set(beam), key=rank)[:beam_width]
        if seen[beam[0]][0] < seen[best][0] - 1e-9:
            best = beam[0]
            stalled = 0
            history.append((rounds, seen[best][0], show(best)))
        else:
            stalled += 1
        if stalled >= patience or len(seen) >= budget:
            break
    bits, program_part, error_part = seen[best]
    exhausted = stalled >= patience and len(seen) < budget
    return Found(program=best, bits=bits, program_bits=program_part,
                 error_bits=error_part, evaluations=len(seen), rounds=rounds,
                 history=history, found_at=first_seen.get(best, len(seen)),
                 termination=SEARCH_EXHAUSTED if exhausted else SEARCH_LIMIT,
                 seconds=time.time() - started,
                 train_errors=int(np.count_nonzero(evaluator(best) != actual)))


def predict(p: Program, columns: Dict[str, Sequence[int]],
            library: Optional[Dict[str, Primitive]] = None) -> np.ndarray:
    return Evaluator(columns, library)(p)
