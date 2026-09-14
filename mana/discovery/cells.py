"""
mana.discovery.cells — understanding a world one region at a time.

Why
---
Step 5.0 found the search's limit structural: past 400k programs every
miss of the narrow beam is a local end. But many rules that are hard as a
whole are easy piece by piece. T2's |y - z| + x is x + y - z where y >= z
and x + z - y where y < z -- five nodes each, no distance anywhere; W0 is
literally two pieces, z and y. So instead of one search for the whole
world: small regions, a model in each, and a global hypothesis made of
them.

The cycle
---------
    cell        a region of situations, written as conditions of the
                language itself -- the same comparisons the search uses
    local model what the ordinary search finds on the cell's points alone
    boundary    a cell is split where its model breaks -- and only in the
                currency everything else is judged in: the condition, plus
                a model on each side, plus what those still get wrong, must
                be shorter than the one model. So a boundary appears where
                there is structure, and noise does not earn one
    global      the tree of boundaries with the local models in its leaves,
                stitched with if: a program of the same language, judged
                against unseen states like any other. And the simplest
                generalisation: where one leaf's program describes all the
                data shorter than the whole tree, it IS the global model

Two ways of cutting
-------------------
    adaptive    start with the whole world, split only where the model
                breaks (experiment B)
    grid        a grid laid over the variables before looking (experiment
                A, the control)

What is fixed: the language of boundaries (the search's own conditions)
and the currency (bits). What is not given: where the boundaries are, how
many, how deep.

Cost is counted honestly: every program any local search evaluated,
including the searches of splits that were refused. Parallelism is a
separate figure -- the time along the longest path of the tree, as if the
cells beside each other ran at once -- and never folded into the count.

Measured 2026-09-14 (scripts/run_cells.py, the 7 questions of step 4 x 10
seeds; solved at N = the rule itself, spending no more than N programs in
all)
----------------------------------------------------------------------
    budget        10k  25k  50k  100k  200k  400k  800k  1.2M   2M
    zero law      29   29   29   39    44    63    63    84    94
    B@25k          0   14   14   14    39    44    44    44    44
    B@50k          0    0   14   14    14    49    54    54    54
    B@100k         0    0    0   24    24    26    69    70    70
    A2 (8 cells)   0    0    0    0    29    29    29    29    29
    A3 (27)        0    0    0    0     0     0     0     0     0
    A10x10 (100)   0    0    0    0     0     0     0     0     0

Adaptive cells pass the narrow beam's plateau -- 70% against 63%, the
structural limit of step 5.0 -- but earn their place under the zero law at
one budget only (69 against 63 at 800k), and lose to the wide beam from
1.2M on (70 against 84 and 94). They do not earn it.

Where the first boundary was the right one, the pieces are the world's
own, found without being told: T2 on 5 seeds of 10 as
if(z < y, (x + y) - z, (x + z) - y), T3 the same way, T4 on 4 seeds with
two boundaries. Where it was not, the first cut went where the
approximate model's errors were -- 0 < z, y == z -- not where the
structure changes; the tree then grew to 5 leaves, exact on what they saw
(0.84 of leaves) and wrong on unseen states (0.17). Noise earned no
boundary on any seed of W3, but proving that cost five searches: the
worlds the plain search solves at 10k cost B 25k to 500k.

Prediction against result: W0 and T1 solved -- yes (T1 10/10 against the
narrow beam's 8/10, but not cheaper); T2, T3, W4 solved and cheaper than
the zero law -- no, 5/10 at up to 1.5M; W3 without splits -- yes; T4
limited -- yes, 4/10 as before.

Why the pieces were not cheap: on the points of T2's own cell y >= z the
plain search first saw (y + x) - z, five nodes, at the 378 798th program,
by way of x -> if(y == z, x, 9) -> if(y == z, x, y + x) ->
if(y == z, x, (y + x) - z). An exact-match error gives x + y no credit
over x, so arithmetic is assembled around conditions, and 5 814 of the
5 889 neighbours of a single leaf are conditions wrapped around it. A cell
makes the rule simpler; it does not make the search for it cheaper. The
limit is inside every cell.

The grid, control A, is worse than no cells at all. Laid down before
looking, it cuts the data into cells of a few points each (A3: about 7)
where a local model is anything; the stitched tree is so long that bits
rightly prefer one cell's wrong model, z, over it -- and lose even W0.
A2 solves W0 and W3 only because one of its cells happened to hold the
boundary x = 5 with enough points around it. Parallel cells would buy B
nothing here (1.0 - 1.2: its trees are nearly a chain); they buy the grid
5 - 40x of a result that is wrong.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import log2
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import description, search
from .language import LESS, Evaluator, Program, cmp, const, get, if_

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Fewest points a cell may have: fewer, and a constant explains anything.
MIN_POINTS = 10
MAX_DEPTH = 4
#: Candidate boundaries actually searched at a cell, in order of how well
#: they separate the model's errors from its right answers.
SPLIT_TRIALS = 2
LOCAL_BUDGET = 25000


@dataclass
class Cell:
    points: int
    found: Optional[search.Found] = None
    #: The outcome given to a cell no point fell into.
    fallback: int = 0
    condition: Optional[Program] = None
    yes: Optional["Cell"] = None           # where the condition holds
    no: Optional["Cell"] = None
    #: Work done at this node: its splits' searches, refused ones included.
    seconds: float = 0.0

    @property
    def split(self) -> bool:
        return self.condition is not None

    def program(self) -> Program:
        if self.split:
            return if_(self.condition, self.yes.program(), self.no.program())
        return self.found.program if self.found is not None else const(self.fallback)

    def bits(self, variables: int) -> float:
        """One bit for leaf-or-split, then the condition and both sides, or
        the leaf's program and the errors it leaves in its cell."""
        if self.split:
            return (1.0 + description.program_bits(self.condition, variables)
                    + self.yes.bits(variables) + self.no.bits(variables))
        if self.found is None:
            return 1.0 + description.program_bits(const(self.fallback), variables)
        return 1.0 + self.found.bits

    def leaves(self) -> List["Cell"]:
        return self.yes.leaves() + self.no.leaves() if self.split else [self]

    def depth(self) -> int:
        return 1 + max(self.yes.depth(), self.no.depth()) if self.split else 0

    def path_seconds(self) -> float:
        """Time along the longest path: siblings as if they ran at once."""
        if not self.split:
            return self.seconds
        return self.seconds + max(self.yes.path_seconds(), self.no.path_seconds())


@dataclass
class Grown:
    tree: Cell
    #: The global hypothesis: the stitched tree, or one leaf's program
    #: where it describes everything shorter.
    program: Program
    bits: float
    #: "tree" or "one cell's model".
    source: str
    evaluations: int
    searches: int
    seconds: float
    #: Total time over time along the longest path: what running the
    #: cells beside each other at once could buy, not what it did.
    parallel: float = field(default=1.0)


def _entropy(share: float) -> float:
    if share <= 0.0 or share >= 1.0:
        return 0.0
    return -(share * log2(share) + (1 - share) * log2(1 - share))


def _ranked(conditions: Sequence[Program], columns: Dict[str, np.ndarray],
            wrong: np.ndarray, min_points: int) -> List[Tuple[Program, np.ndarray]]:
    """Conditions by how much they tell about where the model is wrong.
    One per distinct partition: two texts that cut the same way are one
    boundary."""
    evaluator = Evaluator(columns)
    n = len(wrong)
    base = _entropy(float(wrong.mean()))
    rows, seen = [], set()
    for condition in conditions:
        holds = evaluator(condition) != 0
        k = int(holds.sum())
        if k < min_points or n - k < min_points:
            continue
        key = holds.tobytes()
        if key in seen:
            continue
        seen.add(key)
        gain = (base - (k / n) * _entropy(float(wrong[holds].mean()))
                - ((n - k) / n) * _entropy(float(wrong[~holds].mean())))
        rows.append((-gain, len(rows), condition, holds))
    rows.sort(key=lambda row: (row[0], row[1]))
    return [(condition, holds) for _, _, condition, holds in rows]


class _Grower:
    def __init__(self, columns: Dict[str, Sequence[int]], outcomes: Sequence[int],
                 local_budget: int, min_points: int) -> None:
        self.columns = {name: np.asarray(values) for name, values in columns.items()}
        self.outcomes = np.asarray(outcomes, dtype=np.int64)
        self.local_budget = int(local_budget)
        self.min_points = int(min_points)
        self.variables = len(self.columns)
        self.evaluations = 0
        self.searches = 0
        counts = np.bincount(self.outcomes - self.outcomes.min())
        self.majority = int(np.argmax(counts) + self.outcomes.min())

    def local(self, mask: np.ndarray) -> Optional[search.Found]:
        if not mask.any():
            return None
        found = search.search({name: values[mask] for name, values in self.columns.items()},
                              self.outcomes[mask], budget=self.local_budget)
        self.evaluations += found.evaluations
        self.searches += 1
        return found

    def leaf(self, mask: np.ndarray, found: Optional[search.Found] = None) -> Cell:
        found = found if found is not None else self.local(mask)
        return Cell(points=int(mask.sum()), found=found, fallback=self.majority)

    def sides(self, mask: np.ndarray, holds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        index = np.nonzero(mask)[0]
        yes, no = np.zeros_like(mask), np.zeros_like(mask)
        yes[index[holds]] = True
        no[index[~holds]] = True
        return yes, no

    def finish(self, tree: Cell, started: float) -> Grown:
        stitched = tree.program()
        best = (tree.bits(self.variables), stitched, "tree")
        evaluator = Evaluator(self.columns)
        alphabet = description.alphabet_of(self.outcomes)
        tried = set()
        for cell in tree.leaves():
            if cell.found is None or cell.found.program in tried:
                continue
            tried.add(cell.found.program)
            program = cell.found.program
            bits = (1.0 + description.program_bits(program, self.variables)
                    + description.error_bits(evaluator(program), self.outcomes, alphabet))
            if bits < best[0]:
                best = (bits, program, "one cell's model")
        total = time.time() - started
        path = max(tree.path_seconds(), 1e-9)
        return Grown(tree, best[1], best[0], best[2], self.evaluations, self.searches,
                     total, parallel=total / path)


def adaptive(columns: Dict[str, Sequence[int]], outcomes: Sequence[int],
             local_budget: int = LOCAL_BUDGET, min_points: int = MIN_POINTS,
             max_depth: int = MAX_DEPTH, trials: int = SPLIT_TRIALS) -> Grown:
    """Experiment B: one cell, split only where its model breaks."""
    started = time.time()
    grower = _Grower(columns, outcomes, local_budget, min_points)

    def grow(mask: np.ndarray, depth: int, found: Optional[search.Found]) -> Cell:
        began = time.time()
        cell = grower.leaf(mask, found)
        if (cell.found is None or cell.found.train_errors == 0 or depth >= max_depth
                or cell.points < 2 * min_points):
            cell.seconds = time.time() - began
            return cell
        local_columns = {name: values[mask] for name, values in grower.columns.items()}
        local_outcomes = grower.outcomes[mask]
        wrong = search.predict(cell.found.program, local_columns) != local_outcomes
        _, conditions = search.vocabulary(local_columns, local_outcomes)
        best = None
        for condition, holds in _ranked(conditions, local_columns, wrong, min_points)[:trials]:
            yes_mask, no_mask = grower.sides(mask, holds)
            yes, no = grower.leaf(yes_mask), grower.leaf(no_mask)
            bits = (1.0 + description.program_bits(condition, grower.variables)
                    + yes.bits(grower.variables) + no.bits(grower.variables))
            if best is None or bits < best[0]:
                best = (bits, condition, yes_mask, no_mask, yes.found, no.found)
        cell.seconds = time.time() - began
        if best is None or best[0] >= cell.bits(grower.variables):
            return cell
        _, condition, yes_mask, no_mask, yes_found, no_found = best
        cell.condition = condition
        cell.yes = grow(yes_mask, depth + 1, yes_found)
        cell.no = grow(no_mask, depth + 1, no_found)
        return cell

    everything = np.ones(len(grower.outcomes), dtype=bool)
    return grower.finish(grow(everything, 0, None), started)


def grid(columns: Dict[str, Sequence[int]], outcomes: Sequence[int], bins: int,
         over: Optional[Sequence[str]] = None,
         local_budget: int = LOCAL_BUDGET) -> Grown:
    """Experiment A: equal-width bins on each named variable, laid down
    before looking; a local search in every non-empty cell."""
    started = time.time()
    grower = _Grower(columns, outcomes, local_budget, 0)
    names = list(over) if over else sorted(grower.columns)
    edges: Dict[str, List[int]] = {}
    for name in names:
        low, high = int(grower.columns[name].min()), int(grower.columns[name].max())
        width = (high - low + 1) / float(bins)
        edges[name] = sorted({int(np.ceil(low + width * j)) for j in range(1, bins)}
                             - {low})

    def cut(mask: np.ndarray, level: int) -> Cell:
        if level == len(names):
            began = time.time()
            cell = grower.leaf(mask)
            cell.seconds = time.time() - began
            return cell
        return chain(mask, level, 0)

    def chain(mask: np.ndarray, level: int, j: int) -> Cell:
        name = names[level]
        if j == len(edges[name]):
            return cut(mask, level + 1)
        condition = cmp(LESS, get(name), const(edges[name][j]))
        holds = grower.columns[name] < edges[name][j]
        return Cell(points=int(mask.sum()), condition=condition,
                    yes=cut(mask & holds, level + 1),
                    no=chain(mask & ~holds, level, j + 1),
                    fallback=grower.majority)

    everything = np.ones(len(grower.outcomes), dtype=bool)
    return grower.finish(chain(everything, 0, 0), started)
