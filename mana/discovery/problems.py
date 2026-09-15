"""
mana.discovery.problems — a problem as an object: the unit of reasoning of
H3 (docs/ГЛУБИНА_D0.md, section 2; D1a in docs/ГЛУБИНА_D1.md).

    Problem   inputs, a target, the points where it is set (the domain), and
              where it came from: the problem that asked it, the operator
              that made it, the results its specification was computed from
    Ledger    one budget for a whole tree, spent under four articles --
              search, building questions, assembling answers, verifying
    Node      one problem of a tree, what was spent on it, and its answer
    Solution  the answer to the root, the tree, the ledger

`solve` interprets a problem. D1a asks one thing only: with no question
operators the interpreter is the flat search -- the same program, bits,
evaluations, curve, history and stopping, field for field. Operators come
in D1b; until then a catalogue that is not empty is refused, not ignored.

A domain is a restriction of the data: the points outside it are not in
the subproblem's data, so the flat search and its verdict stay as they are.

Nothing here reads a world.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import policy as P
from .language import Program
from .search import Found

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

SEARCH, BUILD, ASSEMBLE, VERIFY = "search", "build", "assemble", "verify"
ARTICLES = (SEARCH, BUILD, ASSEMBLE, VERIFY)


@dataclass(eq=False)
class Problem:
    inputs: Dict[str, np.ndarray]
    target: np.ndarray
    domain: np.ndarray
    #: The problem that asked this one, by its node's index; None for a root.
    parent: Optional[int] = None
    #: The operator that made it; "" for a root.
    operator: str = ""
    #: The nodes whose results its specification was computed from -- the
    #: edges of the dependency graph that effective depth is measured on.
    depends_on: Tuple[int, ...] = ()

    @classmethod
    def whole(cls, columns: Dict[str, Sequence[int]], outcomes: Sequence[int]) -> "Problem":
        """A root: every point, the data as given."""
        target = np.asarray(outcomes, dtype=np.int64)
        inputs = {name: np.asarray(values, dtype=np.int64) for name, values in columns.items()}
        return cls(inputs, target, np.ones(len(target), dtype=bool))

    def data(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """What the flat search is handed: the domain's points, in order."""
        rows = self.domain
        return {name: values[rows] for name, values in self.inputs.items()}, self.target[rows]


@dataclass
class Ledger:
    """One budget for a tree, and what each article has spent of it."""
    budget: int
    spent: Dict[str, int] = field(default_factory=lambda: {a: 0 for a in ARTICLES})

    def left(self) -> int:
        return self.budget - sum(self.spent.values())

    def charge(self, article: str, evaluations: int) -> None:
        if article not in self.spent:
            raise ValueError(f"no such article: {article!r}")
        self.spent[article] += int(evaluations)


@dataclass(frozen=True)
class Operator:
    """A question operator and its assembly (docs/ГЛУБИНА_D0.md, 2.2):
    `ask(problem, result)` makes a new problem or None; `assemble(result,
    answer)` makes a candidate for the problem that asked. The shape only:
    no operator exists before D1b."""
    name: str
    ask: Callable[[Problem, Program], Optional[Problem]]
    assemble: Callable[[Program, Program], Program]


@dataclass
class Node:
    index: int
    problem: Problem
    found: Found
    program: Program
    cost: Dict[str, int]


@dataclass
class Solution:
    program: Program
    #: The root's flat search, as the flat search returned it.
    found: Found
    nodes: List[Node]
    ledger: Ledger

    def used_depth(self) -> int:
        """The longest chain of the dependency graph, counted in nodes. It is
        reported, never claimed (docs/ГЛУБИНА_D0.md, 3)."""
        depth: Dict[int, int] = {}
        for node in self.nodes:
            before = [depth[i] for i in node.problem.depends_on if i in depth]
            depth[node.index] = 1 + max(before, default=0)
        return max(depth.values(), default=0)


def solve(problem: Problem, policy: P.SearchPolicy, budget: int,
          catalogue: Sequence[Operator] = (), profile: bool = False,
          trace: bool = False) -> Solution:
    """The answer to a problem, within one budget for everything spent on it."""
    if catalogue:
        raise NotImplementedError("операторы вопросов появляются в D1b")
    ledger = Ledger(int(budget))
    columns, target = problem.data()
    found = P.run(P.with_budget(policy, ledger.left()), columns, target,
                  profile=profile, trace=trace)
    ledger.charge(SEARCH, found.evaluations)
    root = Node(0, problem, found, found.program, {SEARCH: found.evaluations})
    return Solution(found.program, found, [root], ledger)
