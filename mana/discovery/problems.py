"""
mana.discovery.problems — a problem as an object: the unit of reasoning of
H3 (docs/ГЛУБИНА_D0.md, section 2; D1 in docs/ГЛУБИНА_D1.md).

    Problem   inputs, a target, the points where it is set (the domain), and
              where it came from: the problem that asked it, the operator
              that made it, the results its specification was computed from
    Ledger    one budget for a whole tree, spent under four articles --
              search, building questions, assembling answers, verifying
    Node      one problem of a tree, what was spent on it, and its answer
    Solution  the answer to the root, the tree, the ledger

`solve` interprets a problem. With no catalogue it is the flat search,
field for field (D1a, measured: 230 instances of 230). With D0's catalogue
it is D1b's one procedure, declared before it was written:

    1  the node's flat search, to its own stop, and no more than a share
       `flat_share` of the node's budget (D1b-A: 1; the diagnostic arms:
       1/2 and 1/4). Its answer is R
    2  R exact on the node's points, or the node at depth 5: R is the answer
    3  what is left of the node's budget is split equally among the branches,
       run in the order given, as the operators stand in D0's table:
           остаток           P' = target - R      candidate R + R'
           остаток-          P' = R - target      candidate R - R'
           промахи и случаи  P' = the target on R's misses -> R' (half the
                             share); the condition where exactly one of R, R'
                             is right -> C (the other half); candidates
                             if(C, R, R') and if(C, R', R) -- both
                             orientations, paid for apart (D0, 9)
    4  the node's answer: the shortest description of R and the candidates,
       by the same measure and the same order at a tie. Once it is exact on
       the node's points, the remaining branches are not run

A node whose share cannot cover its own leaves is not made: the flat
search scores every leaf before it looks at its budget, and the tree must
not spend more than it was given. What a branch does not spend is not
passed on.

A domain is a restriction of the data: the points outside it are not in
the subproblem's data, so the flat search and its verdict stay as they are.

Nothing here reads a world.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import description
from . import policy as P
from .language import Evaluator, Program, add, if_, show, size, sub
from .search import Found, vocabulary

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.2"

SEARCH, BUILD, ASSEMBLE, VERIFY = "search", "build", "assemble", "verify"
ARTICLES = (SEARCH, BUILD, ASSEMBLE, VERIFY)

RESIDUAL, RESIDUAL_MINUS, CASES = "остаток", "остаток-", "промахи и случаи"
#: D0's catalogue, in the order its table lists it (docs/ГЛУБИНА_D0.md, 2.3).
BRANCHES = (RESIDUAL, RESIDUAL_MINUS, CASES)
#: The top of the ladder: a node this deep asks no question.
MAX_DEPTH = 5
FLAT = "плоский"


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
    answer)` makes a candidate for the problem that asked. D1b's branches
    are written into the procedure below; this is the shape D3 will fill."""
    name: str
    ask: Callable[[Problem, Program], Optional[Problem]]
    assemble: Callable[[Program, Program], Program]


@dataclass
class Node:
    index: int
    problem: Problem
    found: Optional[Found]
    program: Optional[Program]
    cost: Dict[str, int]
    #: Which answer the node kept: FLAT, or the branch whose candidate won.
    chosen: str = FLAT
    #: The children whose answers the kept candidate was assembled from.
    used: Tuple[int, ...] = ()
    children: List[int] = field(default_factory=list)
    budget: int = 0


@dataclass
class Solution:
    program: Program
    #: The root's flat search, as the flat search returned it.
    found: Found
    nodes: List[Node]
    ledger: Ledger

    def used_depth(self) -> int:
        """The longest chain of the dependency graph over every node made,
        counted in nodes. Reported, never claimed (docs/ГЛУБИНА_D0.md, 3)."""
        depth: Dict[int, int] = {}
        for node in self.nodes:
            before = [depth[i] for i in node.problem.depends_on if i in depth]
            depth[node.index] = 1 + max(before, default=0)
        return max(depth.values(), default=0)

    def answer_nodes(self) -> List[int]:
        """The nodes the root's answer was assembled from, the root first."""
        out, todo = [], [0]
        while todo:
            i = todo.pop()
            out.append(i)
            todo.extend(self.nodes[i].used)
        return sorted(out)

    def subtree_cost(self, index: int) -> int:
        node = self.nodes[index]
        return sum(node.cost.values()) + sum(self.subtree_cost(c) for c in node.children)


def _key(p: Program, columns: Dict[str, np.ndarray], target: np.ndarray,
         evaluator: Evaluator) -> Tuple[float, int, int, str]:
    """The verdict's order: bits, size, wrong points, text."""
    out = evaluator(p)
    alphabet = description.alphabet_of(target)
    known = description.membership(target)
    bits = (description.program_bits(p, len(columns))
            + description.error_bits(out, target, alphabet, known))
    return (bits, size(p), int(np.count_nonzero(out != target)), show(p))


class _Procedure:
    def __init__(self, policy: P.SearchPolicy, ledger: Ledger, order: Sequence[str],
                 flat_share: float, profile: bool, trace: bool) -> None:
        self.policy = policy
        self.ledger = ledger
        self.order = tuple(order)
        self.flat_share = float(flat_share)
        self.profile = profile
        self.trace = trace
        self.nodes: List[Node] = []

    def solve(self, problem: Problem, budget: int, depth: int) -> Optional[Node]:
        columns, target = problem.data()
        if len(target) == 0:
            return None
        leaves, _ = vocabulary(columns, target)
        cap = min(int(budget * self.flat_share), budget, self.ledger.left())
        if cap <= len(leaves):
            return None
        node = Node(len(self.nodes), problem, None, None, {a: 0 for a in ARTICLES},
                    budget=int(budget))
        self.nodes.append(node)
        if problem.parent is not None:
            self.nodes[problem.parent].children.append(node.index)
        found = P.run(P.with_budget(self.policy, cap), columns, target,
                      profile=self.profile and problem.parent is None, trace=self.trace)
        self._charge(node, SEARCH, found.evaluations)
        node.found, node.program = found, found.program
        evaluator = Evaluator(columns)
        r = found.program
        if np.array_equal(evaluator(r), target) or depth >= MAX_DEPTH:
            return node
        share = (budget - found.evaluations) // len(self.order)
        # A share that cannot cover the node's leaves can make no subproblem:
        # no question is built. Every charge below is made only if the tree's
        # budget can pay it.
        if share <= len(leaves) or not self._afford(1):
            return node
        best = (_key(r, columns, target, evaluator), r, FLAT, ())
        values = evaluator(r)
        self._charge(node, BUILD, 1)
        for branch in self.order:
            made = self._branch(branch, node, columns, target, values, share, depth)
            if made is None:
                continue
            forms, used = made
            short = False
            for candidate in forms:
                if not self._afford(2):
                    short = True
                    break
                self._charge(node, ASSEMBLE, 1)
                self._charge(node, VERIFY, 1)
                key = _key(candidate, columns, target, evaluator)
                if key < best[0]:
                    best = (key, candidate, branch, used)
            if short or best[0][2] == 0:
                break
        _, node.program, node.chosen, node.used = best
        return node

    def _charge(self, node: Node, article: str, evaluations: int) -> None:
        self.ledger.charge(article, evaluations)
        node.cost[article] += int(evaluations)

    def _afford(self, evaluations: int) -> bool:
        return self.ledger.left() >= evaluations

    def _branch(self, branch: str, node: Node, columns, target, values, share, depth):
        r = node.found.program
        everywhere = np.ones(len(target), dtype=bool)
        if not self._afford(1):
            return None
        if branch in (RESIDUAL, RESIDUAL_MINUS):
            asked = target - values if branch == RESIDUAL else values - target
            self._charge(node, BUILD, 1)
            child = self.solve(Problem(columns, asked, everywhere, node.index, branch,
                                       (node.index,)), share, depth + 1)
            if child is None:
                return None
            answer = child.program
            made = add(r, answer) if branch == RESIDUAL else sub(r, answer)
            return (made,), (child.index,)
        if branch == CASES:  # noqa: C901 -- both orientations, D0, 9
            misses = values != target
            self._charge(node, BUILD, 1)
            half = share // 2
            rest = self.solve(Problem(columns, target, misses, node.index, "промахи",
                                      (node.index,)), half, depth + 1)
            if rest is None or not self._afford(1):
                return None
            theirs = Evaluator(columns)(rest.program)
            self._charge(node, BUILD, 1)
            ours_right, theirs_right = values == target, theirs == target
            told = ours_right != theirs_right
            if not told.any():
                return None
            condition = self.solve(Problem(columns, ours_right.astype(np.int64), told,
                                           node.index, "случаи", (node.index, rest.index)),
                                   share - half, depth + 1)
            if condition is None:
                return None
            # Both orientations: on a two-valued question the search may
            # return the condition's complement at no cost in bits, and
            # swapping the branches is how the answer's language says "not"
            # (D0, 9).
            return ((if_(condition.program, r, rest.program),
                     if_(condition.program, rest.program, r)),
                    (rest.index, condition.index))
        raise ValueError(f"no such branch: {branch!r}")


def solve(problem: Problem, policy: P.SearchPolicy, budget: int,
          catalogue: Sequence[str] = (), profile: bool = False, trace: bool = False,
          flat_share: float = 1.0) -> Solution:
    """The answer to a problem, within one budget for everything spent on it.
    `catalogue`: D0's branches to use, in the order they are to be run."""
    ledger = Ledger(int(budget))
    if not catalogue:
        columns, target = problem.data()
        found = P.run(P.with_budget(policy, ledger.left()), columns, target,
                      profile=profile, trace=trace)
        ledger.charge(SEARCH, found.evaluations)
        root = Node(0, problem, found, found.program, {SEARCH: found.evaluations},
                    budget=int(budget))
        return Solution(found.program, found, [root], ledger)
    unknown = [b for b in catalogue if b not in BRANCHES]
    if unknown or len(set(catalogue)) != len(catalogue):
        raise ValueError(f"not D0's catalogue: {list(catalogue)!r}")
    procedure = _Procedure(policy, ledger, catalogue, flat_share, profile, trace)
    root = procedure.solve(problem, int(budget), 1)
    if root is None:
        raise ValueError("the budget does not cover the root's leaves")
    return Solution(root.program, root.found, procedure.nodes, ledger)


def effective_depth(solution: Solution, policy: P.SearchPolicy) -> Tuple[int, int]:
    """Effective depth of a solution (docs/ГЛУБИНА_D0.md, 3) and what measuring
    it cost -- a measurement, not charged to the solution's budget.

    Over the nodes the answer was assembled from: an inner node whose problem
    the flat search solves exactly with the budget its subtree spent is
    collapsed -- that level was not needed. The depth is then the longest
    chain of the dependency graph over what is left."""
    spent = 0
    keep = set(solution.answer_nodes())
    for index in sorted(keep):
        node = solution.nodes[index]
        if index not in keep or not node.used:
            continue
        columns, target = node.problem.data()
        found = P.run(P.with_budget(policy, max(1, solution.subtree_cost(index))),
                      columns, target)
        spent += found.evaluations
        if np.array_equal(Evaluator(columns)(found.program), target):
            drop, todo = set(), list(node.used)
            while todo:
                i = todo.pop()
                drop.add(i)
                todo.extend(solution.nodes[i].used)
            keep -= drop
    depth: Dict[int, int] = {}
    for index in sorted(keep):
        before = [depth[i] for i in solution.nodes[index].problem.depends_on if i in depth]
        depth[index] = 1 + max(before, default=0)
    return max(depth.values(), default=0), spent
