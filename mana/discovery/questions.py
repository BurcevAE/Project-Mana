"""
mana.discovery.questions — the research question: ResearchQuestion -> X ->
DerivedQuestion, with X untyped (docs/ГЛУБИНА_ВОПРОС_ИССЛЕДОВАНИЯ.md).

A solving question is a Problem of D1; its own step is the flat search. A
research question poses a plan on it -- three pieces of data:

    experiment  an expression whose value is the observation X
    entries     derived questions, each an expression giving
                ("question", target, domain[, columns]) from X and the
                answers of the entries before it, and an expression for its
                budget
    rebuild     a template in the answer's language whose holes are X and
                the entries' answers

The interpreter evaluates expressions, solves derived questions by the same
procedure, fills templates, and keeps a record of what was asked and what
it cost. It does not know what any plan means: plans are data, in
mana/discovery/plans.py, and a test reads this file for the name of any
kind of question. Values have forms, not meanings -- a program, a vector
over the points, a mask, a number, a list -- and the interpreter does not
branch on which.

The node's procedure is D1b's, with the plans as its parameter: the node's
own flat search to its own stop, at most `flat_share` of the node's budget;
exact, or at depth 5, or no plan: done; what is left divided equally among the
plans; the node's answer the shortest description of its own answer and the
candidates, the remaining plans skipped once it is exact.

What is charged, one evaluation each: the node's own answer observed on its
data, when the node first asks; every derived question built; every program
an experiment evaluates on the node's data for the first time (all under
"build"); assembling a candidate; verifying it. The flat searches, and every
search an experiment runs, under "search". A record keeps history -- what
was asked, what it cost, what came of it -- and no field for its worth.

Nothing here reads a world.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace as _replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import description
from . import policy as P
from .language import GET, Evaluator, children, rebuild, show
from .problems import (ARTICLES, ASSEMBLE, BUILD, FLAT, MAX_DEPTH, SEARCH, VERIFY, Ledger, Node,
                       Problem, Solution, _key)
from .search import vocabulary

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: A hole of a rebuild template: ("?", name) or ("?", name, item).
HOLE = "?"


@dataclass(frozen=True)
class Entry:
    """A derived question of a plan: its name, the expression of its question,
    and the expression of its budget."""
    name: str
    question: tuple
    budget: tuple


@dataclass(frozen=True)
class Plan:
    name: str
    experiment: tuple
    entries: Tuple[Entry, ...]
    rebuild: tuple


@dataclass
class ResearchRecord:
    """What was asked, what it cost, what came of it -- history, no worth."""
    parent: int
    plan: str
    experiment: tuple
    observation: str = ""
    derived: Tuple[int, ...] = ()
    cost: Dict[str, int] = field(default_factory=dict)
    candidate: str = ""
    kept: bool = False
    stopped: str = ""


@dataclass
class Reasoning(Solution):
    records: List[ResearchRecord] = field(default_factory=list)


class _Short(Exception):
    """What an expression asks for cannot be paid for, or is not there."""


# -- expressions, read as data -------------------------------------------------

def _walk(e, visit) -> None:
    if type(e) is tuple and e:
        visit(e)
        for part in e[1:]:
            _walk(part, visit)


def _names(e, known) -> List[str]:
    """The earlier answers an expression reads, in order."""
    out: List[str] = []

    def visit(node):
        if node[0] == "var" and node[1] in known and node[1] not in out:
            out.append(node[1])

    _walk(e, visit)
    return out


def _uses(plan: Plan, op: str) -> bool:
    found: List[bool] = []

    def visit(node):
        if node[0] == op:
            found.append(True)

    for e in (plan.experiment,) + tuple(x.question for x in plan.entries):
        _walk(e, visit)
    return bool(found)


def _fill(template, env):
    """A rebuild template with its holes filled; the rest copied as it is."""
    if template[0] == HOLE:
        value = env[template[1]]
        return value[template[2]] if len(template) > 2 else value
    if template[0] == "subst":
        return _substitute(_fill(template[1], env), template[2], _fill(template[3], env))
    kids = children(template)
    return rebuild(template, [_fill(kid, env) for kid in kids]) if kids else template


def _substitute(p, name, value):
    if p[0] == GET and p[1] == name:
        return value
    kids = children(p)
    return rebuild(p, [_substitute(kid, name, value) for kid in kids]) if kids else p


def _held(rounds) -> List[tuple]:
    out: Dict[tuple, None] = {}
    for _, beam in rounds or []:
        for p in beam:
            out.setdefault(p, None)
    return list(out)


# -- what an expression may read at one node, and what it costs ---------------

class _Scope:
    def __init__(self, run, node, columns, target, evaluator, leaves, conditions, own,
                 held) -> None:
        self.run, self.node = run, node
        self.columns, self.target, self.evaluator = columns, target, evaluator
        self.leaves, self.conditions, self.own, self.held = leaves, conditions, own, held
        self.seen: set = set()
        both = self._both
        self.ops = {
            "const": lambda e, env, c: e[1],
            "var": lambda e, env, c: env[e[1]],
            "own": lambda e, env, c: self.own,
            "held": lambda e, env, c: list(self.held),
            "leaves": lambda e, env, c: list(self.leaves),
            "conditions": lambda e, env, c: list(self.conditions),
            "target": lambda e, env, c: self.target,
            "domain": lambda e, env, c: np.ones(len(self.target), dtype=bool),
            "eval": self._eval,
            "eq": both(np.equal), "neq": both(np.not_equal),
            "and": both(np.logical_and), "or": both(np.logical_or),
            "vadd": both(np.add), "vsub": both(np.subtract),
            "plus": both(lambda a, b: a + b), "minus": both(lambda a, b: a - b),
            "div": both(lambda a, b: a // b),
            "not": self._one(np.logical_not),
            "mask": self._one(lambda v: np.asarray(v) != 0),
            "bool": self._one(lambda m: np.asarray(m).astype(np.int64)),
            "count": self._one(lambda m: int(np.count_nonzero(m))),
            "bits": self._bits,
            "distinct": lambda e, env, c: description.alphabet_of(
                np.asarray(self.value(e[1], env, c))[np.asarray(self.value(e[2], env, c))]),
            "search": self._search,
            "probe": self._probe,
            "successors": self._successors,
            "take": lambda e, env, c: list(self.value(e[2], env, c))[:int(self.value(e[1], env, c))],
            "flatmap": self._flatmap,
            "argmin": self._pick(min),
            "argmax": self._pick(max),
            "list": lambda e, env, c: tuple(self.value(x, env, c) for x in e[1:]),
            "item": lambda e, env, c: self.value(e[1], env, c)[int(self.value(e[2], env, c))],
            "question": self._question,
        }

    def value(self, e, env, charged):
        return self.ops[e[0]](e, env, charged)

    def _both(self, f):
        return lambda e, env, c: f(self.value(e[1], env, c), self.value(e[2], env, c))

    def _one(self, f):
        return lambda e, env, c: f(self.value(e[1], env, c))

    def _seen(self, program, charged) -> None:
        if charged and program not in self.seen:
            self.run.pay(self.node, BUILD, 1)
            self.seen.add(program)

    def _eval(self, e, env, c):
        program = self.value(e[1], env, c)
        self._seen(program, c)
        return self.evaluator(program)

    def _bits(self, e, env, c):
        program = self.value(e[1], env, c)
        self._seen(program, c)
        return _key(program, self.columns, self.target, self.evaluator)[0]

    def _flat(self, e, env, c):
        target = np.asarray(self.value(e[1], env, c), dtype=np.int64)
        rows = np.asarray(self.value(e[2], env, c), dtype=bool)
        if not rows.any():
            return None
        cols = {k: v[rows] for k, v in self.columns.items()}
        leaves, _ = vocabulary(cols, target[rows])
        cap = min(int(self.value(e[3], env, c)), self.run.ledger.left())
        if cap <= len(leaves):
            raise _Short()
        found = P.run(P.with_budget(self.run.policy, cap), cols, target[rows])
        self.run.pay(self.node, SEARCH, found.evaluations)
        return found

    def _search(self, e, env, c):
        found = self._flat(e, env, c)
        if found is None:
            raise _Short()
        return found.program

    def _probe(self, e, env, c):
        found = self._flat(e, env, c)
        return 0.0 if found is None else found.bits

    def _successors(self, e, env, c):
        ahead = _replace(self.run.policy, rules=tuple(self.value(e[2], env, c)))
        start = self.value(e[1], env, c)
        return list(dict.fromkeys(P.successors(ahead, start, self.leaves, self.conditions)))

    def _apply(self, lam, x, env, c):
        inner = dict(env)
        inner[lam[1]] = x
        return self.value(lam[2], inner, c)

    def _flatmap(self, e, env, c):
        out: Dict[tuple, None] = {}
        for x in self.value(e[1], env, c):
            for y in self._apply(e[2], x, env, c):
                out.setdefault(y, None)
        return list(out)

    def _pick(self, which):
        def pick(e, env, c):
            items = list(self.value(e[1], env, c))
            if not items:
                raise _Short()
            scores = [self._apply(e[2], x, env, c) for x in items]
            order = (lambda i: (scores[i], i)) if which is min else (lambda i: (scores[i], -i))
            return items[which(range(len(items)), key=order)]
        return pick

    def _question(self, e, env, c):
        columns = tuple((col[1], self.evaluator(self.value(col[2], env, c)))
                        for col in (e[3][1:] if len(e) > 3 else ()))
        return ("question", self.value(e[1], env, c), self.value(e[2], env, c), columns)


# -- the interpreter -------------------------------------------------------------

class _Interpreter:
    def __init__(self, policy, ledger, plans, flat_share, profile, trace, env,
                 schedule=None) -> None:
        self.policy, self.ledger, self.plans = policy, ledger, tuple(plans)
        self.flat_share, self.profile, self.trace = float(flat_share), profile, trace
        self.env = dict(env or {})
        self.schedule = None if schedule is None else tuple(tuple(s) for s in schedule)
        self.nodes: List[Node] = []
        self.records: List[ResearchRecord] = []
        every = self.plans + tuple(p for s in (self.schedule or ()) for p in s)
        self.log = any(_uses(plan, "held") for plan in every)

    def _plans(self, depth: int) -> Tuple[Plan, ...]:
        """The plans a node at this depth may pose: the same at every depth, or
        those the schedule gives for it -- none past its end."""
        if self.schedule is None:
            return self.plans
        return self.schedule[depth - 1] if depth <= len(self.schedule) else ()

    def _afford(self, evaluations: int) -> bool:
        return self.ledger.left() >= evaluations

    def _charge(self, node: Node, article: str, evaluations: int) -> None:
        self.ledger.charge(article, evaluations)
        node.cost[article] += int(evaluations)

    def pay(self, node: Node, article: str, evaluations: int) -> None:
        if not self._afford(evaluations):
            raise _Short()
        self._charge(node, article, evaluations)

    def solve(self, problem: Problem, budget: int, depth: int) -> Optional[Node]:
        columns, target = problem.data()
        if len(target) == 0:
            return None
        leaves, conditions = vocabulary(columns, target)
        cap = min(int(budget * self.flat_share), budget, self.ledger.left())
        if cap <= len(leaves):
            return None
        node = Node(len(self.nodes), problem, None, None, {a: 0 for a in ARTICLES},
                    budget=int(budget))
        self.nodes.append(node)
        if problem.parent is not None:
            self.nodes[problem.parent].children.append(node.index)
        rounds = [] if self.log else None
        found = P.run(P.with_budget(self.policy, cap), columns, target,
                      profile=self.profile and problem.parent is None, trace=self.trace,
                      log_rounds=rounds)
        self._charge(node, SEARCH, found.evaluations)
        node.found, node.program = found, found.program
        evaluator = Evaluator(columns)
        own = found.program
        here = self._plans(depth)
        if not here or np.array_equal(evaluator(own), target) or depth >= MAX_DEPTH:
            return node
        share = (budget - found.evaluations) // len(here)
        if share <= len(leaves) or not self._afford(1):
            return node
        best = (_key(own, columns, target, evaluator), own, FLAT, ())
        self._charge(node, BUILD, 1)
        scope = _Scope(self, node, columns, target, evaluator, leaves, conditions, own,
                       _held(rounds))
        for plan in here:
            made = self._pose(plan, node, scope, share, depth)
            if made is None:
                continue
            if not self._afford(2):
                break
            candidate, used = made
            self._charge(node, ASSEMBLE, 1)
            self._charge(node, VERIFY, 1)
            key = _key(candidate, columns, target, evaluator)
            if key < best[0]:
                best = (key, candidate, plan.name, used)
            if best[0][2] == 0:
                break
        _, node.program, node.chosen, node.used = best
        for record in self.records:
            if record.parent == node.index:
                record.kept = record.plan == node.chosen and not record.stopped
        return node

    def _pose(self, plan: Plan, node: Node, scope: _Scope, share: int, depth: int):
        if not self._afford(1):
            return None
        record = ResearchRecord(node.index, plan.name, plan.experiment)
        before = dict(self.ledger.spent)
        env = dict(self.env)
        env["share"] = share
        try:
            observation = scope.value(plan.experiment, env, True)
        except _Short:
            return self._close(record, before, "эксперимент не оплачен")
        env["X"] = observation
        record.observation = repr(observation)[:120]
        answers: Dict[str, Node] = {}
        used: List[int] = []
        for i, entry in enumerate(plan.entries):
            if i and not self._afford(1):
                return self._close(record, before, "вопрос не оплачен")
            self._charge(node, BUILD, 1)
            try:
                spec = scope.value(entry.question, env, False)
                budget = int(scope.value(entry.budget, env, False))
            except _Short:
                return self._close(record, before, "вопрос не построен")
            domain = np.asarray(spec[2], dtype=bool)
            if not domain.any():
                return self._close(record, before, "пустой вопрос")
            inputs = dict(scope.columns)
            for name, column in spec[3]:
                inputs[name] = np.asarray(column, dtype=np.int64)
            depends = (node.index,) + tuple(answers[n].index for n in _names(entry.question, answers))
            child = self.solve(Problem(inputs, np.asarray(spec[1], dtype=np.int64), domain,
                                       node.index, entry.name, depends), budget, depth + 1)
            if child is None:
                return self._close(record, before, "бюджета не хватило на вопрос")
            answers[entry.name] = child
            env[entry.name] = child.program
            used.append(child.index)
        candidate = _fill(plan.rebuild, env)
        record.derived = tuple(used)
        record.candidate = show(candidate)
        self._close(record, before, "")
        return candidate, tuple(used)

    def _close(self, record: ResearchRecord, before: Dict[str, int], stopped: str) -> None:
        record.cost = {a: self.ledger.spent[a] - before[a] for a in ARTICLES}
        record.stopped = stopped
        self.records.append(record)
        return None


def solve(problem: Problem, policy: P.SearchPolicy, budget: int, plans: Sequence[Plan] = (),
          flat_share: float = 1.0, profile: bool = False, trace: bool = False,
          env: Optional[Dict[str, object]] = None,
          schedule: Optional[Sequence[Sequence[Plan]]] = None) -> Reasoning:
    """The answer to a question within one budget, research questions posed by
    the plans given, in the order given. `env` binds names plans may read.
    `schedule`, when given, is the plans by depth -- the first for the root,
    the next for the questions it derives, none past its end -- instead of
    the same plans at every depth."""
    ledger = Ledger(int(budget))
    interpreter = _Interpreter(policy, ledger, plans, flat_share, profile, trace, env, schedule)
    root = interpreter.solve(problem, int(budget), 1)
    if root is None:
        raise ValueError("the budget is smaller than the root's leaves")
    return Reasoning(root.program, root.found, interpreter.nodes, ledger, interpreter.records)
