"""
mana.methods.solvers — four ways of knowing a black box.

Each is a whole way of thinking about the box, not a setting of one:

    exhaust     ask every input; the model is the table. Always right,
                costs LEVELS ** n
    calculate   assume the box is linear: its value at one point and one
                step along each input give every slope. n + 1 questions
    decompose   assume the effects add: each input moved through all its
                values while the others stay put. 1 + n (LEVELS - 1)
    experiment  do not assume, test: for every pair of inputs, does moving
                both differ from the sum of moving each? Inputs that
                interact are grouped, and each group is tabulated with the
                rest held still. Pairs cost about 3 n^2 / 2, the tables
                LEVELS ** (group size) each

Every method then checks itself on CHECKS inputs of its own choosing, and
that check -- not the world's verdict -- is all a method knows about
whether it worked.

What a method sees is a `Session`: how many inputs, how many values each,
and a way to ask. Answers are remembered, so asking again what is known
costs nothing, and one session can serve several methods in turn.

Two regimes. With `announce` (M0) a method knows its own plan's cost
before it spends and declines rather than begin what the budget cannot
finish -- except that experiment learns the size of its groups only after
its tests, and pays for them. Without it (M1) no method says what it will
cost: it asks until it is done or its budget is spent, and whoever
chooses methods has to learn their costs from what they did before.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.2"

CHECKS = 10
BUDGET = 200000
#: Value pairs tried for each pair of inputs: one test can miss an
#: interaction that happens to cancel at the values chosen.
PAIR_TESTS = 3
#: Questions sent to the box at once by the methods that ask many.
CHUNK = 50000

Model = Callable[[np.ndarray], np.ndarray]


class OverBudget(Exception):
    pass


class Decline(Exception):
    def __init__(self, planned: int, why: str) -> None:
        super().__init__(why)
        self.planned = planned
        self.why = why


def _codes(X: np.ndarray, levels: int) -> np.ndarray:
    return X @ (levels ** np.arange(X.shape[1], dtype=np.int64))


def _decode(codes: np.ndarray, k: int, levels: int) -> np.ndarray:
    """The settings of k inputs with these codes, one per row."""
    return (codes[:, None] // (levels ** np.arange(k, dtype=np.int64))) % levels


def _settings(levels: int, k: int):
    """Every setting of k inputs, in chunks, in the order of their codes."""
    total = levels ** k
    for start in range(0, total, CHUNK):
        yield _decode(np.arange(start, min(total, start + CHUNK), dtype=np.int64), k, levels)


class Session:
    """The form of the question and a way to ask it.

    `begin` opens the next method's budget, counted in questions new to
    the session: what an earlier method asked is known, and free."""

    def __init__(self, ask: Callable[[np.ndarray], np.ndarray], n: int, levels: int,
                 budget: int = BUDGET, announce: bool = True) -> None:
        self._ask = ask
        self.n = n
        self.levels = levels
        self.budget = budget
        self.announce = announce
        self.known: Dict[int, int] = {}
        self._start = 0

    def begin(self, budget: int) -> None:
        self._start = len(self.known)
        self.budget = budget

    @property
    def spent(self) -> int:
        """New questions asked since `begin`."""
        return len(self.known) - self._start

    @property
    def total(self) -> int:
        return len(self.known)

    def remaining(self) -> int:
        return self.budget - self.spent

    def ask(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.int64))
        codes = _codes(X, self.levels).tolist()
        fresh: Dict[int, int] = {}
        for i, code in enumerate(codes):
            if code not in self.known and code not in fresh:
                fresh[code] = i
        allowed = self.remaining()
        if len(fresh) > allowed:
            if not self.announce and allowed > 0:
                # Nobody said this would not fit: it asks until it runs out.
                self._store(X, list(fresh.items())[:allowed])
            raise OverBudget(f"{len(fresh)} new questions, {allowed} left")
        self._store(X, list(fresh.items()))
        return np.array([self.known[code] for code in codes], dtype=np.int64)

    def _store(self, X: np.ndarray, items: List[Tuple[int, int]]) -> None:
        if items:
            answers = np.asarray(self._ask(X[[i for _, i in items]])).tolist()
            self.known.update(zip((code for code, _ in items), answers))

    def points(self) -> np.ndarray:
        """The questions asked since `begin`."""
        return np.fromiter(itertools.islice(self.known.keys(), self._start, None),
                           dtype=np.int64, count=self.spent)


def exhaust(s: Session, rng: np.random.Generator) -> Tuple[Model, int]:
    planned = s.levels ** s.n
    if s.announce and planned > s.remaining():
        raise Decline(planned, "the table is larger than the budget")
    table = np.concatenate([s.ask(X) for X in _settings(s.levels, s.n)])
    return (lambda Z: table[_codes(np.asarray(Z, dtype=np.int64), s.levels)]), planned


def calculate(s: Session, rng: np.random.Generator) -> Tuple[Model, int]:
    planned = s.n + 1 + CHECKS
    if s.announce and planned > s.remaining():
        raise Decline(planned, "not enough budget for n + 1 questions")
    base = rng.integers(0, s.levels, size=s.n)
    step = np.where(base < s.levels - 1, 1, -1)
    rows = np.tile(base, (s.n + 1, 1))
    rows[np.arange(1, s.n + 1), np.arange(s.n)] += step
    y = s.ask(rows)
    f0 = int(y[0])
    slopes = (y[1:] - f0) * step
    return (lambda Z: f0 + (np.asarray(Z, dtype=np.int64) - base) @ slopes), planned


def decompose(s: Session, rng: np.random.Generator) -> Tuple[Model, int]:
    planned = 1 + s.n * (s.levels - 1) + CHECKS
    if s.announce and planned > s.remaining():
        raise Decline(planned, "not enough budget to move every input")
    base = rng.integers(0, s.levels, size=s.n)
    rows = np.tile(base, (s.n * s.levels, 1))
    for i in range(s.n):
        rows[i * s.levels:(i + 1) * s.levels, i] = np.arange(s.levels)
    y = s.ask(rows).reshape(s.n, s.levels)
    f0 = int(y[0, base[0]])
    effect = y - f0
    index = np.arange(s.n)[None, :]
    return (lambda Z: f0 + effect[index, np.asarray(Z, dtype=np.int64)].sum(axis=1)), planned


def experiment(s: Session, rng: np.random.Generator) -> Tuple[Model, int]:
    n, levels, t = s.n, s.levels, PAIR_TESTS
    base = rng.integers(0, levels, size=n)
    alt = np.array([rng.choice([v for v in range(levels) if v != base[i]], size=t,
                               replace=False) for i in range(n)])
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rows = [base.copy()]
    for i in range(n):
        for k in range(t):
            row = base.copy()
            row[i] = alt[i, k]
            rows.append(row)
    for i, j in pairs:
        for k in range(t):
            row = base.copy()
            row[i], row[j] = alt[i, k], alt[j, k]
            rows.append(row)
    tests = len(rows)
    if s.announce and tests + CHECKS > s.remaining():
        raise Decline(tests + CHECKS, "not enough budget to test the pairs")
    y = s.ask(np.array(rows))
    f0 = int(y[0])
    single = y[1:1 + n * t].reshape(n, t)
    joint = y[1 + n * t:].reshape(len(pairs), t)
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (i, j), values in zip(pairs, joint):
        if np.any(values - single[i] - single[j] + f0 != 0):
            parent[root(i)] = root(j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    members = sorted(groups.values())
    tables_cost = sum(levels ** len(g) for g in members)
    planned = tests + tables_cost + CHECKS
    if s.announce and tables_cost + CHECKS > s.remaining():
        raise Decline(planned, f"a group of {max(len(g) for g in members)} inputs "
                               "is too large to tabulate")
    tables = []
    for g in members:
        parts = []
        for settings in _settings(levels, len(g)):
            block = np.tile(base, (len(settings), 1))
            block[:, g] = settings
            parts.append(s.ask(block))
        # Codes count the first input fastest: Fortran order.
        tables.append((g, np.concatenate(parts).reshape((levels,) * len(g), order="F")))

    def model(Z) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.int64)
        out = np.full(len(Z), f0, dtype=np.int64)
        for g, table in tables:
            out += table[tuple(Z[:, g].T)] - f0
        return out

    return model, planned


METHODS: Dict[str, Callable[[Session, np.random.Generator], Tuple[Model, int]]] = {
    "exhaust": exhaust, "calculate": calculate,
    "decompose": decompose, "experiment": experiment,
}


@dataclass
class Attempt:
    method: str
    model: Optional[Model]
    #: Its own check passed: all a method knows about whether it worked.
    believed: bool
    #: Questions new to the session it asked, the check's included.
    queries: int
    #: What its plan said it would cost -- recorded, and never shown to
    #: whoever chooses methods in M1.
    planned: int
    #: Why it did not finish, if it did not.
    declined: str = ""
    seconds: float = 0.0
    points: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64), repr=False)


def run(session: Session, name: str, seed: int, budget: int = BUDGET,
        methods: Optional[Dict[str, Callable]] = None) -> Attempt:
    """One method in a session, then its own check. The same seed gives
    every method the same base point and the same check inputs."""
    session.begin(budget)
    started = time.time()
    model: Optional[Model] = None
    planned, why = 0, ""
    try:
        model, planned = (methods or METHODS)[name](session, np.random.default_rng([seed, 1]))
    except Decline as declined:
        planned, why = declined.planned, declined.why
    except OverBudget as over:
        why = str(over)
    believed = False
    if model is not None:
        X = np.random.default_rng([seed, 2]).integers(0, session.levels, size=(CHECKS, session.n))
        try:
            believed = bool(np.array_equal(model(X), session.ask(X)))
        except OverBudget:
            why = "no budget left to check"
    return Attempt(name, model, believed, session.spent, planned, why,
                   time.time() - started, session.points())


def attempt(name: str, ask: Callable[[np.ndarray], np.ndarray], n: int, levels: int,
            seed: int, budget: int = BUDGET, announce: bool = True,
            methods: Optional[Dict[str, Callable]] = None) -> Attempt:
    """One method alone on a box."""
    return run(Session(ask, n, levels, budget, announce), name, seed, budget, methods)
