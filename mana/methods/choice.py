"""
mana.methods.choice — which method next, learnt from what methods did.

What the chooser may know about a method is what it could have seen on
earlier boxes: how many questions each attempt asked, whether it built a
model before its budget ran out, and whether its own check passed. Never
a plan, never a cost formula, never the world's verdict, never the
structure. And one thing about the box in front of it: which methods have
already failed on it -- the only description of the box it has.

    success   how often a method's check passed, among earlier attempts
              made after the same methods had failed (all its attempts,
              if there were none such). Laplace: a method tried once and
              failed is not written off
    cost      the questions it asked, extrapolated in n: a straight line
              through log(questions) against n. So exhaust's 100, 1 000,
              10 000 at n = 2, 3, 4 say a million at n = 6 -- nobody wrote
              10 ** n anywhere
    next      of the methods not yet tried on this box, the one with the
              most success per question (Smith's rule: optimal for trying
              alternatives one after another), among those worth trying
              at all -- expected success times what an answer is worth
              must exceed the questions it would spend. A method expected
              to run past its budget cannot succeed. When none is worth
              it, the chooser gives up
    unknown   a method never tried is assumed cheap: the only way to learn
              what it costs is to try it
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from . import solvers
from .portfolio import Choice

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: What an answer to a box is worth, in questions: as much as one method
#: may spend on it. Fixed before any run.
VALUE = solvers.BUDGET


@dataclass(frozen=True)
class Record:
    n: int
    #: Methods whose check had failed on the same box before this one.
    before: FrozenSet[str]
    method: str
    believed: bool
    queries: int
    finished: bool


class Experience:
    def __init__(self, methods: Sequence[str] = tuple(solvers.METHODS),
                 budget: int = solvers.BUDGET) -> None:
        self.methods = tuple(methods)
        self.budget = budget
        self.records: List[Record] = []

    def add(self, record: Record) -> None:
        self.records.append(record)

    def _rows(self, method: str, before: FrozenSet[str]) -> List[Record]:
        same = [r for r in self.records if r.method == method and r.before == before]
        return same or [r for r in self.records if r.method == method]

    def success(self, method: str, before: FrozenSet[str]) -> float:
        rows = self._rows(method, before)
        return (sum(r.believed for r in rows) + 1) / (len(rows) + 2)

    def cost(self, method: str, n: int, before: FrozenSet[str]) -> Optional[float]:
        rows = self._rows(method, before)
        if not rows:
            return None
        sizes = np.array([r.n for r in rows], dtype=float)
        logs = np.log([max(r.queries, 1) for r in rows])
        if len(set(sizes.tolist())) < 2:
            return float(np.exp(logs.mean()))
        slope, intercept = np.polyfit(sizes, logs, 1)
        return float(np.exp(intercept + slope * n))


class Chooser:
    def __init__(self, experience: Experience, value: float = VALUE, seed: int = 0) -> None:
        self.experience = experience
        self.value = value
        self._rng = np.random.default_rng([seed, 17])

    def next(self, n: int, before: FrozenSet[str], left: Sequence[str]) -> Optional[str]:
        best: Optional[Tuple[Tuple[float, float], str]] = None
        budget = self.experience.budget
        for method in sorted(left):
            cost = self.experience.cost(method, n, before)
            if cost is None:
                chance, spend = 0.5, 1.0
            else:
                chance = self.experience.success(method, before) if cost <= budget else 0.0
                spend = min(max(cost, 1.0), budget)
            if chance * self.value <= spend:
                continue
            key = (chance / spend, float(self._rng.random()))
            if best is None or key > best[0]:
                best = (key, method)
        return best[1] if best else None

    def solve(self, ask: Callable[[np.ndarray], np.ndarray], n: int, levels: int, seed: int,
              learn: bool = True) -> Tuple[Choice, Optional[solvers.Model]]:
        budget = self.experience.budget
        session = solvers.Session(ask, n, levels, budget, announce=False)
        before: FrozenSet[str] = frozenset()
        tried: List[str] = []
        left = list(self.experience.methods)
        while left:
            method = self.next(n, before, left)
            if method is None:
                break
            found = solvers.run(session, method, seed, budget)
            tried.append(method)
            left.remove(method)
            if learn:
                self.experience.add(Record(n, before, method, found.believed, found.queries,
                                           found.model is not None))
            if found.believed:
                return Choice(method, session.total, tuple(tried)), found.model
            before = before | {method}
        return Choice(None, session.total, tuple(tried)), None
