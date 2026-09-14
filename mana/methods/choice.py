"""
mana.methods.choice — which method next, learnt from what methods did.

What a chooser may know about a method is what it could have seen on
earlier boxes: how many questions each attempt asked, whether it built a
model before its budget ran out, and whether its own check passed. Never
a plan, never a cost formula, never the world's verdict, never the
structure. And one thing about the box in front of it: which methods have
already failed on it.

Chooser (M1)
------------
    success   how often a method's check passed, among earlier attempts
              made after the same methods had failed (all its attempts,
              if there were none such). Laplace: a method tried once and
              failed is not written off
    cost      the questions it asked, extrapolated in n: a straight line
              through log(questions) against n
    next      of the methods not yet tried on this box, the one with the
              most success per question (Smith's rule), among those worth
              trying at all -- expected success times what an answer is
              worth must exceed the questions it would spend. A method
              expected to run past its budget cannot succeed. When none is
              worth it, the chooser gives up
    unknown   a method never tried is assumed cheap: the only way to learn
              what it costs is to try it

Explorer (M1b)
--------------
M1 found what the Chooser cannot do: learn how a method's cost grows. One
size says nothing about growth, and it read that as no growth. The
Explorer has a second kind of action -- an experiment about a method:
running it on a smaller version of the box in front of it, the other
inputs held still. Both kinds are weighed by one rule:

    worth of applying m     P(its check passes) x P(it finishes) x VALUE
                            - the questions it is expected to spend
    worth of a probe        how much better the decision about this box is
                            expected to be once the probe's cost is known,
                            - the questions the probe is expected to spend

A probe is made when it is worth more than it costs and more than any
other probe; otherwise the best application is made, or none, and the
Explorer gives up. No rule says "probe when data is scarce": a probe pays
only where what it could show would change what is done.

For that the cost of a method has to be a belief, not a number:

    the kind's line     log(questions) against n, fitted to every earlier
                        run of the method after the same failures -- tasks
                        and probes alike -- with a prior on the slope wide
                        enough for a table's growth. From one size the
                        slope stays as uncertain as the prior
    the box's own line  the box may lie off its kind's line, by a spread
                        measured between boxes and a slope that may differ
                        by BOX_SLOPE; its own runs -- probes, failed
                        attempts -- pull it there
    drift               a line is a guess about sizes not seen, and the
                        guess is doubted in proportion to the distance to
                        the nearest size seen. How much, is learnt: every
                        run is journalled with what was expected of it, and
                        the drift is the typical error per unit of distance
                        of the predictions that were extrapolations

Two journals over one format: the task experience is the entries that
were attempts on boxes; the method experience is every entry, probes
included, each with the cost predicted, its spread, the distance it was
extrapolated over, and the cost that came.

What is declared rather than learnt, said plainly: the priors below, the
value of an answer, and that the value counted is this box's -- a probe's
worth to later boxes is not counted, so it is a lower bound, and a probe
is made only when it already pays on the box at hand. Success is not
explored: which method's check passes is learnt only from attempts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from . import solvers
from .portfolio import Choice

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.1"

#: What an answer to a box is worth, in questions: as much as one method
#: may spend on it. Fixed before any run.
VALUE = solvers.BUDGET

# Where the Explorer's beliefs start, before any experience. In nats of
# log(questions).
#: Prior sd of how much a cost grows per input: wide enough for a table,
#: which grows by ln 10 = 2.3 per input.
SLOPE_PRIOR = 2.5
#: Prior of the spread between boxes of one kind, worth SPREAD_WEIGHT runs.
SPREAD_PRIOR = 1.0
SPREAD_WEIGHT = 2
#: How much one box's growth may differ from its kind's, per input.
BOX_SLOPE = 0.5
#: A box's own run is measured, not estimated.
OBSERVED = 0.05
#: Doubt of an extrapolation per unit of distance, before any surprise,
#: worth DRIFT_WEIGHT surprises.
DRIFT_PRIOR = 0.5
DRIFT_WEIGHT = 2

TASK, PROBE = "task", "probe"
_NODES, _WEIGHTS = np.polynomial.hermite_e.hermegauss(9)
_WEIGHTS = _WEIGHTS / _WEIGHTS.sum()
_PHI = NormalDist().cdf


# --------------------------------------------------------------------------
# M1: the Chooser
# --------------------------------------------------------------------------

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
    def __init__(self, experience: Experience, value: float = VALUE, seed: int = 0,
                 registry: Optional[Dict[str, Callable]] = None) -> None:
        self.experience = experience
        self.value = value
        self.registry = registry
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
            found = solvers.run(session, method, seed, budget, self.registry)
            tried.append(method)
            left.remove(method)
            if learn:
                self.experience.add(Record(n, before, method, found.believed, found.queries,
                                           found.model is not None))
            if found.believed:
                return Choice(method, session.total, tuple(tried)), found.model
            before = before | {method}
        return Choice(None, session.total, tuple(tried)), None


# --------------------------------------------------------------------------
# M1b: the Explorer
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Entry:
    #: TASK: a method applied to a box. PROBE: a method run on a smaller
    #: version of one, to learn what it costs.
    kind: str
    box: int
    n: int
    before: FrozenSet[str]
    method: str
    believed: bool
    finished: bool
    queries: int
    #: What was expected of it: log(questions), its spread, and how far
    #: the size lay from any size seen -- 0 where it was not a guess.
    predicted: float
    spread: float
    distance: float


class Journal:
    """Task experience and method experience, over one format."""

    def __init__(self) -> None:
        self.entries: List[Entry] = []

    def add(self, entry: Entry) -> None:
        self.entries.append(entry)

    def tasks(self) -> List[Entry]:
        return [e for e in self.entries if e.kind == TASK]

    def runs(self, method: str, before: FrozenSet[str], box: int) -> List[Entry]:
        """Every earlier run of a method, on other boxes, after the same
        failures -- or after any, if there were none such."""
        mine = [e for e in self.entries if e.method == method and e.box != box]
        return [e for e in mine if e.before == before] or mine

    def success(self, method: str, before: FrozenSet[str], box: int) -> float:
        done = [e for e in self.entries if e.kind == TASK and e.method == method
                and e.finished and e.box != box]
        same = [e for e in done if e.before == before] or done
        return (sum(e.believed for e in same) + 1) / (len(same) + 2)

    def drift(self) -> float:
        """Typical error of an extrapolation per unit of distance."""
        guesses = [e for e in self.entries if e.distance > 0]
        total = DRIFT_WEIGHT * DRIFT_PRIOR ** 2 + sum(
            ((math.log(max(e.queries, 1)) - e.predicted) / e.distance) ** 2 for e in guesses)
        return math.sqrt(total / (DRIFT_WEIGHT + len(guesses)))


@dataclass
class Belief:
    """log(questions) of one method on one box: (a, b) the kind's line,
    (u, v) the box's offset and change of slope about `centre`."""
    mean: np.ndarray
    cov: np.ndarray
    centre: float
    sizes: FrozenSet[float]
    drift: float

    def _h(self, n: float) -> np.ndarray:
        return np.array([1.0, n, 1.0, n - self.centre])

    def distance(self, n: float) -> float:
        return float(min((abs(n - s) for s in self.sizes), default=n))

    def predict(self, n: float) -> Tuple[float, float]:
        h = self._h(n)
        var = float(h @ self.cov @ h) + (self.drift * self.distance(n)) ** 2
        return float(h @ self.mean), math.sqrt(max(var, 1e-12))

    def observe(self, n: float, y: float) -> "Belief":
        h = self._h(n)
        spread = float(h @ self.cov @ h) + OBSERVED ** 2
        gain = self.cov @ h / spread
        return Belief(self.mean + gain * (y - float(h @ self.mean)),
                      self.cov - np.outer(gain, h @ self.cov), self.centre,
                      self.sizes | {float(n)}, self.drift)


def _worth(mu: float, sd: float, chance: float, value: float, budget: float
           ) -> Tuple[float, float]:
    """Expected worth of running a method whose log cost is N(mu, sd), and
    the questions it is expected to spend: it stops at its budget."""
    cap = math.log(budget)
    z = (cap - mu) / sd
    fits = _PHI(z)
    spend = math.exp(min(mu + sd * sd / 2, 700.0)) * _PHI(z - sd) + budget * (1.0 - fits)
    return chance * fits * value - spend, spend


class Explorer:
    def __init__(self, methods: Sequence[str] = tuple(solvers.METHODS),
                 registry: Optional[Dict[str, Callable]] = None, value: float = VALUE,
                 budget: int = solvers.BUDGET, seed: int = 0) -> None:
        self.methods = tuple(methods)
        self.registry = registry
        self.value = value
        self.budget = budget
        self.journal = Journal()
        self.probes = 0
        #: (method, size, questions) of the probes made on the last box.
        self.last_probes: List[Tuple[str, int, int]] = []
        self._box = 0
        self._seed = seed

    # -- beliefs -----------------------------------------------------------

    def belief(self, method: str, before: FrozenSet[str], box: int,
               own: Sequence[Tuple[int, float]], drift: float) -> Optional[Belief]:
        runs = self.journal.runs(method, before, box)
        if not runs and not own:
            return None
        prior_mean = np.array([0.0, 0.0])
        prior_cov = np.diag([10.0 ** 2, SLOPE_PRIOR ** 2])
        mean, cov, spread2 = prior_mean, prior_cov, SPREAD_PRIOR ** 2
        sizes = [float(e.n) for e in runs]
        if runs:
            X = np.column_stack([np.ones(len(runs)), sizes])
            y = np.log([max(e.queries, 1) for e in runs])
            inverse = np.linalg.inv(prior_cov)
            for _ in range(2):
                cov = np.linalg.inv(X.T @ X / spread2 + inverse)
                mean = cov @ (X.T @ y / spread2 + inverse @ prior_mean)
                rss = float(np.sum((y - X @ mean) ** 2))
                spread2 = (SPREAD_WEIGHT * SPREAD_PRIOR ** 2 + rss) / (SPREAD_WEIGHT + len(runs))
        seen = sizes + [float(n) for n, _ in own]
        full = np.zeros((4, 4))
        full[:2, :2] = cov
        full[2, 2], full[3, 3] = spread2, BOX_SLOPE ** 2
        belief = Belief(np.concatenate([mean, [0.0, 0.0]]), full, float(np.mean(seen)),
                        frozenset(sizes), drift)
        for n, y in own:
            belief = belief.observe(n, y)
        return belief

    def expect(self, method: str, before: FrozenSet[str], n: int) -> Optional[Tuple[float, float]]:
        """What it expects the method to cost on a new box of size n:
        (median questions, spread in nats)."""
        belief = self.belief(method, frozenset(before), -1, (), self.journal.drift())
        if belief is None:
            return None
        mu, sd = belief.predict(n)
        return math.exp(mu), sd

    # -- acting ------------------------------------------------------------

    def _apply_worth(self, belief: Optional[Belief], chance: float, n: int) -> float:
        if belief is None:
            return 0.5 * self.value - 1.0          # never tried: optimism
        mu, sd = belief.predict(n)
        return _worth(mu, sd, chance, self.value, self.budget)[0]

    def _probe(self, session: solvers.Session, method: str, size: int, levels: int,
               seed: int, n: int) -> solvers.Attempt:
        """The method on the first `size` inputs, the rest held at values
        of its own: real questions to the box, remembered by its session."""
        held = np.random.default_rng([seed, 3]).integers(0, levels, size=n)

        def ask(X: np.ndarray) -> np.ndarray:
            full = np.tile(held, (len(X), 1))
            full[:, :size] = X
            return session.ask(full)

        session.begin(self.budget)
        smaller = solvers.Session(ask, size, levels, self.budget, announce=False)
        return solvers.run(smaller, method, seed, self.budget, self.registry)

    def solve(self, ask: Callable[[np.ndarray], np.ndarray], n: int, levels: int, seed: int,
              learn: bool = True) -> Tuple[Choice, Optional[solvers.Model]]:
        self._box += 1
        box = self._box
        session = solvers.Session(ask, n, levels, self.budget, announce=False)
        before: FrozenSet[str] = frozenset()
        tried: List[str] = []
        left = list(self.methods)
        own: Dict[str, List[Tuple[int, float]]] = {m: [] for m in self.methods}
        probed = set()
        self.last_probes = []
        drift = self.journal.drift()
        while left:
            beliefs = {m: self.belief(m, before, box, own[m], drift) for m in left}
            chances = {m: self.journal.success(m, before, box) for m in left}
            worths = {m: self._apply_worth(beliefs[m], chances[m], n) for m in left}
            pick = max(sorted(left), key=lambda m: worths[m])
            now = max(0.0, worths[pick])

            best_net, probe = 0.0, None
            for m in sorted(left):
                belief = beliefs[m]
                if belief is None:
                    continue
                for size in range(2, n):
                    if (m, size) in probed:
                        continue
                    mu, sd = belief.predict(size)
                    spend = _worth(mu, sd, 1.0, 0.0, self.budget)[1]
                    if spend >= self.value - now - best_net:
                        continue                   # could not pay, whatever it showed
                    after = 0.0
                    for node, weight in zip(_NODES, _WEIGHTS):
                        seen = belief.observe(size, mu + sd * node)
                        after += weight * max(0.0, max(
                            self._apply_worth(seen if k == m else beliefs[k], chances[k], n)
                            for k in left))
                    net = after - now - spend
                    if net > best_net:
                        best_net, probe = net, (m, size, mu, sd, belief.distance(size))

            if probe is not None:
                m, size, mu, sd, distance = probe
                found = self._probe(session, m, size, levels, seed, n)
                probed.add((m, size))
                self.probes += 1
                self.last_probes.append((m, size, found.queries))
                own[m].append((size, math.log(max(found.queries, 1))))
                if learn:
                    self.journal.add(Entry(PROBE, box, size, before, m, found.believed,
                                           found.model is not None, found.queries,
                                           mu, sd, distance))
                continue

            if now <= 0.0:
                break
            belief = beliefs[pick]
            mu, sd = belief.predict(n) if belief is not None else (0.0, 0.0)
            distance = belief.distance(n) if belief is not None else 0.0
            found = solvers.run(session, pick, seed, self.budget, self.registry)
            tried.append(pick)
            left.remove(pick)
            if learn:
                self.journal.add(Entry(TASK, box, n, before, pick, found.believed,
                                       found.model is not None, found.queries,
                                       mu, sd, distance))
            if found.believed:
                return Choice(pick, session.total, tuple(tried)), found.model
            before = before | {pick}
        return Choice(None, session.total, tuple(tried)), None
