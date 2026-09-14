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

import itertools
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from . import solvers
from .portfolio import Choice

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.3"

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

    def outlook(self, n: float, budget: float) -> Tuple[float, float]:
        """P(it finishes within the budget) and the questions it is
        expected to spend, at size n."""
        mu, sd = self.predict(n)
        return _PHI((math.log(budget) - mu) / sd), _worth(mu, sd, 1.0, 0.0, budget)[1]

    def outcomes(self, n: float) -> List[Tuple[float, float]]:
        """What a run at size n may cost: (log questions, probability)."""
        mu, sd = self.predict(n)
        return [(mu + sd * node, weight) for node, weight in zip(_NODES, _WEIGHTS)]


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


# --------------------------------------------------------------------------
# M1c: the Planner
# --------------------------------------------------------------------------
#
# The hypothesis under test: what information is worth is how much it
# changes the best plan, not how much better one method is known. M1b
# weighed a probe against the best single application; here the present is
# worth what the best ordered plan costs, and a probe what it is expected to
# save on that plan.
#
#     plan        an order of methods, stopping at the first that succeeds.
#                 Its cost: the questions it is expected to spend, plus the
#                 value of an answer times the chance that none succeeds.
#                 Giving up is the empty plan, and costs the value
#     step        cost and success together: a step succeeds if the method
#                 finishes within its budget and its check passes; it
#                 spends what it spends whether or not -- the whole budget,
#                 if it runs out. Each step's chances are taken after the
#                 steps before it failed
#     probe       worth the cost of the best plan now, less the expected
#                 cost of the best plan once its outcome is known, less its
#                 own expected questions. The outcomes are grouped by the
#                 plan each leads to -- which is why a probe was made, kept
#                 with it
#
# The beliefs about cost -- the kind's line, the box's offset, the drift --
# are the Explorer's, unchanged: not what is under test.

@dataclass(frozen=True)
class Step:
    method: str
    #: P(it finishes and its check passes), the steps before having failed.
    success: float
    #: Questions it is expected to spend.
    spend: float


@dataclass(frozen=True)
class Plan:
    steps: Tuple[Step, ...]
    questions: float
    success: float
    cost: float

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(step.method for step in self.steps)


def plan_cost(order: Sequence[str], before: FrozenSet[str],
              step: Callable[[str, FrozenSet[str]], Tuple[float, float]],
              value: float) -> Plan:
    """One order of methods, weighed. `step(method, failed)` gives
    (success, questions) of a method once the methods in `failed` have
    failed on this box."""
    failed = frozenset(before)
    survive, questions, steps = 1.0, 0.0, []
    for method in order:
        chance, spend = step(method, failed)
        steps.append(Step(method, chance, spend))
        questions += survive * spend
        survive *= 1.0 - chance
        failed = failed | {method}
    return Plan(tuple(steps), questions, 1.0 - survive, questions + value * survive)


def best_plan(left: Sequence[str], before: FrozenSet[str],
              step: Callable[[str, FrozenSet[str]], Tuple[float, float]],
              value: float) -> Plan:
    """The plan of least expected cost among every order of every subset of
    the methods left."""
    best = Plan((), 0.0, 0.0, float(value))
    for k in range(1, len(left) + 1):
        for order in itertools.permutations(sorted(left), k):
            plan = plan_cost(order, before, step, value)
            if plan.cost < best.cost - 1e-9:
                best = plan
    return best


#: How a probe's worth is reckoned. DIFFERENCE, as first run: the cost of
#: the best plan now, less the expected cost of the best plan after the
#: outcome. CHANGE: the expected saving of the best plan after the outcome
#: over the present plan, both weighed with what the outcome would teach --
#: nothing, where the plan would not change. The two agree only when the
#: beliefs are coherent, the expected cost of a plan after an outcome equal
#: to its cost now; the Explorer's are not -- an observation removes the
#: doubt of extrapolation outright, and a cost's expectation falls with its
#: spread -- so DIFFERENCE also pays for spread that goes away.
DIFFERENCE, CHANGE = "difference", "change"


@dataclass(frozen=True)
class Branch:
    """Outcomes of a probe that lead to the same plan."""
    plan: Tuple[str, ...]
    probability: float
    cost: float


@dataclass
class Decision:
    """One decision on a box, and why."""
    box: int
    n: int
    before: Tuple[str, ...]
    plan: Plan
    #: "probe", "apply" or "give up".
    action: str
    method: Optional[str] = None
    size: Optional[int] = None
    #: For a probe: the cost of the plan now less the expected cost of the
    #: plan after it, its expected questions, and the plans it may lead to.
    worth: float = 0.0
    spend: float = 0.0
    branches: Tuple[Branch, ...] = ()
    #: The journal already held this probe -- method, size, failures --
    #: from another box.
    repeat: bool = False
    #: The plan once its actual outcome was known, and what it cost.
    after: Optional[Tuple[str, ...]] = None
    actual: Optional[int] = None

    @property
    def predicted_change(self) -> float:
        """The chance the probe was expected to change the plan."""
        return sum(b.probability for b in self.branches if b.plan != self.plan.names)


class Planner(Explorer):
    """The Explorer's beliefs, weighed as plans."""

    def __init__(self, *args, worth: str = DIFFERENCE, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.worth = worth
        #: Every decision on the last box, in order.
        self.last_decisions: List[Decision] = []

    def _stepper(self, n: int, box: int, own: Dict[str, List[Tuple[int, float]]],
                 base: Callable, chance_of: Callable, drift: float,
                 extra: Optional[Tuple[str, Tuple[int, float]]] = None) -> Callable:
        cache: Dict[Tuple[str, FrozenSet[str]], Tuple[float, float]] = {}

        def step(method: str, failed: FrozenSet[str]) -> Tuple[float, float]:
            key = (method, failed)
            if key not in cache:
                seen = list(own[method])
                if extra is not None and extra[0] == method:
                    seen.append(extra[1])
                belief = base(method, failed)
                if belief is None and seen:
                    belief = self.belief(method, failed, box, seen, drift)
                elif belief is not None:
                    for size, y in seen:
                        belief = belief.observe(size, y)
                if belief is None:
                    cache[key] = (0.5, 1.0)             # never run: optimism
                else:
                    fits, spend = belief.outlook(n, self.budget)
                    cache[key] = (chance_of(method, failed) * fits, spend)
            return cache[key]

        return step

    def _best_probe(self, n, box, before, left, own, base, chance_of, drift, current, probed
                    ) -> Optional[Decision]:
        best: Optional[Tuple[float, Decision]] = None
        for method in sorted(left):
            belief = base(method, before)
            if belief is None:
                continue                                 # never run: it is applied, not probed
            for size, y in own[method]:
                belief = belief.observe(size, y)
            for size in range(2, n):
                if (method, size) in probed:
                    continue
                spend = belief.outlook(size, self.budget)[1]
                if spend >= current.cost:
                    continue                             # no outcome could repay it
                after, saving, groups = 0.0, 0.0, {}
                for y, weight in belief.outcomes(size):
                    step = self._stepper(n, box, own, base, chance_of, drift,
                                         (method, (size, y)))
                    plan = best_plan(left, before, step, self.value)
                    after += weight * plan.cost
                    if self.worth == CHANGE:
                        kept = plan_cost(current.names, before, step, self.value)
                        saving += weight * (kept.cost - plan.cost)
                    mass, total = groups.get(plan.names, (0.0, 0.0))
                    groups[plan.names] = (mass + weight, total + weight * plan.cost)
                worth = saving if self.worth == CHANGE else current.cost - after
                if worth - spend > 0 and (best is None or worth - spend > best[0]):
                    branches = tuple(Branch(names, mass, total / mass) for names, (mass, total)
                                     in sorted(groups.items(), key=lambda kv: -kv[1][0]))
                    repeat = any(e.kind == PROBE and e.method == method and e.n == size
                                 and e.before == before and e.box != box
                                 for e in self.journal.entries)
                    best = (worth - spend, Decision(box, n, tuple(sorted(before)), current,
                                                    "probe", method, size, worth, spend,
                                                    branches, repeat))
        return best[1] if best else None

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
        self.last_probes, self.last_decisions = [], []
        drift = self.journal.drift()
        pending: Optional[Decision] = None
        while True:
            beliefs: Dict[Tuple[str, FrozenSet[str]], Optional[Belief]] = {}
            chances: Dict[Tuple[str, FrozenSet[str]], float] = {}

            def base(method: str, failed: FrozenSet[str]) -> Optional[Belief]:
                if (method, failed) not in beliefs:
                    beliefs[(method, failed)] = self.belief(method, failed, box, (), drift)
                return beliefs[(method, failed)]

            def chance_of(method: str, failed: FrozenSet[str]) -> float:
                if (method, failed) not in chances:
                    chances[(method, failed)] = self.journal.success(method, failed, box)
                return chances[(method, failed)]

            current = best_plan(left, before,
                                self._stepper(n, box, own, base, chance_of, drift), self.value)
            if pending is not None:
                pending.after, pending = current.names, None
            probe = self._best_probe(n, box, before, left, own, base, chance_of, drift,
                                     current, probed)
            if probe is not None:
                belief = base(probe.method, before)
                for size, y in own[probe.method]:
                    belief = belief.observe(size, y)
                mu, sd = belief.predict(probe.size)
                found = self._probe(session, probe.method, probe.size, levels, seed, n)
                probed.add((probe.method, probe.size))
                self.probes += 1
                probe.actual = found.queries
                self.last_probes.append((probe.method, probe.size, found.queries))
                self.last_decisions.append(probe)
                pending = probe
                own[probe.method].append((probe.size, math.log(max(found.queries, 1))))
                if learn:
                    self.journal.add(Entry(PROBE, box, probe.size, before, probe.method,
                                           found.believed, found.model is not None,
                                           found.queries, mu, sd, belief.distance(probe.size)))
                continue
            if not current.steps:
                self.last_decisions.append(Decision(box, n, tuple(sorted(before)), current,
                                                    "give up"))
                break
            pick = current.steps[0].method
            belief = base(pick, before)
            if belief is not None:
                for size, y in own[pick]:
                    belief = belief.observe(size, y)
            mu, sd = belief.predict(n) if belief is not None else (0.0, 0.0)
            distance = belief.distance(n) if belief is not None else 0.0
            found = solvers.run(session, pick, seed, self.budget, self.registry)
            self.last_decisions.append(Decision(box, n, tuple(sorted(before)), current, "apply",
                                                pick, actual=found.queries))
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


# --------------------------------------------------------------------------
# M1d: laws of cost, told apart by experience
# --------------------------------------------------------------------------
#
# M1c showed the value of information working, weighed on beliefs of the
# wrong shape: one straight line of log(questions) against n per method,
# where a method may be cheap and then explode. Here only the belief
# changes; the Planner is the Planner.
#
# The language of laws is small and says nothing of what any law is for:
#
#     one line          log(questions) = a + b n
#     two lines         joined at one of the sizes seen, the second
#                       sloping differently: a + b n + d max(0, n - c)
#
# Which law holds is chosen in the currency used everywhere else: how well
# the law predicts the runs, less what it costs to state. The Bayesian
# evidence of each law is exactly that -- the fit, and an Occam factor for
# every parameter the data had to pay for -- and a law's prior is one bit
# for its class plus log2 K for which of the K sizes it bends at. A run
# that ran out of its budget says only that the cost was at least the
# budget: it weighs the laws by how likely they made that, and is not
# fitted as if it had cost the budget exactly.
#
# The belief is the laws together, each weighted by its evidence: what a
# run at a size may cost is a mixture, and a probe's outcome reweights the
# laws as well as fitting them. A box's own runs move its level against its
# kind's, as the Explorer's did.

#: Prior sd of the change of slope at a bend, in nats per input.
BEND_PRIOR = 2.5
#: Nodes per law when a run's outcome is imagined.
_LAW_NODES, _LAW_WEIGHTS = np.polynomial.hermite_e.hermegauss(5)
_LAW_WEIGHTS = _LAW_WEIGHTS / _LAW_WEIGHTS.sum()
#: Laws weighing less than this are left out of imagined outcomes.
_LAW_FLOOR = 0.01


def _above(mean: float, sd: float, cap: float) -> float:
    """E[y | y >= cap] for y ~ N(mean, sd)."""
    alpha = (cap - mean) / sd
    tail = 1.0 - _PHI(alpha)
    ratio = (NormalDist().pdf(alpha) / tail) if tail > 1e-10 else alpha + 1.0 / alpha
    return mean + sd * ratio


@dataclass
class _Law:
    bend: Optional[float]
    prior: float                 # log prior
    spread2: float               # variance between boxes
    xtwx: np.ndarray             # the kind's runs, as sufficient statistics
    xtwy: np.ndarray
    ytwy: float
    lognorm: float               # sum of log(2 pi sigma^2) over those runs
    count: int
    censored: Tuple[Tuple[float, bool], ...]    # (size, own) of runs cut off
    mean: np.ndarray = None
    cov: np.ndarray = None
    evidence: float = 0.0


class ShapeBelief:
    """What one method will cost on one box: laws of cost weighed by the
    evidence of the runs."""

    def __init__(self, laws: List[_Law], sizes: FrozenSet[float], drift: float,
                 budget: float) -> None:
        self.laws = laws
        self.sizes = sizes
        self.drift = drift
        self.budget = budget
        for law in laws:
            self._solve(law)
        top = max(law.evidence for law in laws)
        weights = np.array([math.exp(law.evidence - top) for law in laws])
        self.weights = weights / weights.sum()

    @staticmethod
    def _x(n: float, bend: Optional[float], own: bool) -> np.ndarray:
        row = [1.0, n] + ([max(0.0, n - bend)] if bend is not None else []) + [1.0 if own else 0.0]
        return np.array(row)

    @classmethod
    def fit(cls, rows: Sequence[Tuple[float, float, bool]], drift: float,
            budget: float) -> "ShapeBelief":
        """rows: (size, log questions, cut off by the budget) of the kind's
        runs on other boxes."""
        sizes = sorted({n for n, _, _ in rows})
        bends = [None] + sizes[1:-1]
        laws = []
        for bend in bends:
            k = 1 if bend is None else len(bends) - 1
            prior = math.log(0.5) - (0.0 if bend is None else math.log(k))
            done = [(n, y) for n, y, cut in rows if not cut]
            cut = tuple((n, False) for n, _, c in rows if c)
            X = np.array([cls._x(n, bend, False) for n, _ in done]) if done else np.zeros((0, 2 + (bend is not None) + 1))
            y = np.array([v for _, v in done])
            spread2 = SPREAD_PRIOR ** 2
            prior_inv = np.linalg.inv(cls._prior(bend, spread2))
            for _ in range(2):
                if not done:
                    break
                cov = np.linalg.inv(X.T @ X / spread2 + prior_inv)
                mean = cov @ (X.T @ y / spread2)
                rss = float(np.sum((y - X @ mean) ** 2))
                spread2 = (SPREAD_WEIGHT * SPREAD_PRIOR ** 2 + rss) / (SPREAD_WEIGHT + len(done))
                prior_inv = np.linalg.inv(cls._prior(bend, spread2))
            laws.append(_Law(bend, prior, spread2, X.T @ X / spread2, X.T @ y / spread2,
                             float(y @ y) / spread2,
                             len(done) * math.log(2 * math.pi * spread2), len(done), cut))
        return cls(laws, frozenset(sizes), drift, budget)

    @staticmethod
    def _prior(bend: Optional[float], spread2: float) -> np.ndarray:
        scales = [10.0 ** 2, SLOPE_PRIOR ** 2] + ([BEND_PRIOR ** 2] if bend is not None else [])
        return np.diag(scales + [spread2])

    def _solve(self, law: _Law) -> None:
        """The law's evidence, from the runs that finished and the chance it
        gave the runs cut off; then its parameters, with each cut-off run
        standing at what it is expected to have cost, given that it cost at
        least the budget -- a bound, refitted a few times, not a cost."""
        prior = self._prior(law.bend, law.spread2)
        prior_inv = np.linalg.inv(prior)
        cov = np.linalg.inv(law.xtwx + prior_inv)
        mean = cov @ law.xtwy
        fit = law.ytwy - 2 * float(mean @ law.xtwy) + float(mean @ law.xtwx @ mean)
        evidence = (-0.5 * law.lognorm - 0.5 * fit - 0.5 * float(mean @ prior_inv @ mean)
                    - 0.5 * np.linalg.slogdet(prior)[1] + 0.5 * np.linalg.slogdet(cov)[1])
        cap = math.log(self.budget)
        cut = [(self._x(n, law.bend, own), OBSERVED ** 2 if own else law.spread2)
               for n, own in law.censored]
        for x, noise in cut:
            sd = math.sqrt(float(x @ cov @ x) + noise)
            evidence += math.log(max(1.0 - _PHI((cap - float(x @ mean)) / sd), 1e-12))
        for _ in range(4 if cut else 0):
            xtwx, xtwy = law.xtwx.copy(), law.xtwy.copy()
            for x, noise in cut:
                y = _above(float(x @ mean), math.sqrt(float(x @ cov @ x) + noise), cap)
                xtwx += np.outer(x, x) / noise
                xtwy += y * x / noise
            cov = np.linalg.inv(xtwx + prior_inv)
            mean = cov @ xtwy
        law.mean, law.cov, law.evidence = mean, cov, evidence + law.prior

    def observe(self, n: float, y: float) -> "ShapeBelief":
        """This box's own run at size n: its level, and the laws' weights."""
        laws = []
        cut = y >= math.log(self.budget) - 1e-9
        for law in self.laws:
            if cut:
                laws.append(_Law(law.bend, law.prior, law.spread2, law.xtwx, law.xtwy, law.ytwy,
                                 law.lognorm, law.count, law.censored + ((n, True),)))
                continue
            x = self._x(n, law.bend, True)
            w = 1.0 / OBSERVED ** 2
            laws.append(_Law(law.bend, law.prior, law.spread2, law.xtwx + w * np.outer(x, x),
                             law.xtwy + w * y * x, law.ytwy + w * y * y,
                             law.lognorm + math.log(2 * math.pi * OBSERVED ** 2),
                             law.count + 1, law.censored))
        return ShapeBelief(laws, self.sizes | {float(n)}, self.drift, self.budget)

    def distance(self, n: float) -> float:
        return float(min((abs(n - s) for s in self.sizes), default=n))

    def _parts(self, n: float) -> List[Tuple[float, float, float]]:
        """(weight, mean, sd) of each law's log questions for this box at n."""
        extra = OBSERVED ** 2 + (self.drift * self.distance(n)) ** 2
        out = []
        for weight, law in zip(self.weights, self.laws):
            x = self._x(n, law.bend, True)
            out.append((float(weight), float(x @ law.mean), math.sqrt(float(x @ law.cov @ x) + extra)))
        return out

    def predict(self, n: float) -> Tuple[float, float]:
        parts = self._parts(n)
        mu = sum(w * m for w, m, _ in parts)
        var = sum(w * (s * s + m * m) for w, m, s in parts) - mu * mu
        return mu, math.sqrt(max(var, 1e-12))

    def outlook(self, n: float, budget: float) -> Tuple[float, float]:
        fits = spend = 0.0
        cap = math.log(budget)
        for w, m, s in self._parts(n):
            fits += w * _PHI((cap - m) / s)
            spend += w * _worth(m, s, 1.0, 0.0, budget)[1]
        return fits, spend

    def outcomes(self, n: float) -> List[Tuple[float, float]]:
        points = [(m + s * node, w * nw) for w, m, s in self._parts(n) if w >= _LAW_FLOOR
                  for node, nw in zip(_LAW_NODES, _LAW_WEIGHTS)]
        total = sum(p for _, p in points)
        return [(y, p / total) for y, p in points]

    def law(self) -> Dict[str, float]:
        """For reports only: how much weight bends, and where."""
        bent = [(w, law.bend) for w, law in zip(self.weights, self.laws) if law.bend is not None]
        line = float(sum(w for w, law in zip(self.weights, self.laws) if law.bend is None))
        where = max(bent)[1] if bent else None
        return {"line": line, "bend": 1.0 - line, "at": where}


class ShapePlanner(Planner):
    """The Planner, weighing laws of cost instead of one line."""

    def belief(self, method: str, before: FrozenSet[str], box: int,
               own: Sequence[Tuple[int, float]], drift: float) -> Optional[ShapeBelief]:
        runs = self.journal.runs(method, before, box)
        if not runs and not own:
            return None
        rows = [(float(e.n), math.log(max(e.queries, 1)), not e.finished) for e in runs]
        belief = ShapeBelief.fit(rows, drift, self.budget)
        for n, y in own:
            belief = belief.observe(n, y)
        return belief
