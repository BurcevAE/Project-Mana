"""
mana.discovery.frontier — which states of a search are worth keeping,
learnt from the search's own history.

Step 7. The control of step 6 showed where answers are lost: T4's rule, a
way to it, and the state it is reached from all exist, and the beam --
ranked by each state's description length alone -- drops that state.
Keeping it by hand solved T4 on 10 seeds of 10 instead of 4. The question
here is whether what makes a state worth keeping can be learnt instead of
fixed, from nothing but the search's own successful traces.

What is fixed, said plainly -- this is not yet a measure MANA invents:

    the space of scores   weighted sums of four things the search already
                          computes about every state: its program's bits,
                          its errors' bits, its size, and how many points
                          it gets wrong. The measure as it stands is one
                          point of that space: bits, (1, 1, 0, 0)
    what a score decides  only which states the beam keeps. The answer is
                          still the shortest description of everything
                          evaluated, judged by the same gates

What is learnt: the weights. The teacher is the search's own history,
bought with an expensive search and read afterwards. At every step of a
derivation the frontier made a decision -- of the programs made from the
same parent, the one that led on to the answer is known now. A score is
good where it puts that one above its siblings, and a step counts more
the closer it stood to the answer (1 / (1 + edits still to go)). The
answer itself is never shown: only which of its own states led to it.

Nothing here reads a world.

Measured 2026-09-14 (the table is in the package docstring): on the
history of T1..T3 the measure as it stands already put the state that led
on first among its siblings at the median, and ordered every decision
right; the learnt weights -- mostly "small and short", almost nothing on
errors -- ordered them right too, and in a search solved 20 of 70 against
54, no better than the same weights permuted. A successful history holds
only the frontier's successes; what it throws away never reaches it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from . import description
from .language import Evaluator, Program, size
from .search import MAX_SIZE, neighbours, vocabulary

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: What a score may read of a state -- what the search computes anyway.
FEATURES = ("program bits", "error bits", "size", "wrong points")
#: The measure as it stands: bits, program plus errors.
BASE = np.array([1.0, 1.0, 0.0, 0.0])
#: Siblings sampled at each step of a derivation.
SIBLINGS = 1500


@dataclass
class Score:
    """A weighted sum of a state's features, lower kept first. Weights are
    held on standardised features, so that a control can permute them
    between features without mixing units."""
    name: str
    weights: np.ndarray
    scale: np.ndarray

    def __call__(self, program_bits: float, error_bits: float, size_: int,
                 wrong: int) -> float:
        x = np.array([program_bits, error_bits, size_, wrong], dtype=float)
        return float((x / self.scale) @ self.weights)

    def value(self, features: np.ndarray) -> np.ndarray:
        return (features / self.scale) @ self.weights

    def raw(self) -> np.ndarray:
        """Weights on the features as the search computes them."""
        return self.weights / self.scale


@dataclass
class Step:
    """One decision of a frontier on the way to an answer: the state that
    led on, and the other states made from the same parent."""
    chosen: np.ndarray
    others: np.ndarray
    weight: float
    #: Where the measure as it stands put the chosen state among them, and
    #: how many there were in all.
    base_rank: int
    made: int


def _features(programs: Sequence[Program], evaluator: Evaluator, actual: np.ndarray,
              alphabet: int, known, variables: int) -> np.ndarray:
    rows = []
    for p in programs:
        predicted = evaluator(p)
        rows.append([description.program_bits(p, variables),
                     description.error_bits(predicted, actual, alphabet, known),
                     size(p), int(np.count_nonzero(predicted != actual))])
    return np.array(rows, dtype=float)


def steps(derivation: Sequence[Program], columns, outcomes, seed: int = 0,
          siblings: int = SIBLINGS) -> List[Step]:
    """The frontier's decisions along one derivation."""
    actual = np.asarray(outcomes, dtype=np.int64)
    evaluator = Evaluator(columns)
    variables = len(columns)
    alphabet = description.alphabet_of(actual)
    known = description.membership(actual)
    leaves, conditions = vocabulary(columns, actual)
    rng = np.random.default_rng([seed, 29])
    out: List[Step] = []
    k = len(derivation) - 1
    for i in range(1, k + 1):
        parent, chosen = derivation[i - 1], derivation[i]
        made = [q for q in dict.fromkeys(neighbours(parent, leaves, conditions, MAX_SIZE))
                if q != chosen]
        if not made:
            continue
        pick = sorted(rng.choice(len(made), size=min(siblings, len(made)), replace=False))
        others = _features([made[j] for j in pick], evaluator, actual, alphabet, known, variables)
        mine = _features([chosen], evaluator, actual, alphabet, known, variables)[0]
        below = int(np.sum(others @ BASE < mine @ BASE))
        rank = 1 + round(below * len(made) / len(pick))
        out.append(Step(mine, others, 1.0 / (1 + k - i), rank, len(made)))
    return out


def base(scale: np.ndarray) -> Score:
    return Score("как есть", BASE * scale, scale)


def accuracy(score: Score, data: Sequence[Step]) -> float:
    """Weighted share of sibling pairs the score puts in the order the
    history showed: the state that led on first."""
    total = sum(s.weight for s in data)
    right = sum(s.weight * float(np.mean(score.value(s.chosen[None, :])[0] < score.value(s.others)))
                for s in data)
    return right / total


def fit(data: Sequence[Step], iterations: int = 600, rate: float = 0.5,
        l2: float = 1e-2) -> Score:
    """Weights that put the state that led on above its siblings, from the
    measure as it stands. Pairwise logistic loss, plain gradient descent,
    deterministic."""
    pooled = np.vstack([s.others for s in data] + [s.chosen[None, :] for s in data])
    scale = pooled.std(axis=0) + 1e-9
    v = BASE * scale
    v = v / np.linalg.norm(v)
    diffs = [((s.chosen[None, :] - s.others) / scale, s.weight) for s in data]
    total = sum(w for _, w in diffs)
    for _ in range(iterations):
        grad = l2 * v
        for d, w in diffs:
            margin = np.clip(d @ v, -30, 30)
            grad += w * ((1.0 / (1.0 + np.exp(-margin)))[:, None] * d).mean(axis=0) / total
        v = v - rate * grad
    return Score("выучена", v, scale)


def permuted(score: Score, seed: int) -> Score:
    """The learnt weights, moved onto other features: the same sizes, no
    lesson."""
    rng = np.random.default_rng([seed, 31])
    while True:
        order = rng.permutation(len(score.weights))
        if not np.array_equal(order, np.arange(len(score.weights))):
            return Score("переставлена", score.weights[order], score.scale)


def random_like(score: Score, seed: int) -> Score:
    rng = np.random.default_rng([seed, 37])
    v = rng.normal(size=len(score.weights))
    return Score("случайная", v * np.linalg.norm(score.weights) / np.linalg.norm(v), score.scale)
