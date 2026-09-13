"""
mana.discovery.worlds — small worlds whose truth is written down here, and
only here.

An oracle, not a demo: every world keeps its true rule so what a learner
finds can be graded against it -- over every state, not over a sample.
The learner modules (`language`, `description`, `search`, `baselines`) must
never import this one; a test reads their source to hold them to it.

    W0   outcome = if(x > 5, y, z)                x, y, z in 0..9
    W3   W0, and one outcome in ten replaced by a random value

W0 checks that the search finds a short program where one exists. W3
checks that it does not buy back noise with exceptions.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import numpy as np

from .language import Program, cmp, const, get, if_, LESS

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

State = Dict[str, int]


@dataclass(frozen=True)
class Split:
    train: Dict[str, np.ndarray]
    train_outcomes: np.ndarray
    test: Dict[str, np.ndarray]
    #: What the world reported on the held-out states, noise included.
    test_outcomes: np.ndarray
    #: The rule itself on those states, for grading only.
    test_truth: np.ndarray


@dataclass(frozen=True)
class World:
    name: str
    variables: Tuple[str, ...]
    low: int
    high: int
    rule: Callable[[State], int]
    #: The rule in the language, for the oracle comparison only.
    truth: Program
    noise: float = 0.0
    note: str = ""

    @property
    def values(self) -> int:
        return self.high - self.low + 1

    def states(self) -> Dict[str, np.ndarray]:
        grid = list(itertools.product(range(self.low, self.high + 1),
                                      repeat=len(self.variables)))
        return {name: np.array([row[i] for row in grid], dtype=np.int64)
                for i, name in enumerate(self.variables)}

    def _rule_on(self, columns: Dict[str, np.ndarray]) -> np.ndarray:
        n = len(next(iter(columns.values())))
        return np.array([self.rule({name: int(columns[name][i])
                                    for name in self.variables})
                         for i in range(n)], dtype=np.int64)

    def _observed(self, clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        noisy = clean.copy()
        if self.noise:
            hit = rng.random(len(clean)) < self.noise
            noisy[hit] = rng.integers(self.low, self.high + 1, int(hit.sum()))
        return noisy

    def split(self, n_train: int, n_test: int, seed: int) -> Split:
        """Distinct states: the held-out ones were never seen in training."""
        everything = self.states()
        total = len(next(iter(everything.values())))
        rng = np.random.default_rng(seed)
        order = rng.permutation(total)
        chosen_train, chosen_test = order[:n_train], order[n_train:n_train + n_test]
        train = {name: values[chosen_train] for name, values in everything.items()}
        test = {name: values[chosen_test] for name, values in everything.items()}
        train_clean, test_clean = self._rule_on(train), self._rule_on(test)
        return Split(train=train, train_outcomes=self._observed(train_clean, rng),
                     test=test, test_outcomes=self._observed(test_clean, rng),
                     test_truth=test_clean)

    def grade(self, predict: Callable[[Dict[str, np.ndarray]], np.ndarray]) -> float:
        """Agreement with the rule over every state of the world."""
        everything = self.states()
        return float(np.mean(predict(everything) == self._rule_on(everything)))


def _w0(state: State) -> int:
    return state["y"] if state["x"] > 5 else state["z"]


_W0_TRUTH = if_(cmp(LESS, const(5), get("x")), get("y"), get("z"))

W0 = World("W0", ("x", "y", "z"), 0, 9, _w0, _W0_TRUTH,
           note="if(x > 5, y, z)")
W3 = World("W3", ("x", "y", "z"), 0, 9, _w0, _W0_TRUTH, noise=0.10,
           note="W0 и 10% случайных исходов")

WORLDS: Dict[str, World] = {"W0": W0, "W3": W3}


# ------------------------------------------------------------ with a memory

@dataclass(frozen=True)
class Sequences:
    """Episodes, each a row of `steps` observations in time order."""
    columns: Dict[str, np.ndarray]          # name -> (episodes, steps)
    outcomes: np.ndarray                    # what the world reported
    #: For grading only: the rule's own outcome and its hidden state.
    clean: np.ndarray
    hidden: np.ndarray


@dataclass(frozen=True)
class SequenceWorld:
    """A world whose rule reads something the observation does not show.

    W2: x, y in 0..9 and an event a in {0, 1} are seen at every step. A
    switch h is not: it starts off, and every a = 1 flips it. The outcome
    is x while h is on and y while it is off. No program over what is seen
    at one step can do much better than a coin between x and y; the rule
    needs a quantity the language does not have.
    """
    name: str
    variables: Tuple[str, ...]
    steps: int
    press: float
    noise: float = 0.0
    note: str = ""

    def episodes(self, n: int, rng: np.random.Generator) -> Sequences:
        x = rng.integers(0, 10, (n, self.steps))
        y = rng.integers(0, 10, (n, self.steps))
        a = (rng.random((n, self.steps)) < self.press).astype(np.int64)
        hidden = np.cumsum(a, axis=1) % 2          # off at the start; a flips
        clean = np.where(hidden == 1, x, y).astype(np.int64)
        reported = clean.copy()
        if self.noise:
            hit = rng.random(clean.shape) < self.noise
            reported[hit] = rng.integers(0, 10, int(hit.sum()))
        return Sequences(columns={"x": x, "y": y, "a": a}, outcomes=reported,
                         clean=clean, hidden=hidden)

    def split(self, n_train: int, n_test: int, seed: int) -> Tuple[Sequences, Sequences]:
        """Held-out episodes are new episodes: fresh switches, fresh events."""
        rng = np.random.default_rng(seed)
        return self.episodes(n_train, rng), self.episodes(n_test, rng)


W2 = SequenceWorld("W2", ("x", "y", "a"), steps=20, press=0.3,
                   note="скрытый переключатель: a=1 переключает h; исход x при h, иначе y")

SEQUENCE_WORLDS: Dict[str, SequenceWorld] = {"W2": W2}
