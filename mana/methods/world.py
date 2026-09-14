"""
mana.methods.world — a black box with knobs.

A hidden function of n inputs, each taking one of LEVELS values. Asking
the box about an input costs one question; its answer is exact. The goal
of every method is the same: predict the box on inputs it never asked
about, and the box judges that on its own held-out inputs.

The knob is the structure, and each structure is a different answer to
"what kind of thinking fits here":

    linear    intercept + sum of slope_i * x_i
    additive  sum of a table per input: the effects add, but each is
              arbitrary
    blocks    the inputs paired at random, a table per pair: interactions,
              but local
    global    a hash of the whole input: every input interacts with every
              other, and nothing short of the full table describes it

Nothing in `solvers` may import this module: a method that could read
the structure would choose perfectly and mean nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.1"

LEVELS = 10
LINEAR, ADDITIVE, BLOCKS, GLOBAL = "linear", "additive", "blocks", "global"
STRUCTURES = (LINEAR, ADDITIVE, BLOCKS, GLOBAL)
HELD_OUT = 500


def _codes(X: np.ndarray, levels: int = LEVELS) -> np.ndarray:
    return X @ (levels ** np.arange(X.shape[1], dtype=np.int64))


def _mix(codes: np.ndarray, seed: int, levels: int = LEVELS) -> np.ndarray:
    """splitmix64 of the input's code: a value per input, no structure."""
    with np.errstate(over="ignore"):
        z = codes.astype(np.uint64) + np.uint64(seed + 1) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z % np.uint64(levels)).astype(np.int64)


@dataclass
class BlackBox:
    n: int
    structure: str
    seed: int
    #: Values each input takes.
    levels: int = LEVELS
    #: Questions answered, repeats included.
    asked: int = 0
    blocks: List[List[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        rng = np.random.default_rng([self.seed, self.n, STRUCTURES.index(self.structure)])
        if self.structure == LINEAR:
            self._intercept = int(rng.integers(0, self.levels))
            self._slopes = rng.integers(-3, 4, size=self.n)
        elif self.structure == ADDITIVE:
            self._tables = rng.integers(0, self.levels, size=(self.n, self.levels))
        elif self.structure == BLOCKS:
            order = [int(i) for i in rng.permutation(self.n)]
            self.blocks = [sorted(order[i:i + 2]) for i in range(0, self.n, 2)]
            self._block_tables = [rng.integers(0, self.levels, size=(self.levels,) * len(b))
                                  for b in self.blocks]
        elif self.structure != GLOBAL:
            raise ValueError(f"unknown structure {self.structure!r}")

    def _truth(self, X: np.ndarray) -> np.ndarray:
        if self.structure == LINEAR:
            return self._intercept + X @ self._slopes
        if self.structure == ADDITIVE:
            return self._tables[np.arange(self.n)[None, :], X].sum(axis=1)
        if self.structure == BLOCKS:
            out = np.zeros(len(X), dtype=np.int64)
            for block, table in zip(self.blocks, self._block_tables):
                out += table[tuple(X[:, j] for j in block)]
            return out
        return _mix(_codes(X, self.levels), self.seed, self.levels)

    def answer(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.int64))
        self.asked += len(X)
        return self._truth(X)

    def verdict(self, model: Optional[Callable[[np.ndarray], np.ndarray]],
                held_out: int = HELD_OUT) -> float:
        """The share of the box's own held-out inputs the model gets
        exactly right. No model: nothing right."""
        if model is None:
            return 0.0
        X = np.random.default_rng([self.seed, 7919]).integers(0, self.levels, size=(held_out, self.n))
        return float(np.mean(np.asarray(model(X)) == self._truth(X)))
