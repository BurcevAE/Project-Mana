"""
mana.methods.trap — two stand-ins with the costs of a scaling trap.

Both are right on a linear box: each is calculate, after asking a number
of other questions that gives it the cost curve of a kind of method:

    explosive   cheap while n is small -- 50 n questions up to n = 4 --
                then ten times more for every input: 2 000 at n = 5,
                20 000 at 6, 200 000 at 7, 2 000 000 at 8
    steady      dear while n is small -- 1 000 n -- and no worse later:
                8 000 at n = 8

Stand-ins, said plainly: nothing here is a way of knowing a box that
anyone would write. What they test is the chooser -- whether, having
seen only n = 2..4, where explosive is twenty times cheaper, it believes
that holds at n = 8 or finds out first, and whether it changes its
strategy once it has been wrong. The padding questions are real questions
to the box, each method in its own half of the input space, so one's
padding is never the other's free answers.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from . import solvers

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Values per input on a trap box: enough inputs to pad with at n = 2.
LEVELS = 100


def explosive_cost(n: int) -> int:
    return 50 * n if n <= 4 else 200 * 10 ** (n - 4)


def steady_cost(n: int) -> int:
    return 1000 * n


def _pad(s: solvers.Session, count: int, half: int) -> None:
    """Ask `count` questions from this method's own half of the inputs."""
    size = s.levels ** s.n // 2
    if count > size:
        raise ValueError(f"{count} questions do not fit in half of {s.levels} ** {s.n}")
    start = half * size
    for low in range(start, start + count, solvers.CHUNK):
        codes = np.arange(low, min(start + count, low + solvers.CHUNK), dtype=np.int64)
        s.ask(solvers._decode(codes, s.n, s.levels))


def explosive(s: solvers.Session, rng: np.random.Generator) -> Tuple[solvers.Model, int]:
    _pad(s, explosive_cost(s.n), 0)
    return solvers.calculate(s, rng)


def steady(s: solvers.Session, rng: np.random.Generator) -> Tuple[solvers.Model, int]:
    _pad(s, steady_cost(s.n), 1)
    return solvers.calculate(s, rng)


METHODS: Dict[str, object] = {"explosive": explosive, "steady": steady}
