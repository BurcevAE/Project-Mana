"""
mana.discovery.description — the one number: bits to describe the data.

Two-part code:

    L = bits(program) + bits(errors | program)

The program is written as a prefix code: every node names its kind, a
variable names which, a constant is an Elias-gamma integer, a comparison
names its operator. The errors are written as how many there are, which
positions, and the right value at each.

Why this and not "accuracy, with a penalty": both halves are in the same
unit, so there is nothing to tune between them. A condition that fixes one
exception costs more bits than the exception it removes; a condition that
fixes a hundred pays for itself many times over. Memorising loses to
structure exactly when there is structure, and to noise it loses always --
which is the property a learner of worlds needs, stated as arithmetic.
"""
from __future__ import annotations

from math import floor, lgamma, log, log2
from typing import Optional, Sequence, Tuple

import numpy as np

from .language import CMP, COMPARISONS, CONST, GET, KINDS, Program, children

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.1"

_LOG2 = log(2.0)


def integer_bits(value: int) -> float:
    """Elias gamma on the zig-zag of an integer: 0 -> 1 bit, 5 -> 7."""
    value = int(value)
    folded = 2 * value if value >= 0 else -2 * value - 1
    return float(2 * floor(log2(folded + 1)) + 1)


def program_bits(p: Program, variables: int) -> float:
    kind = p[0]
    bits = log2(len(KINDS))
    if kind == GET:
        bits += log2(max(1, variables))
    elif kind == CONST:
        bits += integer_bits(p[1])
    elif kind == CMP:
        bits += log2(len(COMPARISONS))
    return bits + sum(program_bits(child, variables) for child in children(p))


def _log2_choose(n: int, k: int) -> float:
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) / _LOG2


def membership(actual: np.ndarray) -> Tuple[int, np.ndarray]:
    """The outcome alphabet as a lookup: (lowest value, is-a-value table).

    Computed once per data set; asking it per program would cost as much
    as the program itself."""
    values = np.unique(np.asarray(actual, dtype=np.int64))
    low = int(values[0]) if len(values) else 0
    table = np.zeros(int(values[-1]) - low + 1 if len(values) else 1, dtype=bool)
    table[values - low] = True
    return low, table


def error_bits(predicted: np.ndarray, actual: np.ndarray, alphabet: int,
               known: Optional[Tuple[int, np.ndarray]] = None) -> float:
    """How many are wrong, which ones, and what each should have been.

    A wrong guess that is itself one of the outcomes leaves K-1 values the
    truth can be; a guess outside them leaves all K. On two outcomes a
    wrong guess inside names the right one for free. Step 1 charged every
    wrong guess log2(K), and on two outcomes that made "always wrong" a
    shorter description than "right seven times in ten" -- found in step 2,
    where the search for a hidden switch returned the constant 2.
    """
    n = int(len(actual))
    wrong_mask = predicted != actual
    wrong = int(np.count_nonzero(wrong_mask))
    low, table = known if known is not None else membership(actual)
    size = max(int(alphabet), int(table.sum()))
    guesses = predicted[wrong_mask] - low
    in_range = (guesses >= 0) & (guesses < len(table))
    inside = int(np.count_nonzero(table[guesses[in_range]]))
    return (log2(n + 1) + _log2_choose(n, wrong)
            + inside * log2(max(1, size - 1))
            + (wrong - inside) * log2(max(2, size)))


def table_bits(entries: int, variables: int, values_per_variable: int,
               alphabet: int) -> float:
    """A lookup table in the same currency: every remembered state and its
    outcome, written out. No errors on what it remembered, by construction."""
    per_entry = (variables * log2(max(2, values_per_variable))
                 + log2(max(2, alphabet)))
    return log2(entries + 1) + entries * per_entry


def total_bits(program_part: float, error_part: float) -> float:
    return float(program_part + error_part)


def alphabet_of(outcomes: Sequence[int]) -> int:
    return len(set(int(value) for value in outcomes))
