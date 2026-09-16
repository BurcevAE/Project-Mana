"""
mana.discovery.prior — programs drawn with their prior probability, two to
the minus their description bits (docs/РАСШИРЕНИЕ_ПРОСТРАНСТВА.md, 3.1).

`description.program_bits` is a prefix code: a node names its kind among
KINDS, a variable which of the variables, a constant the Elias-gamma code of
its zig-zag, a comparison which operator. Drawing every symbol of that code
uniformly gives each program exactly 2^-bits. No distribution is designed
here: the measure already is one.

Programs larger than `most` nodes are rejected whole, which keeps the
relative weights of all the others.
"""
from __future__ import annotations

import random
from typing import Sequence

from .language import ADD, CMP, COMPARISONS, CONST, GET, KINDS, SUB, Program, add, cmp, const, get, if_, sub
from .search import MAX_SIZE

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"


class _TooLarge(Exception):
    """The draw outgrew the language's size limit: rejected whole."""


def _integer(rng: random.Random) -> int:
    """An integer with probability 2^-integer_bits: the leading zeros of the
    gamma code, each with probability one half, then as many uniform bits."""
    zeros = 0
    while rng.random() < 0.5:
        zeros += 1
        if zeros > 40:
            raise _TooLarge()
    folded = (1 << zeros) + rng.randrange(1 << zeros) - 1
    return folded // 2 if folded % 2 == 0 else -(folded + 1) // 2


def _draw(variables: Sequence[str], rng: random.Random, room: list) -> Program:
    if room[0] <= 0:
        raise _TooLarge()
    room[0] -= 1
    kind = rng.choice(KINDS)
    if kind == GET:
        return get(rng.choice(variables))
    if kind == CONST:
        return const(_integer(rng))
    if kind == CMP:
        op = rng.choice(COMPARISONS)
        return cmp(op, _draw(variables, rng, room), _draw(variables, rng, room))
    if kind == ADD:
        return add(_draw(variables, rng, room), _draw(variables, rng, room))
    if kind == SUB:
        return sub(_draw(variables, rng, room), _draw(variables, rng, room))
    return if_(_draw(variables, rng, room), _draw(variables, rng, room),
               _draw(variables, rng, room))


def sample(variables: Sequence[str], rng: random.Random, most: int = MAX_SIZE) -> Program:
    """One program of the language, drawn with probability 2^-bits among the
    programs of at most `most` nodes."""
    names = sorted(variables)
    while True:
        try:
            return _draw(names, rng, [most])
        except _TooLarge:
            continue
