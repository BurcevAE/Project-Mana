"""
mana.discovery.language — the programs models are written in, and nothing
about what they mean.

Six kinds of node, integers only:

    get(v)          a variable of the observation
    const(c)        an integer
    cmp(op, a, b)   a < b or a == b, as 1 or 0
    add(a, b)       a + b
    sub(a, b)       a - b
    if(c, a, b)     a where c is non-zero, b elsewhere

A program is a nested tuple, so it is a value: two equal programs are the
same computation, hash alike, and are evaluated once. That is what makes a
search over hundreds of thousands of them affordable -- most candidates
share almost every subtree with the program they were edited from.
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

GET, CONST, CMP, ADD, SUB, IF = "get", "const", "cmp", "add", "sub", "if"
KINDS = (GET, CONST, CMP, ADD, SUB, IF)

LESS, EQUAL = "<", "=="
COMPARISONS = (LESS, EQUAL)

Program = tuple

#: Subtree results kept per evaluator. Bounded: an array per subtree for a
#: few hundred observations is small, but a search visits millions.
MEMO_LIMIT = 40000


def get(name: str) -> Program:
    return (GET, str(name))


def const(value: int) -> Program:
    return (CONST, int(value))


def cmp(op: str, a: Program, b: Program) -> Program:
    if op not in COMPARISONS:
        raise ValueError(f"unknown comparison {op!r}")
    return (CMP, op, a, b)


def add(a: Program, b: Program) -> Program:
    return (ADD, a, b)


def sub(a: Program, b: Program) -> Program:
    return (SUB, a, b)


def if_(cond: Program, then: Program, other: Program) -> Program:
    return (IF, cond, then, other)


def children(p: Program) -> Tuple[Program, ...]:
    kind = p[0]
    if kind in (GET, CONST):
        return ()
    if kind == CMP:
        return (p[2], p[3])
    if kind in (ADD, SUB):
        return (p[1], p[2])
    return (p[1], p[2], p[3])


def rebuild(p: Program, kids: Sequence[Program]) -> Program:
    kind = p[0]
    if kind == CMP:
        return (CMP, p[1], kids[0], kids[1])
    if kind in (ADD, SUB):
        return (kind, kids[0], kids[1])
    if kind == IF:
        return (IF, kids[0], kids[1], kids[2])
    return p


def size(p: Program) -> int:
    return 1 + sum(size(child) for child in children(p))


def nodes(p: Program, path: Tuple[int, ...] = ()) -> Iterator[Tuple[Tuple[int, ...], Program]]:
    """Every subtree with the path to it, root first."""
    yield path, p
    for index, child in enumerate(children(p)):
        yield from nodes(child, path + (index,))


def replace(p: Program, path: Tuple[int, ...], new: Program) -> Program:
    if not path:
        return new
    kids: List[Program] = list(children(p))
    kids[path[0]] = replace(kids[path[0]], path[1:], new)
    return rebuild(p, kids)


def show(p: Program) -> str:
    kind = p[0]
    if kind == GET:
        return p[1]
    if kind == CONST:
        return str(p[1])
    if kind == CMP:
        return f"({show(p[2])} {p[1]} {show(p[3])})"
    if kind == ADD:
        return f"({show(p[1])} + {show(p[2])})"
    if kind == SUB:
        return f"({show(p[1])} - {show(p[2])})"
    return f"if({show(p[1])}, {show(p[2])}, {show(p[3])})"


class Evaluator:
    """Evaluates programs on one data set, remembering every subtree."""

    def __init__(self, columns: Dict[str, Sequence[int]]) -> None:
        self.columns = {name: np.asarray(values, dtype=np.int64)
                        for name, values in columns.items()}
        self.n = len(next(iter(self.columns.values()))) if self.columns else 0
        self._memo: Dict[Program, np.ndarray] = {}

    def __call__(self, p: Program) -> np.ndarray:
        hit = self._memo.get(p)
        if hit is not None:
            return hit
        kind = p[0]
        if kind == GET:
            out = self.columns[p[1]]
        elif kind == CONST:
            out = np.full(self.n, p[1], dtype=np.int64)
        elif kind == CMP:
            left, right = self(p[2]), self(p[3])
            out = (left < right if p[1] == LESS else left == right).astype(np.int64)
        elif kind == ADD:
            out = self(p[1]) + self(p[2])
        elif kind == SUB:
            out = self(p[1]) - self(p[2])
        else:
            out = np.where(self(p[1]) != 0, self(p[2]), self(p[3]))
        if len(self._memo) >= MEMO_LIMIT:
            self._memo.clear()
        self._memo[p] = out
        return out
