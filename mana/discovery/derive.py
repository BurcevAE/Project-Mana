"""
mana.discovery.derive — how answers were reached, compressed into moves.

Step 6, the breakthrough experiment (docs/АУДИТ_ПОРОЖДЕНИЯ.md). Everything
MANA grew before was about the world: programs, a hidden variable, words
of the language of models. The search that found them stayed Python, out
of reach of anything that grows. Here the object compressed is not an
answer but the way it was found.

A derivation is a program of solving, written in a language that already
existed: the search's own edits. The search keeps, for every program, the
one it was first made from; followed back from an answer, that is a chain
from a starting leaf to the answer, one edit per step.

    x -> if(y == z, x, 9) -> if(y == z, x, y + x) -> ... -> (y + x) - z

Where several steps of several derivations did the same thing to a node,
that thing can become one move -- a macro-edit, n -> template(n, leaves).
The five starting edits are moves of exactly this form, one step long, so
a macro is not a new kind of object: it is the language of the solver
growing from its own use.

A move is kept in the currency everything else is kept in. A derivation
costs, at each step, log2 of how many programs were one edit away -- the
index of the next program in that neighbourhood, and exactly the number
of programs a search has to look at to take that step. A macro makes
every neighbourhood larger, so every step dearer, and collapses the runs
of steps it stands for into one. It is kept when the derivations it
shortens, plus its own definition, cost less than without it.

Nothing here reads a world. It sees derivations and the vocabulary their
edits drew on.

Measured 2026-09-14 (scripts/run_derive.py; the table is in the package
docstring): from 30 derivations of T1..T3 one move was kept, written both
ways round -- n -> |n - #0| -- halving the derivations' bits. On unseen
data of the same family it took a step in 26 of 27 answers and brought
the answer four times sooner; on T4, another surface, it took none.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import description
from .language import ADD, CMP, CONST, GET, IF, SUB, Program, children, nodes, rebuild, show, size
from .library import MAX_HOLES, _same_head, generalise, match
from .search import SELF, Macro

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Longest run of consecutive steps one move may stand for.
MAX_WINDOW = 3
#: Smallest template worth a name: anything smaller is one starting edit.
MIN_TEMPLATE = 4


@dataclass
class Derivation:
    """How one answer was reached, and the vocabulary its edits drew on."""
    programs: List[Program]
    leaves: List[Program]
    conditions: List[Program]
    max_size: int
    #: Which question it answered: moves are sought across questions.
    task: str = ""

    def __post_init__(self) -> None:
        self._leaf_set = set(self.leaves)
        self._basic: Dict[Program, int] = {}


def difference(p: Program, q: Program) -> Optional[Tuple[Tuple[int, ...], Program, Program]]:
    """Where two programs differ: the path to the smallest subtree that
    holds every difference, and that subtree before and after."""
    if p == q:
        return None
    path: Tuple[int, ...] = ()
    while _same_head(p, q) and children(p):
        differing = [i for i, (a, b) in enumerate(zip(children(p), children(q))) if a != b]
        if len(differing) != 1:
            break
        i = differing[0]
        path += (i,)
        p, q = children(p)[i], children(q)[i]
    return path, p, q


def _subtree(p: Program, path: Tuple[int, ...]) -> Program:
    for i in path:
        p = children(p)[i]
    return p


def _abstract(before: Program, after: Program) -> Program:
    """`after`, with the node that was rewritten written as SELF."""
    if after == before:
        return SELF
    kids = children(after)
    if not kids:
        return after
    return rebuild(after, [_abstract(before, kid) for kid in kids])


def _fill_self(template: Program, node: Program) -> Program:
    if template == SELF:
        return node
    kids = children(template)
    if not kids:
        return template
    return rebuild(template, [_fill_self(kid, node) for kid in kids])


def windows(d: Derivation) -> List[Tuple[int, int, Program]]:
    """(start, end, template) for every run of 2..MAX_WINDOW steps: what
    the run did to the one place it changed."""
    out = []
    for i in range(len(d.programs)):
        for j in range(i + 2, min(len(d.programs), i + MAX_WINDOW + 1)):
            diff = difference(d.programs[i], d.programs[j])
            if diff is not None:
                out.append((i, j, _abstract(diff[1], diff[2])))
    return out


def _is_leaf(p: Program) -> bool:
    return p[0] in (GET, CONST)


def applies(macro: Macro, p: Program, q: Program, d: Derivation) -> bool:
    """Is q one application of this macro to p, with leaves of the vocabulary?"""
    diff = difference(p, q)
    if diff is None:
        return False
    path = diff[0]
    for cut in range(len(path), -1, -1):
        where = path[:cut]
        before, after = _subtree(p, where), _subtree(q, where)
        bound = match(_fill_self(macro.template, before), after)
        if (bound is not None and len(bound) == macro.arity
                and all(_is_leaf(v) and v in d._leaf_set for v in bound.values())):
            return True
    return False


def _basic_count(p: Program, d: Derivation) -> int:
    """How many programs the five starting edits make from p."""
    hit = d._basic.get(p)
    if hit is not None:
        return hit
    whole, leaves, conds = size(p), len(d.leaves), len(d.conditions)
    total = 0
    for _, node in nodes(p):
        total += leaves - (1 if node in d._leaf_set else 0)
        if node[0] in (ADD, SUB, CMP):
            total += 2
        elif node[0] == IF:
            total += 3
        if whole + 4 <= d.max_size:
            total += 2 * conds * leaves
        if whole + 2 <= d.max_size:
            total += 3 * leaves
    d._basic[p] = total
    return total


def _macro_count(p: Program, macro: Macro, d: Derivation) -> int:
    whole = size(p)
    selves = sum(1 for _, part in nodes(macro.template) if part == SELF)
    count = 0
    for _, node in nodes(p):
        if whole - size(node) + size(macro.template) + selves * (size(node) - 1) <= d.max_size:
            count += len(d.leaves) ** macro.arity
    return count


def derivation_bits(d: Derivation, macros: Sequence[Macro]) -> Tuple[float, int]:
    """The shortest description of a derivation with these moves, and how
    many steps of it a macro took."""
    programs = d.programs
    cost = [math.log2(max(1, _basic_count(p, d) + sum(_macro_count(p, m, d) for m in macros)))
            for p in programs]
    best: List[Tuple[float, int]] = [(math.log2(max(1, len(d.leaves))), 0)]
    for j in range(1, len(programs)):
        row = (best[j - 1][0] + cost[j - 1], best[j - 1][1])
        for i in range(max(0, j - MAX_WINDOW), j - 1):
            if any(applies(m, programs[i], programs[j], d) for m in macros):
                if best[i][0] + cost[i] < row[0]:
                    row = (best[i][0] + cost[i], best[i][1] + 1)
        best.append(row)
    return best[-1]


def candidates(derivations: Sequence[Derivation]) -> List[Macro]:
    """Moves shared by runs of steps in different derivations."""
    per = [sorted({t for _, _, t in windows(d)}, key=show) for d in derivations]
    found: Dict[Program, int] = {}
    for a, b in itertools.combinations(range(len(per)), 2):
        for x in per[a]:
            for y in per[b]:
                template, arity = generalise(x, y)
                if (arity <= MAX_HOLES and size(template) >= MIN_TEMPLATE
                        and any(part == SELF for _, part in nodes(template))):
                    found.setdefault(template, arity)
    ordered = sorted(found.items(), key=lambda row: (-size(row[0]), show(row[0])))
    return [Macro(f"m{i + 1}", template, arity) for i, (template, arity) in enumerate(ordered)]


@dataclass
class Learned:
    macros: List[Macro]
    #: (name, template, bits saved, steps it took) for every move kept.
    kept: List[Tuple[str, str, float, int]] = field(default_factory=list)
    bits_before: float = 0.0
    bits_after: float = 0.0
    considered: int = 0


def _total(derivations: Sequence[Derivation], macros: Sequence[Macro]) -> Tuple[float, int]:
    bits, uses = 0.0, 0
    for d in derivations:
        b, u = derivation_bits(d, macros)
        bits += b
        uses += u
    return bits, uses


def compress(derivations: Sequence[Derivation], variables: int) -> Learned:
    """Add moves, one at a time, while each makes the derivations shorter."""
    pool = candidates(derivations)
    # A move that shortens no derivation cannot pay; most candidates do not,
    # and are dropped before the greedy rounds.
    pool = [m for m in pool
            if sum(1 for d in derivations
                   if any(applies(m, d.programs[i], d.programs[j], d)
                          for i, j, _ in windows(d))) >= 2]
    chosen: List[Macro] = []
    definitions = 0.0
    before, _ = _total(derivations, chosen)
    now = before
    kept: List[Tuple[str, str, float, int]] = []
    while True:
        best = None
        for macro in pool:
            if macro in chosen:
                continue
            cost = description.primitive_bits(macro.template, variables, len(chosen))
            bits, uses = _total(derivations, chosen + [macro])
            after = bits + definitions + cost
            if best is None or after < best[0]:
                best = (after, macro, cost, uses)
        if best is None or best[0] >= now - 1e-9:
            break
        after, macro, cost, uses = best
        chosen.append(macro)
        definitions += cost
        kept.append((macro.name, show(macro.template), now - after, uses))
        now = after
    return Learned(chosen, kept, before, now, len(pool))
