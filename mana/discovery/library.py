"""
mana.discovery.library — the language grows a word when a word pays.

Programs found for different questions sometimes contain the same piece,
up to which variables it is applied to. Such a piece becomes a primitive:
a template with holes, defined once, called by name. It is kept only in
the currency everything else is kept in -- the programs rewritten with it,
plus its definition, must be shorter than the programs without it. A word
used once never pays for its definition; a word used everywhere pays many
times over. Nobody decides which pieces deserve names.

How a template is found: two subtrees, from programs found for different
questions, are generalised against each other -- where they agree the
template keeps the node, where two leaves differ it puts a hole (at most
two). The same two leaves meeting twice get the same hole, so
if(y < x, x - y, y - x) against if(z < y, y - z, z - y) gives
if(#1 < #0, #0 - #1, #1 - #0) -- a function of two arguments, found, not
written.

The limit, said plainly: only structure written the same way can be
shared. Where the search expressed |x - y| < 3 as two comparisons, no
template of |x - y| can match it; the library learns what the search
already happened to say alike.

Measured 2026-09-13 (scripts/run_library.py, 5 seeds; solve every question
of the family, compress what was found for T1..T4, solve again; W4 is the
same kind of question over other names and never compressed from)
-------------------------------------------------------------------------
    solved without -> with the grown language, and the median count of
    programs before the answer was first seen:

        T1  |x - y|              4/5 -> 4/5     87 712 -> 15
        T2  |y - z| + x          2/5 -> 3/5    286 046 -> 32 161
        T3  |x - z| + y          3/5 -> 3/5    280 527 -> 125 969
        T4  if(|x-y| < 3, z, y)  3/5 -> 1/5    138 816 -> 172 882
        W4  |q - r| + p          2/5 -> 3/5    286 046 -> 216 096
        W0  control              5/5 -> 5/5        417 -> 516

    Fourteen of twenty-five solved either way: what the word won on T2
    and W4 it lost on T4.

What that is made of:
  * The general word, if(#0 < #1, #1 - #0, #0 - #1), was learned on two
    seeds of five. On two others the bits chose a narrower one with a
    variable baked in -- |#0 - y|, |#0 - x| -- which is shorter inside the
    family and means nothing over p, q, r. Description length rewards
    what repeats in what was seen, not what would carry elsewhere; nothing
    here pays for generality.
  * Where the general word existed, it carried: W4 solved on seed 0 where
    the starting language could not, and found seventeen times sooner on
    seed 1.
  * On T4 the word misleads. The shortest answer says |x - y| < 3 as two
    comparisons; with a distance in the vocabulary the search goes through
    it and does not reach the comparison within the budget.
  * On one seed of five nothing was learned: the starting search failed
    three questions of the family and there was nothing shared to compress.
    A library is only as good as what the search already found.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import description
from .language import (CMP, HOLE, PRIM, Primitive, Program, children, hole,
                       nodes, prim, rebuild, show, size)

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Smallest piece worth naming, and most holes a template may have.
MIN_FRAGMENT = 4
MAX_HOLES = 2


def _same_head(a: Program, b: Program) -> bool:
    if a[0] != b[0] or len(children(a)) != len(children(b)):
        return False
    if a[0] == CMP:
        return a[1] == b[1]
    if a[0] == PRIM:
        return a[1] == b[1]
    return True


def generalise(a: Program, b: Program) -> Tuple[Program, int]:
    """The template two programs share, with a hole where leaves differ."""
    holes: Dict[Tuple[Program, Program], int] = {}

    def walk(x: Program, y: Program) -> Program:
        if x == y:
            return x
        if _same_head(x, y) and children(x):
            return rebuild(x, [walk(cx, cy) for cx, cy in zip(children(x), children(y))])
        if (x, y) not in holes:
            holes[(x, y)] = len(holes)
        return hole(holes[(x, y)])

    return walk(a, b), len(holes)


def _solid(template: Program) -> int:
    """Nodes that are not holes: how much of the template is structure."""
    return sum(1 for _, node in nodes(template) if node[0] != HOLE)


def candidates(programs: Sequence[Program]) -> List[Tuple[Program, int]]:
    """Templates shared by pieces of different programs."""
    pieces = [[node for _, node in nodes(p) if size(node) >= MIN_FRAGMENT]
              for p in programs]
    found: Dict[Program, int] = {}
    for i, j in itertools.combinations(range(len(pieces)), 2):
        for a in pieces[i]:
            for b in pieces[j]:
                template, arity = generalise(a, b)
                if arity <= MAX_HOLES and _solid(template) >= MIN_FRAGMENT - 1:
                    found.setdefault(template, arity)
    return sorted(found.items(), key=lambda row: (-size(row[0]), show(row[0])))


def match(template: Program, node: Program,
          bound: Optional[Dict[int, Program]] = None) -> Optional[Dict[int, Program]]:
    """Hole bindings that make the template equal to this node, or None."""
    bound = {} if bound is None else bound
    if template[0] == HOLE:
        index = template[1]
        if index in bound:
            return bound if bound[index] == node else None
        bound[index] = node
        return bound
    if not _same_head(template, node):
        return None
    if not children(template):
        return bound if template == node else None
    for ct, cn in zip(children(template), children(node)):
        if match(ct, cn, bound) is None:
            return None
    return bound


def rewrite(p: Program, word: Primitive) -> Program:
    """The program with every piece the template matches called by name."""
    bound = match(word.template, p)
    if bound is not None and len(bound) == word.arity:
        return prim(word.name, *[rewrite(bound[i], word) for i in range(word.arity)])
    kids = children(p)
    if not kids:
        return p
    return rebuild(p, [rewrite(kid, word) for kid in kids])


@dataclass
class Compressed:
    library: Dict[str, Primitive]
    programs: List[Program]
    #: (name, template, bits saved) for every word kept, in order.
    kept: List[Tuple[str, str, float]] = field(default_factory=list)
    bits_before: float = 0.0
    bits_after: float = 0.0


def _total(programs: Sequence[Program], variables: int, library: int,
           definitions: float) -> float:
    return definitions + sum(description.program_bits(p, variables, library)
                             for p in programs)


def compress(programs: Sequence[Program], variables: int) -> Compressed:
    """Add words, one at a time, while each makes everything shorter.

    Rewriting keeps what every program computes, so the errors do not
    move and only the programs and the definitions are counted.
    """
    library: Dict[str, Primitive] = {}
    current = list(programs)
    definitions = 0.0
    before = _total(current, variables, 0, 0.0)
    kept: List[Tuple[str, str, float]] = []
    while True:
        now = _total(current, variables, len(library), definitions)
        best = None
        for template, arity in candidates(current):
            name = f"f{len(library) + 1}"
            word = Primitive(name, template, arity)
            rewritten = [rewrite(p, word) for p in current]
            if rewritten == current:
                continue
            cost = description.primitive_bits(template, variables, len(library))
            after = _total(rewritten, variables, len(library) + 1, definitions + cost)
            if best is None or after < best[0]:
                best = (after, word, rewritten, cost)
        if best is None or best[0] >= now - 1e-9:
            break
        after, word, rewritten, cost = best
        library[word.name] = word
        current = rewritten
        definitions += cost
        kept.append((word.name, show(word.template), now - after))
    return Compressed(library, current, kept, before,
                      _total(current, variables, len(library), definitions))
