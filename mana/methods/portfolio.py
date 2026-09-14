"""
mana.methods.portfolio — the controls: choosing a method without
knowing anything.

    cascade   try the methods in a fixed order, stop at the first whose
              own check passes. The order ORDER is the cheapest announced
              plan first -- the trivial rule any meta-knowledge must beat
    oracle    the cheapest method the world's verdict says is right, as if
              it were known in advance; when none is, it spends nothing

A cascade remembers what it asked: a question one method asked is not
paid again by the next, so its cost is the union of what the methods it
tried asked. Every question counts, including those of methods that
declined or were wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from .solvers import Attempt

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

ORDER = ("calculate", "decompose", "experiment", "exhaust")


@dataclass
class Choice:
    method: Optional[str]
    cost: int
    tried: Tuple[str, ...]


def cascade(attempts: Dict[str, Attempt], order: Sequence[str] = ORDER) -> Choice:
    known: set = set()
    tried = []
    for name in order:
        known.update(attempts[name].points.tolist())
        tried.append(name)
        if attempts[name].believed:
            return Choice(name, len(known), tuple(tried))
    return Choice(None, len(known), tuple(tried))


def oracle(attempts: Dict[str, Attempt], verdicts: Dict[str, float]) -> Choice:
    right = [a for a in attempts.values() if verdicts[a.method] == 1.0]
    if not right:
        return Choice(None, 0, ())
    best = min(right, key=lambda a: (a.queries, a.method))
    return Choice(best.method, best.queries, (best.method,))
