"""
mana.methods.portfolio — the controls: choosing a method without
knowing anything.

    cascade   try the methods in a fixed order, stop at the first whose
              own check passes. The order ORDER is the cheapest announced
              plan first -- the trivial rule any meta-knowledge must beat,
              and, once plans are no longer announced (M1), knowledge
              handed in by whoever wrote the methods
    oracle    the cheapest method the world's verdict says is right, as if
              it were known in advance; when none is, it spends nothing

A cascade remembers what it asked: a question one method asked is not
paid again by the next. `cascade` reckons that from attempts run apart,
as M0 did; `run_order` runs the methods one after another in one session,
which is what M1 compares with.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .solvers import BUDGET, Attempt, Model, Session, run

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.1"

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


def run_order(ask: Callable[[np.ndarray], np.ndarray], n: int, levels: int, seed: int,
              order: Sequence[str] = ORDER, budget: int = BUDGET,
              announce: bool = False) -> Tuple[Choice, Optional[Model], List[Attempt]]:
    """The methods in this order in one session, until one believes itself."""
    session = Session(ask, n, levels, budget, announce)
    tried: List[Attempt] = []
    for name in order:
        tried.append(run(session, name, seed, budget))
        if tried[-1].believed:
            return (Choice(name, session.total, tuple(a.method for a in tried)),
                    tried[-1].model, tried)
    return Choice(None, session.total, tuple(a.method for a in tried)), None, tried


def oracle(attempts: Dict[str, Attempt], verdicts: Dict[str, float]) -> Choice:
    right = [a for a in attempts.values() if verdicts[a.method] == 1.0]
    if not right:
        return Choice(None, 0, ())
    best = min(right, key=lambda a: (a.queries, a.method))
    return Choice(best.method, best.queries, (best.method,))
