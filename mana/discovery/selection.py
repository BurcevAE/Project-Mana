"""
mana.discovery.selection — which states the beam keeps, in a language made
of the interpreter's own operations.

Probe P1 of N3 (docs/АУДИТ_ПОРОЖДЕНИЯ.md). The policy language of R1 can
say only one kind of selection: order the pool by a score of each state
on its own, keep the first k. Steps 7 and 7b found that kind is the wrong
one for the decision a frontier makes. Before any search for another
kind, P1 asks whether a language assembled from nothing but what the
interpreter already does can express a selection of another type at all,
and whether one written by hand does better -- an existence proof, as the
kept state of step 6's control was.

The language -- every operation is one the interpreter already performs:

    collections   the pool; order by a key (ties, as ever, by size and
                  text); the first k; the first; a collection without
                  another; one appended; repeat n times with an
                  accumulator (the interpreter's own rounds)
    one state     its score (the policy's weighted sum); its answers on
                  the data; where it is right -- its answers compared with
                  the data, which the interpreter computes to count errors
    vectors       pointwise and, or, not, not-equal, and a count: the same
                  pointwise operations the DSL evaluator is made of
                  (cmp, if); and the or of where a collection is right
    successors    a state's successors under the policy's rules -- what the
                  interpreter makes of it every round, and the one
                  operation that reaches a state's future. Left out of P1 by
                  mistake, restored in P1b. Every successor a selection
                  looks at is evaluated like any program, counted in the
                  same budget, and can be the answer

R1's selection is the program `take 4 (order by score (pool))`, and it
reproduces the search exactly. The two programs of another kind below are
written by hand, as a diagnostic: nothing of them enters an N3 experiment.

A selection is of another kind than R1's when no score of a state on its
own can reproduce it: remove one of the states it kept, choose again, and
a top-k by any score keeps all the others and adds the next; a selection
that decides about the set may change them. `reversals` looks for that.

Nothing here reads a world.

Measured 2026-09-14 (scripts/run_p1.py): coverage and diversity are of
another kind (reversals in 3 and 6 of 9 rounds; R1 in none), and neither
does better -- T4 4 and 0 of 10 against R1's 4, the other six questions
27 and 24 of 60 against 40, most of the run spent choosing. The state
that leads to T4's rule is set apart by where it leads, not by what it
gets right; this language reads only the present of a state. One of the
interpreter's operations was left out of it -- making a state's
successors -- and it is the only one that reaches a state's future.

P1b, measured 2026-09-14 (scripts/run_p1b.py), with successors restored
and every look paid for in the search's budget: a look one step ahead of
the first 32 or 256 states is a score of one state (no reversals), and
it spent 98-100% of the budget looking -- T4 0 of 10, the other six 29
and 20 of 60, against R1's 4 and 40. The state that leads to T4's rule
was in the pool and never kept: hundreds deep by score, and a look costs
a whole neighbourhood of about 6 000 programs.

P2, measured 2026-09-14 (scripts/run_p2.py): the same look, its
successors made with narrower rules (policy.Selection.ahead) -- A' of R2,
without "add a condition". A look of the first 256 then solved T4 on 9 of
10 by 400k against R1's 4, and so did the look through A' with its grown
move taken out; through R2b', the same move with the width kept, 0. The
width of the look decided, not the move.

P2c, declared before the run and measured 2026-09-14
(scripts/run_look_claim.py): on fresh seeds core.gates ACCEPTED the look
through A' as a better search for T4 at 400k (dev 0.37 -> 0.90, hidden
0.27 -> 0.83, no counterexample in 60) and REJECTED the look without the
move on 9 counterexamples in the family.

P2d, a diagnostic (scripts/run_look_anatomy.py): in every solved run of
P2c's dev seeds the exact answer was first evaluated by a look, from a
state earlier picks of the look had carried the way to; a pick and its
dropped twin differ ten times more one step on than by their own score.
What a look finds is computed here and thrown away: nothing in MANA's
experience records it.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.1"

S, KEPT, T = ("var", "s"), ("var", "kept"), ("var", "t")
POOL = ("pool",)


def key(*exprs) -> tuple:
    return ("key", "s", tuple(exprs))


def by_score() -> tuple:
    return key(("score", S))


#: R1's selection, written in the language: order by score, keep the first 4.
R1 = ("take", 4, ("sort", by_score(), POOL))

_BEST = ("take", 1, ("sort", by_score(), POOL))

#: Hand-written, a diagnostic. Keep the best by score; then, three times,
#: the state right on the most points the kept ones get wrong.
COVER = ("iterate", 3, _BEST,
         ("lambda", "kept",
          ("append", KEPT,
           ("first", ("sort", key(("neg", ("count", ("and", ("right", S),
                                                      ("not", ("cover", KEPT))))),
                                  ("score", S)),
                      ("without", POOL, KEPT))))))

#: Hand-written, a diagnostic. Keep the best by score; then, three times,
#: the state whose answers differ most from the nearest of the kept ones.
DIVERSE = ("iterate", 3, _BEST,
           ("lambda", "kept",
            ("append", KEPT,
             ("first", ("sort", key(("neg", ("min_over", KEPT,
                                              ("lambda", "t",
                                               ("count", ("neq", ("answers", S),
                                                          ("answers", T)))))),
                                    ("score", S)),
                        ("without", POOL, KEPT))))))


def look(shortlist: int) -> tuple:
    """Hand-written, a diagnostic (P1b). Keep the best by score; then,
    three times, of the first `shortlist` by score, the state whose best
    successor is best. A score of one state that reads its future -- not a
    decision about the set."""
    return ("iterate", 3, _BEST,
            ("lambda", "kept",
             ("append", KEPT,
              ("first", ("sort", key(("min_over", ("successors", S),
                                      ("lambda", "t", ("score", T))),
                                     ("score", S)),
                         ("without", ("take", shortlist, ("sort", by_score(), POOL)), KEPT))))))


def size_of(program) -> int:
    """Nodes of a selection program, as a DSL program is measured."""
    if not isinstance(program, tuple):
        return 1
    return 1 + sum(size_of(part) for part in program[1:] if isinstance(part, tuple))


def _free(expr, cache: Dict[int, frozenset]) -> frozenset:
    hit = cache.get(id(expr))
    if hit is not None:
        return hit
    kind = expr[0]
    if kind == "var":
        out = frozenset([expr[1]])
    else:
        parts = [p for p in expr[1:] if isinstance(p, tuple)]
        if kind == "key":
            parts = list(expr[2])
        out = frozenset().union(*[_free(p, cache) for p in parts]) if parts else frozenset()
        if kind in ("lambda", "key"):
            out = out - {expr[1]}
    cache[id(expr)] = out
    return out


class _Run:
    def __init__(self, pool: List, ctx) -> None:
        self.pool = pool
        self.ctx = ctx
        self.free: Dict[int, frozenset] = {}

    def ev(self, e, env: Dict, memo: Optional[Dict] = None, bound: Optional[str] = None):
        if memo is not None and bound is not None and bound not in _free(e, self.free):
            if id(e) in memo:
                return memo[id(e)]
            value = self._ev(e, env, None, None)
            memo[id(e)] = value
            return value
        return self._ev(e, env, memo, bound)

    def _ev(self, e, env, memo, bound):
        kind, ctx = e[0], self.ctx
        if kind == "pool":
            return self.pool
        if kind == "var":
            return env[e[1]]
        if kind == "take":
            return self.ev(e[2], env, memo, bound)[:e[1]]
        if kind == "sort":
            coll = self.ev(e[2], env, memo, bound)
            param, exprs = e[1][1], e[1][2]
            inner: Dict[int, object] = {}

            def rank(state):
                local = dict(env)
                local[param] = state
                return tuple(self.ev(x, local, inner, param) for x in exprs) + ctx.tie(state)

            return sorted(coll, key=rank)
        if kind == "first":
            coll = self.ev(e[1], env, memo, bound)
            return coll[0] if coll else None
        if kind == "without":
            drop = set(self.ev(e[2], env, memo, bound))
            return [x for x in self.ev(e[1], env, memo, bound) if x not in drop]
        if kind == "append":
            coll = list(self.ev(e[1], env, memo, bound))
            item = self.ev(e[2], env, memo, bound)
            return coll + ([item] if item is not None else [])
        if kind == "iterate":
            acc = self.ev(e[2], env, memo, bound)
            param, body = e[3][1], e[3][2]
            for _ in range(e[1]):
                local = dict(env)
                local[param] = acc
                acc = self.ev(body, local)
            return acc
        if kind == "score":
            return ctx.score(self.ev(e[1], env, memo, bound))
        if kind == "right":
            return ctx.right(self.ev(e[1], env, memo, bound))
        if kind == "answers":
            return ctx.answers(self.ev(e[1], env, memo, bound))
        if kind == "count":
            return int(np.count_nonzero(self.ev(e[1], env, memo, bound)))
        if kind == "neg":
            return -self.ev(e[1], env, memo, bound)
        if kind == "and":
            return self.ev(e[1], env, memo, bound) & self.ev(e[2], env, memo, bound)
        if kind == "or":
            return self.ev(e[1], env, memo, bound) | self.ev(e[2], env, memo, bound)
        if kind == "not":
            return ~self.ev(e[1], env, memo, bound)
        if kind == "neq":
            return self.ev(e[1], env, memo, bound) != self.ev(e[2], env, memo, bound)
        if kind == "cover":
            out = None
            for state in self.ev(e[1], env, memo, bound):
                right = ctx.right(state)
                out = right.copy() if out is None else (out | right)
            return out if out is not None else np.zeros(ctx.points, dtype=bool)
        if kind == "min_over":
            coll = self.ev(e[1], env, memo, bound)
            param, body = e[2][1], e[2][2]
            values = []
            for t in coll:
                local = dict(env)
                local[param] = t
                values.append(self.ev(body, local))
            return min(values) if values else float("inf")
        if kind == "successors":
            return ctx.successors(self.ev(e[1], env, memo, bound))
        if kind == "const":
            return e[1]
        raise ValueError(f"unknown operation {kind!r}")


def choose(program, pool: List, ctx) -> List:
    """The states a selection program keeps from a pool. `ctx` answers
    score(p), right(p), answers(p), tie(p) and has `points`."""
    return list(_Run(list(pool), ctx).ev(program, {}))


def reversals(program, pool: List, ctx) -> bool:
    """Does removing one kept state change which of the others are kept?
    Never, for a top-k by any score of a state on its own."""
    kept = choose(program, pool, ctx)
    for gone in kept:
        again = choose(program, [p for p in pool if p != gone], ctx)
        if any(p not in again for p in kept if p != gone):
            return True
    return False
