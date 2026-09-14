"""
mana.discovery.reflect — MANA changes its own search, from its own
experience, and the core decides.

Stage R2. R1 made the search an object (policy.py). Here MANA builds a
change of that object itself: nobody hands it a list of changes to try --
no "change the beam", "change the score", "reorder the rules". One
principle, the one that grew the move of step 6:

    a policy is the language MANA's successful experience is written in.
    The better policy is the one in which that experience -- the
    derivations of its own answers -- is written most briefly, together
    with the definitions of whatever the policy had to add.

Every step of a derivation costs log2 of the number of programs the
policy makes at that point: exactly what a search pays to take the step.
From this one criterion the changes come by themselves, of two kinds and
no others:

    a rule is added      where runs of steps in several derivations did
                         the same thing -- the move of step 6, now a rule
                         of the policy
    a rule is dropped    where the experience never needed it: every step
                         pays for every program each rule makes, so a
                         rule no answer went through only makes the
                         experience longer to write

A rule is kept dropped only if every derivation can still be written
without it. What the generator cannot change, said plainly: the selection
and the resources. How long the experience is to write does not depend on
them, so this principle has nothing to say about them -- the limit of R2,
not a choice.

The experience is MANA's own: derivations of the answers a dear run of the
policy found (its beam widened), on questions and seeds set apart for it.
No answer is shown and no question outside the experience is looked at.
Whether the new policy is better is not decided here: it goes to
core.gates, as every claim does.

Measured 2026-09-14 (scripts/run_reflect.py; the table is in the package
docstring): from the search as it stands and its experience of T1..T3,
MANA added the distance move and dropped "add a condition", "replace by a
leaf" and "swap add". On unseen seeds of the family the new policy solved
28 of 30 from 10k programs (the old: none below 100k, 23 at 400k); on W0,
W3, W4 it lost 20 answers the old one had, and T4 went from 4 of 10 to 0.
core.gates rejected it on significance and counterexamples. A second
starting policy got the same change and shed the idle rule it had been
given. Compressing one family's experience made a specialist.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from . import derive, description
from . import policy as P
from .language import Program
from .search import SELF

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

MAX_WINDOW = derive.MAX_WINDOW


@dataclass
class Experience:
    """One derivation of one answer, and the vocabulary it was made in."""
    programs: List[Program]
    leaves: List[Program]
    conditions: List[Program]
    task: str = ""


class _Cache:
    def __init__(self) -> None:
        self.makes: Dict[Tuple[P.Rule, Program, Program], bool] = {}
        self.sizes: Dict[Tuple[Tuple[P.Rule, ...], Program], int] = {}


def _makes(policy: P.SearchPolicy, rule: P.Rule, p: Program, q: Program,
           e: Experience, cache: _Cache) -> bool:
    key = (rule, p, q)
    if key not in cache.makes:
        cache.makes[key] = P.produces(policy, rule, p, q, e.leaves, e.conditions)
    return cache.makes[key]


def _size(policy: P.SearchPolicy, p: Program, e: Experience, cache: _Cache) -> int:
    key = (policy.rules, p)
    if key not in cache.sizes:
        cache.sizes[key] = P.neighbourhood(policy, p, e.leaves, e.conditions)
    return cache.sizes[key]


def written(policy: P.SearchPolicy, e: Experience, cache: _Cache) -> float:
    """Bits to write one derivation in this policy; infinite if it cannot."""
    programs = e.programs
    cost = [math.log2(max(1, _size(policy, p, e, cache))) for p in programs]
    best = [math.log2(max(1, len(e.leaves)))] + [math.inf] * (len(programs) - 1)
    for j in range(1, len(programs)):
        for i in range(max(0, j - MAX_WINDOW), j):
            if best[i] + cost[i] >= best[j]:
                continue
            if any(_makes(policy, rule, programs[i], programs[j], e, cache)
                   for rule in policy.rules):
                best[j] = best[i] + cost[i]
    return best[-1]


def total(policy: P.SearchPolicy, experience: Sequence[Experience], added: float,
          cache: _Cache) -> float:
    return added + sum(written(policy, e, cache) for e in experience)


@dataclass
class Change:
    before: P.SearchPolicy
    after: P.SearchPolicy
    #: (what was done, bits saved) in order.
    steps: List[Tuple[str, float]] = field(default_factory=list)
    bits_before: float = 0.0
    bits_after: float = 0.0
    considered: int = 0


def improve(policy: P.SearchPolicy, experience: Sequence[Experience],
            variables: int = 3) -> Change:
    """The policy in which the experience is written most briefly: add or
    drop one rule at a time while that shortens it."""
    cache = _Cache()
    as_derivations = [derive.Derivation(e.programs, e.leaves, e.conditions, policy.max_size,
                                        e.task) for e in experience]
    pool = [(P.macro_rule(f"grown {m.name}", m.template, m.arity, SELF),
             description.primitive_bits(m.template, variables, 0))
            for m in derive.candidates(as_derivations)]
    current, added = policy, 0.0
    before = total(current, experience, added, cache)
    now = before
    steps: List[Tuple[str, float]] = []
    while True:
        options = []
        for rule, cost in pool:
            if rule not in current.rules:
                options.append((f"добавлено правило {rule.name}: "
                                f"{P._show_template(rule.lhs)} -> {P._show_template(rule.rhs[0])}",
                                P.with_rule(current, rule), added + cost))
        for rule in current.rules:
            options.append((f"убрано правило «{rule.name}»", P.without(current, rule.name), added))
        best = None
        for what, candidate, cost in options:
            bits = total(candidate, experience, cost, cache)
            if best is None or bits < best[0]:
                best = (bits, what, candidate, cost)
        if best is None or best[0] >= now - 1e-9:
            break
        bits, what, current, added = best
        steps.append((what, now - bits))
        now = bits
    return Change(policy, current, steps, before, now, len(pool) + len(policy.rules))
