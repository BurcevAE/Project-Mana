"""
mana.discovery.policy — the search, as an object MANA can read and change.

Stage R1 (docs/АУДИТ_ПОРОЖДЕНИЯ.md). Everything MANA has grown lives inside
a mechanism written in Python: what a search is, how it makes candidates,
how it compares them, which it keeps. Before asking whether MANA can change
that mechanism, the question is whether the mechanism can be an object at
all -- something to read, copy, compare, edit and run -- without touching
the immutable core. Nothing here tries to make the search better.

A `SearchPolicy` is data:

    rules       how candidates are made: rewrite rules at one node, a
                pattern on the left (?0, ?1 match any subtree) and what it
                becomes on the right, drawing leaves and conditions from
                the vocabulary. The five edits of search.py are eight such
                rules, in the order search.py tries them
    selection   which candidates are kept: order them by a weighted sum of
                a state's program bits, error bits, size and wrong points,
                then by size, then by text, and keep the first k of what
                the round made together with what was kept
    resources   the size limit, the rounds, the patience, the budget

`run` interprets a policy. The interpreter is the fixed part: the rounds,
the budget, and the verdict -- the answer returned is the shortest
description of everything evaluated, whatever the policy kept, judged by
the same gates as ever. The current search is one policy, `CURRENT`, and
running it must give what search.search gives, program for program.

What the policy language cannot say -- the wall, drawn here on purpose:

    selection   only "order by a weighted sum, keep the first k". Nothing
                about the set kept as a set -- its diversity, what it
                already covers -- no memory of states dropped, no chance
    state       a single program. Not a set of programs, not a trajectory
    rounds      every kept state expanded in full, in order; no partial
                expansion, no priority across rounds, no restart
    rules       rewrites at one node, parameters from the vocabulary; no
                rule that depends on the data, on scores, or on history
    the loop    itself -- rounds, budget, stopping, the verdict -- is the
                interpreter, Python, outside the object

Each of those is a place where a different kind of search would need a
different kind of object, and none of them can be reached by editing a
policy.

Measured 2026-09-14 (scripts/run_policy.py): CURRENT reproduced
search.search on 70 runs of 70, every field and every point of the budget
curve, 1.1 times slower. Deliberately meaningless edits did what was
predicted of them: a beam of 2 made 0.39 of the programs per round and
solved the family less often; the rules reversed changed no answer and,
where the search stopped by itself, not a single evaluation; without
"combine with a leaf" no question of the family was solved and W0, W3 were
untouched.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, replace as _replace
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import description
from .language import (CMP, CONST, EQUAL, GET, HOLE, LESS, Evaluator, Program, add, children,
                       cmp, hole, if_, nodes, rebuild, replace, show, size, sub)
from .library import match
from .search import (BUDGET, MAX_ROUNDS, MAX_SIZE, PATIENCE, SEARCH_EXHAUSTED, SEARCH_LIMIT,
                     Found, vocabulary)

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

PARAM = "param"


def param(kind: str, index: int) -> Program:
    """A value a rule draws when it is applied: the index-th of its
    parameters, a "leaf" or a "cond" of the vocabulary."""
    return (PARAM, kind, int(index))


@dataclass(frozen=True)
class Rule:
    name: str
    lhs: Program
    rhs: Tuple[Program, ...]
    #: What each application draws, outermost first.
    params: Tuple[str, ...] = ()
    #: The rule applies only while size(program) + margin <= the size limit.
    margin: Optional[int] = None


@dataclass(frozen=True)
class Selection:
    keep: int = 4
    #: "made+kept": this round's new programs with the beam; "made": only them.
    pool: str = "made+kept"
    #: The order's score on (program bits, error bits, size, wrong points).
    weights: Tuple[float, float, float, float] = (1.0, 1.0, 0.0, 0.0)


@dataclass(frozen=True)
class SearchPolicy:
    rules: Tuple[Rule, ...]
    selection: Selection = Selection()
    max_size: int = MAX_SIZE
    max_rounds: int = MAX_ROUNDS
    patience: int = PATIENCE
    budget: int = BUDGET


N, A, B, C = hole(0), hole(0), hole(1), hole(2)
LEAF, COND = param("leaf", 0), param("cond", 0)

#: The search as it stands, written as a policy: search.neighbours, rule by
#: rule, in the order it tries them at every node.
CURRENT = SearchPolicy(rules=(
    Rule("replace by a leaf", N, (LEAF,), ("leaf",)),
    Rule("swap add", add(A, B), (sub(A, B), add(B, A))),
    Rule("swap sub", sub(A, B), (add(A, B), sub(B, A))),
    Rule("swap <", cmp(LESS, A, B), (cmp(EQUAL, A, B), cmp(LESS, B, A))),
    Rule("swap ==", cmp(EQUAL, A, B), (cmp(LESS, A, B), cmp(EQUAL, B, A))),
    Rule("drop or turn a condition", if_(A, B, C), (B, C, if_(A, C, B))),
    Rule("add a condition", N, (if_(COND, N, param("leaf", 1)), if_(COND, param("leaf", 1), N)),
         ("cond", "leaf"), 4),
    Rule("combine with a leaf", N, (add(N, LEAF), sub(N, LEAF), sub(LEAF, N)), ("leaf",), 2),
))


def _fill(template: Program, bound: Dict[int, Program], drawn: Tuple[Program, ...]) -> Program:
    if template[0] == HOLE:
        return bound[template[1]]
    if template[0] == PARAM:
        return drawn[template[2]]
    kids = children(template)
    if not kids:
        return template
    return rebuild(template, [_fill(kid, bound, drawn) for kid in kids])


def successors(policy: SearchPolicy, p: Program, leaves: Sequence[Program],
               conditions: Sequence[Program]) -> Iterator[Program]:
    """Every program one rule away, in the order the policy says."""
    pools = {"leaf": leaves, "cond": conditions}
    whole = size(p)
    for path, node in nodes(p):
        for rule in policy.rules:
            if rule.margin is not None and whole + rule.margin > policy.max_size:
                continue
            bound = match(rule.lhs, node, {})
            if bound is None:
                continue
            for drawn in itertools.product(*[pools[kind] for kind in rule.params]):
                for template in rule.rhs:
                    new = _fill(template, bound, drawn)
                    if new != node:
                        yield replace(p, path, new)


def run(policy: SearchPolicy, columns, outcomes, profile: bool = False) -> Found:
    """Interpret a policy on one data set. The answer is the shortest
    description of everything evaluated, whatever the policy kept."""
    started = time.time()
    actual = np.asarray(outcomes, dtype=np.int64)
    evaluator = Evaluator(columns)
    variables = len(columns)
    alphabet = description.alphabet_of(actual)
    known = description.membership(actual)
    leaves, conditions = vocabulary(columns, actual)
    w = policy.selection.weights
    seen: Dict[Program, Tuple[float, float, float]] = {}
    value: Dict[Program, float] = {}
    first_seen: Dict[Program, int] = {}
    anytime: List[Tuple[int, Program, float]] = []
    leader: Dict[str, object] = {"key": None, "program": None}
    shortest: Dict[str, object] = {"key": None, "program": None}

    def better(held: Dict[str, object], key, p) -> bool:
        return (held["key"] is None or key < held["key"]
                or (key == held["key"] and show(p) < show(held["program"])))

    def score(p: Program) -> Tuple[float, float, float]:
        first_seen.setdefault(p, len(first_seen) + 1)
        program_part = description.program_bits(p, variables)
        error_part = description.error_bits(evaluator(p), actual, alphabet, known)
        total = program_part + error_part
        wrong = int(np.count_nonzero(evaluator(p) != actual)) if w[3] else 0
        value[p] = program_part * w[0] + error_part * w[1] + size(p) * w[2] + wrong * w[3]
        key = (total, size(p))
        if better(shortest, key, p):
            shortest["key"], shortest["program"] = key, p
        if profile and better(leader, key, p):
            leader["key"], leader["program"] = key, p
            anytime.append((first_seen[p], p, total))
        return (total, program_part, error_part)

    def order(p: Program) -> Tuple[float, int, str]:
        return (value[p], size(p), show(p))

    for leaf in leaves:
        seen[leaf] = score(leaf)
    k = policy.selection.keep
    beam = sorted(seen, key=order)[:k]
    best = shortest["program"]
    history = [(0, seen[best][0], show(best))]
    stalled = 0
    rounds = 0
    for rounds in range(1, policy.max_rounds + 1):
        fresh: List[Program] = []
        for p in beam:
            for q in successors(policy, p, leaves, conditions):
                if q in seen:
                    continue
                if len(seen) >= policy.budget:
                    break
                seen[q] = score(q)
                fresh.append(q)
        pool = set(fresh) | set(beam) if policy.selection.pool == "made+kept" else set(fresh) or set(beam)
        beam = sorted(pool, key=order)[:k]
        if seen[shortest["program"]][0] < seen[best][0] - 1e-9:
            best = shortest["program"]
            stalled = 0
            history.append((rounds, seen[best][0], show(best)))
        else:
            stalled += 1
        if stalled >= policy.patience or len(seen) >= policy.budget:
            break
    bits, program_part, error_part = seen[best]
    exhausted = stalled >= policy.patience and len(seen) < policy.budget
    return Found(program=best, bits=bits, program_bits=program_part, error_bits=error_part,
                 evaluations=len(seen), rounds=rounds, history=history,
                 found_at=first_seen.get(best, len(seen)),
                 termination=SEARCH_EXHAUSTED if exhausted else SEARCH_LIMIT,
                 seconds=time.time() - started,
                 train_errors=int(np.count_nonzero(evaluator(best) != actual)),
                 anytime=anytime)


# -- what MANA can do with a policy -----------------------------------------

def _show_template(t: Program) -> str:
    if t[0] == HOLE:
        return f"?{t[1]}"
    if t[0] == PARAM:
        return f"#{t[1]}{t[2]}"
    if t[0] in (GET, CONST):
        return show(t)
    kids = [_show_template(kid) for kid in children(t)]
    if t[0] == CMP:
        return f"({kids[0]} {t[1]} {kids[1]})"
    if t[0] in ("add", "sub"):
        return f"({kids[0]} {'+' if t[0] == 'add' else '-'} {kids[1]})"
    return f"if({', '.join(kids)})"


def describe(policy: SearchPolicy) -> str:
    """The policy, readable."""
    lines = []
    for rule in policy.rules:
        guard = f", пока размер + {rule.margin} <= {policy.max_size}" if rule.margin else ""
        lines.append(f"  {rule.name}: {_show_template(rule.lhs)} -> "
                     f"{' | '.join(_show_template(t) for t in rule.rhs)}{guard}")
    s = policy.selection
    lines.append(f"  отбор: {s.keep} первых из «{s.pool}» по "
                 f"{s.weights[0]:g}·биты программы + {s.weights[1]:g}·биты ошибок + "
                 f"{s.weights[2]:g}·размер + {s.weights[3]:g}·ошибок")
    lines.append(f"  ресурсы: размер <= {policy.max_size}, раундов <= {policy.max_rounds}, "
                 f"терпение {policy.patience}, бюджет {policy.budget}")
    return "\n".join(lines)


def copy(policy: SearchPolicy) -> SearchPolicy:
    return _replace(policy)


def difference(a: SearchPolicy, b: SearchPolicy) -> List[str]:
    """Where two policies differ, part by part."""
    out = []
    names_a, names_b = [r.name for r in a.rules], [r.name for r in b.rules]
    for name in names_a:
        if name not in names_b:
            out.append(f"нет правила «{name}»")
    for name in names_b:
        if name not in names_a:
            out.append(f"новое правило «{name}»")
    common = [n for n in names_a if n in names_b]
    if [n for n in names_b if n in names_a] != common:
        out.append("другой порядок правил")
    for rule_a in a.rules:
        for rule_b in b.rules:
            if rule_a.name == rule_b.name and rule_a != rule_b:
                out.append(f"правило «{rule_a.name}» изменено")
    if a.selection != b.selection:
        out.append(f"отбор: {a.selection} -> {b.selection}")
    for field_ in ("max_size", "max_rounds", "patience", "budget"):
        if getattr(a, field_) != getattr(b, field_):
            out.append(f"{field_}: {getattr(a, field_)} -> {getattr(b, field_)}")
    return out


def with_keep(policy: SearchPolicy, k: int) -> SearchPolicy:
    return _replace(policy, selection=_replace(policy.selection, keep=k))


def reordered(policy: SearchPolicy, order: Sequence[int]) -> SearchPolicy:
    return _replace(policy, rules=tuple(policy.rules[i] for i in order))


def without(policy: SearchPolicy, name: str) -> SearchPolicy:
    return _replace(policy, rules=tuple(r for r in policy.rules if r.name != name))


def with_budget(policy: SearchPolicy, budget: int) -> SearchPolicy:
    return _replace(policy, budget=int(budget))
