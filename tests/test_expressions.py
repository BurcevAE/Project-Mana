"""The space of experiment programs (D3-prep-1, docs/ГЛУБИНА_D3.md, 10):
counted without being built, enumerated in one order, sampled uniformly, and
held to the restrictions the declaration names."""
import random
from collections import Counter

import numpy as np
import pytest

from mana.discovery import expressions as X
from mana.discovery import policy as P
from mana.discovery import problems, questions
from mana.discovery.search import vocabulary
from mana.discovery.worlds import W0

GIVES = {name: gives for name, gives, _ in X.OPS}
ASKS = {name: asks for name, _, asks in X.OPS}
#: What running an expression of the wrong shape may raise. Not _Short:
#: that is the budget speaking, and it is allowed too.
FAULTS = (TypeError, ValueError, IndexError, KeyError, ZeroDivisionError,
          AttributeError, questions._Short)


def _form(e, lam=False):
    """The form of an expression, or None if some part does not fit."""
    if e == X.BOUND:
        return "P" if lam else None
    for form, atoms in X.ATOMS.items():
        if e in atoms:
            return form
    name = e[0]
    if name not in GIVES or (lam and name in X.NOT_IN_LAMBDA):
        return None
    asks, parts = ASKS[name], e[1:]
    if len(asks) != len(parts):
        return None
    for want, part in zip(asks, parts):
        if want == "K":
            if part[0] != "const" or part[1] not in X.CANDIDATE_SIZES:
                return None
        elif want == "Λ":
            if part[0] != "lambda" or part[1] != "x" or _form(part[2], True) != "N":
                return None
        elif _form(part, lam) != want:
            return None
    return GIVES[name]


def _holds(e, name):
    if type(e) is not tuple or not e:
        return 0
    return (e == name if type(name) is tuple else e[0] == name) \
        + sum(_holds(part, name) for part in e[1:])


def _arithmetic_over_constants(e):
    if type(e) is not tuple or not e:
        return False
    if e[0] in X.ARITHMETIC and all(p[0] == "const" for p in e[1:]):
        return True
    return any(_arithmetic_over_constants(p) for p in e[1:])


def _bound_outside_a_lambda(e, lam=False):
    if type(e) is not tuple or not e:
        return False
    if e == X.BOUND:
        return not lam
    if e[0] == "lambda":
        return _bound_outside_a_lambda(e[2], True)
    return any(_bound_outside_a_lambda(part, lam) for part in e[1:])


def _inside_lambdas(e):
    if type(e) is not tuple or not e:
        return []
    if e[0] == "lambda":
        return [e[2]] + _inside_lambdas(e[2])
    return [x for part in e[1:] for x in _inside_lambdas(part)]


def _size(e):
    if type(e) is not tuple or not e:
        return 0
    if e[0] == "lambda":
        return 1 + _size(e[2])
    if e[0] in ("const", "var", "own", "target", "domain", "leaves", "conditions", "held"):
        return 1
    return 1 + sum(_size(part) for part in e[1:])


# -- the space is what the declaration says ------------------------------------

@pytest.mark.parametrize("size", (1, 2, 3, 4, 5))
def test_what_is_counted_is_what_is_enumerated(size):
    for form in X.FORMS:
        listed = list(X.generate(size, form))
        assert len(listed) == X.total(size, form)
        assert len(set(listed)) == len(listed)


def test_every_expression_has_the_form_asked_of_it_and_the_size_asked_of_it():
    for size in range(1, 6):
        for e in X.generate(size):
            assert _form(e) == "P" and _size(e) == size


def test_the_restrictions_of_the_declaration_hold():
    for size in range(1, 7):
        for e in X.generate(size):
            assert _holds(e, "candidates") <= 1
            assert not _holds(e, "flatmap") and not _holds(e, "list")
            assert not _holds(e, "successors")
            for body in _inside_lambdas(e):
                for forbidden in X.NOT_IN_LAMBDA:
                    assert not _holds(body, forbidden)
            assert not _bound_outside_a_lambda(e)
            assert not _arithmetic_over_constants(e)


def test_the_order_is_one_order_and_does_not_change():
    once, again = list(X.generate(5)), list(X.generate(5))
    assert once == again
    assert once[:1] == [] or _size(once[0]) == 5


def test_the_ladder_of_sizes_grows_and_is_counted_before_it_is_built():
    counted = [X.total(n) for n in range(1, 9)]
    assert counted[0] == 1 and all(counted[i] <= counted[i + 1] for i in (4, 5, 6))
    assert X.total(12) > 10 ** 6
    assert X.LADDER == (4, 6, 8, 10, 12) and X.MOST == 12


# -- the uniform sample of the same space --------------------------------------

def test_a_sample_is_of_the_space_and_repeats_with_its_seed():
    space = set(X.generate(4))
    rng = random.Random(7)
    drawn = [X.sample(4, rng) for _ in range(500)]
    assert set(drawn) <= space
    again = random.Random(7)
    assert drawn == [X.sample(4, again) for _ in range(500)]
    rng = random.Random(11)
    seen = Counter(X.sample(4, rng) for _ in range(4000))
    assert len(seen) > min(20, len(space)) and max(seen.values()) < 4000 // 4


# -- and the interpreter runs them ---------------------------------------------

def test_the_interpreter_runs_what_is_enumerated_or_says_it_cannot():
    split = W0.split(120, 50, seed=0)
    columns, target = split.train, np.asarray(split.train_outcomes)
    problem = problems.Problem.whole(columns, target)
    leaves, conditions = vocabulary(columns, target)
    ledger = problems.Ledger(40000)
    run = questions._Interpreter(P.CURRENT, ledger, (), 1.0, False, False, None)
    node = problems.Node(0, problem, None, None, {a: 0 for a in problems.ARTICLES},
                         budget=40000)
    run.nodes.append(node)
    scope = questions._Scope(run, node, columns, target, questions.Evaluator(columns),
                             leaves, conditions, leaves[0], list(leaves[:4]))
    answered, refused = 0, 0
    for size in (1, 3, 4):
        for e in list(X.generate(size))[:40]:
            try:
                assert scope.value(e, {"share": 2000}, True) is not None
                answered += 1
            except FAULTS:
                refused += 1
    assert answered + refused > 60 and answered > 10
