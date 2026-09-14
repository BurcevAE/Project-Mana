"""
Experiment M, step M0: a black box with knobs, four ways of knowing it,
and the controls that choose without knowing anything. What is pinned:
each method is right exactly where its assumption holds and says so
through its own check, a session never pays twice for one question, and
no method can see the world's structure.
"""
from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from mana.methods import portfolio, solvers
from mana.methods.world import (ADDITIVE, BLOCKS, GLOBAL, LEVELS, LINEAR, STRUCTURES,
                                BlackBox)


def _run(structure, n, name, seed=0, budget=solvers.BUDGET):
    box = BlackBox(n, structure, seed)
    found = solvers.attempt(name, box.answer, n, LEVELS, seed, budget)
    return box, found, box.verdict(found.model)


def test_the_box_is_a_function_and_its_own_truth_scores_perfectly():
    for structure in STRUCTURES:
        box = BlackBox(4, structure, seed=3)
        X = np.random.default_rng(0).integers(0, LEVELS, size=(50, 4))
        assert np.array_equal(box.answer(X), box.answer(X))
        assert box.verdict(box._truth) == 1.0
        assert box.verdict(None) == 0.0


def test_blocks_pair_every_input_once():
    box = BlackBox(7, BLOCKS, seed=1)
    assert sorted(i for block in box.blocks for i in block) == list(range(7))
    assert sorted(len(b) for b in box.blocks) == [1, 2, 2, 2]


def test_a_question_asked_twice_is_paid_once():
    box = BlackBox(3, GLOBAL, seed=0)
    session = solvers.Session(box.answer, 3, LEVELS, budget=5)
    X = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
    first = session.ask(X)
    again = session.ask(X[:1])
    assert session.spent == 2 and box.asked == 2 and again[0] == first[0]
    with pytest.raises(solvers.OverBudget):
        session.ask(np.random.default_rng(0).integers(0, LEVELS, size=(10, 3)))


def test_calculation_is_right_on_a_linear_box_and_knows_it_is_wrong_elsewhere():
    _, found, verdict = _run(LINEAR, 6, "calculate")
    assert found.believed and verdict == 1.0 and found.queries == 6 + 1 + solvers.CHECKS
    _, found, verdict = _run(ADDITIVE, 6, "calculate")
    assert not found.believed and verdict < 1.0


def test_decomposition_is_right_where_effects_add_and_only_there():
    _, found, verdict = _run(ADDITIVE, 6, "decompose")
    assert found.believed and verdict == 1.0
    _, found, verdict = _run(BLOCKS, 6, "decompose")
    assert not found.believed and verdict < 1.0


def test_experiment_finds_the_blocks_it_was_not_told():
    box, found, verdict = _run(BLOCKS, 8, "experiment")
    assert found.believed and verdict == 1.0
    # 8 inputs in 4 pairs: the tests, then four tables of 100.
    assert found.queries < 4 * LEVELS ** 2 + 200


def test_exhaustion_declines_what_the_budget_cannot_finish_without_spending():
    _, found, verdict = _run(GLOBAL, 6, "exhaust")
    assert found.model is None and found.queries == 0 and found.declined
    _, found, verdict = _run(GLOBAL, 3, "exhaust")
    assert found.believed and verdict == 1.0 and found.queries == LEVELS ** 3


def test_a_cascade_stops_at_the_first_method_that_believes_itself_and_pays_the_union():
    box = BlackBox(5, ADDITIVE, seed=2)
    attempts = {name: solvers.attempt(name, box.answer, 5, LEVELS, 2)
                for name in solvers.METHODS}
    choice = portfolio.cascade(attempts)
    assert choice.method == "decompose" and choice.tried == ("calculate", "decompose")
    union = set(attempts["calculate"].points.tolist()) | set(attempts["decompose"].points.tolist())
    assert choice.cost == len(union) < attempts["calculate"].queries + attempts["decompose"].queries
    verdicts = {name: box.verdict(a.model) for name, a in attempts.items()}
    assert portfolio.oracle(attempts, verdicts).method == "decompose"


def test_methods_cannot_see_the_world():
    for module in (solvers, portfolio):
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                named = [getattr(node, "module", "") or ""] + [a.name for a in node.names]
                assert not any("world" in name for name in named), module.__name__


# --------------------------------------------------------------------------
# M1: no method says what it will cost; the chooser learns it
# --------------------------------------------------------------------------

def test_without_announcing_a_method_asks_until_its_budget_is_spent():
    box = BlackBox(6, GLOBAL, seed=0)
    found = solvers.attempt("exhaust", box.answer, 6, LEVELS, 0, budget=1000, announce=False)
    assert found.model is None and found.queries == 1000 and found.declined != ""


def test_in_one_session_a_later_method_does_not_pay_for_what_an_earlier_asked():
    box = BlackBox(5, ADDITIVE, seed=4)
    alone = solvers.attempt("decompose", box.answer, 5, LEVELS, 4, announce=False)
    choice, model, tried = portfolio.run_order(box.answer, 5, LEVELS, 4)
    assert choice.method == "decompose" and [a.method for a in tried] == ["calculate", "decompose"]
    assert tried[1].queries < alone.queries
    assert choice.cost == tried[0].queries + tried[1].queries
    assert box.verdict(model) == 1.0


def test_the_cost_of_a_method_is_extrapolated_from_what_it_spent_before():
    from mana.methods.choice import Experience, Record

    experience = Experience()
    for n in (2, 3, 4):
        experience.add(Record(n, frozenset(), "exhaust", True, LEVELS ** n, True))
    assert abs(np.log10(experience.cost("exhaust", 6, frozenset())) - 6) < 1e-6
    assert experience.cost("calculate", 6, frozenset()) is None


def test_the_chooser_tries_what_it_never_tried_and_skips_what_will_not_finish():
    from mana.methods.choice import Chooser, Experience, Record

    experience = Experience()
    chooser = Chooser(experience)
    assert chooser.next(4, frozenset(), ["exhaust"]) == "exhaust"      # never tried
    for n in (2, 3, 4):
        experience.add(Record(n, frozenset(), "exhaust", True, LEVELS ** n, True))
    assert chooser.next(4, frozenset(), ["exhaust"]) == "exhaust"      # 10 000: worth it
    assert chooser.next(6, frozenset(), ["exhaust"]) is None           # a million: not


def test_the_chooser_learns_from_boxes_and_keeps_to_what_it_may_see():
    from mana.methods import choice

    chooser = choice.Chooser(choice.Experience(), seed=1)
    for i, structure in enumerate(STRUCTURES * 3):
        box = BlackBox(3, structure, seed=100 + i)
        found, model = chooser.solve(box.answer, 3, LEVELS, 100 + i)
        assert found.method is not None and box.verdict(model) == 1.0
    assert len(chooser.experience.records) >= 12
    source = inspect.getsource(choice)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            named = [getattr(node, "module", "") or ""] + [a.name for a in node.names]
            assert not any("world" in name for name in named)
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("planned", "verdict", "structure", "declined"), node.attr
