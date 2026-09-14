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
