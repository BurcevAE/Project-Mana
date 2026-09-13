"""
Discovery, step 1: a poor language, syntactic edits, and one currency --
bits. What is pinned: the arithmetic of the currency, that the search finds
the rule where one exists and does not buy back noise, and that no learner
module can see the world's truth.
"""
from __future__ import annotations

import ast
import inspect

import numpy as np

from mana.discovery import baselines, description
from mana.discovery import search as discovery
from mana.discovery.language import (EQUAL, LESS, Evaluator, cmp, const, get,
                                     if_, replace, show, size)
from mana.discovery.worlds import W0, W3


def test_a_program_is_a_value_that_evaluates_on_columns():
    rule = if_(cmp(LESS, const(5), get("x")), get("y"), get("z"))
    out = Evaluator({"x": [6, 2], "y": [1, 1], "z": [7, 7]})(rule)
    assert out.tolist() == [1, 7]
    assert show(rule) == "if((5 < x), y, z)" and size(rule) == 6
    assert replace(rule, (2,), get("x")) == if_(rule[1], get("y"), get("x"))
    assert rule == if_(cmp(LESS, const(5), get("x")), get("y"), get("z"))


def test_integers_are_written_in_elias_gamma_on_the_zigzag():
    assert description.integer_bits(0) == 1
    assert description.integer_bits(-1) == 3
    assert description.integer_bits(5) == 7


def _bits(split, program):
    on_train = Evaluator(split.train)
    alphabet = description.alphabet_of(split.train_outcomes)
    return (description.program_bits(program, 3)
            + description.error_bits(on_train(program), split.train_outcomes, alphabet))


def test_remembering_one_exception_costs_more_than_it_saves():
    """The currency, in one case: fixing a single noisy point with a
    condition that names it is a worse description, not a better one."""
    split = W3.split(200, 50, seed=0)
    rule = W3.truth
    on_train = Evaluator(split.train)
    wrong = np.nonzero(on_train(rule) != split.train_outcomes)[0]
    assert len(wrong)
    i = int(wrong[0])
    point = {name: int(split.train[name][i]) for name in ("x", "y", "z")}
    patched = if_(cmp(EQUAL, get("x"), const(point["x"])),
                  if_(cmp(EQUAL, get("y"), const(point["y"])),
                      if_(cmp(EQUAL, get("z"), const(point["z"])),
                          const(int(split.train_outcomes[i])), rule), rule), rule)
    assert int(np.count_nonzero(on_train(patched) != split.train_outcomes)) == len(wrong) - 1
    assert _bits(split, patched) > _bits(split, rule)


def test_a_table_of_the_training_states_costs_more_than_the_rule():
    split = W0.split(200, 50, seed=0)
    alphabet = description.alphabet_of(split.train_outcomes)
    assert _bits(split, W0.truth) < description.table_bits(200, 3, 10, alphabet)


def test_the_search_finds_the_rule_of_w0():
    split = W0.split(150, 100, seed=0)
    found = discovery.search(split.train, split.train_outcomes)
    assert W0.grade(lambda cols: discovery.predict(found.program, cols)) == 1.0
    assert found.bits <= _bits(split, W0.truth) + 1e-6


def test_on_noise_it_keeps_the_rule_and_buys_no_exceptions():
    split = W3.split(200, 100, seed=0)
    found = discovery.search(split.train, split.train_outcomes)
    assert W3.grade(lambda cols: discovery.predict(found.program, cols)) == 1.0
    assert size(found.program) <= size(W3.truth)


def test_the_budget_is_respected():
    split = W0.split(150, 50, seed=1)
    found = discovery.search(split.train, split.train_outcomes, budget=500)
    assert found.evaluations <= 500


def test_the_floor_is_what_it_says():
    split = W0.split(200, 300, seed=0)
    table = baselines.table_predict(split.train, split.train_outcomes, split.test)
    tree, nodes = baselines.tree_predict(split.train, split.train_outcomes, split.test)
    assert len(table) == len(tree) == 300 and nodes > 1


def test_no_learner_module_can_see_the_worlds():
    """A learner that peeks scores perfectly and means nothing."""
    from mana.discovery import language, search

    for module in (language, description, search, baselines):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                named = [node.module or ""] + [alias.name for alias in node.names]
            elif isinstance(node, ast.Import):
                named = [alias.name for alias in node.names]
            else:
                continue
            assert not any("worlds" in name for name in named), module.__name__


# --------------------------------------------------------------------------
# step 2: a variable nobody observed
# --------------------------------------------------------------------------

def test_a_transition_is_run_along_each_episode_from_its_start():
    from mana.discovery import invent
    from mana.discovery.language import SUB

    flips = if_(get("a"), (SUB, const(1), get(invent.PREV)), get(invent.PREV))
    a = np.array([[0, 1, 0, 1, 1], [1, 0, 0, 0, 0]])
    assert invent.run(flips, 0, {"a": a}).tolist() == [[0, 1, 1, 0, 1], [1, 1, 1, 1, 1]]


def test_the_hidden_switch_of_w2_is_invented_and_pays_for_itself():
    from mana.discovery import invent
    from mana.discovery.worlds import W2

    train, test = W2.split(20, 50, seed=0)
    found = invent.invent(train.columns, train.outcomes)
    assert found.accepted, found.note
    assert found.bits < found.base.bits
    predicted = invent.predict(found, test.columns)
    assert float(np.mean(predicted == test.clean)) >= 0.99
    value = invent.hidden(found, test.columns)
    recovered = max(np.mean(value == test.hidden), np.mean(value != test.hidden))
    assert recovered >= 0.99
    # Without the new variable the same search is near a coin between x and y.
    flat = {name: values.reshape(-1) for name, values in test.columns.items()}
    base = discovery.predict(found.base.program, flat)
    assert float(np.mean(base == test.clean.reshape(-1))) <= 0.65


def test_where_the_program_is_right_there_is_nothing_to_invent():
    from mana.discovery import invent

    split = W0.split(200, 50, seed=0)
    shaped = {name: values.reshape(10, 20) for name, values in split.train.items()}
    found = invent.invent(shaped, split.train_outcomes.reshape(10, 20))
    assert not found.accepted and "объяснять нечего" in found.note


def test_noise_does_not_earn_an_invented_cause():
    """The same currency that refuses a table of exceptions refuses a
    hidden variable made up to explain them."""
    from mana.discovery import invent

    split = W3.split(200, 50, seed=0)
    shaped = {name: values.reshape(10, 20) for name, values in split.train.items()}
    found = invent.invent(shaped, split.train_outcomes.reshape(10, 20))
    assert not found.accepted, found.describe()


def test_the_inventor_cannot_see_the_worlds_either():
    from mana.discovery import invent

    tree = ast.parse(inspect.getsource(invent))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            named = ([getattr(node, "module", "") or ""]
                     + [alias.name for alias in node.names])
            assert not any("worlds" in name for name in named)
