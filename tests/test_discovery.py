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
