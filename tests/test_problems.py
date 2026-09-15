"""D1a: a problem as an object is the flat search when it asks no question."""
import ast
import inspect

import numpy as np
import pytest

from mana.discovery import policy as P
from mana.discovery import problems
from mana.discovery.worlds import W0

FIELDS = ("program", "bits", "program_bits", "error_bits", "evaluations", "rounds",
          "history", "found_at", "termination", "train_errors", "anytime")


def _same(a, b):
    return all(getattr(a, f) == getattr(b, f) for f in FIELDS)


def test_with_no_operators_a_problem_is_the_flat_search():
    split = W0.split(200, 50, seed=0)
    policy = P.with_budget(P.CURRENT, 30000)
    flat = P.run(policy, split.train, split.train_outcomes, profile=True)
    solved = problems.solve(problems.Problem.whole(split.train, split.train_outcomes),
                            P.CURRENT, 30000, profile=True)
    assert _same(flat, solved.found) and solved.program == flat.program


def test_a_problem_on_part_of_the_points_is_the_flat_search_on_those_points():
    split = W0.split(200, 50, seed=1)
    whole = problems.Problem.whole(split.train, split.train_outcomes)
    whole.domain = np.arange(200) % 3 != 0
    rows = whole.domain
    flat = P.run(P.with_budget(P.CURRENT, 20000), {k: v[rows] for k, v in split.train.items()},
                 np.asarray(split.train_outcomes)[rows], profile=True)
    assert _same(flat, problems.solve(whole, P.CURRENT, 20000, profile=True).found)


def test_every_evaluation_is_charged_to_search_and_nothing_else():
    split = W0.split(200, 50, seed=2)
    solved = problems.solve(problems.Problem.whole(split.train, split.train_outcomes),
                            P.CURRENT, 20000)
    spent = solved.ledger.spent
    assert spent[problems.SEARCH] == solved.found.evaluations
    assert all(spent[a] == 0 for a in problems.ARTICLES if a != problems.SEARCH)
    assert solved.ledger.left() == 20000 - solved.found.evaluations
    assert len(solved.nodes) == 1 and solved.used_depth() == 1


def test_a_catalogue_is_refused_before_d1b_not_ignored():
    split = W0.split(200, 50, seed=0)
    noop = problems.Operator("ничего", lambda problem, result: None, lambda a, b: a)
    with pytest.raises(NotImplementedError):
        problems.solve(problems.Problem.whole(split.train, split.train_outcomes),
                       P.CURRENT, 10000, catalogue=[noop])


def test_the_ledger_knows_its_four_articles_only():
    ledger = problems.Ledger(100)
    ledger.charge(problems.BUILD, 7)
    assert ledger.left() == 93
    with pytest.raises(ValueError):
        ledger.charge("другое", 1)


def test_problems_cannot_see_the_worlds():
    for node in ast.walk(ast.parse(inspect.getsource(problems))):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            named = [getattr(node, "module", "") or ""] + [a.name for a in node.names]
            assert not any("worlds" in name or "ladder" in name for name in named)
