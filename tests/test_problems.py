"""D1: a problem as an object. D1a -- the flat search when it asks nothing;
D1b -- the one declared procedure with D0's catalogue, held to its declaration."""
import ast
import inspect

import numpy as np
import pytest

from mana.discovery import policy as P
from mana.discovery import problems
from mana.discovery.language import Evaluator, show
from mana.discovery.worlds import W0

FIELDS = ("program", "bits", "program_bits", "error_bits", "evaluations", "rounds",
          "history", "found_at", "termination", "train_errors", "anytime")


def _same(a, b):
    return all(getattr(a, f) == getattr(b, f) for f in FIELDS)


def _sum_of_two(seed=0):
    """A target no single condition writes: if(x > 5, y, z) + if(y < 3, 1, 0)."""
    split = W0.split(200, 50, seed=seed)
    cols = split.train
    target = (np.where(cols["x"] > 5, cols["y"], cols["z"])
              + (cols["y"] < 3).astype(np.int64))
    return cols, target


# -- D1a ---------------------------------------------------------------------

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


# -- D1b ---------------------------------------------------------------------

def test_only_d0s_catalogue_is_accepted():
    cols, target = _sum_of_two()
    with pytest.raises(ValueError):
        problems.solve(problems.Problem.whole(cols, target), P.CURRENT, 10000,
                       catalogue=("остаток", "новый элемент"))
    with pytest.raises(ValueError):
        problems.solve(problems.Problem.whole(cols, target), P.CURRENT, 10000,
                       catalogue=("остаток", "остаток"))


def test_a_problem_the_flat_search_solves_asks_no_question():
    split = W0.split(200, 50, seed=0)
    flat = P.run(P.with_budget(P.CURRENT, 30000), split.train, split.train_outcomes)
    solved = problems.solve(problems.Problem.whole(split.train, split.train_outcomes),
                            P.CURRENT, 30000, catalogue=problems.BRANCHES)
    assert flat.train_errors == 0
    assert len(solved.nodes) == 1 and solved.program == flat.program
    assert solved.ledger.spent[problems.BUILD] == 0


def test_the_tree_never_spends_more_than_its_budget_and_says_on_what():
    cols, target = _sum_of_two()
    solved = problems.solve(problems.Problem.whole(cols, target), P.CURRENT, 30000,
                            catalogue=problems.BRANCHES, flat_share=0.5)
    assert sum(solved.ledger.spent.values()) <= 30000
    assert len(solved.nodes) > 1
    assert solved.ledger.spent[problems.BUILD] > 0 and solved.ledger.spent[problems.VERIFY] > 0
    assert sum(sum(n.cost.values()) for n in solved.nodes) == sum(solved.ledger.spent.values())


def test_a_flat_search_that_spends_the_whole_budget_leaves_no_question_and_no_charge():
    cols, target = _sum_of_two()
    solved = problems.solve(problems.Problem.whole(cols, target), P.CURRENT, 30000,
                            catalogue=problems.BRANCHES)
    assert sum(solved.ledger.spent.values()) <= 30000
    if solved.found.evaluations >= 30000 - 40:
        assert len(solved.nodes) == 1
        assert solved.ledger.spent[problems.BUILD] == 0


def test_a_share_caps_the_flat_search_of_every_node():
    cols, target = _sum_of_two()
    solved = problems.solve(problems.Problem.whole(cols, target), P.CURRENT, 40000,
                            catalogue=problems.BRANCHES, flat_share=0.25)
    for node in solved.nodes:
        assert node.found.evaluations <= max(node.budget // 4, 1) + 40


def test_children_depend_on_the_results_they_were_asked_from():
    cols, target = _sum_of_two()
    solved = problems.solve(problems.Problem.whole(cols, target), P.CURRENT, 60000,
                            catalogue=problems.BRANCHES, flat_share=0.5)
    for node in solved.nodes[1:]:
        p = node.problem
        assert p.parent is not None and p.parent in p.depends_on
        if p.operator == "случаи":
            rest = [i for i in p.depends_on if i != p.parent]
            assert len(rest) == 1 and solved.nodes[rest[0]].problem.operator == "промахи"


def test_the_answer_of_a_node_is_the_shortest_of_its_flat_answer_and_candidates():
    cols, target = _sum_of_two()
    solved = problems.solve(problems.Problem.whole(cols, target), P.CURRENT, 60000,
                            catalogue=problems.BRANCHES, flat_share=0.5)
    root = solved.nodes[0]
    ev = Evaluator(cols)
    mine = problems._key(solved.program, cols, target, ev)
    assert mine <= problems._key(root.found.program, cols, target, ev)


def test_the_order_is_the_order_given():
    cols, target = _sum_of_two(seed=3)
    forward = problems.solve(problems.Problem.whole(cols, target), P.CURRENT, 30000,
                             catalogue=problems.BRANCHES, flat_share=0.5)
    backward = problems.solve(problems.Problem.whole(cols, target), P.CURRENT, 30000,
                              catalogue=tuple(reversed(problems.BRANCHES)), flat_share=0.5)
    first = lambda s: [n.problem.operator for n in s.nodes[1:2]]  # noqa: E731
    if len(forward.nodes) > 1 and len(backward.nodes) > 1:
        assert first(forward) == ["остаток"] and first(backward) == ["промахи"]


def test_effective_depth_is_measured_apart_from_the_budget():
    cols, target = _sum_of_two()
    solved = problems.solve(problems.Problem.whole(cols, target), P.CURRENT, 60000,
                            catalogue=problems.BRANCHES, flat_share=0.5)
    before = dict(solved.ledger.spent)
    depth, spent = problems.effective_depth(solved, P.CURRENT)
    assert solved.ledger.spent == before
    assert 1 <= depth <= solved.used_depth() and spent >= 0
    assert show(solved.program)
