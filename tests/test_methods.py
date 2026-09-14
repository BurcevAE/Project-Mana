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

from mana.methods import portfolio, solvers, trap
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


# --------------------------------------------------------------------------
# M1b: experiments about methods
# --------------------------------------------------------------------------

def _entry(n, queries, before=frozenset(), method="exhaust", box=0, predicted=0.0, distance=0.0):
    from mana.methods.choice import TASK, Entry
    return Entry(TASK, box, n, before, method, True, True, queries, predicted, 0.0, distance)


def test_one_size_says_nothing_about_growth():
    from mana.methods.choice import Explorer

    explorer = Explorer()
    for box in range(5):
        explorer.journal.add(_entry(4, LEVELS ** 4, box=box))
    belief = explorer.belief("exhaust", frozenset(), -1, (), explorer.journal.drift())
    _, at_seen = belief.predict(4)
    _, two_away = belief.predict(6)
    assert at_seen < 1.0 and two_away > 3.0          # nats: a factor of 20 either way


def test_a_line_seen_at_three_sizes_is_trusted_near_them_and_doubted_far_away():
    from mana.methods.choice import Explorer

    explorer = Explorer()
    for box, n in enumerate((2, 3, 4) * 3):
        explorer.journal.add(_entry(n, LEVELS ** n, box=box))
    belief = explorer.belief("exhaust", frozenset(), -1, (), explorer.journal.drift())
    mu, near = belief.predict(5)
    _, far = belief.predict(8)
    assert abs(mu - np.log(LEVELS ** 5)) < 0.2 and near < far


def test_a_surprise_far_from_what_was_seen_raises_the_doubt_of_extrapolation():
    from mana.methods.choice import Journal

    journal = Journal()
    calm = journal.drift()
    journal.add(_entry(8, 200000, predicted=np.log(600), distance=4))
    assert journal.drift() > 1.5 * calm              # one surprise against the prior's two


def test_a_probe_asks_the_box_itself_and_what_it_learnt_is_remembered():
    from mana.methods.choice import Explorer

    box = BlackBox(6, BLOCKS, seed=5)
    explorer = Explorer()
    session = solvers.Session(box.answer, 6, LEVELS, announce=False)
    found = explorer._probe(session, "experiment", 4, LEVELS, 5, 6)
    assert found.believed and found.queries == session.total == box.asked
    after = solvers.run(session, "exhaust", 5, 10 ** 6)
    assert after.queries == LEVELS ** 6 - found.queries + 0   # the probe's answers are free now


def test_the_trap_costs_what_it_says():
    for n, cost in ((3, trap.explosive_cost(3)), (6, trap.explosive_cost(6)),
                    (3, trap.steady_cost(3))):
        name = "explosive" if cost != trap.steady_cost(3) else "steady"
        box = BlackBox(n, LINEAR, seed=n, levels=trap.LEVELS)
        found = solvers.attempt(name, box.answer, n, trap.LEVELS, n, announce=False,
                                methods=trap.METHODS)
        assert found.believed and box.verdict(found.model) == 1.0
        assert cost <= found.queries <= cost + n + 1 + solvers.CHECKS


def test_the_explorer_and_the_trap_cannot_see_the_world():
    from mana.methods import choice

    for module in (choice, trap):
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                named = [getattr(node, "module", "") or ""] + [a.name for a in node.names]
                assert not any("world" in name for name in named), module.__name__
            if isinstance(node, ast.Attribute) and module is choice:
                assert node.attr not in ("planned", "verdict", "structure", "declined")


def test_after_small_boxes_only_the_explorer_probes_before_the_trap_and_the_chooser_falls_in():
    """The trap as posed, measured 2026-09-14 on 10 streams of 10: after
    n = 2..4, a box of n = 8. The Explorer runs explosive on six of its
    inputs, sees it explode, and applies steady; the Chooser runs
    explosive to its budget first."""
    from mana.methods import choice

    names = tuple(trap.METHODS)
    rng = np.random.default_rng([0, 13])
    sizes = [int(rng.choice((2, 3, 4))) for _ in range(20)]
    chooser = choice.Chooser(choice.Experience(methods=names), registry=trap.METHODS)
    explorer = choice.Explorer(methods=names, registry=trap.METHODS)
    for i, n in enumerate(sizes):
        for agent in (chooser, explorer):
            box = BlackBox(n, LINEAR, i, levels=trap.LEVELS)
            agent.solve(box.answer, n, trap.LEVELS, i)
    box = BlackBox(8, LINEAR, 500, levels=trap.LEVELS)
    fell, _ = chooser.solve(box.answer, 8, trap.LEVELS, 500)
    box = BlackBox(8, LINEAR, 500, levels=trap.LEVELS)
    looked, model = explorer.solve(box.answer, 8, trap.LEVELS, 500)
    assert fell.tried == ("explosive", "steady") and fell.cost > solvers.BUDGET
    assert [(m, size) for m, size, _ in explorer.last_probes] == [("explosive", 6)]
    assert looked.tried == ("steady",) and looked.cost < 30000 and box.verdict(model) == 1.0


# --------------------------------------------------------------------------
# M1c: a probe is worth what it changes in the best plan
# --------------------------------------------------------------------------

def test_the_best_plan_orders_methods_and_counts_what_happens_if_all_fail():
    from mana.methods.choice import best_plan

    table = {"a": (0.6, 50000.0), "b": (0.9, 20000.0)}
    plan = best_plan(["a", "b"], frozenset(), lambda m, failed: table[m], 200000.0)
    # b first, then a: 20 000 + 0.1 x 50 000 + 0.1 x 0.4 x 200 000.
    assert plan.names == ("b", "a") and abs(plan.cost - 33000.0) < 1e-6
    assert abs(plan.success - 0.96) < 1e-9
    from mana.methods.choice import plan_cost

    other = plan_cost(("a", "b"), frozenset(), lambda m, failed: table[m], 200000.0)
    # 50 000 + 0.4 x 20 000 + 0.4 x 0.1 x 200 000: the same methods, dearer in that order.
    assert abs(other.cost - 66000.0) < 1e-6 and other.cost > plan.cost


def test_giving_up_is_a_plan_and_is_chosen_when_every_method_costs_more_than_an_answer():
    from mana.methods.choice import best_plan

    plan = best_plan(["a"], frozenset(), lambda m, failed: (0.1, 50000.0), 200000.0)
    assert plan.names == () and plan.cost == 200000.0


def test_a_step_is_weighed_after_the_steps_before_it_failed():
    from mana.methods.choice import best_plan

    def step(method, failed):
        if method == "second":
            return (0.9 if "first" in failed else 0.1, 1000.0)
        return (0.5, 1000.0)

    plan = best_plan(["first", "second"], frozenset(), step, 200000.0)
    assert plan.names == ("first", "second")


def test_every_decision_says_what_the_plan_was_and_why():
    from mana.methods import choice

    planner = choice.Planner()
    box = BlackBox(3, ADDITIVE, seed=9)
    found, model = planner.solve(box.answer, 3, LEVELS, 9)
    assert box.verdict(model) == 1.0
    applied = [d for d in planner.last_decisions if d.action == "apply"]
    assert applied and all(d.plan.steps and d.method == d.plan.names[0] for d in applied)
    assert [d.method for d in applied] == list(found.tried)


# --------------------------------------------------------------------------
# M1d: laws of cost told apart by experience
# --------------------------------------------------------------------------

def _runs(pairs, repeat=3):
    return [(float(n), float(np.log(cost)), False) for n, cost in pairs] * repeat


def test_runs_that_bend_are_told_from_a_line_and_the_wall_is_expected():
    from mana.methods.choice import ShapeBelief

    bent = ShapeBelief.fit(_runs(((2, 100), (3, 150), (4, 200), (6, 20000))), 0.5, 200000)
    assert bent.law()["bend"] > 0.9 and bent.law()["at"] == 4
    assert bent.outlook(8, 200000)[0] < 0.1                  # will not finish at n = 8
    table = ShapeBelief.fit(_runs(((2, 100), (3, 1000), (4, 10000))), 0.5, 200000)
    assert table.law()["line"] > 0.5
    assert abs(table.predict(6)[0] - np.log(10 ** 6)) < np.log(2)


def test_a_run_cut_off_by_the_budget_is_a_bound_not_a_cost():
    from mana.methods.choice import ShapeBelief

    rows = _runs(((2, 100), (3, 150), (4, 200))) + [(7.0, float(np.log(200000)), True)]
    belief = ShapeBelief.fit(rows, 0.5, 200000)
    assert belief.law()["bend"] > 0.9
    assert belief.outlook(8, 200000)[0] < 0.3


def test_a_box_s_own_run_reweighs_the_laws():
    from mana.methods.choice import ShapeBelief

    kind = ShapeBelief.fit(_runs(((2, 100), (3, 150), (4, 200))), 0.5, 200000)
    assert kind.law()["line"] > 0.5
    seen = kind.observe(6, float(np.log(20000)))
    assert seen.law()["bend"] > 0.9 and seen.predict(8)[0] > np.log(200000)
