"""
The sequential verdict: how much experience is enough, decided by the
evidence rather than by a fixed count.

The exact paths pin the arithmetic; the simulations pin the guarantees --
under no effect it accepts no more often than Wald's bound, an effect of
exactly `min_effect` is accepted at least as often as the bound promises,
and the forecast of how much is needed agrees with what it took.
"""
from __future__ import annotations

import math
import random

import pytest

from mana import core
from mana.core import sequential as seq
from mana.core.gates import ACCEPTED, NOT_EVALUATED, REJECTED


def test_it_lives_inside_the_immutable_core():
    assert core.is_immutable_path(seq.__file__)


def test_the_bounds_are_walds():
    plan = seq.Plan(min_effect=0.10)
    assert plan.alpha == 0.05 and plan.beta == 0.20
    assert plan.accept_at == pytest.approx(math.log(16.0))
    assert plan.reject_at == pytest.approx(math.log(0.2 / 0.95))
    strict = seq.Plan(min_effect=0.10, consequence=seq.IRREVERSIBLE)
    assert strict.accept_at == pytest.approx(math.log(0.9 / 0.01))
    assert strict.reject_at == pytest.approx(math.log(0.1 / 0.99))


@pytest.mark.parametrize("bad", [{"min_effect": 0.0}, {"min_effect": 0.5},
                                 {"min_effect": -0.1},
                                 {"min_effect": 0.1, "consequence": "как-нибудь"},
                                 {"min_effect": 0.1, "ceiling": -1}])
def test_a_plan_that_means_nothing_is_refused(bad):
    with pytest.raises(ValueError):
        seq.Plan(**bad)


def test_sixteen_wins_in_a_row_accept_and_fifteen_do_not():
    test = seq.SequentialTest(seq.Plan(min_effect=0.10))
    test.observe_all([True] * 15)
    assert test.status == seq.CONTINUE
    assert test.observe(True) == ACCEPTED and test.trials == 16


def test_seven_losses_in_a_row_reject():
    test = seq.SequentialTest(seq.Plan(min_effect=0.10))
    test.observe_all([False] * 6)
    assert test.status == seq.CONTINUE
    assert test.observe(False) == REJECTED and test.trials == 7


def test_running_out_of_budget_is_not_a_refutation():
    test = seq.SequentialTest(seq.Plan(min_effect=0.10, ceiling=10))
    test.observe_all([True, False] * 5)
    assert test.status == NOT_EVALUATED
    assert "это не отказ" in test.note()


def test_the_verdict_is_the_first_crossing_and_does_not_move():
    """Looking again after a decision is how a test gets fooled."""
    test = seq.SequentialTest(seq.Plan(min_effect=0.10))
    test.observe_all([True] * 16 + [False] * 40)
    assert test.status == ACCEPTED
    assert (test.won, test.lost, test.late) == (16, 0, 40)


def test_a_record_survives_the_disk_and_cannot_claim_what_its_numbers_do_not():
    test = seq.SequentialTest(seq.Plan(min_effect=0.10, ceiling=200))
    test.observe_all([True, True, False] * 4)
    back = seq.SequentialTest.from_dict(test.as_dict())
    assert back.as_dict() == test.as_dict()
    forged = dict(test.as_dict(), status=ACCEPTED)
    with pytest.raises(ValueError):
        seq.SequentialTest.from_dict(forged)


def test_the_forecast_is_walds_at_both_hypotheses():
    plan = seq.Plan(min_effect=0.10)
    at_null = seq.forecast(plan, 0.5)
    at_target = seq.forecast(plan, 0.6)
    assert at_null["accept"] == pytest.approx(plan.alpha, abs=1e-6)
    assert at_target["accept"] == pytest.approx(1 - plan.beta, abs=1e-6)
    assert at_null["trials"] == pytest.approx(65.7, abs=0.5)
    assert at_target["trials"] == pytest.approx(94.7, abs=0.5)


def test_a_running_test_says_how_far_it_still_has_to_go():
    test = seq.SequentialTest(seq.Plan(min_effect=0.10))
    test.observe_all([True, False, True])
    assert "до решения примерно ещё" in test.note()
    assert test.forecast()["trials"] > 0


def _simulate(share, runs, plan, seed):
    rng = random.Random(seed)
    counts = {ACCEPTED: 0, REJECTED: 0, NOT_EVALUATED: 0}
    trials = 0
    for _ in range(runs):
        test = seq.SequentialTest(plan)
        while not test.decided:
            test.observe(rng.random() < share)
        counts[test.status] += 1
        trials += test.trials
    return {k: v / runs for k, v in counts.items()}, trials / runs


def test_under_no_effect_it_accepts_no_more_often_than_walds_bound():
    plan = seq.Plan(min_effect=0.10)
    rates, _ = _simulate(0.5, 2000, plan, seed=1)
    assert rates[ACCEPTED] <= plan.alpha / (1 - plan.beta)


def test_an_effect_of_exactly_min_effect_is_accepted_as_promised():
    plan = seq.Plan(min_effect=0.10)
    rates, _ = _simulate(0.6, 2000, plan, seed=2)
    assert rates[ACCEPTED] >= 1 - plan.beta / (1 - plan.alpha)


def test_a_harmful_change_is_all_but_never_accepted():
    rates, mean = _simulate(0.4, 2000, seq.Plan(min_effect=0.10), seed=3)
    assert rates[ACCEPTED] <= 0.005
    assert mean < 40                       # and it is found out quickly


@pytest.mark.parametrize("share", [0.5, 0.6, 0.7])
def test_the_forecast_of_how_much_is_needed_matches_what_it_took(share):
    plan = seq.Plan(min_effect=0.10)
    _, mean = _simulate(share, 2000, plan, seed=4)
    assert mean == pytest.approx(seq.forecast(plan, share)["trials"], rel=0.2)
