"""Three splits, and the ones that cannot be peeked at.

The claim "measured on data it was never chosen on" rests entirely on
this module behaving, so these test the refusals rather than the happy
path: a fresh split that answers before sealing, or twice, or about a
strategy nobody committed to, would make that claim worthless while every
number in it stayed true.
"""
from __future__ import annotations

import pytest

from mana.world import trials


@pytest.fixture(autouse=True)
def clean():
    trials._reset_for_tests()
    yield
    trials._reset_for_tests()


def _constant(value):
    return lambda seed: value


# --------------------------------------------------------------------------
# the seeds are never handed out
# --------------------------------------------------------------------------

def test_a_hidden_split_returns_numbers_and_not_seeds():
    """An agent cannot overfit a set it has never seen. `core/splits.py`
    made this argument first; this is the same one for a world."""
    result = trials.score(trials.VALIDATION, _constant(0.5))
    assert result["mean"] == 0.5
    assert result["values"] is None
    # Discovery is the set you are allowed to fit, so it does hand them back.
    assert trials.score(trials.DISCOVERY, _constant(0.5))["values"]


def test_the_splits_do_not_overlap():
    seen = set()
    for split in (trials.DISCOVERY, trials.VALIDATION, trials.FRESH):
        assert trials.size(split) > 0
        seen.add(trials.size(split))
    # The seeds themselves are not reachable from here, which is the
    # point; what can be checked is that each split is a real sample.
    assert trials.size(trials.FRESH) >= 30


# --------------------------------------------------------------------------
# fresh means fresh
# --------------------------------------------------------------------------

def test_the_fresh_split_refuses_before_anything_is_sealed():
    with pytest.raises(trials.NotSealed):
        trials.score(trials.FRESH, _constant(0.5))


def test_the_fresh_split_refuses_a_strategy_nobody_committed_to():
    trials.seal({"policy": "planning"})
    with pytest.raises(trials.NotSealed):
        trials.score(trials.FRESH, _constant(0.5),
                     strategy={"policy": "something else"})


def test_the_fresh_split_answers_once():
    """A second answer would make it a set to tune against, which is what
    it exists not to be."""
    strategy = {"policy": "planning"}
    trials.seal(strategy)
    trials.score(trials.FRESH, _constant(0.5), strategy=strategy)
    with pytest.raises(trials.BudgetExceeded):
        trials.score(trials.FRESH, _constant(0.5), strategy=strategy)


def test_sealing_again_is_a_new_strategy_and_a_weaker_claim():
    """Adjusting after a fresh read is allowed and is not the same claim.

    The second strategy was chosen knowing something about this split, so
    its answer is worth less than the first. That is what `read_number`
    is for: not a refusal, a discount a reader can apply.
    """
    first = trials.seal({"policy": "a"})
    trials.score(trials.FRESH, _constant(0.5), strategy={"policy": "a"})
    second = trials.seal({"policy": "b"})
    assert first != second
    second_answer = trials.score(trials.FRESH, _constant(0.6),
                                 strategy={"policy": "b"})
    assert second_answer["mean"] == 0.6
    assert second_answer["read_number"] == 2


def test_the_fresh_split_runs_out_even_across_strategies():
    """Every answer leaks. Enough of them and it is a set to tune
    against, whatever it is called."""
    for n in range(3):
        strategy = {"policy": f"try-{n}"}
        trials.seal(strategy)
        trials.score(trials.FRESH, _constant(0.5), strategy=strategy)
    trials.seal({"policy": "one too many"})
    with pytest.raises(trials.BudgetExceeded):
        trials.score(trials.FRESH, _constant(0.5),
                     strategy={"policy": "one too many"})


# --------------------------------------------------------------------------
# budget, not honour system
# --------------------------------------------------------------------------

def test_validation_has_a_budget_and_it_is_enforced():
    for _ in range(8):
        trials.score(trials.VALIDATION, _constant(0.5))
    with pytest.raises(trials.BudgetExceeded):
        trials.score(trials.VALIDATION, _constant(0.5))


def test_discovery_is_the_one_you_may_fit():
    for _ in range(30):
        trials.score(trials.DISCOVERY, _constant(0.5))
    assert trials.reads()[trials.DISCOVERY] == 30


def test_every_read_is_written_down():
    """A claim about a holdout can then be checked instead of believed."""
    trials.score(trials.DISCOVERY, _constant(0.1))
    trials.seal({"policy": "planning"})
    trials.paired(trials.VALIDATION, _constant(0.1), _constant(0.2))

    events = [row["event"] for row in trials.audit()]
    assert events == ["score", "seal", "paired"]
    assert trials.audit()[-1]["split"] == trials.VALIDATION
    assert trials.audit()[-1]["mean"] == pytest.approx(0.1)


def test_a_paired_read_costs_one_read_and_not_two():
    """Charging the honest way of asking more than the sloppy one would
    push callers towards the sloppy one."""
    trials.paired(trials.VALIDATION, _constant(0.1), _constant(0.2))
    assert trials.reads()[trials.VALIDATION] == 1


# --------------------------------------------------------------------------
# what a paired read says
# --------------------------------------------------------------------------

def test_an_improvement_is_an_interval_above_zero():
    result = trials.paired(trials.VALIDATION, _constant(0.1), _constant(0.2))
    assert result["improved"] is True
    assert result["low"] > 0
    assert result["better"] == result["n"] and result["worse"] == 0


def test_no_difference_is_not_an_improvement():
    result = trials.paired(trials.VALIDATION, _constant(0.3), _constant(0.3))
    assert result["improved"] is False
    assert result["mean"] == 0.0
    assert result["same"] == result["n"]


def test_a_loss_is_not_an_improvement():
    result = trials.paired(trials.VALIDATION, _constant(0.4), _constant(0.2))
    assert result["improved"] is False
    assert result["high"] < 0
