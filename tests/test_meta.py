"""
tests/test_meta.py — evolving the search, and the four guards that keep
it from eating its own criteria.

The tests that matter most here are the refusals. A meta layer that
works is unremarkable; a meta layer that cannot reach the acceptance
rules is the whole claim.
"""
from __future__ import annotations

import pytest

from mana.cognition import meta
from mana.cognition.meta import (EPISODE_BAR, EpisodeResult, MAX_CONSECUTIVE,
                                 MetaError, MetaEvolution, MetaParameter,
                                 baseline_parameters, check_tunable, judge,
                                 paired_outcomes, propose, run_episodes,
                                 yield_report)


def parameter(name="gap.cost", module="mana.cognition.gaps", key="cost",
              value=0.5, lo=-2.0, hi=2.0):
    return MetaParameter(name, module, key, value, lo, hi, "тест")


def episodes(seed, policy):
    """A world where a higher weight genuinely searches better."""
    weight = list(policy.values())[0]
    return EpisodeResult(seed=seed, policy_id=str(weight),
                         resolved=0.3 + 0.5 * weight + (seed % 3) * 0.02,
                         capability_gain=0.0, calls_used=200,
                         accepted_claims=1)


def flat_episodes(seed, policy):
    """A world where the parameter changes nothing."""
    return EpisodeResult(seed=seed, policy_id="flat", resolved=0.6,
                         capability_gain=0.0, calls_used=200, accepted_claims=1)


SEEDS = tuple(range(40))


# ---------------------------------------------------------------------------
# guard 1: it cannot reach what counts as a result
# ---------------------------------------------------------------------------

def test_a_core_module_is_refused_by_path():
    """By path, not by a name list -- so an acceptance rule added to the
    core later is covered without anyone remembering to list it."""
    with pytest.raises(MetaError, match="ядру"):
        check_tunable(parameter(name="gates.alpha", module="mana.core.gates",
                                key="ALPHA"))


def test_every_core_submodule_is_refused():
    for module in ("mana.core", "mana.core.oracle", "mana.core.splits",
                   "mana.core.tasks", "mana.core.evaluation"):
        with pytest.raises(MetaError):
            check_tunable(parameter(name="x", module=module, key="anything"))


def test_an_acceptance_threshold_outside_core_is_refused_by_name():
    """The core check has a hole the name check covers: PASS_BAR lives in
    the curriculum, and moving it would lower the bar for passing a
    stage without touching core at all."""
    with pytest.raises(MetaError, match="критерий приёмки"):
        check_tunable(parameter(name="curriculum.pass_bar",
                                module="mana.cognition.curriculum", key="PASS_BAR"))


def test_the_bar_this_module_judges_by_is_itself_refused():
    """Otherwise the first thing a search learns is to lower the bar."""
    with pytest.raises(MetaError):
        check_tunable(parameter(name="meta.bar", module="mana.cognition.meta",
                                key="EPISODE_BAR"))


def test_a_module_not_on_the_list_is_refused_even_if_it_looks_harmless():
    with pytest.raises(MetaError, match="не в списке"):
        check_tunable(parameter(name="brains.timeout", module="mana.brains",
                                key="timeout"))


def test_every_baseline_parameter_passes_its_own_check():
    """The starting set must not contain anything the guard forbids."""
    for name, p in baseline_parameters().items():
        check_tunable(p)


def test_the_parameters_are_read_from_the_modules_that_use_them():
    """A copy here would drift from the value actually in force, and the
    layer would be tuning a number nothing reads."""
    from mana.cognition import experiments, gaps
    params = baseline_parameters()
    assert params["gap.cost"].value == gaps.PRIORITY_WEIGHTS["cost"]
    assert (params["experiment.information_gain"].value ==
            experiments.VALUE_WEIGHTS["information_gain"])


# ---------------------------------------------------------------------------
# guard 2: judged by the same gates as everything else
# ---------------------------------------------------------------------------

def test_a_real_improvement_is_accepted(isolated_config):
    p = propose(parameter(value=0.2), 1.2, "выше вес — шире поиск")
    run_episodes(p, SEEDS, episodes)
    verdict = judge(p, hidden=(0.55, 0.70), counterexamples=(4, 0))
    assert verdict.accepted
    assert p.status == "ACCEPTED"


def test_a_change_that_does_nothing_is_rejected(isolated_config):
    p = propose(parameter(value=0.2), 1.2, "предположительно лучше")
    run_episodes(p, SEEDS, flat_episodes)
    verdict = judge(p, hidden=(0.55, 0.70), counterexamples=(4, 0))
    assert not verdict.accepted
    assert "dev_improvement" in verdict.failed_gates


def test_too_few_episodes_is_refused(isolated_config):
    """Being meta earns no easier ruling -- and this is what makes the
    layer honestly expensive rather than quietly cheap."""
    p = propose(parameter(value=0.2), 1.2)
    run_episodes(p, (1, 2, 3), episodes)
    verdict = judge(p, hidden=(0.55, 0.70), counterexamples=(4, 0))
    assert not verdict.accepted
    assert "sample_size" in verdict.failed_gates


def test_the_comparison_is_paired_on_the_same_seeds(isolated_config):
    p = propose(parameter(value=0.2), 1.2)
    run_episodes(p, SEEDS, episodes)
    assert [e.seed for e in p.baseline] == [e.seed for e in p.candidate]
    assert len(paired_outcomes(p)) == len(SEEDS)


def test_judging_leaves_a_finished_transaction(isolated_config):
    from mana.core import transaction
    p = propose(parameter(value=0.2), 1.2)
    run_episodes(p, SEEDS, episodes)
    judge(p, hidden=(0.55, 0.70), counterexamples=(4, 0))
    assert transaction.unfinished() == []
    assert "meta" in {t.kind for t in transaction.read_journal()}


# ---------------------------------------------------------------------------
# guard 3: yield is what the search was for, not how much it accepted
# ---------------------------------------------------------------------------

def test_a_policy_that_accepts_more_but_resolves_less_does_not_win(isolated_config):
    """Counting accepted claims rewards a policy proposing safe trivia:
    twenty tiny true claims beat one real mechanism on that scale."""
    def trivia(seed, policy):
        prolific = list(policy.values())[0] > 0.5
        return EpisodeResult(seed=seed, policy_id="x",
                             resolved=0.2 if prolific else 0.9,
                             capability_gain=0.0, calls_used=200,
                             accepted_claims=20 if prolific else 1)

    p = propose(parameter(value=0.2), 1.2, "принимает больше заявок")
    run_episodes(p, SEEDS, trivia)
    verdict = judge(p, hidden=(0.55, 0.70), counterexamples=(4, 0))
    assert not verdict.accepted
    report = yield_report(p)
    assert report["candidate_accepted"] > report["baseline_accepted"]
    assert report["candidate_value"] < report["baseline_value"]


def test_accepted_claims_are_reported_but_never_scored():
    import inspect
    source = inspect.getsource(meta.EpisodeResult.value.fget)
    assert "accepted_claims" not in source
    assert "accepted_claims" not in inspect.getsource(meta.paired_outcomes)


def test_capability_gain_counts_toward_the_value():
    e = EpisodeResult(1, "p", resolved=0.3, capability_gain=0.4, calls_used=10)
    assert e.value == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# guard 4: the bar is fixed before the episodes run
# ---------------------------------------------------------------------------

def test_the_bar_is_declared_in_source_not_computed():
    """A threshold chosen after seeing the results is not a threshold."""
    assert isinstance(EPISODE_BAR, float)
    import inspect
    source = inspect.getsource(meta)
    assert f"EPISODE_BAR = {EPISODE_BAR}" in source


def test_the_default_bar_is_used_unless_a_caller_says_otherwise(isolated_config):
    p = propose(parameter(value=0.2), 1.2)
    run_episodes(p, SEEDS, episodes)
    assert paired_outcomes(p) == paired_outcomes(p, EPISODE_BAR)


def test_a_custom_bar_goes_into_the_record(isolated_config):
    from mana.core import transaction
    p = propose(parameter(value=0.2), 1.2)
    run_episodes(p, SEEDS, episodes)
    judge(p, bar=0.9, hidden=(0.55, 0.70), counterexamples=(4, 0))
    measured = [t for t in transaction.read_journal() if t.kind == "meta"]
    assert measured
    assert any(t.data.get("bar") == 0.9 for t in measured)


# ---------------------------------------------------------------------------
# runaway prevention
# ---------------------------------------------------------------------------

def test_a_value_outside_the_bounds_is_clamped_not_refused():
    """Bounds contain the search; they do not make it fail."""
    p = propose(parameter(value=0.5, lo=0.0, hi=1.0), 9.0)
    assert p.new_value == 1.0


def test_a_parameter_cannot_widen_its_own_range():
    param = parameter(value=0.5, lo=0.0, hi=1.0)
    assert param.clamped(50.0) == 1.0
    assert param.clamped(-50.0) == 0.0


def test_a_change_that_changes_nothing_is_refused():
    with pytest.raises(MetaError, match="не меняется"):
        propose(parameter(value=0.5), 0.5)


def test_a_run_of_accepted_changes_in_one_direction_is_stopped():
    """Each step looks locally justified, which is exactly how a search
    walks a parameter to its bound and calls it learning."""
    history = []
    param = parameter(value=0.2)
    for i in range(MAX_CONSECUTIVE):
        p = propose(param, 0.3 + i * 0.1, history=history)
        p.status = "ACCEPTED"
        history.append(p)
    with pytest.raises(MetaError, match="дрейф"):
        propose(param, 1.5, history=history)


def test_a_reversal_resets_the_run():
    history = []
    param = parameter(value=0.5)
    for i in range(MAX_CONSECUTIVE):
        p = propose(param, 0.6 + i * 0.1, history=history)
        p.status = "ACCEPTED"
        history.append(p)
    down = propose(param, 0.1, history=history)
    down.status = "ACCEPTED"
    history.append(down)
    propose(param, 1.5, history=history)        # must not raise


def test_a_rejected_change_does_not_count_toward_the_run():
    history = []
    param = parameter(value=0.2)
    for i in range(MAX_CONSECUTIVE + 2):
        p = propose(param, 0.3 + i * 0.1, history=history)
        p.status = "REJECTED"
        history.append(p)
    propose(param, 1.5, history=history)        # must not raise


# ---------------------------------------------------------------------------
# the evolution object
# ---------------------------------------------------------------------------

def test_an_accepted_change_moves_the_policy_in_force(isolated_config):
    m = MetaEvolution()
    before = m.policy()["gap.cost"]
    p = m.propose("gap.cost", before + 0.9, "шире поиск")
    assert m.evaluate(p, SEEDS, episodes, hidden=(0.55, 0.70), counterexamples=(4, 0))
    assert m.policy()["gap.cost"] != before


def test_a_rejected_change_leaves_the_policy_alone(isolated_config):
    m = MetaEvolution()
    before = m.policy()["gap.cost"]
    p = m.propose("gap.cost", before + 0.9)
    assert not m.evaluate(p, SEEDS, flat_episodes,
                          hidden=(0.55, 0.70), counterexamples=(4, 0))
    assert m.policy()["gap.cost"] == before


def test_a_change_can_be_rolled_back_without_a_gate(isolated_config):
    """Requiring evidence to undo a change leaves it in force exactly
    when the evidence for it has gone."""
    m = MetaEvolution()
    before = m.policy()["gap.cost"]
    p = m.propose("gap.cost", before + 0.9, "шире поиск")
    m.evaluate(p, SEEDS, episodes, hidden=(0.55, 0.70), counterexamples=(4, 0))
    m.rollback(p)
    assert m.policy()["gap.cost"] == before
    assert p.status == "ROLLED_BACK"

    import inspect
    assert "judge" not in inspect.getsource(MetaEvolution.rollback)


def test_an_unknown_parameter_is_refused(isolated_config):
    with pytest.raises(MetaError, match="неизвестный"):
        MetaEvolution().propose("gates.alpha", 0.99)


def test_the_report_shows_the_policy_and_how_it_got_there(isolated_config):
    m = MetaEvolution()
    p = m.propose("gap.cost", m.policy()["gap.cost"] + 0.9, "шире поиск")
    m.evaluate(p, SEEDS, episodes, hidden=(0.55, 0.70), counterexamples=(4, 0))
    report = m.report()
    assert "gap.cost" in report["policy"]
    assert report["accepted"] == ["gap.cost"]
    assert report["changes"][0]["rationale"] == "шире поиск"


def test_the_cost_of_a_meta_conclusion_is_reported_honestly(isolated_config):
    """Tens of thousands of calls for one conclusion. Reported rather
    than hidden behind an encouraging default."""
    p = propose(parameter(value=0.2), 1.2)
    run_episodes(p, SEEDS, episodes)
    report = yield_report(p)
    assert report["episodes"] == len(SEEDS)
    assert report["calls"] == 2 * len(SEEDS) * 200
