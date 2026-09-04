"""
tests/test_meta.py — evolving the search, and the four guards that keep
it from eating its own criteria.

The tests that matter most here are the refusals. A meta layer that
works is unremarkable; a meta layer that cannot reach the acceptance
rules is the whole claim.
"""
from __future__ import annotations

import pytest

from mana.core import instrument
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
    """A world where a higher weight genuinely searches better.

    The `record_read` is not decoration: a real episode reads the weight
    inside `gaps._build`, and without the equivalent here the activation
    guard correctly refuses to rule -- which is exactly what it did when
    it was added, to every test in this file at once.
    """
    weight = list(policy.values())[0]
    for name in policy:
        instrument.record_read(name)
    return EpisodeResult(seed=seed, policy_id=str(weight),
                         resolved=0.3 + 0.5 * weight + (seed % 3) * 0.02,
                         capability_gain=0.0, calls_used=200,
                         accepted_claims=1)


def flat_episodes(seed, policy):
    """A world where the parameter is read and changes nothing."""
    for name in policy:
        instrument.record_read(name)
    return EpisodeResult(seed=seed, policy_id="flat", resolved=0.6,
                         capability_gain=0.0, calls_used=200, accepted_claims=1)


def blind_episodes(seed, policy):
    """A world where the parameter is never consulted at all."""
    return EpisodeResult(seed=seed, policy_id="blind", resolved=0.9,
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
# guard 2b: an experiment that never touched the parameter is not a result
# ---------------------------------------------------------------------------

def test_an_episode_that_never_reads_the_parameter_is_refused(isolated_config):
    """The trap a live run walked into: two policies scored an identical
    1.398 and the natural reading was "this parameter does not matter".
    The true reading was that the code reading it never ran.

    Both readings produce the same numbers, so no statistic separates
    them. Only a counter at the point of use does.
    """
    p = propose(parameter(value=0.2), 1.2, "должен помочь")
    run_episodes(p, SEEDS, blind_episodes)
    verdict = judge(p, hidden=(0.55, 0.70), counterexamples=(4, 0))
    assert not verdict.accepted
    assert verdict.failed_gates == ("parameter_not_exercised",)
    assert p.parameter_reads == 0


def test_not_exercised_is_recorded_apart_from_refuted(isolated_config):
    """A refusal to rule must not go into the record as evidence against
    the parameter -- that would stop it being tried again on the strength
    of an experiment that never touched it."""
    p = propose(parameter(value=0.2), 1.2)
    run_episodes(p, SEEDS, blind_episodes)
    judge(p, hidden=(0.55, 0.70), counterexamples=(4, 0))
    assert p.status == "NOT_EXERCISED"
    assert p.status != "REJECTED"


def test_a_read_parameter_is_counted_and_ruled_on(isolated_config):
    p = propose(parameter(value=0.2), 1.2)
    run_episodes(p, SEEDS, episodes)
    assert p.parameter_reads >= len(SEEDS) * 2
    verdict = judge(p, hidden=(0.55, 0.70), counterexamples=(4, 0))
    assert "parameter_not_exercised" not in verdict.failed_gates


def test_the_real_gap_ranking_reads_its_weights(isolated_config):
    """The counter has to fire from the production path, not only from a
    test double -- otherwise it proves nothing about real episodes."""
    from mana.cognition import gaps
    from mana.cognition.self_model import Observation, SelfModel
    model = SelfModel()
    for i in range(24):
        model.record(Observation(f"t{i}", "arithmetic", 0.8, False,
                                 reason="wrong", calls=1))
    with instrument.watching() as used:
        assert gaps.detect(model)
    assert used.get("gap.severity", 0) > 0
    assert used.get("gap.cost", 0) > 0


# ---------------------------------------------------------------------------
# guard 3: yield is what the search was for, not how much it accepted
# ---------------------------------------------------------------------------

def test_a_policy_that_accepts_more_but_resolves_less_does_not_win(isolated_config):
    """Counting accepted claims rewards a policy proposing safe trivia:
    twenty tiny true claims beat one real mechanism on that scale."""
    def trivia(seed, policy):
        for name in policy:
            instrument.record_read(name)
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


# ---------------------------------------------------------------------------
# guard 2c: read is necessary, not sufficient
# ---------------------------------------------------------------------------

def decided(same: bool):
    """Episodes that record what they chose. `same` makes both arms
    choose identically however the weight is set."""
    def runner(seed, policy):
        weight = list(policy.values())[0]
        for name in policy:
            instrument.record_read(name)
        picks = ("measure-a", "measure-b") if same else (
            ("measure-a",) if weight < 0.5 else ("experiment-x",))
        return EpisodeResult(seed=seed, policy_id=str(weight), resolved=0.9,
                             capability_gain=0.0, calls_used=200,
                             accepted_claims=1, decisions=picks)
    return runner


def test_a_parameter_read_and_discarded_is_still_not_a_result(isolated_config):
    """Found by fixing the previous guard. A live run counted 48 reads of
    gap.cost and both arms still scored an identical 1.398: the options
    it ranks lost to a category of options not ranked by it at all.

    Identical decisions on the same seed give identical outcomes
    necessarily, so "no effect" there is vacuous rather than measured.
    """
    p = propose(parameter(value=0.2), 1.2)
    run_episodes(p, SEEDS, decided(same=True))
    verdict = judge(p, hidden=(0.55, 0.70), counterexamples=(4, 0))
    assert p.parameter_reads > 0, "the parameter WAS consulted"
    assert p.decisions_differed == 0
    assert verdict.failed_gates == ("parameter_had_no_influence",)


def test_a_parameter_that_changed_a_decision_is_ruled_on_normally(isolated_config):
    p = propose(parameter(value=0.2), 1.2)
    run_episodes(p, SEEDS, decided(same=False))
    verdict = judge(p, hidden=(0.55, 0.70), counterexamples=(4, 0))
    assert p.decisions_differed == len(SEEDS)
    assert "parameter_had_no_influence" not in verdict.failed_gates


def test_without_recorded_decisions_the_influence_check_stays_silent(isolated_config):
    """It cannot tell, so it does not claim to. A runner that reports no
    decisions gets the ordinary verdict, and the honest cost of that is
    that this particular trap goes undetected for it."""
    p = propose(parameter(value=0.2), 1.2)
    run_episodes(p, SEEDS, episodes)
    verdict = judge(p, hidden=(0.55, 0.70), counterexamples=(4, 0))
    assert p.decisions_differed == 0
    assert "parameter_had_no_influence" not in verdict.failed_gates
    assert verdict.accepted


# ---------------------------------------------------------------------------
# the far end of the loop
# ---------------------------------------------------------------------------

def test_an_accepted_change_is_written_where_the_search_reads_it(isolated_config):
    """From phase 13 until now an accepted meta-change updated a field on
    the MetaEvolution object and nothing else. The search kept reading
    the old number, and the live script papered over it by patching the
    module itself -- which is how an architectural gap becomes "it
    works"."""
    from mana.cognition import gaps
    m = MetaEvolution()
    before = gaps.PRIORITY_WEIGHTS["cost"]
    p = m.propose("gap.cost", before + 0.9, "шире поиск")
    assert m.evaluate(p, SEEDS, episodes, hidden=(0.55, 0.70), counterexamples=(4, 0))
    assert gaps.PRIORITY_WEIGHTS["cost"] != before
    assert gaps.PRIORITY_WEIGHTS["cost"] == m.policy()["gap.cost"]


def test_a_rollback_puts_the_module_back_too(isolated_config):
    from mana.cognition import gaps
    m = MetaEvolution()
    before = gaps.PRIORITY_WEIGHTS["cost"]
    p = m.propose("gap.cost", before + 0.9, "шире поиск")
    m.evaluate(p, SEEDS, episodes, hidden=(0.55, 0.70), counterexamples=(4, 0))
    m.rollback(p)
    assert gaps.PRIORITY_WEIGHTS["cost"] == before


def test_the_write_path_goes_through_the_same_guard(isolated_config):
    """`propose` guards what may be proposed. Without the same check on
    the write path, applying would be a way around it."""
    from mana.cognition.meta import put_in_force
    with pytest.raises(MetaError):
        put_in_force(parameter(name="gates.alpha", module="mana.core.gates",
                               key="ALPHA"), 0.99)


def test_the_policy_survives_a_restart(isolated_config, tmp_path):
    """"Accepted" means nothing if the next process reads the old number."""
    from mana.cognition import gaps
    path = tmp_path / "policy.json"
    m = MetaEvolution()
    p = m.propose("gap.cost", m.policy()["gap.cost"] + 0.9, "шире поиск")
    m.evaluate(p, SEEDS, episodes, hidden=(0.55, 0.70), counterexamples=(4, 0))
    adopted = m.policy()["gap.cost"]
    m.save(path)

    # A fresh process: the module is back to its declared default.
    gaps.PRIORITY_WEIGHTS["cost"] = -0.4
    restarted = MetaEvolution()
    assert restarted.policy()["gap.cost"] == -0.4
    restored = restarted.load(path)
    assert restored["gap.cost"] == adopted
    assert gaps.PRIORITY_WEIGHTS["cost"] == adopted


def test_verify_in_force_catches_a_module_that_disagrees(isolated_config):
    from mana.cognition import gaps
    m = MetaEvolution()
    assert m.verify_in_force()["ok"] is True
    gaps.PRIORITY_WEIGHTS["cost"] = 0.777
    report = m.verify_in_force()
    assert report["ok"] is False
    assert "gap.cost" in report["mismatches"]


def test_a_policy_file_cannot_smuggle_in_a_forbidden_parameter(isolated_config, tmp_path):
    """A hand-edited file must not be a way past check_tunable."""
    import json
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"policy": {"gates.alpha": 0.99,
                                           "unknown.thing": 1.0}}),
                    encoding="utf-8")
    restored = MetaEvolution().load(path)
    assert restored == {}


def test_a_policy_file_from_a_later_version_does_not_stop_startup(isolated_config,
                                                                 tmp_path):
    import json
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"policy": {"gap.cost": -0.5,
                                           "future.parameter": 3.0}}),
                    encoding="utf-8")
    restored = MetaEvolution().load(path)
    assert restored == {"gap.cost": -0.5}
