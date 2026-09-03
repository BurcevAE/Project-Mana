"""
tests/test_experiments.py — the research loop, and the things that stop it
from becoming a compute shredder.

The properties worth protecting:

  1. hypotheses come from failure patterns, not from a model's imagination;
  2. the lab picks the experiment that resolves the most, not the one that
     scores best;
  3. it stops -- on budget, on count, and on having stopped learning.
"""
from __future__ import annotations

import pytest

from mana.cognition import experiments as lab
from mana.cognition.experiments import ExperimentLab, Hypothesis
from mana.cognition.gaps import detect
from mana.cognition.self_model import Observation, SelfModel


class FakeTask:
    def __init__(self, task_id, domain="arithmetic", difficulty=0.8):
        self.task_id = task_id
        self.domain = domain
        self.difficulty = difficulty


def model_with(domain, difficulty, correct, n, reason="wrong", brain=""):
    model = SelfModel()
    for i in range(n):
        model.record(Observation(f"{domain}{i}", domain, difficulty, correct,
                                 reason="" if correct else reason, brain=brain, calls=1))
    return model


@pytest.fixture
def journal(tmp_path, monkeypatch):
    monkeypatch.setenv("MANA_DATA_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# hypotheses come from evidence
# ---------------------------------------------------------------------------

def test_a_hypothesis_names_a_runnable_intervention_not_a_sentiment():
    """"Be more careful" cannot be run. A chain of steps can."""
    model = model_with("arithmetic", 0.9, False, 30, reason="format")
    hypotheses = lab.hypotheses_from_model(model)
    assert hypotheses
    h = hypotheses[0]
    assert h.candidate_steps != h.baseline_steps
    assert all(isinstance(s, str) for s in h.candidate_steps)


def test_the_failure_pattern_chooses_the_intervention():
    """Format failures and wrong answers are different problems."""
    format_model = model_with("arithmetic", 0.9, False, 30, reason="format")
    wrong_model = model_with("arithmetic", 0.9, False, 30, reason="wrong")
    format_h = lab.hypotheses_from_model(format_model)[0]
    wrong_h = lab.hypotheses_from_model(wrong_model)[0]
    assert "CRITIQUE" in format_h.candidate_steps
    assert "VERIFY" in wrong_h.candidate_steps


def test_a_knowledge_gap_produces_no_hypothesis():
    """Proposing a mechanism for something not yet known to be broken is
    how a search spends its budget on phantoms."""
    model = model_with("logic", 0.5, True, 3)
    knowledge_gaps = [g for g in detect(model) if g.kind == "knowledge"]
    assert knowledge_gaps
    assert all(lab.hypotheses_from_gap(g) == [] for g in knowledge_gaps)


def test_a_slice_with_two_failure_patterns_gets_two_hypotheses():
    """An intervention aimed at the average of two problems addresses
    neither."""
    # A medium band, so only the two failure patterns drive the result --
    # the "hard" band adds a decomposition hypothesis of its own, which is
    # correct behaviour but would obscure what this test is about.
    model = SelfModel()
    for i in range(15):
        model.record(Observation(f"a{i}", "arithmetic", 0.5, False, reason="format"))
    for i in range(15):
        model.record(Observation(f"b{i}", "arithmetic", 0.5, False, reason="wrong"))
    gap = next(g for g in detect(model) if g.capability_id == "arithmetic/medium")
    hypotheses = lab.hypotheses_from_gap(gap)
    assert len(hypotheses) == 2
    assert {"CRITIQUE" in h.candidate_steps for h in hypotheses} == {True}
    assert any("VERIFY" in h.candidate_steps for h in hypotheses)


def test_a_hypothesis_is_narrow_enough_to_be_refuted():
    model = model_with("sequence", 0.9, False, 30)
    h = lab.hypotheses_from_model(model)[0]
    assert "sequence/hard" in h.statement
    assert "→" in h.statement


# ---------------------------------------------------------------------------
# choosing what to run
# ---------------------------------------------------------------------------

def test_the_more_uncertain_question_is_worth_more():
    """§12: the experiment that resolves the most, not the one that scores
    best."""
    thin = model_with("arithmetic", 0.9, False, 8)
    thick = model_with("arithmetic", 0.9, False, 300)
    h_thin = lab.hypotheses_from_model(thin)[0]
    h_thick = lab.hypotheses_from_model(thick)[0]
    assert lab.plan(h_thin, thin).value > lab.plan(h_thick, thick).value


def test_cost_is_estimated_before_anything_is_spent():
    model = model_with("arithmetic", 0.9, False, 30, reason="wrong")
    p = lab.plan(lab.hypotheses_from_model(model)[0], model, trials=30)
    assert p.estimated_calls > 0
    # baseline GENERATE = 1 brain call; candidate GENERATE+CRITIQUE+REPAIR = 3.
    # VERIFY costs nothing here: it is a tool, not a brain, which is
    # exactly the distinction that makes verification cheap to add.
    assert p.estimated_calls == 30 * (1 + 3)


def test_an_experiment_that_does_not_fit_the_budget_is_not_selected():
    model = model_with("arithmetic", 0.9, False, 30)
    plans = [lab.plan(h, model) for h in lab.hypotheses_from_model(model)]
    assert lab.select(plans, budget_left=5) is None


def test_a_worthless_experiment_is_refused_even_with_budget_left():
    """Running one because there is budget left converts compute into
    noise."""
    model = model_with("arithmetic", 0.5, True, 500)
    h = Hypothesis("h1", "s", "g", "arithmetic", "medium",
                   ("OBSERVE", "GENERATE", "ANSWER"),
                   ("OBSERVE", "GENERATE", "CRITIQUE", "ANSWER"))
    p = lab.plan(h, model, trials=30)
    assert p.value < lab.MIN_EXPERIMENT_VALUE
    assert lab.select([p], budget_left=100_000) is None


def test_the_value_weights_are_declared_rather_than_buried():
    assert set(lab.VALUE_WEIGHTS) == {"information_gain", "capability_gain", "cost"}
    assert lab.VALUE_WEIGHTS["cost"] < 0


# ---------------------------------------------------------------------------
# running and concluding
# ---------------------------------------------------------------------------

def runner_where_candidate_wins(steps, task):
    """Baseline fails, candidate succeeds -- a clean effect."""
    is_candidate = len(steps) > 3
    return (is_candidate, len([s for s in steps if s in lab._GENERATIVE]))


def runner_with_no_difference(steps, task):
    return (task.task_id.endswith(("0", "2", "4", "6", "8")), 1)


def test_a_real_effect_is_measured_paired(journal):
    model = model_with("arithmetic", 0.9, False, 30)
    lab_ = ExperimentLab(model)
    p = lab.plan(lab.hypotheses_from_model(model)[0], model, trials=40)
    tasks = [FakeTask(f"t{i}") for i in range(40)]
    m = lab_.run_experiment(p, tasks, runner_where_candidate_wins)
    assert len(m.outcomes) == 40
    assert m.summary()["baseline"] == 0.0
    assert m.summary()["candidate"] == 1.0


def test_the_lab_does_not_decide_the_gate_does(journal):
    """The lab assembles, asks and writes down the answer."""
    model = model_with("arithmetic", 0.9, False, 30)
    lab_ = ExperimentLab(model)
    p = lab.plan(lab.hypotheses_from_model(model)[0], model, trials=40)
    tasks = [FakeTask(f"t{i}") for i in range(40)]
    m = lab_.run_experiment(p, tasks, runner_where_candidate_wins)
    discovery = lab_.conclude(p, m, hidden=(0.30, 0.80), counterexamples=(20, 0))
    assert discovery.status == lab.SUPPORTED
    assert discovery.verdict["accepted"] is True


def test_an_effect_with_incomplete_evidence_is_refuted_not_accepted(journal):
    model = model_with("arithmetic", 0.9, False, 30)
    lab_ = ExperimentLab(model)
    p = lab.plan(lab.hypotheses_from_model(model)[0], model, trials=40)
    tasks = [FakeTask(f"t{i}") for i in range(40)]
    m = lab_.run_experiment(p, tasks, runner_where_candidate_wins)
    discovery = lab_.conclude(p, m)          # no hidden, no counterexamples
    assert discovery.status == lab.REFUTED
    assert "hidden_confirms" in discovery.verdict["failed_gates"]
    assert "counterexamples" in discovery.verdict["failed_gates"]


def test_refutations_are_recorded_too(journal):
    """A lab that records only successes cannot tell learning from luck."""
    model = model_with("arithmetic", 0.9, False, 30)
    lab_ = ExperimentLab(model)
    p = lab.plan(lab.hypotheses_from_model(model)[0], model, trials=40)
    tasks = [FakeTask(f"t{i}") for i in range(40)]
    lab_.conclude(p, lab_.run_experiment(p, tasks, runner_with_no_difference))
    assert len(lab_.discoveries) == 1
    assert lab_.report()["refuted"] == 1
    assert lab_.report()["discoveries"] == []


def test_every_experiment_leaves_a_transaction_record(journal):
    from mana.core import transaction
    model = model_with("arithmetic", 0.9, False, 30)
    lab_ = ExperimentLab(model)
    p = lab.plan(lab.hypotheses_from_model(model)[0], model, trials=40)
    tasks = [FakeTask(f"t{i}") for i in range(40)]
    lab_.conclude(p, lab_.run_experiment(p, tasks, runner_where_candidate_wins),
                  hidden=(0.3, 0.8), counterexamples=(20, 0))
    entries = transaction.read_journal()
    assert entries and entries[0].last_state == transaction.COMMITTED
    assert transaction.unfinished() == []


# ---------------------------------------------------------------------------
# stopping
# ---------------------------------------------------------------------------

def test_the_budget_stops_the_loop(journal):
    lab_ = ExperimentLab(model_with("arithmetic", 0.9, False, 30), budget_calls=10)
    lab_.calls_used = 10
    assert "бюджет исчерпан" in lab_.stop_reason()


def test_the_experiment_count_stops_the_loop(journal):
    lab_ = ExperimentLab(model_with("arithmetic", 0.9, False, 30), max_experiments=1)
    lab_.discoveries.append(lab.Discovery("d", {}, {}, {}, lab.REFUTED))
    assert "предел экспериментов" in lab_.stop_reason()


def test_a_loop_that_has_stopped_learning_stops_running(journal):
    """Without this an autonomous researcher runs forever, because each
    individual experiment stays affordable."""
    lab_ = ExperimentLab(model_with("arithmetic", 0.9, False, 30))
    lab_._recent_values = [0.0] * lab.DIMINISHING_WINDOW
    assert "убывающая отдача" in lab_.stop_reason()


def test_a_run_of_rejections_that_still_resolve_things_is_not_stopped(journal):
    """Measured on realised information, not on acceptance: rejections
    that narrow intervals are productive."""
    lab_ = ExperimentLab(model_with("arithmetic", 0.9, False, 30))
    lab_._recent_values = [0.25] * lab.DIMINISHING_WINDOW
    assert lab_.stop_reason() == ""


def test_a_fresh_lab_has_no_reason_to_stop(journal):
    assert ExperimentLab(model_with("arithmetic", 0.9, False, 30)).stop_reason() == ""


def test_the_report_separates_supported_from_merely_run(journal):
    lab_ = ExperimentLab(model_with("arithmetic", 0.9, False, 30))
    lab_.discoveries = [lab.Discovery("a", {}, {}, {}, lab.SUPPORTED),
                        lab.Discovery("b", {}, {}, {}, lab.REFUTED)]
    report = lab_.report()
    assert report["experiments"] == 2
    assert report["supported"] == 1
    assert len(report["discoveries"]) == 1
