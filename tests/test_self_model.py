"""
tests/test_self_model.py — what MANA may claim about itself, and what it
may not.

The rule under test throughout: a capability exists because something
measured it. There is no path from a model's opinion to a score, and no
way to be confident from three samples.
"""
from __future__ import annotations

import json

import pytest

from mana.cognition import gaps
from mana.cognition.self_model import (BANDS, MIN_OBSERVATIONS, Observation,
                                       SelfModel, band_of, wilson_interval)


def observations(domain, difficulty, correct, n, reason="wrong", brain="", program=""):
    return [Observation(f"{domain}-{difficulty}-{i}", domain, difficulty, correct,
                        reason="" if correct else reason, brain=brain, program=program,
                        calls=1)
            for i in range(n)]


def model_with(*groups):
    model = SelfModel()
    for group in groups:
        for o in group:
            model.record(o)
    return model


# ---------------------------------------------------------------------------
# no self-assessment
# ---------------------------------------------------------------------------

def test_there_is_no_way_to_assert_a_capability():
    """A self-model an LLM can write to is a self-report, which is exactly
    the evidence this project may not accept."""
    model = SelfModel()
    assert not hasattr(model, "set_capability")
    assert not hasattr(model, "set_score")
    public = [n for n in dir(model) if not n.startswith("_")]
    assert not any("set" in n or "assert" in n or "claim" in n for n in public)


def test_capabilities_come_only_from_recorded_observations():
    model = model_with(observations("arithmetic", 0.2, True, 8))
    caps = model.capabilities()
    assert caps["arithmetic"].observations == 8
    assert caps["arithmetic"].score == 1.0


def test_grades_are_ingested_straight_from_the_core_grader(isolated_config):
    """The only route in runs through a grader that computed the answer."""
    from mana.core import oracle, tasks
    generated = tasks.generate("arithmetic", 6, seed=3)
    grades = [oracle.grade(t, str(t.answer)) for t in generated]
    model = SelfModel()
    added = model.record_grades(generated, grades, program="direct")
    assert added == 6
    assert model.capabilities()["arithmetic"].score == 1.0


# ---------------------------------------------------------------------------
# confidence is an interval
# ---------------------------------------------------------------------------

def test_three_out_of_three_is_not_certainty():
    """The obvious encoding makes a capability measured three times
    indistinguishable from one measured three hundred times."""
    lo, hi = wilson_interval(3, 3)
    assert lo < 0.5
    assert hi <= 1.0


def test_the_interval_narrows_as_evidence_accumulates():
    narrow = wilson_interval(90, 100)
    wide = wilson_interval(9, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_the_interval_never_leaves_the_unit_range():
    """The normal approximation runs below 0 at the sample sizes this
    system actually collects."""
    for successes, trials in ((0, 3), (3, 3), (1, 100), (99, 100)):
        lo, hi = wilson_interval(successes, trials)
        assert 0.0 <= lo <= hi <= 1.0


def test_too_little_evidence_is_reported_as_unmeasured_not_as_a_low_score():
    """"We do not know" and "it is bad" lead to different actions."""
    model = model_with(observations("logic", 0.5, False, 2))
    cap = model.capabilities()["logic"]
    assert cap.measured is False
    assert "не измерено" in cap.describe()


# ---------------------------------------------------------------------------
# conditions, not averages
# ---------------------------------------------------------------------------

def test_capability_is_sliced_by_difficulty_band():
    """"MANA is 0.7 at arithmetic" is nearly useless; 0.9 easy and 0.3
    hard is a statement a learning goal can be written against."""
    model = model_with(observations("arithmetic", 0.1, True, 10),
                       observations("arithmetic", 0.9, False, 10))
    caps = model.capabilities()
    assert caps["arithmetic/easy"].score == 1.0
    assert caps["arithmetic/hard"].score == 0.0
    assert 0.4 < caps["arithmetic"].score < 0.6


def test_failure_modes_are_kept_apart():
    """A capability failing on output format needs a different fix from
    one computing the wrong number."""
    model = model_with(observations("sequence", 0.5, False, 6, reason="format"),
                       observations("sequence", 0.55, False, 4, reason="wrong"))
    cap = model.capabilities()["sequence"]
    assert cap.failure_modes == {"format": 6, "wrong": 4}


def test_ungradable_attempts_do_not_count_against_the_score():
    """A missing sandbox is a measurement failure, not a capability one."""
    model = model_with(observations("code", 0.5, True, 6),
                       observations("code", 0.5, False, 4, reason="ungradable"))
    cap = model.capabilities()["code"]
    assert cap.observations == 6
    assert cap.score == 1.0


def test_a_per_brain_rate_needs_its_own_evidence():
    """Two attempts is noise wearing a decimal point."""
    model = model_with(observations("arithmetic", 0.3, True, 6, brain="groq"),
                       observations("arithmetic", 0.3, False, 2, brain="ollama"))
    cap = model.capabilities()["arithmetic"]
    assert "groq" in cap.by_brain
    assert "ollama" not in cap.by_brain


def test_transfer_profile_separates_one_capability_from_two_sharing_a_name():
    model = model_with(observations("arithmetic", 0.3, True, 9),
                       observations("logic", 0.3, False, 9))
    profile = model.transfer_profile()
    assert profile["spread"] == 1.0
    assert profile["does_not_transfer"]
    assert profile["transfers"] == []


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def test_observations_are_stored_and_conclusions_are_recomputed(tmp_path):
    """A stored conclusion can drift away from the evidence that produced
    it; a stored observation cannot."""
    model = model_with(observations("arithmetic", 0.2, True, 7))
    path = tmp_path / "self_model.json"
    model.save(path)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert "capabilities" not in stored
    assert len(stored["observations"]) == 7
    assert SelfModel.load(path).capabilities()["arithmetic"].score == 1.0


def test_a_corrupt_self_model_does_not_stop_mana(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    restored = SelfModel.load(path)
    assert restored.observations == []


# ---------------------------------------------------------------------------
# gap detection
# ---------------------------------------------------------------------------

def test_a_confidently_low_capability_becomes_a_competence_gap():
    model = model_with(observations("arithmetic", 0.9, False, 30))
    found = [g for g in gaps.detect(model) if g.capability_id == "arithmetic/hard"]
    assert found and found[0].kind == gaps.COMPETENCE
    assert "весь интервал ниже" in found[0].description


def test_a_capability_whose_interval_straddles_the_bar_is_a_knowledge_gap():
    """The honest reading is that we cannot yet tell, even though it looks
    like a competence problem."""
    model = model_with(observations("sequence", 0.5, True, 7),
                       observations("sequence", 0.5, False, 3))
    found = [g for g in gaps.detect(model) if g.capability_id == "sequence/medium"]
    assert found and found[0].kind == gaps.KNOWLEDGE
    assert "различить нельзя" in found[0].description


def test_an_unmeasured_slice_is_a_knowledge_gap_with_low_severity():
    """Not knowing is not being bad; treating them alike sends the system
    to fix things that may not be broken."""
    model = model_with(observations("logic", 0.8, False, 2))
    found = [g for g in gaps.detect(model) if g.band == "hard"]
    assert found and found[0].kind == gaps.KNOWLEDGE
    assert found[0].severity <= 0.3


def test_measuring_again_is_worth_more_when_less_is_known():
    """§12: an experiment with a worse immediate score can outrank one
    with a better score if it resolves more."""
    thin = model_with(observations("arithmetic", 0.5, True, 6)).capabilities()["arithmetic/medium"]
    thick = model_with(observations("arithmetic", 0.5, True, 200)).capabilities()["arithmetic/medium"]
    assert gaps.expected_information_gain(thin) > gaps.expected_information_gain(thick)


def test_information_gain_collapses_once_a_capability_is_well_measured():
    thick = model_with(observations("arithmetic", 0.5, True, 400)).capabilities()["arithmetic/medium"]
    assert gaps.expected_information_gain(thick) < 0.02


def test_the_dominant_failure_pattern_names_the_action():
    """"Improve arithmetic" is not an action."""
    model = model_with(observations("arithmetic", 0.9, False, 30, reason="format"))
    found = [g for g in gaps.detect(model) if g.capability_id == "arithmetic/hard"]
    assert "формату" in found[0].suggested_action


def test_a_brain_split_is_surfaced_as_a_routing_action():
    model = model_with(observations("sequence", 0.9, True, 15, brain="groq"),
                       observations("sequence", 0.9, False, 15, brain="ollama"))
    found = [g for g in gaps.detect(model) if g.capability_id == "sequence/hard"]
    assert found and "groq" in found[0].suggested_action


def test_a_weakness_in_rare_work_ranks_below_the_same_weakness_in_common_work():
    """A weakness in something that never comes up is a curiosity."""
    model = model_with(observations("arithmetic", 0.9, False, 30),
                       observations("logic", 0.9, False, 6))
    ranked = {g.capability_id: g.priority for g in gaps.detect(model)}
    assert ranked["arithmetic/hard"] > ranked.get("logic/hard", 0)


def test_the_priority_weights_are_declared_rather_than_hidden_in_a_sum():
    """Changing what MANA studies next is too consequential to be
    implicit."""
    assert set(gaps.PRIORITY_WEIGHTS) == {
        "severity", "uncertainty", "frequency",
        "information_gain", "capability_gain", "cost"}
    assert gaps.PRIORITY_WEIGHTS["cost"] < 0


def test_learning_goals_are_few_and_actionable():
    """A curriculum that pursues everything pursues nothing."""
    model = model_with(observations("arithmetic", 0.9, False, 30),
                       observations("sequence", 0.9, False, 30),
                       observations("logic", 0.9, False, 30))
    goals = gaps.learning_goals(model, limit=2)
    assert len(goals) == 2
    assert all(g["goal"] and g["why"] for g in goals)


def test_nothing_is_a_gap_when_everything_is_measured_and_competent():
    model = model_with(observations("arithmetic", 0.2, True, 30),
                       observations("arithmetic", 0.5, True, 30),
                       observations("arithmetic", 0.9, True, 30))
    assert gaps.detect(model) == []
