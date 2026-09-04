"""
tests/test_curriculum.py — choosing what to practise, without being able
to rig the exam.

A system that picks its own curriculum can pick an easy one. Most of these
tests are about the structural reasons it cannot: it chooses a generator
and a difficulty, and everything else -- the tasks, the answers, the
grading -- comes from the immutable core.
"""
from __future__ import annotations

import pytest

from mana.cognition import curriculum as cur
from mana.cognition.curriculum import (ADVERSARIAL, EASY, HARD, MEDIUM, STAGES,
                                       TRANSFER, Curriculum, LearningGoal)
from mana.cognition.self_model import Observation, SelfModel


def model_with(domain, difficulty, correct, n, reason="wrong"):
    model = SelfModel()
    for i in range(n):
        model.record(Observation(f"{domain}{i}", domain, difficulty, correct,
                                 reason="" if correct else reason, calls=1))
    return model


def always(correct):
    def runner(task):
        return correct, "brain", 1
    return runner


def by_difficulty(threshold):
    """Succeeds below a difficulty, fails above -- a system with a real
    ceiling, which is what a curriculum is supposed to find."""
    def runner(task):
        return task.difficulty < threshold, "brain", 1
    return runner


# ---------------------------------------------------------------------------
# it cannot rig the exam
# ---------------------------------------------------------------------------

def test_the_curriculum_chooses_only_the_domain_and_the_difficulty():
    """The tasks, the answers and the grading all come from core."""
    c = Curriculum(SelfModel())
    lesson = c.plan_lesson(LearningGoal("g", "arithmetic", HARD, "why"))
    assert lesson.difficulty == cur._STAGE_DIFFICULTY[HARD]
    assert not hasattr(c, "generate_task")
    assert not hasattr(c, "grade")


def test_an_unknown_domain_produces_no_lesson():
    """It cannot invent a domain to practise in either."""
    c = Curriculum(SelfModel())
    assert c.plan_lesson(LearningGoal("g", "telepathy", EASY, "why")) is None


def test_every_lesson_draws_a_fresh_seed():
    """Re-serving tasks already seen measures memorisation and reports it
    as progress -- the quietest way a self-directed curriculum can lie."""
    c = Curriculum(SelfModel())
    seeds = {c.plan_lesson(LearningGoal(f"g{i}", "arithmetic", EASY, "w")).seed
             for i in range(12)}
    assert len(seeds) == 12


def test_used_seeds_are_remembered_so_a_resumed_run_cannot_repeat_one():
    c = Curriculum(SelfModel())
    first = c.next_seed()
    c.used_seeds.add(first)
    assert c.next_seed() != first


def test_the_same_seed_twice_would_have_produced_the_same_tasks():
    """Why fresh seeds matter, demonstrated rather than asserted."""
    from mana.core import tasks
    a = tasks.generate("arithmetic", 5, seed=4242)
    b = tasks.generate("arithmetic", 5, seed=4242)
    assert [t.prompt for t in a] == [t.prompt for t in b]


# ---------------------------------------------------------------------------
# goals come from evidence
# ---------------------------------------------------------------------------

def test_a_competence_gap_is_practised_one_stage_below_where_it_failed():
    """Practising at the level something is already failing at produces
    more failures; the band underneath is where the missing ability
    lives."""
    model = model_with("arithmetic", 0.9, False, 30)
    goals = cur.goals_from_model(model)
    hard_goal = next(g for g in goals if g.domain == "arithmetic")
    assert hard_goal.stage == MEDIUM


def test_a_knowledge_gap_is_practised_where_it_is_unmeasured():
    """It needs measurement, and measurement is what a lesson produces."""
    model = model_with("logic", 0.8, True, 3)
    goals = cur.goals_from_model(model)
    assert goals
    assert goals[0].kind == "knowledge"
    assert goals[0].stage == HARD


def test_an_untouched_niche_becomes_a_goal_even_though_it_has_no_gap():
    """A gap needs observations to exist at all, so a niche nothing has
    attempted is the blind spot a purely gap-driven curriculum cannot
    see."""
    goals = cur.goals_from_coverage({"empty": ["text_ops/hard", "logic/easy"]})
    assert {g.domain for g in goals} == {"text_ops", "logic"}
    assert "ни разу не пробовалась" in goals[0].reason


def test_goals_can_be_restricted_to_chosen_domains():
    model = SelfModel()
    for i in range(30):
        model.record(Observation(f"a{i}", "arithmetic", 0.9, False, reason="wrong"))
        model.record(Observation(f"l{i}", "logic", 0.9, False, reason="wrong"))
    goals = cur.goals_from_model(model, domains=("logic",))
    assert {g.domain for g in goals} == {"logic"}


# ---------------------------------------------------------------------------
# progression is earned
# ---------------------------------------------------------------------------

def test_a_stage_is_passed_only_when_the_whole_interval_clears_the_bar():
    """Passing on a lucky mean walks the curriculum into material the
    system cannot do, then records the failures as a capability profile."""
    c = Curriculum(SelfModel(), lesson_size=12)
    lesson = c.run_lesson(c.plan_lesson(LearningGoal("g", "arithmetic", EASY, "w")),
                          always(True))
    assert lesson.accuracy == 1.0
    assert lesson.passed is True

    c2 = Curriculum(SelfModel(), lesson_size=12)
    mixed = c2.plan_lesson(LearningGoal("g", "arithmetic", EASY, "w"))
    calls = {"n": 0}

    def sometimes(task):
        calls["n"] += 1
        return calls["n"] % 10 < 8, "brain", 1        # 80% mean, interval below bar

    result = c2.run_lesson(mixed, sometimes)
    assert result.accuracy >= 0.7
    assert result.passed is False, "a mean above the bar is not an interval above it"


def test_a_failed_stage_does_not_advance():
    c = Curriculum(SelfModel(), lesson_size=12)
    lesson = c.run_lesson(c.plan_lesson(LearningGoal("g", "arithmetic", EASY, "w")),
                          always(False))
    assert lesson.passed is False


def test_stages_run_from_difficulty_through_adversarial_to_transfer():
    c = Curriculum(SelfModel())
    goal = LearningGoal("g", "arithmetic", EASY, "w")
    seen = [goal.stage]
    while (goal := c.next_stage(goal)) is not None:
        seen.append(goal.stage)
    assert seen == list(STAGES)


def test_a_mastered_domain_stops_consuming_budget():
    """Returning None at the end rather than looping."""
    c = Curriculum(SelfModel())
    assert c.next_stage(LearningGoal("g", "arithmetic", TRANSFER, "w")) is None


# ---------------------------------------------------------------------------
# evidence, and what a lesson is worth
# ---------------------------------------------------------------------------

def test_a_lesson_writes_its_evidence_into_the_self_model():
    model = SelfModel()
    c = Curriculum(model, lesson_size=10)
    c.run_lesson(c.plan_lesson(LearningGoal("g", "sequence", MEDIUM, "w")), always(True))
    assert len(model.observations) == 10
    assert model.capabilities()["sequence"].score == 1.0


def test_a_lesson_is_valued_on_what_it_resolved_not_on_the_score():
    """A lesson confirming a capability is poor has taught something; one
    that changes nothing has not, whatever the score was."""
    model = SelfModel()
    c = Curriculum(model, lesson_size=12)
    first = c.run_lesson(c.plan_lesson(LearningGoal("g", "arithmetic", MEDIUM, "w")),
                         always(False))
    second = c.run_lesson(c.plan_lesson(LearningGoal("g2", "arithmetic", MEDIUM, "w")),
                          always(False))
    assert first.information > second.information, "the first lesson resolved more"


def test_a_task_that_cannot_be_run_counts_as_wrong_not_as_missing():
    """Dropping it would quietly bias the sample toward tasks that
    happened to run."""
    model = SelfModel()
    c = Curriculum(model, lesson_size=8)

    def broken(task):
        raise RuntimeError("brain died")

    lesson = c.run_lesson(c.plan_lesson(LearningGoal("g", "arithmetic", EASY, "w")), broken)
    assert lesson.observations == 8
    assert lesson.correct == 0


def test_a_task_that_could_not_be_run_is_recorded_as_an_error_not_a_wrong_answer():
    """Recording both the same way hid an unwired runner behind a
    capability profile of solid zeros that read like a weak model."""
    model = SelfModel()
    c = Curriculum(model, lesson_size=6)

    def broken(task):
        raise RuntimeError("brain died")

    c.run_lesson(c.plan_lesson(LearningGoal("g", "arithmetic", EASY, "w")), broken)
    modes = {m for o in model.observations for m in [o.reason] if m}
    assert modes == {"error"}, "a dead brain is not a wrong answer"


def test_the_curriculum_finds_a_real_ceiling():
    """The point of the whole thing: run easy and hard, and the evidence
    should show where the system stops working."""
    model = SelfModel()
    c = Curriculum(model, lesson_size=12)
    c.run_lesson(c.plan_lesson(LearningGoal("g1", "arithmetic", EASY, "w")),
                 by_difficulty(0.45))
    c.run_lesson(c.plan_lesson(LearningGoal("g2", "arithmetic", HARD, "w")),
                 by_difficulty(0.45))
    caps = model.capabilities()
    assert caps["arithmetic/easy"].score > caps["arithmetic/hard"].score


# ---------------------------------------------------------------------------
# stopping
# ---------------------------------------------------------------------------

def test_the_budget_stops_the_curriculum():
    c = Curriculum(SelfModel(), budget_calls=5)
    c.calls_used = 5
    assert "бюджет исчерпан" in c.stop_reason()


def test_the_lesson_cap_stops_it():
    c = Curriculum(SelfModel(), max_lessons=1, lesson_size=4)
    c.run_lesson(c.plan_lesson(LearningGoal("g", "arithmetic", EASY, "w")), always(True))
    assert "предел уроков" in c.stop_reason()


def test_a_curriculum_that_has_stopped_learning_stops_running():
    """Practising something already known consumes the budget an
    unexplored niche needed."""
    model = SelfModel()
    c = Curriculum(model, lesson_size=6, max_lessons=99)
    # Enough lessons that the Wilson interval has stopped moving. Early
    # repetitions of a known result still narrow it, and stopping while it
    # narrows would be stopping on real progress.
    for i in range(10):
        c.run_lesson(c.plan_lesson(LearningGoal(f"g{i}", "arithmetic", EASY, "w")),
                     always(True))
    assert max(l.information for l in c.lessons[-cur.DIMINISHING_WINDOW:]) < cur.MIN_LESSON_VALUE
    assert "убывающая отдача" in c.stop_reason()


def test_step_returns_nothing_once_a_stop_condition_holds():
    c = Curriculum(SelfModel(), budget_calls=1)
    c.calls_used = 1
    assert c.step([LearningGoal("g", "arithmetic", EASY, "w")], always(True)) is None


def test_step_takes_the_highest_priority_goal():
    c = Curriculum(SelfModel(), lesson_size=4)
    low = LearningGoal("low", "arithmetic", EASY, "w", priority=0.1)
    high = LearningGoal("high", "sequence", EASY, "w", priority=0.9)
    lesson = c.step([low, high], always(True))
    assert lesson.goal.domain == "sequence"


def test_the_report_can_be_read_backwards_to_the_weakness(tmp_path):
    model = model_with("arithmetic", 0.9, False, 30)
    c = Curriculum(model, lesson_size=6)
    goal = cur.goals_from_model(model)[0]
    c.run_lesson(c.plan_lesson(goal), always(True))
    report = c.report()
    assert report["lessons"] == 1
    assert report["history"][0]["domain"] == goal.domain
    c.save(tmp_path / "curriculum.json")
    assert (tmp_path / "curriculum.json").exists()
