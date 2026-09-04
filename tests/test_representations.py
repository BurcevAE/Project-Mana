"""
tests/test_representations.py — Level 3, and the difference between
evolving a vocabulary and renaming one.

The claim under test: "my current representation is insufficient" is a
computable statement, not a feeling. A representation is insufficient
exactly when it maps tasks that behaved differently onto the same
description, and those collisions are countable.
"""
from __future__ import annotations

import pytest

from mana.cognition.genome import Representation
from mana.cognition.representations import (FIELD_LIBRARY, describe, enriched,
                                            insufficiency_gap,
                                            measure_insufficiency, propose_fields)
from mana.cognition.self_model import Observation


def rep(name, *fields):
    return Representation(name, tuple(fields), f"{name} view")


def observation(task_id, correct, domain="arithmetic", difficulty=0.5):
    return Observation(task_id, domain, difficulty, correct)


def corpus(rows):
    """rows: (task_id, text, correct) -> (observations, task_texts)."""
    observations = [observation(tid, ok) for tid, _text, ok in rows]
    texts = {tid: text for tid, text, _ok in rows}
    return observations, texts


# ---------------------------------------------------------------------------
# a description is what makes two tasks the same
# ---------------------------------------------------------------------------

def test_two_tasks_alike_under_the_fields_get_the_same_description():
    a = describe("Вычисли 12 + 7", ("number_count", "operator_count"))
    b = describe("Вычисли 45 + 3", ("number_count", "operator_count"))
    assert a == b


def test_a_field_that_separates_them_changes_the_description():
    flat = describe("Вычисли 12 + 7", ("nesting_depth",))
    nested = describe("Вычисли (12 + 7) * (3 - 1)", ("nesting_depth",))
    assert flat != nested


def test_the_task_text_itself_is_never_a_field():
    """A field carrying the text makes every description unique, the
    collision rate zero, and the whole measurement meaningless."""
    a = describe("Вычисли 12 + 7", ("task", "number_count"))
    b = describe("Вычисли 99 + 1", ("task", "number_count"))
    assert a == b


def test_fields_are_buckets_not_continuous_values():
    """A field returning a unique value per task would give a perfect
    score and explain nothing."""
    values = {FIELD_LIBRARY["length_band"](f"слово " * n) for n in range(1, 60)}
    assert len(values) <= 4


# ---------------------------------------------------------------------------
# insufficiency is countable
# ---------------------------------------------------------------------------

def test_a_representation_that_explains_every_difference_has_no_collisions():
    observations, texts = corpus([
        ("a", "Вычисли 2 + 2", True),
        ("b", "Вычисли (((5 + 3) * 2) - 1)", False),
    ])
    result = measure_insufficiency(rep("v", "nesting_depth"), observations, texts)
    assert result.colliding_pairs == 0
    assert result.rate == 0.0


def test_a_representation_that_cannot_tell_them_apart_collides():
    observations, texts = corpus([
        ("a", "Вычисли 2 + 2", True),
        ("b", "Вычисли 7 + 9", False),
    ])
    result = measure_insufficiency(rep("v", "operator_count"), observations, texts)
    assert result.colliding_pairs == 1
    assert result.rate == 1.0
    assert "неразличимы" in result.describe()


def test_collisions_are_counted_as_contradicting_pairs_not_group_size():
    """One success against nine failures is far less confusing than five
    against five, and group size cannot see the difference."""
    lopsided, lop_texts = corpus([("t0", "Вычисли 1 + 1", True)] +
                                 [(f"t{i}", f"Вычисли {i} + 1", False) for i in range(1, 10)])
    balanced, bal_texts = corpus([(f"s{i}", f"Вычисли {i} + 1", i < 5) for i in range(10)])
    lop = measure_insufficiency(rep("v", "operator_count"), lopsided, lop_texts)
    bal = measure_insufficiency(rep("v", "operator_count"), balanced, bal_texts)
    assert lop.collisions[0].size == bal.collisions[0].size == 10
    assert bal.colliding_pairs > lop.colliding_pairs


def test_tasks_with_no_recorded_text_are_skipped_not_treated_as_empty():
    """An unknown task collides with every other unknown one and would
    manufacture insufficiency out of missing data."""
    observations = [observation("known", True), observation("missing", False)]
    result = measure_insufficiency(rep("v", "operator_count"), observations,
                                   {"known": "Вычисли 2 + 2"})
    assert result.observations == 1
    assert result.colliding_pairs == 0


def test_adding_a_separating_field_lowers_the_rate():
    """The whole point: a richer vocabulary explains more of what was
    observed."""
    observations, texts = corpus([
        ("a", "Вычисли 2 + 2 * 3", True),
        ("b", "Вычисли 3 + 4 * 5", True),
        ("c", "Вычисли (2 + 2) * 3", False),
        ("d", "Вычисли (3 + 4) * 5", False),
    ])
    before = measure_insufficiency(rep("v", "operator_count"), observations, texts)
    after = measure_insufficiency(rep("v", "operator_count", "nesting_depth"),
                                  observations, texts)
    assert before.rate > after.rate
    assert after.colliding_pairs == 0


# ---------------------------------------------------------------------------
# where a new field comes from
# ---------------------------------------------------------------------------

def test_the_proposed_field_is_the_one_that_separates_the_collisions():
    """Not from an LLM's imagination -- from the collisions themselves."""
    observations, texts = corpus([
        ("a", "Вычисли 2 + 2 * 3", True),
        ("b", "Вычисли 3 + 4 * 5", True),
        ("c", "Вычисли (2 + 2) * 3", False),
        ("d", "Вычисли (3 + 4) * 5", False),
    ])
    result = measure_insufficiency(rep("v", "operator_count"), observations, texts)
    proposals = propose_fields(result, observations, texts)
    assert proposals
    assert proposals[0].field_name == "nesting_depth"
    assert proposals[0].remaining_pairs == 0


def test_a_field_that_splits_nothing_is_not_proposed():
    observations, texts = corpus([
        ("a", "Вычисли 2 + 2", True),
        ("b", "Вычисли 3 + 4", False),
    ])
    result = measure_insufficiency(rep("v", "operator_count"), observations, texts)
    names = {p.field_name for p in propose_fields(result, observations, texts, limit=10)}
    assert "operator_count" not in names
    assert "nesting_depth" not in names, "identical structure cannot separate them"


def test_a_field_already_in_the_representation_is_never_proposed_again():
    observations, texts = corpus([
        ("a", "Вычисли 2 + 2", True),
        ("b", "Вычисли (((9 * 8) + 7) - 6)", False),
        ("c", "Вычисли 3 + 4", False),
    ])
    result = measure_insufficiency(rep("v", "nesting_depth"), observations, texts)
    assert "nesting_depth" not in {p.field_name for p in propose_fields(result, observations, texts)}


def test_proposals_are_ranked_by_how_much_confusion_they_remove():
    observations, texts = corpus(
        [(f"short{i}", "Сколько будет 2 + 2", True) for i in range(4)] +
        [(f"long{i}", "Внимательно прочитай условие и аккуратно вычисли "
                      "результат следующего выражения: 2 + 2", False) for i in range(4)])
    result = measure_insufficiency(rep("v", "operator_count"), observations, texts)
    proposals = propose_fields(result, observations, texts, limit=3)
    assert proposals[0].reduction >= proposals[-1].reduction
    assert proposals[0].separates_pairs > 0


def test_enrichment_adds_exactly_one_field_and_records_its_ancestry():
    """Adding three at once produces a representation that works without
    saying which addition did it."""
    observations, texts = corpus([
        ("a", "Вычисли 2 + 2 * 3", True),
        ("b", "Вычисли (2 + 2) * 3", False),
    ])
    base = rep("task_view", "operator_count")
    result = measure_insufficiency(base, observations, texts)
    proposals = propose_fields(result, observations, texts)
    richer = enriched(base, proposals[0])
    assert len(richer.fields) == len(base.fields) + 1
    assert richer.derived_from == "task_view"
    assert str(proposals[0].separates_pairs) in richer.description


# ---------------------------------------------------------------------------
# when the vocabulary itself becomes the thing to work on
# ---------------------------------------------------------------------------

def test_a_mostly_explanatory_representation_is_not_a_gap():
    """Every representation collides somewhat; treating a small rate as a
    problem is the Level-3 equivalent of chasing a lucky streak.

    Note that a single description group always gives a rate of 1.0 -- if
    every task looks the same, the vocabulary explains nothing however the
    outcomes fall. So the data here contains pairs the field does separate.
    """
    rows = [(f"flat{i}", f"Вычисли {i} + 1", True) for i in range(10)]
    rows += [(f"deep{i}", f"Вычисли ({i} + 1) * 2", False) for i in range(10)]
    rows.append(("odd", "Вычисли 99 + 1", False))     # one exception inside a clean group
    observations, texts = corpus(rows)
    result = measure_insufficiency(rep("v", "nesting_depth"), observations, texts)
    assert 0 < result.rate < 0.2
    assert insufficiency_gap(result) is None


def test_a_large_collision_rate_is_reported_as_a_representation_gap():
    observations, texts = corpus([(f"t{i}", f"Вычисли {i} + 1", i % 2 == 0)
                                  for i in range(24)])
    result = measure_insufficiency(rep("task_view", "operator_count"), observations, texts)
    gap = insufficiency_gap(result)
    assert gap is not None
    assert gap["kind"] == "representation"
    assert "недостаточен" in gap["description"]


def test_too_few_observations_is_not_a_gap_either():
    observations, texts = corpus([("a", "Вычисли 1 + 1", True),
                                  ("b", "Вычисли 2 + 2", False)])
    result = measure_insufficiency(rep("v", "operator_count"), observations, texts)
    assert insufficiency_gap(result) is None


def test_the_field_library_needs_no_model_to_extract_anything():
    """Semantic features would put a model's judgement inside the
    definition of MANA's own representation space."""
    sample = "Вычисли (((9 * 8) + 7) - 6) и объясни ход решения"
    for name, extractor in FIELD_LIBRARY.items():
        value = extractor(sample)
        assert isinstance(value, (str, bool, int)), f"{name} returned {type(value)}"


def test_the_baseline_representation_can_actually_be_measured(isolated_config):
    """The starting genome's task_view must work with this machinery, or
    Level 3 has nothing to improve on."""
    from mana.cognition.genome import CognitiveGenome
    from mana.core import tasks
    generated = tasks.generate("arithmetic", 12, seed=5)
    observations = [observation(t.task_id, i % 3 == 0, "arithmetic", t.difficulty)
                    for i, t in enumerate(generated)]
    texts = {t.task_id: t.prompt for t in generated}
    baseline = CognitiveGenome().representations["task_view"]
    result = measure_insufficiency(baseline, observations, texts)
    assert result.observations == 12
    assert 0.0 <= result.rate <= 1.0


# ---------------------------------------------------------------------------
# the description space stops being a list
# ---------------------------------------------------------------------------

def _corpus(n=30):
    from mana.cognition.self_model import Observation
    from mana.core import tasks as core_tasks
    observations, texts = [], {}
    for domain in ("arithmetic", "logic", "text_ops"):
        for task in core_tasks.generate(domain, n, seed=21):
            texts[task.task_id] = task.prompt
            observations.append(Observation(task.task_id, domain, task.difficulty,
                                            task.difficulty < 0.5, calls=1))
    return observations, texts


def test_a_generated_threshold_is_not_in_the_hand_written_library():
    """Eleven extractors were the whole space. A cutoff chosen from the
    observations is a field nobody wrote."""
    from mana.cognition import fields as field_gen
    observations, texts = _corpus()
    generated = field_gen.generate(observations, texts, limit=8)
    assert generated
    assert any(g.kind == "threshold" for g in generated)
    assert not any(g.name in FIELD_LIBRARY for g in generated)


def test_a_field_can_ask_which_cheap_mechanism_handles_the_task():
    """A boolean nothing in the library expresses, costing microseconds,
    and the single most useful thing to know in a system whose design is
    routing to the cheapest sufficient mechanism."""
    from mana.cognition import fields as field_gen
    extract = field_gen.mechanism_field("arithmetic")
    assert extract("Вычисли: 2 + 2") is True
    assert extract("Объясни, почему небо синее") is False


def test_a_generated_field_is_scored_where_it_was_not_fitted():
    """A cutoff chosen to separate the observed collisions will separate
    the observed collisions. Only the held-out half says whether it found
    anything."""
    from mana.cognition import fields as field_gen
    observations, texts = _corpus()
    for generated in field_gen.generate(observations, texts, limit=8):
        assert generated.held_out_pairs > 0
        assert generated.separates_held_out <= generated.held_out_pairs


def test_proposals_are_ranked_by_the_held_out_score():
    """Ranking by the fitted score puts the best-overfitted field first
    every time, which is exactly the ordering that makes fitting
    invisible."""
    from mana.cognition import fields as field_gen
    observations, texts = _corpus()
    generated = field_gen.generate(observations, texts, limit=8)
    scores = [g.separates_held_out for g in generated]
    assert scores == sorted(scores, reverse=True)


def test_a_field_that_only_fits_is_not_reported_as_generalising():
    from mana.cognition.fields import GeneratedField
    fitted_only = GeneratedField(name="x>=1", kind="threshold", source="words",
                                 threshold=1.0, separates_fitted=50,
                                 separates_held_out=0, held_out_pairs=40)
    assert fitted_only.generalises is False


def test_the_generator_is_deterministic():
    """A generator whose output moves between runs cannot be audited, and
    two runs disagreeing about what the vocabulary needs is worse than
    either answer."""
    from mana.cognition import fields as field_gen
    observations, texts = _corpus()
    first = [g.name for g in field_gen.generate(observations, texts, limit=8)]
    second = [g.name for g in field_gen.generate(list(reversed(observations)),
                                                 texts, limit=8)]
    assert first == second


def test_the_generator_needs_no_model():
    import inspect
    from mana.cognition import fields as field_gen
    source = inspect.getsource(field_gen).lower()
    for forbidden in (".ask(", "llm", "prompt"):
        assert forbidden not in source


def test_too_few_observations_generate_nothing():
    """A cutoff derived from four tasks describes those four tasks."""
    from mana.cognition import fields as field_gen
    observations, texts = _corpus(n=1)
    assert field_gen.generate(observations, texts) == []
