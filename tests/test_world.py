"""Can a frame for "what is a world" be filled by acting in one?

The experiment, not a demo. The world's true rules live in `universe.py`
and the explorer must never read them, so the first test in this file
checks that by reading the explorer's source: a learner that peeks scores
perfectly and means nothing.

The load-bearing assertion is `wrongly_certain == 0`. A model may be
incomplete without being dishonest -- and this one is incomplete, on
purpose and measurably -- but a claim marked KNOWN that turns out wrong
is a different kind of failure, and it is counted on its own.
"""
from __future__ import annotations

import inspect

import pytest

from mana.world import explore as explore_mod
from mana.world.explore import Explorer, MIN_ATTEMPTS_FOR_CANNOT
from mana.world.schema import (BELIEVED, CAN, CANNOT, KNOWN, Observation,
                               StateFact, UNKNOWN, situation_of)
from mana.world.universe import SmallWorld, grade


def _explored(steps: int = 1000, seed: int = 7):
    world = SmallWorld(seed=seed)
    return Explorer().explore(world, steps=steps, seed=seed).model()


# --------------------------------------------------------------------------
# the discipline the experiment rests on
# --------------------------------------------------------------------------

def test_the_explorer_never_reads_the_worlds_rules():
    source = inspect.getsource(explore_mod)
    for secret in ("TRUE_RULES", "NEVER_POSSIBLE", "DOMAIN_OF", "INITIAL"):
        assert secret not in source, f"{secret} reached the learner"
    # It may not import the module that holds them at all: an import is
    # one line away from a peek, and this is the whole experiment.
    assert "universe" not in source


def test_the_world_reports_two_different_impossibilities_the_same_way():
    """`open_door` has no actuator; `write_file` has unmet preconditions.

    If the world said which, "I cannot do this" would be read off an
    error message instead of concluded from evidence -- and the thing
    being tested would not be tested.
    """
    world = SmallWorld(seed=1)
    no_actuator = world.act("open_door")
    unmet = world.act("write_file")
    assert no_actuator.succeeded is False and unmet.succeeded is False
    assert no_actuator.changed == unmet.changed == frozenset()
    assert no_actuator.note == unmet.note == ""


# --------------------------------------------------------------------------
# what may be called knowledge
# --------------------------------------------------------------------------

def test_a_fact_that_never_varies_is_never_claimed_as_a_condition():
    """Nothing in this world deletes the file, so "the file exists" holds
    in every success of everything. Asserting it everywhere would fill
    the model with true claims nobody earned."""
    model = _explored()
    for rule in model.rules.values():
        assert ("file", "exists", True) not in rule.preconditions
        assert ("app", "installed", True) not in rule.preconditions
        assert ("service", "up", True) not in rule.preconditions
    # And it is reported as unlearnable rather than silently dropped.
    evidence = model.rules["open_file"].evidence
    assert "file.exists" in evidence["unlearnable"]


def test_a_condition_is_known_only_with_a_discriminating_failure():
    model = _explored()
    known = model.rules["open_file"]
    assert known.status == KNOWN
    assert ("power", "on", True) in known.preconditions
    assert ["power", "on", True] in known.evidence["discriminated"]


def test_a_condition_nothing_singled_out_stays_believed():
    """`request` really needs a reachable host, and a reachable host is
    only ever produced through an up link, so no state in this world has
    one without the other. The learner cannot separate them -- and marks
    the pair BELIEVED rather than KNOWN, which is the correct answer to
    evidence that cannot decide."""
    model = _explored()
    request = model.rules["request"]
    assert request.status == BELIEVED
    assert ("link", "up", True) in request.preconditions      # not a true one
    assert request.evidence["assumed"], "an assumption must be listed as one"


def test_the_model_is_never_confidently_wrong():
    """The assertion this file exists for."""
    assert grade(_explored()).wrongly_certain == []


# --------------------------------------------------------------------------
# effects, and what is not one
# --------------------------------------------------------------------------

def test_an_effect_that_follows_only_sometimes_is_not_a_rule():
    model = _explored()
    heating = model.rules["heat_rod"]
    assert ("rod", "expanded", True) not in heating.effects
    shares = {fact: share for fact, share in heating.regularities}
    assert ("rod", "expanded", True) in shares
    assert 0.6 < shares[("rod", "expanded", True)] < 1.0


def test_effects_are_recovered_in_full():
    score = grade(_explored())
    assert score.effect_recall == 1.0
    assert score.effects_invented == 0


# --------------------------------------------------------------------------
# what I can and cannot do
# --------------------------------------------------------------------------

def test_an_action_with_no_actuator_is_learned_from_evidence():
    model = _explored()
    door = model.capabilities["open_door"]
    assert door.verdict == CANNOT
    assert door.attempts >= MIN_ATTEMPTS_FOR_CANNOT
    assert door.successes == 0


def test_a_negative_is_believed_and_never_known():
    """No number of failures proves that no situation allowing it exists.
    This project has already read "no counterexamples in seven episodes"
    as safety once."""
    model = _explored()
    assert model.capabilities["open_door"].status == BELIEVED
    assert all(c.status == BELIEVED for c in model.cannot())


def test_one_success_settles_a_positive():
    model = _explored()
    assert model.capabilities["open_file"].status == KNOWN
    assert model.capabilities["open_file"].verdict == CAN


def test_too_few_attempts_is_unknown_rather_than_cannot():
    world = SmallWorld(seed=3)
    short = Explorer().explore(world, steps=11, seed=3).model()
    door = short.capabilities.get("open_door")
    if door is not None and door.attempts < MIN_ATTEMPTS_FOR_CANNOT:
        assert door.verdict == UNKNOWN


def test_every_capability_verdict_is_right():
    score = grade(_explored())
    assert score.capabilities_wrong == 0
    assert score.capabilities_unknown == 0
    assert score.capabilities_right == 11


# --------------------------------------------------------------------------
# layers, and time
# --------------------------------------------------------------------------

def test_the_layers_are_recovered_from_where_observations_came_from():
    model = _explored()
    assert set(model.domains) == {"PHYSICAL", "DEVICE", "LOCAL_SYSTEM",
                                  "NETWORK"}
    assert "power" in model.domains["DEVICE"]
    assert "door" in model.domains["PHYSICAL"]
    assert grade(model).domains_right == 4


def test_a_state_carries_when_it_was_seen():
    """Memory is an archive of observations, not the world. A fact with
    no time on it cannot be told from one that is still true."""
    fact = StateFact(entity="app", attribute="running", value=True,
                     observed_at=100.0, source="LOCAL_SYSTEM")
    assert fact.age(now=160.0) == 60.0
    assert fact.as_dict()["observed_at"] == 100.0


# --------------------------------------------------------------------------
# the grader itself
# --------------------------------------------------------------------------

def test_the_grader_counts_an_invented_condition():
    model = _explored()
    score = grade(model)
    assert score.preconditions_invented >= 1
    assert score.precondition_precision < 1.0
    # And says which, so a reader can disagree with the count.
    assert score.detail["request"]["pre_got"] != score.detail["request"]["pre_want"]


def test_an_empty_model_scores_nothing_rather_than_everything():
    from mana.world.schema import WorldModel

    score = grade(WorldModel())
    assert score.precondition_recall == 0.0
    assert score.capabilities_unknown == 11
    assert score.wrongly_certain == []


def test_more_steps_do_not_lift_the_unlearnable():
    """The ceiling is structural, not a sample size. Three facts in this
    world never vary, so nothing an explorer does can establish them, and
    a run ten times longer says exactly the same thing."""
    short = grade(_explored(steps=600))
    long = grade(_explored(steps=6000))
    assert short.precondition_recall == long.precondition_recall
