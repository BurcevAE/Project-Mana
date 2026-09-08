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


# --------------------------------------------------------------------------
# choosing what to try, and what the choice was measured to be worth
# --------------------------------------------------------------------------

def test_an_untried_action_is_worth_the_most():
    explorer = Explorer()
    value, why = explorer.value_of("open_file", frozenset())
    assert value == explore_mod.UNTRIED
    assert "не пробовали" in why


def test_a_situation_with_one_false_candidate_is_the_valuable_one():
    """A failure confirms that candidate and a success kills it. The only
    shape where both outcomes conclude something.

    Built by hand rather than by running: whether a real run has already
    discriminated a given candidate depends on how long it ran, and a test
    that depends on that is testing the run, not the rule.
    """
    on = frozenset({("power", "on", True)})
    off = frozenset({("power", "on", False)})
    explorer = Explorer()
    for _ in range(3):
        explorer.record(Observation(action="launch", target="", succeeded=True,
                                    before=on, after=on))
    # Both values have been seen, so the fact is learnable; the candidate
    # held in every success and no failure has singled it out yet.
    explorer.record(Observation(action="other", target="", succeeded=True,
                                before=off, after=off))

    value, why = explorer.value_of("launch", off)
    assert value == explore_mod.DISCRIMINATING
    assert "ровно одно условие ложно" in why

    # And once a failure has singled it out, there is nothing left to ask.
    explorer.record(Observation(action="launch", target="", succeeded=False,
                                before=off, after=off))
    settled, _ = explorer.value_of("launch", off)
    assert settled < explore_mod.DISCRIMINATING


def test_repeating_the_same_attempt_in_the_same_state_is_worth_less():
    """Otherwise a greedy policy picks the same action in the same
    unchanged situation for ever, which is a confident way to learn
    nothing."""
    world = SmallWorld(seed=5)
    explorer = Explorer().explore(world, steps=200, seed=5)
    situation = world.situation()
    before, _ = explorer.value_of("heat_rod", situation)
    explorer.record(world.act("heat_rod"))
    after, _ = explorer.value_of("heat_rod", situation)
    assert after < before or before == 0.0


def test_choosing_falls_back_when_nothing_clears_the_floor():
    explorer = Explorer()
    import random as _random

    action, value, why = explorer.choose(["open_file", "heat_rod"],
                                         frozenset(), _random.Random(0))
    assert action in {"open_file", "heat_rod"}
    assert value == explore_mod.UNTRIED


def test_the_default_policy_is_the_one_that_earned_it_on_a_holdout():
    """The default moved because a number moved, and only after the
    number came from data the strategy was not chosen on: +0.0769 on
    validation and +0.0641 on a fresh split read once after sealing,
    worse in no pair out of forty on either.

    The information policy is still not the default -- it was proposed
    just as plausibly and measured -0.107.
    """
    assert explore_mod.DEFAULT_POLICY == explore_mod.BY_PLANNING
    assert explore_mod.DEFAULT_POLICY != explore_mod.BY_INFORMATION
    signature = inspect.signature(Explorer.explore)
    assert signature.parameters["policy"].default == explore_mod.BY_PLANNING


def test_both_policies_reach_the_same_ceiling():
    """The rejected policy is not broken -- it is not better. Given
    enough steps the two models are the same, which is why the finding is
    WORSE rather than a bug report."""
    long = 2500
    by_coverage = grade(Explorer().explore(
        SmallWorld(seed=4), steps=long, seed=4,
        policy=explore_mod.BY_COVERAGE).model())
    by_information = grade(Explorer().explore(
        SmallWorld(seed=4), steps=long, seed=4,
        policy=explore_mod.BY_INFORMATION).model())
    assert by_coverage.precondition_precision == by_information.precondition_precision
    assert by_coverage.precondition_recall == by_information.precondition_recall
    assert by_information.wrongly_certain == []


def test_neither_policy_drops_the_unfalsifiable_belief():
    """`request` is credited with needing an up link. It does not -- and
    no state in this world has a reachable host without one, so nothing
    an explorer does can refute it. A policy that dropped it would have
    stopped believing something for no reason, which is not an
    improvement."""
    for policy in (explore_mod.BY_COVERAGE, explore_mod.BY_INFORMATION):
        model = Explorer().explore(SmallWorld(seed=4), steps=2500, seed=4,
                                   policy=policy).model()
        assert ("link", "up", True) in model.rules["request"].preconditions
        assert model.rules["request"].status == BELIEVED


# --------------------------------------------------------------------------
# the schedule that watches the world, and what a holdout said about it
# --------------------------------------------------------------------------

def test_resetting_on_failures_replaces_the_clock():
    """Both schedules explore; the question is only which is better, and
    that is not settled here."""
    on_the_clock = Explorer().explore(SmallWorld(seed=6), steps=300, seed=6)
    on_failures = Explorer().explore(SmallWorld(seed=6), steps=300, seed=6,
                                     reset_after_failures=2)
    assert len(on_the_clock.attempts) == len(on_failures.attempts) == 300
    # A different schedule visits different situations; if it did not,
    # there would be nothing to measure.
    assert ({a.before for a in on_the_clock.attempts}
            != {a.before for a in on_failures.attempts})


def test_the_default_schedule_is_the_one_the_holdout_did_not_refute():
    """Chosen on seeds 1..40 it looked better by +0.0035; on forty seeds
    nothing had read it came to -0.0057 [-0.0133, 0.0000], better in no
    pair out of forty. The discovery win was fitted to its own seeds, and
    the default did not move."""
    assert explore_mod.RESET_AFTER_FAILURES == 0
    signature = inspect.signature(Explorer.explore)
    assert signature.parameters["reset_after_failures"].default == 0


# --------------------------------------------------------------------------
# walking to the question
# --------------------------------------------------------------------------

def test_the_planner_uses_its_own_beliefs_as_the_map():
    """The first thing in this project that uses a model of a world to
    decide what to do in it, rather than to report on what was done."""
    world = SmallWorld(seed=8)
    explorer = Explorer().explore(world, steps=120, seed=8,
                                  policy=explore_mod.BY_COVERAGE)
    plan = explorer.plan_to_discriminate(world.situation())
    if plan is None:
        pytest.skip("nothing open to discriminate from this state")
    route, target, fact = plan
    assert isinstance(route, list) and isinstance(target, str)
    # The route is made of actions the explorer believes it can perform.
    beliefs = explorer._beliefs()
    assert all(step in beliefs for step in route)
    # And the target is an action with that fact still undiscriminated.
    assert fact in beliefs[target]["open"]


def test_a_plan_is_never_longer_than_the_limit():
    world = SmallWorld(seed=9)
    explorer = Explorer().explore(world, steps=200, seed=9,
                                  policy=explore_mod.BY_COVERAGE)
    plan = explorer.plan_to_discriminate(world.situation())
    if plan is not None:
        assert len(plan[0]) <= explore_mod.MAX_PLAN_DEPTH


def test_no_route_falls_back_rather_than_standing_still():
    """A world it has not finished touching is not one it has finished
    learning, so a planner with nothing to plan keeps exploring."""
    world = SmallWorld(seed=10)
    explorer = Explorer().explore(world, steps=300, seed=10,
                                  policy=explore_mod.BY_PLANNING)
    assert len(explorer.attempts) == 300
    assert explorer.plans + explorer.planless > 0


def test_planning_does_not_break_what_coverage_got_right():
    """Precision is what the hypothesis said would move. Everything else
    must not: measured over forty seeds, recall, effects, capability
    verdicts and the count of confidently wrong claims are identical."""
    by_coverage = grade(Explorer().explore(
        SmallWorld(seed=11), steps=400, seed=11,
        policy=explore_mod.BY_COVERAGE).model())
    by_planning = grade(Explorer().explore(
        SmallWorld(seed=11), steps=400, seed=11,
        policy=explore_mod.BY_PLANNING).model())

    assert by_planning.effect_recall == by_coverage.effect_recall
    assert by_planning.capabilities_wrong == by_coverage.capabilities_wrong == 0
    assert by_planning.wrongly_certain == [] == by_coverage.wrongly_certain
