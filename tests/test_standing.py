"""
core/standing: what an answer may claim to know for a task. Unit tests for
the definitions first, then the same definitions against the research loop
they were measured in (`cognition/inquiry.py` 1.14), bridged from outside
core -- core never imports the loop.
"""
from __future__ import annotations

from mana import core
from mana.cognition import inquiry
from mana.core import standing
from mana.world import device as dev


def _space(names):
    configs = [()]
    for _ in names:
        configs = [c + (v,) for c in configs for v in (False, True)]
    return [(c, dict(zip(names, c))) for c in configs]


# --------------------------------------------------------------------------
# the definitions
# --------------------------------------------------------------------------

def test_standing_is_inside_the_immutable_core():
    assert core.is_immutable_path(standing.__file__)


def test_three_rules_reach_three_distances_from_what_was_seen():
    """A model reading s0, one independent observation at (1,1,0): observed
    and claims keep that point, pattern takes all of s0=1; a second one at
    (1,0,1) completes the claims for all of s0=1."""
    space = _space(["s0", "s1", "s2"])
    first = {"s0": True, "s1": True, "s2": False}
    second = {"s0": True, "s1": False, "s2": True}
    s0_on = {(True, b, e) for b in (False, True) for e in (False, True)}
    one = {(True, True, False)}
    two = one | {(True, False, True)}
    assert standing.inside(space, ["s0"], one, [first], standing.OBSERVED) == one
    assert standing.inside(space, ["s0"], one, [first], standing.CLAIMS) == one
    assert standing.inside(space, ["s0"], one, [first], standing.PATTERN) == s0_on
    assert standing.inside(space, ["s0"], two, [first, second], standing.CLAIMS) == s0_on
    assert standing.inside(space, ["s0"], two, [first, second], standing.OBSERVED) == two


def test_an_answer_is_verified_only_where_the_task_relies_on_it():
    """The same boundary, two tasks: verified for the one inside it,
    conditional -- with the part left out -- for the one that is not."""
    space = _space(["s0", "s1"])
    seen = {(True, True)}
    inside_task = {(True, True): 1.0}
    wider_task = {(True, True): 1.0, (False, False): 3.0}
    a = standing.standing("b", "m", space, ["s0"], seen, [], standing.OBSERVED, inside_task)
    b = standing.standing("b", "m", space, ["s0"], seen, [], standing.OBSERVED, wider_task)
    assert a.status == standing.VERIFIED_FOR_TASK and a.uncovered == ()
    assert b.status == standing.CONDITIONAL
    assert b.uncovered == ((False, False),) and b.uncovered_share == 0.75
    assert b.assumption == standing.ASSUMPTIONS[standing.OBSERVED]


def test_an_answer_says_whether_repeatable_outcomes_were_assumed_or_checked():
    space = _space(["s0"])
    task = {key: 1.0 for key, _ in space}
    assumed = standing.standing("b", "m", space, ["s0"], {(True,), (False,)}, [],
                                standing.OBSERVED, task)
    checked = standing.standing("b", "m", space, ["s0"], {(True,)}, [], standing.OBSERVED,
                                task, repeatability=standing.REPEATABLE_CHECKED)
    assert assumed.repeatability == standing.REPEATABLE_ASSUMED
    assert assumed.status == standing.VERIFIED_FOR_TASK
    assert checked.repeatability == standing.REPEATABLE_CHECKED
    assert checked.status == standing.CONDITIONAL and checked.uncovered == ((False,),)
    try:
        standing.standing("b", "m", space, ["s0"], set(), [], standing.OBSERVED, task,
                          repeatability="hoped")
    except ValueError:
        return
    raise AssertionError("an unknown repeatability must be refused")


def test_an_unexplained_answer_claims_nothing():
    answer = standing.unexplained("b0", observed=12)
    assert answer.inside == frozenset() and standing.audit(answer, {(True,)}).honest


def test_the_audit_splits_errors_by_the_boundary():
    space = _space(["s0", "s1"])
    answer = standing.standing("b", "m", space, ["s0"], {(True, True)},
                               [{"s0": True, "s1": True}], standing.PATTERN,
                               {key: 1.0 for key, _ in space})
    result = standing.audit(answer, {(True, False), (False, False)})
    assert result.inside == {(True, False)} and result.outside == {(False, False)}
    assert not result.honest


def test_an_unknown_rule_is_refused():
    try:
        standing.inside(_space(["s0"]), ["s0"], set(), [], "everything")
    except ValueError:
        return
    raise AssertionError("an unknown inclusion rule must be refused")


# --------------------------------------------------------------------------
# the same definitions, against the loop they were measured in
# --------------------------------------------------------------------------

def _run(seed, options, rule, cost):
    device = dev.random_device(seed, **options)
    knowledge = dev.knowledge_for(device)
    stakes = inquiry.Stakes(error_cost=cost)
    inquiry.inquire(lambda: inquiry.unsettled(knowledge), dev.DeviceProbes(device), 250,
                    challenge=dev.challenge_for(device), stakes=stakes, boundary=rule)
    return device, knowledge, stakes


def _bridge(button, hs, stakes):
    """The loop's state, as core's inputs -- for the member of the answering
    class the loop closed on. The loop counts evidence per member, so
    another member of the same class can have a different boundary."""
    _, _, answer = hs.leading_class(hs.space)
    space = [(hs._config_key(p), p["conditions"]) for p in hs.space]
    independent = [p["conditions"] for (p, o), r in zip(hs.history, hs.evidence)
                   if hs._vouches(r, p, o, answer)]
    task = {hs._config_key(p): stakes.weight(p) for p in hs.space}
    return answer, standing.standing(button, answer.name, space, hs._relevant(answer),
                                     hs.observed_coverage(), independent, hs.boundary, task)


def test_core_draws_the_boundary_the_loop_drew():
    """On A, C, D and E, for both rules the loop measured: the same
    boundary, the same standing, the same uncovered part of the task."""
    worlds = ({}, {"needs": 2}, {"needs": 3}, {"hidden_pair": True})
    for options in worlds:
        for seed in range(2):
            for rule in (standing.CLAIMS, standing.PATTERN):
                device, knowledge, stakes = _run(seed, options, rule, 64.0)
                for button, hs in knowledge.items():
                    if not (hs.settled and hs.settled[0] == inquiry.ANSWERED):
                        continue
                    answer, record = _bridge(button, hs, stakes)
                    where = (options, seed, rule, button)
                    assert record.inside == frozenset(hs._inside(answer)), where
                    assert record.status == hs.applicability["status"], where
                    assert set(record.uncovered) == set(hs.applicability["uncovered"]), where


def test_the_audit_catches_the_claims_rule_where_two_ignored_conditions_interact():
    """World E, device 80: the model reads s1 and ignores s0 and s5, whose
    claims were confirmed by different observations; (1,1,1) with both on is
    inside the claims boundary and wrong there."""
    device, knowledge, stakes = _run(80, {"hidden_pair": True}, standing.CLAIMS, 16.0)
    hs = knowledge["b0"]
    answer, record = _bridge("b0", hs, stakes)
    probes = dev.DeviceProbes(device)
    rule = device.rules["b0"]
    wrong = {c for c in device.reachable()
             if (answer.predict(probes.params("b0", c)).get(True, 0.0) > 0.5) != rule.holds(c)}
    result = standing.audit(record, wrong)
    assert not result.honest
    a, e, f = rule.switches
    assert all(c[a] and c[e] and c[f] for c in result.inside)
