"""
mana/research on a world whose situations are not given: `world/universe`.
The adapter must never read how that world really works; the oracle below
does, for the audit only.
"""
from __future__ import annotations

import pathlib

import pytest

from mana.core import standing
from mana.research import contract, loop
from mana.research.adapters import universe as adapter
from mana.world.universe import SmallWorld, TRUE_RULES

EFFECTS = (("heat_rod", "rod.expanded"), ("open_file", "file.open"),
           ("write_file", "file.written"))


def truth(action, fact, conditions):
    """How the world really answers; None where it answers only sometimes."""
    rule = TRUE_RULES[action]
    works = (not rule.get("no_actuator")
             and all(conditions[adapter.fact_name(e, a)] == v for (e, a), v in rule["pre"].items()))
    if fact is None:
        return works
    before = conditions[fact]
    if not works:
        return before
    entity, attribute = fact.split(".")
    if (entity, attribute) in rule["eff"]:
        return bool(rule["eff"][(entity, attribute)])
    if (entity, attribute) in (rule.get("sometimes") or {}):
        return True if before else None
    return before


def run(seed=0, cost=64.0, budget=300):
    world = adapter.UniverseWorld(SmallWorld(seed), effects=EFFECTS)
    knowledge = adapter.knowledge_for(world)
    report = loop.inquire(lambda: loop.unsettled(knowledge), world, budget,
                          contract.Stakes(error_cost=cost), challenge=adapter.challenge_for(world))
    return world, knowledge, report


def audit(world, knowledge):
    """Per answered question: the known situations where it is wrong, split
    by its boundary; situations the world answers only sometimes apart."""
    out = {}
    for subject, e in knowledge.items():
        if not (e.settled and e.settled[0] == loop.ANSWERED):
            continue
        action, fact = world.questions[subject]
        _, _, lead = e.leading_class(e.space)
        wrong, uncertain = set(), set()
        for params in e.space:
            want = truth(action, fact, params["conditions"])
            if want is None:
                uncertain.add(params["key"])
            elif (lead.predict(params).get(True, 0.0) > 0.5) != want:
                wrong.add(params["key"])
        out[subject] = (standing.audit(e.standing, wrong), uncertain & e.standing.inside)
    return out


def test_the_adapter_never_reads_how_the_world_works():
    source = pathlib.Path(adapter.__file__).read_text(encoding="utf-8")
    for secret in ("TRUE_RULES", "NEVER_POSSIBLE", "DOMAIN_OF", "INITIAL"):
        assert secret not in source, secret


def test_an_incomplete_world_takes_an_undeclared_reader_to_read_everything():
    e = loop.Explanations([contract.Explanation("x", lambda p: {True: 1.0})])
    e.set_space([{"key": 1, "conditions": {"a": True, "b": False}}])
    e.complete = False
    assert e.reads(e.members[0]) == ("a", "b")
    declared = contract.Explanation("y", lambda p: {True: 1.0}, reads=("a",))
    assert e.reads(declared) == ("a",)


def test_the_space_grows_with_what_the_world_shows():
    world, knowledge, report = run()
    assert len(world.known) > 10
    assert all(len(e.space) <= len(world.known) for e in knowledge.values())


def test_the_audit_finds_where_two_ignored_preconditions_meet():
    """Step 3, seed 5: "what does write_file leave file.written at" is
    answered by a model reading file.written alone. Its claims about
    file.open and about file.writable were each confirmed apart, so the
    situation with both -- where the world writes -- lies inside the
    boundary, and the model is wrong there. The claims rule's declared
    assumption, broken by a conjunction of two preconditions."""
    world, knowledge, report = run(seed=5, cost=64.0, budget=300)
    subject = adapter.effect_subject("write_file", "file.written")
    result, _ = audit(world, knowledge)[subject]
    assert not result.honest
    e = knowledge[subject]
    _, _, lead = e.leading_class(e.space)
    assert {"file.open", "file.writable"}.isdisjoint(e.reads(lead))
    for key in result.inside:
        conditions = dict(key)
        assert conditions["file.open"] and conditions["file.writable"]


def test_only_the_known_failure_is_wrong_inside_a_boundary():
    """Across ten runs, every answer wrong inside its boundary is the one
    above; no other question breaks its boundary."""
    for seed in range(10):
        world, knowledge, report = run(seed=seed, cost=64.0, budget=300)
        for subject, (result, _) in audit(world, knowledge).items():
            if not result.honest:
                assert subject == adapter.effect_subject("write_file", "file.written"), seed


def test_chance_is_not_called_verified_when_repeatability_is_checked():
    """The same world, the task no longer assuming an outcome seen once is
    the outcome: the situations the heat was seen in are repeated, and here
    the answer is not called verified. Not a guarantee -- no number of
    repeats proves an outcome repeatable. Measured with a budget of 600:
    verified in 3 runs of 20 at error costs 64 and 256 (7 of 10 when
    assumed); unexplained in 16 of 20."""
    world = adapter.UniverseWorld(SmallWorld(0), effects=EFFECTS)
    knowledge = adapter.knowledge_for(world)
    loop.inquire(lambda: loop.unsettled(knowledge), world, 300,
                 contract.Stakes(error_cost=16.0, assume_repeatable=False),
                 challenge=adapter.challenge_for(world))
    heat = knowledge[adapter.effect_subject("heat_rod", "rod.expanded")]
    assert heat.standing is None or heat.standing.status != standing.VERIFIED_FOR_TASK
    for subject, (result, _) in audit(world, knowledge).items():
        assert result.honest, subject


@pytest.mark.xfail(strict=True, reason=(
    "найдено замером шага 3: эффект нагрева (стержень расширяется 4 раза из 5) "
    "получил «проверено для задачи» в 7 из 10 запусков при любой цене ошибки -- "
    "пока мир не показал ни одного расхождения, контракт принимает увиденное "
    "за известное; бюджет 600 вместо 300 ничего не меняет, цикл сам считает "
    "вопрос закрытым"))
def test_chance_is_never_called_verified():
    world, knowledge, report = run()
    heat = knowledge[adapter.effect_subject("heat_rod", "rod.expanded")]
    assert heat.standing is None or heat.standing.status != standing.VERIFIED_FOR_TASK
