"""
Step 4, first run: the research contract on the agent's real turns. Not new
knowledge -- that the contract lives through the agent's life: the turn is
an observation, the standing is drawn by core, and nothing about the answer
changes.
"""
from __future__ import annotations

from pathlib import Path

from mana.core import standing
from mana.research.adapters import turns

ARITHMETIC = "Сколько будет 17 * 23? Ответь кратко."
CHAT = "Расскажи про Git"


def _path(agent):
    return Path(agent.config.findings_path).with_name("turn_observations.jsonl")


def test_a_broken_standing_never_breaks_the_turn(isolated_agent, monkeypatch):
    monkeypatch.setattr(turns, "situation_of", lambda result: 1 / 0)
    result = isolated_agent.solve_task(CHAT)
    assert result["answer"]
    assert "standing" not in result


def test_the_standing_does_not_change_the_answer(isolated_agent_exec_enabled, monkeypatch):
    agent = isolated_agent_exec_enabled
    with_standing = agent.solve_task(ARITHMETIC)
    monkeypatch.setattr(type(agent), "_note_standing", lambda self, task, result: None)
    without = agent.solve_task(ARITHMETIC)
    assert with_standing["answer"] == without["answer"]
    assert "standing" in with_standing and "standing" not in without


def test_a_kind_of_turn_is_verified_only_once_its_outcome_repeated(isolated_agent_exec_enabled):
    """Seen once, the kind of turn is conditional: repeatability is checked,
    not assumed. Served again, it is verified for turns of that kind -- and
    a new kind of turn stays outside the boundary."""
    agent = isolated_agent_exec_enabled
    first = agent.solve_task(ARITHMETIC)
    capability = first["plan"]["capability"]
    assert first["standing"]["capability"] == capability
    assert first["standing"]["status"] == standing.CONDITIONAL
    second = agent.solve_task(ARITHMETIC)
    assert second["standing"]["status"] == standing.VERIFIED_FOR_TASK
    assert second["standing"]["rule"] == standing.OBSERVED
    assert second["standing"]["repeatability"] == standing.REPEATABLE_CHECKED
    third = agent.solve_task(CHAT)
    assert third["plan"]["capability"] == capability
    assert third["standing"]["status"] == standing.CONDITIONAL
    new_kind = turns.situation_of(third)["key"]
    assert [tuple(tuple(pair) for pair in key) for key in third["standing"]["uncovered"]] \
        == [new_kind]
    fourth = agent.solve_task(CHAT)
    assert fourth["standing"]["status"] == standing.VERIFIED_FOR_TASK
    # served where its answer was checked, not where it was not
    assert "зависит от проверка: есть" in fourth["standing"]["class"]


def test_nothing_inside_the_boundary_contradicts_a_turn_seen_there(isolated_agent_exec_enabled):
    """No oracle in real work, so honesty is checked against the turns
    themselves: wherever the boundary reaches, the standing's explanation
    says what every turn there did."""
    agent = isolated_agent_exec_enabled
    for task in (ARITHMETIC, ARITHMETIC, CHAT, CHAT):
        result = agent.solve_task(task)
    capability = result["plan"]["capability"]
    e = agent._turn_standings.explanations(capability)
    names, mass, lead = e.leading_class(e.space)
    answer = e.answer_for(lead)
    for params, served in e.history:
        if params["key"] in answer.inside:
            assert (lead.predict(params).get(True, 0.0) > 0.5) == served


def test_the_standing_survives_a_restart(isolated_agent_exec_enabled):
    agent = isolated_agent_exec_enabled
    agent.solve_task(ARITHMETIC)
    last = agent.solve_task(ARITHMETIC)["standing"]
    reloaded = turns.TurnStandings(_path(agent)).standing_of(last["capability"])
    for field in ("status", "model", "uncovered", "rule", "repeatability", "observations"):
        assert reloaded[field] == last[field], field
