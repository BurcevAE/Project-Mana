"""
Phase 3: AGENT -> GAP. A turn the agent could not serve is written down,
with what would close it, and a turn it did serve is not.
"""
from __future__ import annotations

from pathlib import Path

from mana.cognition import gaps
from mana.cognition.findings import Ledger

ARITHMETIC = "Сколько будет 17 * 23? Ответь кратко."


def _gaps(agent):
    book = Ledger(Path(agent.config.findings_path))
    return [f for f in book.latest() if f.question.startswith("разрыв способности")]


def test_an_instruction_that_was_not_carried_out_is_one_gap(isolated_agent, monkeypatch):
    from mana.apps import intent as app_intent

    monkeypatch.setattr(app_intent, "perform", lambda found, registry, goal="": {
        "ok": False, "output": None, "error": "не найден Notepad++"})
    result = isolated_agent.solve_task("Открой Notepad++")
    found = _gaps(isolated_agent)
    assert len(found) == 1
    assert found[0].question.endswith(result["plan"]["capability"])
    assert found[0].measurement["observed_failures"] == 1


def test_an_answer_refuted_and_not_corrected_is_one_gap(isolated_agent_exec_enabled,
                                                        monkeypatch):
    agent = isolated_agent_exec_enabled
    monkeypatch.setattr(type(agent), "_local_fallback", staticmethod(lambda task: "392"))
    monkeypatch.setattr(type(agent), "_correct_refuted_answer",
                        lambda self, *args, **kwargs: None)
    result = agent.solve_task(ARITHMETIC)
    assert result["verification"]["verified"] is False
    found = _gaps(agent)
    assert len(found) == 1
    assert found[0].question.endswith("@arithmetic")
    assert "algorithmic" in found[0].note            # what would close it


def test_a_turn_that_went_fine_is_no_gap(isolated_agent_exec_enabled):
    result = isolated_agent_exec_enabled.solve_task(ARITHMETIC)
    assert result["answer"] == "391"
    assert _gaps(isolated_agent_exec_enabled) == []


def test_a_turn_it_corrected_itself_is_no_gap(isolated_agent_exec_enabled, monkeypatch):
    agent = isolated_agent_exec_enabled
    monkeypatch.setattr(type(agent), "_local_fallback", staticmethod(lambda task: "392"))
    result = agent.solve_task(ARITHMETIC)
    assert result["answer"].strip().startswith("391")
    assert _gaps(agent) == []


def test_no_brain_is_one_gap_that_counts_up(isolated_agent):
    isolated_agent.solve_task("Расскажи про квантовые компьютеры")
    isolated_agent.solve_task("Что такое фотосинтез?")
    found = _gaps(isolated_agent)
    assert len(found) == 1
    assert found[0].question.endswith("brain:pool")
    assert found[0].measurement["observed_failures"] == 2


def test_a_broken_detector_never_breaks_the_turn(isolated_agent, monkeypatch):
    monkeypatch.setattr(gaps, "from_turn", lambda *args, **kwargs: 1 / 0)
    result = isolated_agent.solve_task("Расскажи про Git")
    assert result["answer"]


def test_recording_a_gap_does_not_change_the_answer(isolated_agent):
    first = isolated_agent.solve_task("Расскажи про Git")
    assert _gaps(isolated_agent)            # the fallback answer is a gap here
    second = isolated_agent.solve_task("Расскажи про Git")
    assert first["answer"] == second["answer"]
