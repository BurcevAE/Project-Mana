"""What may be stored as a conclusion, and what may not.

Measured on a real memory: all twelve stored items were junk.

    [graph_turn]   [alpha] короткий ответ на: Сожми это в ОДНО предложение
    [graph_turn]   Не нашлось.
    [graph_turn]   Не найдено информации о Мане.
    [graph_turn]   未找到 Voronezh 实时天气信息，请查官方气象网站。
    [graph_entity] одно
    [graph_entity] сожми

`distill_turn` asked a model for a conclusion and stored whatever came
back. Two ways that goes wrong, both present: the model refused, and the
refusal became the turn's conclusion; the model echoed the instruction,
and "Сожми это в ОДНО предложение" was stored and mined for entities --
which is where "одно" and "сожми" came from.
"""
from __future__ import annotations

import pytest

from mana.graph_memory import is_refusal, usable_summary

REFUSALS = [
    "Не нашлось.",
    "Не найдено информации о Мане.",
    "未找到 Voronezh 实时天气信息，请查官方气象网站。",
    "К сожалению, данных нет.",
    "Не могу ответить на этот вопрос.",
    "I don't know.",
]

INSTRUCTION_ECHOES = [
    "[alpha] короткий ответ на: Сожми это в ОДНО предложение — только вывод",
    "Сожми это в ОДНО предложение, максимум 240 символов",
    "Ответ без вводных слов и воды",
]

CONCLUSIONS = [
    "Столица Франции — Париж.",
    "1С запускается через 1cestart.exe, база выбирается из списка.",
    "Пользователь работает с базой UT11-ER на файловом варианте.",
]


# --------------------------------------------------------------------------
# a refusal is not a conclusion
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", REFUSALS)
def test_a_refusal_is_recognised(text):
    """"Не нашлось" is a fact about an attempt, not about the world.
    Recalled later it teaches nothing except how to refuse."""
    assert is_refusal(text) is True
    assert usable_summary(text) is False


@pytest.mark.parametrize("text", CONCLUSIONS)
def test_a_conclusion_is_kept(text):
    assert is_refusal(text) is False
    assert usable_summary(text) is True


def test_an_honest_failure_message_is_not_a_refusal():
    """"Не получилось: базы нет в списке" reports an outcome and names
    the reason. It concludes something."""
    assert usable_summary(
        "Не получилось запустить базу: её нет в списке 1С, доступны "
        "UT11-ER и Информационная база.") is True


# --------------------------------------------------------------------------
# the instruction is not the answer
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", INSTRUCTION_ECHOES)
def test_the_summarising_instruction_is_never_stored(text):
    assert usable_summary(text) is False


def test_something_returned_is_not_the_same_as_a_conclusion():
    """The old check was `if summary:` -- the model returning bytes was
    taken as the model returning a conclusion."""
    import inspect

    from mana import graph_memory

    source = inspect.getsource(graph_memory.distill_turn)
    assert "usable_summary(summary)" in source


def test_a_turn_that_declined_records_nothing(isolated_agent):
    """The journal already holds every exchange; this graph is for what
    was concluded."""
    graph = isolated_agent.graph_memory
    session = isolated_agent.session_id
    assert graph.record_turn(session, "какая погода?", "Не нашлось.") == 0
    assert graph.junk_nodes() == []


def test_a_turn_with_a_conclusion_is_recorded(isolated_agent):
    graph = isolated_agent.graph_memory
    node = graph.record_turn(isolated_agent.session_id,
                             "какая столица Франции?",
                             "Столица Франции — Париж, город на Сене.")
    assert node


# --------------------------------------------------------------------------
# empty turns
# --------------------------------------------------------------------------

def test_an_empty_message_is_not_stored(isolated_agent):
    """A blank "USER:" sat at the top of every recalled transcript."""
    memory = isolated_agent.persistent_memory
    session = isolated_agent.session_id
    assert memory.remember_user(session, "") == 0
    assert memory.remember_user(session, "   ") == 0
    assert memory.remember_assistant(session, "") == 0
    assert memory.remember_user(session, "настоящий вопрос") != 0


# --------------------------------------------------------------------------
# clearing what was stored before the guards
# --------------------------------------------------------------------------

def _plant(agent, texts):
    from mana.graph_memory import NODE_TURN

    for text in texts:
        agent.persistent_memory.upsert_memory_item(
            NODE_TURN, text, agent.session_id, 0.5, 0.5, {"kind": "turn"})


def test_junk_is_reported_before_anything_is_removed(isolated_agent):
    """Deleting somebody's memory is not something to do quietly."""
    _plant(isolated_agent, REFUSALS[:2] + CONCLUSIONS[:1])
    report = isolated_agent.graph_memory.forget_junk()
    assert report["removed"] is False
    assert len(report["nodes"]) == 2
    assert all("Париж" not in row["text"] for row in report["nodes"])


def test_a_cleanup_names_every_row_not_just_a_count(isolated_agent):
    """A cleanup nobody can check is one nobody should trust."""
    _plant(isolated_agent, REFUSALS[:2])
    for row in isolated_agent.graph_memory.forget_junk()["nodes"]:
        assert row["text"].strip()
        assert "id" in row


def test_consent_removes_the_junk_and_keeps_the_rest(isolated_agent):
    _plant(isolated_agent, REFUSALS[:2] + CONCLUSIONS[:1])
    graph = isolated_agent.graph_memory
    assert graph.forget_junk(consented=True)["removed"] is True

    assert graph.junk_nodes() == []
    remaining = isolated_agent.persistent_memory.semantic_search(
        "какая столица Франции", 5, session_id="", cross_session=True)
    assert any("Париж" in row["text"] for row in remaining)


def test_a_clean_memory_removes_nothing(isolated_agent):
    _plant(isolated_agent, CONCLUSIONS)
    report = isolated_agent.graph_memory.forget_junk(consented=True)
    assert report["nodes"] == [] and report["entities"] == []
    assert report["removed"] is False


def test_the_command_needs_consent_on_the_command_line():
    """A separate word, so a script or a hook cannot imply it."""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent / "app.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    literals = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "--forget-junk" in literals
    assert "--yes" in literals
