"""
tests/test_app_intent.py — an instruction is carried out, not narrated.

Measured before any of this existed:

    Открой Notepad++  ->  "Открываю Notepad++."   tools called: none
    Запусти 1С        ->  "Запускаю 1С."          tools called: none

Ten application tools were registered and not one was reachable: the
answer pipeline calls a fixed list and never consults the registry for a
tool matching the request. So the model narrated the action and nothing
happened -- a claim resting on nothing, in a project built so that every
claim rests on a measurement.
"""
from __future__ import annotations

import pytest

from mana.apps import intent


# ------------------------------------------------------------- recognising

@pytest.mark.parametrize("text,action", [
    ("Открой Notepad++", "launch_editor"),
    ("открой notepad++", "launch_editor"),
    ("Запусти Notepad++", "launch_editor"),
    ("запусти блокнот++", "launch_editor"),
    ("Запусти 1С", "launch_onec"),
    ("запусти 1C", "launch_onec"),
    ("Открой 1с", "launch_onec"),
])
def test_an_instruction_is_recognised(text, action):
    found = intent.match(text)
    assert found is not None, text
    assert found.action == action


@pytest.mark.parametrize("text", [
    "Как открыть Notepad++?",
    "расскажи, как запустить 1С",
    "что такое Notepad++",
    "Привет, Мана",
    "6+5",
    "",
])
def test_a_question_about_an_application_is_not_an_instruction(text):
    """A matcher that fired on both would start launching programs at
    people who asked for advice. Only an imperative at the start counts."""
    assert intent.match(text) is None


def test_a_named_base_is_carried_through():
    found = intent.match('Запусти 1С базу "UT11-ER"')
    assert found.action == "launch_onec"
    assert found.params["base"] == "UT11-ER"


def test_the_configurator_is_asked_for_explicitly():
    found = intent.match("Открой 1С конфигуратор")
    assert found.params.get("designer") is True


def test_a_path_opens_that_file():
    found = intent.match(r"Открой C:\Работа\отчёт.txt")
    assert found.action == "open_file"
    assert found.params["path"].endswith("отчёт.txt")


def test_a_path_with_a_named_editor_still_opens_the_file():
    found = intent.match(r"Открой в Notepad++ C:\Работа\заметки.md")
    assert found.action == "open_file"
    assert found.params["path"].endswith("заметки.md")


# --------------------------------------------------- reporting the outcome

class FakeResult:
    def __init__(self, ok, output=None, error=""):
        self.ok, self.output, self.error = ok, output, error


class FakeRegistry:
    def __init__(self, result):
        self.result, self.calls = result, []

    def call(self, name, **kwargs):
        self.calls.append((name, kwargs))
        return self.result


def test_the_answer_comes_from_what_happened_not_from_what_was_asked():
    """The whole defect in one property. An answer written from the
    request says "открываю" whether or not anything opened."""
    found = intent.match("Открой Notepad++")
    registry = FakeRegistry(FakeResult(True, {"pid": 4242, "path": ""}))
    outcome = intent.perform(found, registry)
    assert registry.calls[0][0] == "open_in_editor"
    assert registry.calls[0][1]["launch_only"] is True
    assert "4242" in intent.describe(found, outcome)


def test_a_refusal_is_passed_through_as_a_refusal():
    """"не найден Notepad++" is a useful answer. "Открываю" is a false
    one, and it was what the user got."""
    found = intent.match("Открой Notepad++")
    registry = FakeRegistry(FakeResult(False, None, "не найден notepad++.exe"))
    outcome = intent.perform(found, registry)
    answer = intent.describe(found, outcome)
    assert "Не получилось" in answer
    assert "не найден" in answer


def test_the_password_warning_survives_into_the_answer():
    found = intent.match('Запусти 1С базу "UT11-ER"')
    registry = FakeRegistry(FakeResult(True, {
        "base": "UT11-ER", "kind": "клиент-сервер", "location": "srv/base",
        "mode": "предприятие", "warning": "пароль виден в командной строке"}))
    answer = intent.describe(found, intent.perform(found, registry))
    assert "пароль виден" in answer


def test_the_selector_answer_names_the_bases_offered():
    found = intent.match("Запусти 1С")
    registry = FakeRegistry(FakeResult(True, {
        "bases_offered": ["UT11-ER", "Информационная база"]}))
    answer = intent.describe(found, intent.perform(found, registry))
    assert "UT11-ER" in answer
    assert "выбираете вы" in answer


# ------------------------------------------------------------- the wiring

def test_the_intent_is_checked_before_the_model():
    """So it works on an installation with no language model -- which is
    the state a fresh one is in, and the state the laptop was in when it
    refused every question."""
    import inspect

    from mana.agent_parts import core

    source = inspect.getsource(core.CoreMixin.solve_task)
    assert "_perform_app_intent" in source
    assert source.index("_perform_app_intent") < source.index("self.answer(task")


# ------------------------------------------------- naming a specific base

def test_a_base_named_without_quotes_is_recognised(monkeypatch):
    """Reported: asked for a specific base, got the chooser.

    The first version only saw a name in quotes, so "Запусти 1С базу
    UT11-ER" -- how a person actually types it -- opened the chooser and
    MANA looked like it had ignored half the instruction.
    """
    monkeypatch.setattr(intent, "_known_bases",
                        lambda: ["UT11-ER", "Информационная база"])
    found = intent.match("Запусти 1С базу UT11-ER")
    assert found.params["base"] == "UT11-ER"


def test_a_base_name_alone_is_enough(monkeypatch):
    """A name out of the machine's own list is as strong a signal as the
    word "1С", and grounded rather than guessed."""
    monkeypatch.setattr(intent, "_known_bases", lambda: ["UT11-ER"])
    found = intent.match("Запусти UT11-ER")
    assert found is not None
    assert found.action == "launch_onec"
    assert found.params["base"] == "UT11-ER"


def test_cyrillic_lookalikes_still_find_the_base(monkeypatch):
    """"UT11-ER" typed as "ут11-er" looks identical to a person and
    matches nothing at all to a comparison over code points -- the same
    trap the "1С" pattern already had to allow for."""
    monkeypatch.setattr(intent, "_known_bases", lambda: ["UT11-ER"])
    found = intent.match("запусти ут11-er")
    assert found is not None
    assert found.params["base"] == "UT11-ER"


def test_the_longest_matching_base_wins(monkeypatch):
    """A base called "УТ" must not win over "УТ11-ER" in a message that
    names the longer one."""
    monkeypatch.setattr(intent, "_known_bases", lambda: ["УТ", "УТ11-ER"])
    assert intent.match("Запусти УТ11-ER").params["base"] == "УТ11-ER"


def test_an_unknown_base_is_passed_on_rather_than_dropped(monkeypatch):
    """`onec_launch` refuses with the list of what exists, which tells
    the person more than silently opening the chooser."""
    monkeypatch.setattr(intent, "_known_bases", lambda: ["UT11-ER"])
    found = intent.match("Запусти 1С базу Бухгалтерия")
    assert found.params.get("base")


def test_a_question_naming_a_base_is_still_not_an_instruction(monkeypatch):
    monkeypatch.setattr(intent, "_known_bases", lambda: ["UT11-ER"])
    assert intent.match("Как запустить UT11-ER?") is None
