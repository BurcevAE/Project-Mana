"""
tests/test_echo_guard.py — an answer that merely repeats the last one.

Reproduced on a clean state directory before any of this existed:

    Какая погода в Воронеже?  -> погода в Воронеже, +12°, ветер 3,3 м/с
    Привет, Мана              -> погода в Воронеже, +12°, ветер 3,3 м/с

`[RECENT CONVERSATION]` puts prior turns into the prompt as
`MANA: <answer>`, and a small model reading six thousand characters of
preamble finds that the most answer-shaped text available is its own last
answer. A greeting gives it nothing better to do.
"""
from __future__ import annotations

import pytest

from mana import echo_guard

WEATHER = ("Погода в Воронеже сейчас ясная, температура воздуха +12°, "
           "ощущается как +9°. Ветер 3,3 м/с, СЗ, влажность 71%, "
           "атмосферное давление 745 мм рт. ст.")
WEATHER_REWORDED = ("Погода в Воронеже сейчас ясная, температура воздуха "
                    "+12°, ощущается как +9°. Ветер СЗ 3,3 м/с, влажность "
                    "71%, атмосферное давление 745 мм рт. ст.")


def test_the_real_failure_is_caught():
    """The case a user reported and a clean run reproduced."""
    echo = echo_guard.repeats_previous(
        "Привет, Мана", WEATHER, "Какая погода в Воронеже?", WEATHER)
    assert echo is not None
    assert echo["answer_similarity"] > 0.95
    assert echo["question_similarity"] < 0.3


def test_a_lightly_reworded_repeat_is_still_a_repeat():
    """Measured at 0.856, which a 0.90 bar let through. The threshold
    leans permissive because a false positive costs one extra call and a
    false negative costs a wrong answer shown to a person."""
    echo = echo_guard.repeats_previous(
        "Привет, Мана", WEATHER_REWORDED, "Какая погода в Воронеже?", WEATHER)
    assert echo is not None


def test_the_same_question_twice_is_consistency_not_an_echo():
    """Getting this backwards would make MANA refuse to be consistent,
    which is worse than the bug it fixes."""
    assert echo_guard.repeats_previous(
        "Какая погода в Воронеже?", WEATHER,
        "Какая погода в Воронеже?", WEATHER) is None


def test_a_genuinely_different_answer_passes():
    assert echo_guard.repeats_previous(
        "Как тебя зовут?", "Меня зовут Мана. Я когнитивный агент, и моя "
        "задача — отвечать точно либо отказываться.",
        "Какая погода в Воронеже?", WEATHER) is None


def test_a_short_answer_is_not_evidence_of_anything():
    """"Не нашлось." twice is not a repeat worth acting on -- there is
    not enough text for similarity to mean anything."""
    assert echo_guard.repeats_previous(
        "Привет", "Не нашлось.", "Погода?", "Не нашлось.") is None


def test_no_previous_answer_means_nothing_to_repeat():
    assert echo_guard.repeats_previous("Привет", WEATHER, "", "") is None


# ------------------------------------------------------- reading the history

class FakeMemory:
    def __init__(self, records):
        self.records = records

    def recent(self, session_id, limit=None):
        return self.records[-(limit or len(self.records)):]


def test_the_previous_exchange_is_the_one_before_this_turn():
    memory = FakeMemory([
        {"kind": "USER_MESSAGE", "content": "Какая погода в Воронеже?"},
        {"kind": "MANA_RESPONSE", "content": WEATHER},
        {"kind": "DECISION", "content": "task=... | route="},
    ])
    question, answer = echo_guard.previous_exchange(memory, "default")
    assert question == "Какая погода в Воронеже?"
    assert answer == WEATHER


def test_an_unreadable_history_degrades_to_no_check():
    """Losing the check is acceptable; failing the answer over it is
    not."""
    class Broken:
        def recent(self, *a, **k):
            raise RuntimeError("база занята")

    assert echo_guard.previous_exchange(Broken(), "default") == ("", "")


def test_the_guard_reads_before_the_turn_is_written():
    """The bug in the first wiring, and the reason it looked like the
    check did nothing.

    `answer(save_memory=True)` writes the exchange as it goes, so asking
    afterwards returns the CURRENT question and answer as "previous". The
    guard then compared the answer with itself, saw identical questions,
    called it consistency and passed everything through.
    """
    import inspect

    from mana.agent_parts import core

    # `solve_task` is now a thin journal wrapper; the ordering this test
    # guards lives in the body it delegates to.
    source = inspect.getsource(core.CoreMixin._solve_task)
    assert source.index("_previous_exchange") < source.index(
        'self.answer(task, self.pipeline, save_memory=True')


def test_the_retry_suppresses_all_recalled_context():
    """Suppressing only [RECENT CONVERSATION] left the same material
    reaching the model through the evidence blocks, and the retry came
    back reciting it again."""
    import inspect

    from mana.agent_parts import core

    source = inspect.getsource(core.CoreMixin._reject_echo)
    assert "memory_recent_messages = 0" in source
    assert "memory_retrieval_limit = 0" in source


def test_a_repeating_retry_keeps_the_original_answer():
    """Refusing outright would trade a wrong answer for no answer on the
    cases where the repetition was correct and the check was wrong."""
    import inspect

    from mana.agent_parts import core

    source = inspect.getsource(core.CoreMixin._reject_echo)
    assert "retry_helped" in source
    assert source.count("return result") >= 3


# --------------------------------------------------------------------------
# the knob has to reach the live path
#
# It did not. `echo_lookback` was declared, `previous_exchanges` respected
# it, the candidate generator proposed it -- and `core.py` called
# `previous_exchange`, one turn, and never saw any of it. The test that
# existed checked the function rather than the wiring, so it passed while
# the knob did nothing.
#
# Measured on a real session afterwards: an answer to "Мана, привет!" came
# back as a configurator instruction and reappeared word for word four
# turns later. At depth 1 that repeat is invisible; at 6 all four repeats
# in the session are caught.
# --------------------------------------------------------------------------

def test_the_agent_reads_as_many_exchanges_as_the_policy_says():
    import inspect

    from mana.agent_parts import core

    source = inspect.getsource(core.CoreMixin._previous_exchange)
    assert "previous_exchanges" in source
    assert "echo_lookback" in source


def test_the_agent_checks_against_all_of_them():
    import inspect

    from mana.agent_parts import core

    source = inspect.getsource(core.CoreMixin._reject_echo)
    assert "repeats_any_previous" in source


def test_a_repeat_older_than_one_turn_is_caught_at_depth(isolated_agent):
    """The live path, not the helper: the agent reads history through
    `persistent_memory`, so this drives it the way a session does."""
    from mana import policy as policy_mod
    from mana.policy import Policy

    long_answer = ("Поняла, вы хотите запустить конфигуратор информационной "
                   "базы 1С. Давайте это сделаем. Запустите программу и "
                   "выберите нужную базу в списке.")
    session = isolated_agent.session_id
    memory = isolated_agent.persistent_memory
    memory.remember_user(session, "запусти конфигуратор")
    memory.remember_assistant(session, long_answer)
    memory.remember_user(session, "спасибо")
    memory.remember_assistant(session, "Пожалуйста, обращайтесь ещё когда угодно.")
    memory.remember_user(session, "а что такое регистр сведений")
    memory.remember_assistant(session, "Регистр сведений хранит записи по измерениям.")

    with policy_mod.use(Policy.of(echo_lookback=1)):
        shallow = isolated_agent._previous_exchange()
    with policy_mod.use(Policy.of(echo_lookback=6)):
        deep = isolated_agent._previous_exchange()

    assert len(shallow) == 1
    assert len(deep) >= 3

    # "Мана, привет!" answered with the configurator text repeats
    # something three exchanges back -- invisible at depth 1.
    result = {"answer": long_answer}
    assert isolated_agent._reject_echo("Мана, привет!", dict(result),
                                       shallow) == result
    caught = isolated_agent._reject_echo("Мана, привет!", dict(result), deep)
    assert caught.get("echo") or caught["answer"] != long_answer
