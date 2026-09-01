"""
tests/test_intent.py — the conversation-reference detector, and the
accuracy number that justifies it existing.

Background: two independent measurements (lexical and embedding paths)
showed no similarity threshold separates "asking about X" from
"complaining that you gave me X" -- the complaint scored HIGHER than the
weakest genuine request on both. It was never a relevance problem: such a
turn refers to the previous exchange, not to storage.

These tests measure precision and recall on a labelled set rather than
asserting the detector "works". When patterns change, the number moves,
and that is the point.

Operating point, chosen deliberately: HIGH PRECISION over high recall.
A false negative means memory is searched when it needn't be -- bounded,
because the lexical gate already blocks the worst irrelevant hits. A false
positive means memory is skipped when it was needed -- the agent loses
context it had. Losing information is the worse failure, so the patterns
are narrow on purpose and recall on unseen phrasings is genuinely low.
"""
from __future__ import annotations

import pytest

from mana.intent import refers_to_previous_turn

# Turns that comment on the exchange itself.
META = [
    "Разве я просил новости?",
    "Я не спрашивал про новости",
    "Зачем ты дал мне новости?",
    "Хватит про новости",
    "Почему ты опять вспомнил про ИИ?",
    "При чём тут новости?",
    "Это не то, что я просил",
    "Я имел в виду другое",
    "Ты меня не понял",
    "Ты ошибся",
]

# Genuine requests about the world -- must never be mistaken for meta.
GENUINE = [
    "какие последние новости про ИИ",
    "новости про искусственный интеллект",
    "что там было про ИИ в новостях",
    "расскажи про новости ИИ ещё раз",
    "Привет, Мана",
    "напиши функцию сортировки",
    "сколько будет 17 умножить на 23",
    "почему небо голубое",
    "зачем нужны модульные тесты",
    "зачем ты нужен вообще?",          # about the agent, but an honest question
    "почему ты работаешь на Python?",
    "расскажи, зачем нужны юнит-тесты",
    "я не понимаю рекурсию, объясни",
]


@pytest.mark.parametrize("text", META)
def test_meta_remarks_are_detected(text):
    assert refers_to_previous_turn(text, has_previous_assistant_turn=True), text


@pytest.mark.parametrize("text", GENUINE)
def test_genuine_requests_are_not_flagged(text):
    result = refers_to_previous_turn(text, has_previous_assistant_turn=True)
    assert not result, f"{text!r} wrongly flagged as a conversation reference ({result.matched})"


def test_precision_is_perfect_on_the_labelled_set():
    """False positives cost more than false negatives here -- see the
    module docstring. This asserts the operating point, not just 'it works'."""
    tp = sum(1 for t in META if refers_to_previous_turn(t))
    fp = sum(1 for t in GENUINE if refers_to_previous_turn(t))
    precision = tp / max(1, tp + fp)
    assert precision == 1.0, f"precision dropped to {precision:.2f} ({fp} false positives)"


def test_recall_on_this_set_is_tracked():
    tp = sum(1 for t in META if refers_to_previous_turn(t))
    assert tp / len(META) == 1.0


def test_known_limitation_unseen_phrasings_are_missed():
    """KNOWN LIMITATION, documented rather than hidden.

    This is lexical pattern matching, not understanding. Phrasings with no
    matching marker are missed. Held-out examples measured at the time of
    writing: 0/6 caught before the pattern list was widened. Recall on
    genuinely novel wording remains low by design -- widening the patterns
    until this test fails would trade precision for recall in the wrong
    direction.
    """
    unseen = [
        "Ответь на мой вопрос, а не про новости",
        "Не надо мне новостей",
        "Это неправильный ответ",
    ]
    missed = [t for t in unseen if not refers_to_previous_turn(t)]
    assert missed, ("if all of these are now caught, recall improved -- re-check "
                    "precision on GENUINE before celebrating, and update this test")


def test_no_previous_assistant_turn_suppresses_detection():
    """At the start of a session nothing can refer back. Treating an
    opening message as a correction would skip memory for no reason."""
    result = refers_to_previous_turn("Разве я просил новости?", has_previous_assistant_turn=False)
    assert not result
    assert "no previous assistant turn" in result.reason


def test_empty_input_is_not_a_reference():
    assert not refers_to_previous_turn("", has_previous_assistant_turn=True)
    assert not refers_to_previous_turn("   ", has_previous_assistant_turn=True)


def test_result_explains_itself():
    """A wrong decision must be diagnosable, not guessed at."""
    hit = refers_to_previous_turn("Разве я просил новости?")
    assert hit.kind == "rejection" and hit.matched and hit.reason
    miss = refers_to_previous_turn("какие последние новости про ИИ")
    assert miss.reason == "no conversation-reference marker found"


# --- integration: the gate actually skips memory -------------------------

def test_conversation_reference_skips_long_term_memory(isolated_agent):
    from dataclasses import asdict

    from mana.pipeline import PipelineSpec

    agent = isolated_agent
    agent.tools.call("write_memory",
                     content="Задача: какие последние новости про ИИ\nОтвет: РИА Новости 18 июня.",
                     source="llm", confidence=0.55)
    agent.persistent_memory.remember_assistant(agent.session_id, "Вот последние новости про ИИ...")
    spec = PipelineSpec(**asdict(agent.pipeline)).normalize(agent.config)

    _, trace = agent._build_context("Разве я просил новости?", spec)
    assert trace["memory"] == 0
    assert trace.get("memory_skipped") == "conversation_reference"

    _, trace = agent._build_context("какие последние новости про ИИ", spec)
    assert trace["memory"] > 0, "a genuine request must still reach memory"
    assert trace.get("memory_skipped") is None


def test_gate_is_inactive_before_any_assistant_turn(isolated_agent):
    from dataclasses import asdict

    from mana.pipeline import PipelineSpec

    agent = isolated_agent
    agent.tools.call("write_memory",
                     content="Задача: какие последние новости про ИИ\nОтвет: РИА Новости 18 июня.",
                     source="llm", confidence=0.55)
    spec = PipelineSpec(**asdict(agent.pipeline)).normalize(agent.config)
    _, trace = agent._build_context("Разве я просил новости?", spec)
    assert trace.get("memory_skipped") is None, (
        "with no prior assistant turn this cannot be a correction")


def test_conversation_reference_also_skips_web(isolated_agent):
    """Observed on real hardware in 5.7.9: "Разве я просил новости?" still
    issued a live web search ("веб: 3 результатов" in the trace), because
    the first version of this gate only skipped KnowledgeBase. A remark
    about the previous turn is not a request for current information, so
    no retrieval of any kind belongs there -- neither stored nor live."""
    from dataclasses import asdict

    from mana.pipeline import PipelineSpec

    agent = isolated_agent
    agent.persistent_memory.remember_assistant(agent.session_id, "Вот последние новости про ИИ...")
    spec = PipelineSpec(**asdict(agent.pipeline))
    spec.web_mode = "always"
    spec.use_web = True
    spec = spec.normalize(agent.config)

    _, trace = agent._build_context("Разве я просил новости?", spec)
    assert trace["web"] == 0, "a meta-remark triggered a web search"
    assert trace.get("web_skipped") == "conversation_reference"
    assert trace.get("memory_skipped") == "conversation_reference"


def test_genuine_current_events_question_still_reaches_web_path(isolated_agent):
    """The gate must not suppress retrieval for real questions."""
    from dataclasses import asdict

    from mana.pipeline import PipelineSpec

    agent = isolated_agent
    agent.persistent_memory.remember_assistant(agent.session_id, "Вот последние новости про ИИ...")
    spec = PipelineSpec(**asdict(agent.pipeline))
    spec.web_mode = "always"
    spec.use_web = True
    spec = spec.normalize(agent.config)

    _, trace = agent._build_context("какие последние новости про ИИ", spec)
    assert trace.get("web_skipped") is None
    assert trace.get("memory_skipped") is None


def test_correction_turn_gets_a_response_instruction(isolated_agent):
    """Blocking retrieval was not sufficient.

    Measured over 5 live trials of 5.7.11: the gate held (web=0, memory=0)
    on every correction turn, yet the answer still recapped the topic --
    "Хватит про новости" was answered with another paragraph of news. The
    cause is that RECENT CONVERSATION legitimately stays in context (you
    cannot answer a correction without knowing what is being corrected),
    so the fix is not less context but an explicit instruction on how to
    respond to one.
    """
    from dataclasses import asdict

    from mana.pipeline import PipelineSpec

    agent = isolated_agent
    agent.persistent_memory.remember_assistant(agent.session_id,
                                                "Вот последние новости про ИИ: РИА, Коммерсантъ...")
    spec = PipelineSpec(**asdict(agent.pipeline)).normalize(agent.config)

    context, trace = agent._build_context("Хватит про новости", spec)
    assert "[РЕПЛИКА О РАЗГОВОРЕ]" in context
    assert "НЕ пересказывай" in context
    assert trace.get("conversation_reference_kind")


def test_no_response_instruction_for_a_genuine_question(isolated_agent):
    """The instruction must appear only where it applies -- adding it to
    ordinary turns would suppress legitimate answers."""
    from dataclasses import asdict

    from mana.pipeline import PipelineSpec

    agent = isolated_agent
    agent.persistent_memory.remember_assistant(agent.session_id, "Вот новости про ИИ...")
    spec = PipelineSpec(**asdict(agent.pipeline)).normalize(agent.config)
    context, _ = agent._build_context("какие последние новости про ИИ", spec)
    assert "[РЕПЛИКА О РАЗГОВОРЕ]" not in context
