"""
tests/test_clarification.py — ask instead of guessing, but rarely.

Scenario that motivated this:
    1. "Какие последние новости про ИИ?"
    2. "Какой результат матча между Спартаком и ЦСКА?"
    3. "Какие последние новости?"   <- which of the two?

Guessing here is worse than asking, and asking is what a person would do.

The governing constraint is the COST ASYMMETRY, and it runs opposite to
most gates in this codebase: a clarifying question is a REFUSAL TO ANSWER.
Over-asking is therefore worse than the occasional wrong guess it
prevents, so the trigger is narrow and the false-ask rate is measured
below rather than assumed.

Three bugs were found while building this, all by running it:
  * `новост\\b` never matched "новости" -- a word boundary after a stem is
    a contradiction, and the exact reported phrase went undetected;
  * counting entities from ASSISTANT answers made a single-subject chat
    ask "про ИИ или РИА?", offering a news agency the user never raised;
  * counting entities rather than TURNS turned one football question into
    two candidates: "про ЦСКА или Спартак?".
"""
from __future__ import annotations

import pytest

from mana.intent import format_clarifying_question, is_ambiguous_followup

TWO_TOPICS = [["ии"], ["цска", "спартаком"]]
ONE_TOPIC = [["ии"]]


# --- it asks exactly when it should --------------------------------------

@pytest.mark.parametrize("query", [
    "Какие последние новости?",
    "А что там дальше?",
    "Какой результат?",
    "Есть что-то свежее?",
])
def test_asks_when_subject_is_missing_and_topics_compete(query):
    assert is_ambiguous_followup(query, TWO_TOPICS), query


def test_the_reported_scenario(self=None):
    """The literal example: two subjects raised, third turn names neither."""
    result = is_ambiguous_followup("Какие последние новости?", TWO_TOPICS)
    assert result.is_ambiguous
    assert result.candidates == ["ии", "цска"]
    assert "ИИ" in format_clarifying_question(result.candidates)
    assert "ЦСКА" in format_clarifying_question(result.candidates)


# --- and stays quiet otherwise (the expensive direction) -----------------

def test_does_not_ask_when_only_one_topic_is_in_play():
    """With nothing to guess between, asking is pure friction."""
    result = is_ambiguous_followup("Какие последние новости?", ONE_TOPIC)
    assert not result
    assert "nothing to guess between" in result.reason


@pytest.mark.parametrize("query", [
    "Какие последние новости про ИИ?",
    "Какой результат матча Спартак ЦСКА?",
    "Расскажи про Git",
])
def test_does_not_ask_when_the_question_names_its_subject(query):
    result = is_ambiguous_followup(query, TWO_TOPICS)
    assert not result
    assert "names its own subject" in result.reason


@pytest.mark.parametrize("query", [
    "Сколько будет 17 умножить на 23?",
    "Привет, Мана",
    "Напиши функцию сортировки",
    "Почему небо голубое?",
    "Спасибо",
])
def test_does_not_ask_on_self_contained_questions(query):
    assert not is_ambiguous_followup(query, TWO_TOPICS), query


def test_false_ask_rate_is_zero_on_the_labelled_set():
    """The measurement that matters. Asking is a refusal to answer, so a
    false ask costs more than a missed one -- this asserts the operating
    point, not merely that the feature works."""
    should_not_ask = [
        "Сколько будет 17 умножить на 23?", "Привет, Мана", "Напиши функцию сортировки",
        "Почему небо голубое?", "Спасибо", "Какие последние новости про ИИ?",
        "Расскажи про Git", "зачем нужны модульные тесты",
    ]
    false_asks = [q for q in should_not_ask if is_ambiguous_followup(q, TWO_TOPICS)]
    assert not false_asks, f"asked unnecessarily on: {false_asks}"


def test_empty_input_never_asks():
    assert not is_ambiguous_followup("", TWO_TOPICS)
    assert not is_ambiguous_followup("   ", TWO_TOPICS)


def test_no_history_never_asks():
    assert not is_ambiguous_followup("Какие последние новости?", [])


# --- one candidate per turn, not per entity ------------------------------

def test_one_question_with_two_names_is_one_candidate():
    """"Спартак — ЦСКА" is one subject the user raised once."""
    result = is_ambiguous_followup("Какие последние новости?",
                                    [["цска", "спартаком"], ["ии"]])
    assert result.candidates == ["цска", "ии"], result.candidates


# --- the question is built by code, not invented by a model --------------

def test_question_lists_the_detected_topics_only():
    question = format_clarifying_question(["ии", "цска"])
    assert "ИИ" in question and "ЦСКА" in question
    assert question.endswith("?")


def test_question_degrades_gracefully_with_too_few_candidates():
    assert format_clarifying_question(["ии"]).endswith("?")
    assert format_clarifying_question([]).endswith("?")


def test_acronyms_are_shown_uppercase_not_as_index_keys():
    """Topics are stored lowercased because they are graph keys. Showing
    the user "про ии, цска" would expose internal index form."""
    question = format_clarifying_question(["ии", "цска"])
    assert "ИИ" in question and "ЦСКА" in question
    assert "ии" not in question and "цска" not in question


# --- integration ---------------------------------------------------------

def test_agent_asks_instead_of_guessing(isolated_agent):
    agent = isolated_agent
    for user, mana in [("Какие последние новости про ИИ?", "РИА сообщает о сервисе ИИ."),
                       ("Какой результат матча Спартак ЦСКА?", "ЦСКА выиграл 2:1.")]:
        agent.persistent_memory.remember_user(agent.session_id, user)
        agent.persistent_memory.remember_assistant(agent.session_id, mana)

    result = agent.solve_task("Какие последние новости?")
    assert result["trace"].get("clarification_requested") is True
    assert "ИИ" in result["answer"] and "ЦСКА" in result["answer"]
    assert result["llm_ok"] is False, "asking must not require an LLM call"


def test_agent_does_not_ask_when_subject_is_named(isolated_agent):
    agent = isolated_agent
    for user, mana in [("Какие последние новости про ИИ?", "РИА сообщает о сервисе ИИ."),
                       ("Какой результат матча Спартак ЦСКА?", "ЦСКА выиграл 2:1.")]:
        agent.persistent_memory.remember_user(agent.session_id, user)
        agent.persistent_memory.remember_assistant(agent.session_id, mana)
    result = agent.solve_task("Какие новости про ИИ?")
    assert result["trace"].get("clarification_requested") is not True


def test_assistant_entities_do_not_become_candidates(isolated_agent):
    """A single-subject conversation must not start asking about a news
    agency the user never mentioned."""
    agent = isolated_agent
    agent.persistent_memory.remember_user(agent.session_id, "Какие последние новости про ИИ?")
    agent.persistent_memory.remember_assistant(
        agent.session_id, "РИА Новости и Коммерсантъ сообщают о запуске сервиса ИИ.")
    result = agent.solve_task("Какие последние новости?")
    assert result["trace"].get("clarification_requested") is not True


def test_clarification_can_be_disabled(isolated_agent):
    agent = isolated_agent
    agent.config.clarify_ambiguous_followups = False
    for user, mana in [("новости про ИИ?", "РИА."), ("матч Спартак ЦСКА?", "ЦСКА 2:1.")]:
        agent.persistent_memory.remember_user(agent.session_id, user)
        agent.persistent_memory.remember_assistant(agent.session_id, mana)
    result = agent.solve_task("Какие последние новости?")
    assert result["trace"].get("clarification_requested") is not True


# --- length: found by a false ask in live use ----------------------------

#: Verbatim from a real session. MANA asked "Уточни: про ИИ или ЦСКА?"
#: about this, which is a refusal to answer a perfectly clear instruction.
LIVE_FALSE_ASK = ("Я не говорил что матч был сегодня, узнай когды был последний "
                  "матч и с каким счетом закончился.")


def test_live_false_ask_is_not_repeated():
    """The labelled set above was too easy -- every "should not ask" case
    was a short self-contained question. This one names its subject with
    ORDINARY NOUNS ("матч", "счёт"), which extract_entities cannot see,
    so "no entities found" wrongly read as "no subject named".

    Length is what separates them: a turn that inherits its topic is a few
    words; a sentence long enough to explain itself carries its subject.
    """
    assert not is_ambiguous_followup(LIVE_FALSE_ASK, TWO_TOPICS), (
        "asked for clarification on an explicit 17-word instruction")


@pytest.mark.parametrize("query", [
    "Расскажи подробнее про последние события в мире технологий",
    "узнай когда был последний матч и с каким счётом он закончился",
    "покажи что там было самое свежее за последние несколько дней",
])
def test_long_requests_are_treated_as_self_contained(query):
    assert not is_ambiguous_followup(query, TWO_TOPICS), f"{len(query.split())} words: {query}"


@pytest.mark.parametrize("query", [
    "Какие последние новости?",
    "А что там дальше?",
    "Какой результат?",
    "Есть что-то свежее?",
])
def test_short_followups_still_ask(query):
    """The ceiling must not silence genuine ambiguity."""
    assert is_ambiguous_followup(query, TWO_TOPICS), query


def test_word_ceiling_is_configurable():
    long_followup = "какие там были самые последние новости"
    assert not is_ambiguous_followup(long_followup, TWO_TOPICS, max_words=3)
    assert is_ambiguous_followup(long_followup, TWO_TOPICS, max_words=10)


def test_agent_does_not_ask_on_the_live_false_ask_case(isolated_agent):
    agent = isolated_agent
    for user, mana in [("Какие последние новости про ИИ?", "РИА сообщает о сервисе ИИ."),
                       ("Как сыграли Зенит и ЦСКА?", "Данных нет.")]:
        agent.persistent_memory.remember_user(agent.session_id, user)
        agent.persistent_memory.remember_assistant(agent.session_id, mana)
    result = agent.solve_task(LIVE_FALSE_ASK)
    assert result["trace"].get("clarification_requested") is not True


def test_history_window_counts_user_turns_not_events(isolated_agent):
    """UNITS BUG found live. `recent()` returns EVENTS -- user and
    assistant interleaved -- so asking for `clarify_history_turns` (6)
    showed only 3 user turns. In the observed session the AI-news and
    football topics had scrolled out of view, and a question about news
    was answered with "Уточни: про Воронеже или Gismeteo?" -- offering two
    weather entities and omitting the topic actually asked about.
    """
    agent = isolated_agent
    topics = ["Расскажи последние новости про ИИ",
              "А как сыграли ЦСКА и Локомотив?",
              "А за какую дату есть результаты?",
              "Мана, какая завтра погода в Воронеже?",
              "Посмотри на сайте Gismeteo"]
    for user in topics:
        agent.persistent_memory.remember_user(agent.session_id, user)
        agent.persistent_memory.remember_assistant(agent.session_id, "ответ")

    result = agent._ambiguous_followup("Напомни какие были там последние новости?")
    assert result.is_ambiguous
    assert "ии" in result.candidates, (
        f"the AI topic scrolled out of the window: {result.candidates}")
    assert "цска" in result.candidates, result.candidates
