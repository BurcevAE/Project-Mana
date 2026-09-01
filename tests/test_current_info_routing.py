"""
tests/test_current_info_routing.py — questions needing fresh information
must actually reach the web.

Found in a live session, and both failures were silent -- MANA did not
say it had skipped the search, it said it had no information:

    "Мана, какая завтра погода в Воронеже?"  -> general, web NOT called
    "какая сегодня погода в Воронеже"        -> current, web called

The only difference is "завтра" vs "сегодня". "сегодня" was in the
keyword lists; "завтра" and "вчера" were in none of them. Sports results
had no coverage at all.

Root cause was structural, not a missing word: the same list was written
out FOUR times in routing.py (_task_category, _should_use_web,
_route_signature, classify_route) and had already drifted. This is the
third time in this project that duplicated keyword lists disagreed -- the
programming classifiers did it too. CURRENT_INFO_TERMS is now the single
source.
"""
from __future__ import annotations

from dataclasses import asdict

import pytest

from mana.agent_parts.routing import CURRENT_INFO_TERMS
from mana.pipeline import PipelineSpec


def _spec(agent):
    spec = PipelineSpec(**asdict(agent.pipeline))
    spec.web_mode = "auto"
    spec.use_web = True
    return spec.normalize(agent.config)


@pytest.mark.parametrize("query", [
    "Мана, какая завтра погода в Воронеже?",
    "какая сегодня погода в Воронеже",
    "что было вчера на бирже",
    "А как сыграли ЦСКА и Локомотив?",
    "Какой был счёт в матче ЦСКА Локомотив?",
    "результаты матча ЦСКА Локомотив",
    "Расскажи последние новости про ИИ",
    "какой сейчас курс доллара",
])
def test_current_information_questions_reach_the_web(isolated_agent, query):
    isolated_agent.config.enable_web = True
    assert isolated_agent._should_use_web(query, _spec(isolated_agent)), query


@pytest.mark.parametrize("query", [
    "Сколько будет 17 умножить на 23?",
    "напиши функцию сортировки",
    "Привет, Мана",
    "объясни, что такое рекурсия",
])
def test_self_contained_questions_do_not_reach_the_web(isolated_agent, query):
    """Searching when there is nothing to search for costs latency and
    pollutes context with irrelevant snippets."""
    isolated_agent.config.enable_web = True
    assert not isolated_agent._should_use_web(query, _spec(isolated_agent)), query


def test_all_four_call_sites_share_one_list():
    """Guard against the drift that caused this. If a literal keyword list
    reappears, the lists will diverge again exactly as they did before."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "mana" / "agent_parts" / "routing.py").read_text(encoding="utf-8")
    body = src.split("CURRENT_INFO_TERMS = (", 1)[1].split(")", 1)[1]
    assert '"сегодня", "сейчас", "актуаль"' not in body, (
        "a literal current-info keyword list reappeared outside CURRENT_INFO_TERMS")


def test_temporal_terms_that_were_missing_are_present():
    for term in ("завтра", "вчера", "погод", "матч", "результат"):
        assert term in CURRENT_INFO_TERMS, term


def test_known_limitation_typos_defeat_substring_matching(isolated_agent):
    """KNOWN LIMITATION, documented rather than papered over.

    The live session contained "А как сыглали ЦСКА и Локомотив?" -- a typo
    for "сыграли". Substring matching cannot see through it, so the
    question still does not reach the web, while the correctly spelled
    version now does.

    Not fixed here on purpose: fuzzy matching would fire on unrelated
    words too, and the cost of a wrong web call is real. Doing it properly
    means edit-distance matching with a measured false-positive rate --
    its own change, with its own benchmark.
    """
    isolated_agent.config.enable_web = True
    spec = _spec(isolated_agent)
    assert isolated_agent._should_use_web("А как сыграли ЦСКА и Локомотив?", spec)
    assert not isolated_agent._should_use_web("А как сыглали ЦСКА и Локомотив?", spec), (
        "if typos now route correctly, fuzzy matching was added -- check the "
        "false-positive rate before celebrating and update this test")


# --- the agent must not invent limitations it does not have --------------

def test_prompt_lists_the_tools_that_are_actually_available(isolated_agent_exec_enabled):
    """Observed: asked "ты можешь посмотреть в интернете?", MANA replied
    that it cannot work in real time -- right after running three
    searches. Nothing in the prompt ever told it what it can do."""
    agent = isolated_agent_exec_enabled
    spec = PipelineSpec(**asdict(agent.pipeline)).normalize(agent.config)
    prompt = agent._compose_prompt("ты можешь посмотреть в интернете?", "", spec)
    assert "Что ты умеешь прямо сейчас" in prompt
    assert "verify_arithmetic" in prompt
    assert "run_code" in prompt, "exec is enabled in this fixture, so it must be listed"


def test_unavailable_tools_are_not_advertised(isolated_agent):
    """isolated_agent has LLM and web disabled -- claiming otherwise would
    be the same false-capability problem in reverse."""
    agent = isolated_agent
    spec = PipelineSpec(**asdict(agent.pipeline)).normalize(agent.config)
    prompt = agent._compose_prompt("что ты умеешь?", "", spec)
    # Only the generated LIST is checked: the surrounding instruction
    # sentence legitimately names web_search while telling the model not
    # to deny having it.
    listing = agent._available_tools_line()
    assert "web_search" not in listing
    assert "run_code" not in listing
    assert "llm_generate" not in listing


def test_tool_line_never_raises(isolated_agent):
    assert isinstance(isolated_agent._available_tools_line(), str)
