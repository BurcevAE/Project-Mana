"""
tests/test_memory_relevance.py — irrelevant memory must not be injected as
context.

Observed in a real session: after one earlier question about AI news, the
greeting "Привет, Мана. Я твой создатель" came back with a recital of that
news answer, and so did every following turn. It looked like the agent had
lost the thread of the conversation. It had not -- KnowledgeBase.search
returned `scored[:top_k]` with NO relevance floor, so with a single entry
in memory EVERY query retrieved it, and the model faithfully repeated the
context it was handed.

`min_confidence` did not help: it filters on an entry's own stored
confidence, not on its similarity to the query. Two different notions of
"good enough" that were easy to confuse.
"""
from __future__ import annotations

import pytest

from mana.knowledge import KnowledgeBase


@pytest.fixture
def kb_with_news_entry(isolated_config):
    isolated_config.use_embeddings = False   # exercise the TF-IDF / overlap paths
    kb = KnowledgeBase(isolated_config)
    kb.add("Задача: какие последние новости про ИИ\n"
           "Ответ: РИА Новости 18 июня, Известия — прямая трансляция.",
           source="llm", confidence=0.55, status="unverified")
    return kb


@pytest.mark.parametrize("greeting", [
    "Привет, Мана. Я твой создатель, Алексей.",
    "Мана, ты меня понимаешь?",
    "Как дела?",
    "Который час?",
])
def test_unrelated_query_does_not_retrieve_stored_answer(kb_with_news_entry, greeting):
    """THE observed failure: a greeting must not drag in an unrelated
    stored answer just because it is the only thing in memory."""
    hits = kb_with_news_entry.search(greeting, top_k=3, min_confidence=0.3)
    assert hits == [], f"{greeting!r} retrieved irrelevant memory: {[h.content[:40] for h in hits]}"


@pytest.mark.parametrize("query", [
    "какие последние новости про ИИ",
    "новости про ИИ",
])
def test_relevant_query_still_retrieves(kb_with_news_entry, query):
    """The floor must not be so aggressive that memory stops working --
    that would trade one failure for a worse one."""
    hits = kb_with_news_entry.search(query, top_k=3, min_confidence=0.3)
    assert hits, f"{query!r} should still find the stored answer"


def test_min_confidence_and_min_relevance_are_different_filters(kb_with_news_entry):
    """Regression guard for the confusion behind the bug: a high stored
    confidence must not buy an entry its way into an irrelevant answer."""
    kb_with_news_entry.entries[0].confidence = 0.99
    assert kb_with_news_entry.search("Привет", top_k=3, min_confidence=0.0) == [], (
        "confidence is about the entry's own reliability, not about whether "
        "it answers THIS query")


def test_known_limitation_shared_keyword_still_matches(kb_with_news_entry):
    """KNOWN LIMITATION -- documented, deliberately not "fixed".

    "Разве я просил новости?" retrieves the stored news answer, because it
    literally contains the word "новости". The match is real, not spurious.

    A similarity search cannot distinguish "I am asking about news" from
    "I am complaining that you gave me news" -- that needs intent
    understanding (negation, meta-commentary about the conversation
    itself), which no relevance threshold can supply. Raising the floor
    until this case fails would also start rejecting genuinely relevant
    queries, trading a visible annoyance for silent memory loss.

    The honest fix is intent detection, which is a separate change with
    its own evaluation. This test pins current behaviour so that fix is
    deliberate and visible when it lands.
    """
    hits = kb_with_news_entry.search("Разве я просил новости?", top_k=3, min_confidence=0.3)
    assert hits, ("if this now returns nothing, intent detection was added -- "
                  "update this test to assert the improved behaviour")


def test_embedding_and_lexical_paths_have_separate_floors(isolated_config):
    """Regression guard for the second bug: one threshold across scoring
    paths that are not on the same scale. Embedding cosine between
    unrelated text sits ~0.2-0.4; lexical scores sit near 0. A single
    0.25 floor filtered nothing wherever embeddings were available -- and
    that was invisible in an environment without sentence-transformers.
    """
    assert isolated_config.memory_min_relevance_embedding > isolated_config.memory_min_relevance_lexical


def test_relevance_floor_is_configurable(kb_with_news_entry):
    loose = kb_with_news_entry.search("Привет, Мана", top_k=3, min_relevance=0.0)
    strict = kb_with_news_entry.search("Привет, Мана", top_k=3, min_relevance=0.9)
    assert loose and not strict, "min_relevance must actually control the cut-off"


def test_empty_memory_is_still_handled(isolated_config):
    isolated_config.use_embeddings = False
    assert KnowledgeBase(isolated_config).search("что угодно") == []


def test_context_build_does_not_inject_irrelevant_memory(isolated_agent):
    """End-to-end: the same check at the level that actually builds the
    prompt, not just at the store's API."""
    from dataclasses import asdict
    from mana.pipeline import PipelineSpec
    isolated_agent.tools.call(
        "write_memory",
        content="Задача: какие последние новости про ИИ\nОтвет: РИА Новости 18 июня.",
        source="llm", confidence=0.55)
    spec = PipelineSpec(**asdict(isolated_agent.pipeline)).normalize(isolated_agent.config)
    context, trace = isolated_agent._build_context("Привет, Мана. Я твой создатель.", spec)
    assert trace["memory"] == 0, "a greeting pulled stored memory into the prompt"
    assert "РИА Новости" not in context


def test_lexical_gate_survives_when_embeddings_are_available(isolated_config):
    """The heart of the third fix.

    The old code was `if embeddings: ... elif tfidf: ...` -- so the moment
    a sentence-transformers model was installed, the lexical signal was
    discarded. Measured on real hardware, that was exactly backwards:

        query                            embedding   lexical
        "Привет, Мана. Я твой создатель"    0.577      0.000
        "напиши функцию сортировки"         0.546      0.000
        "что там было про ИИ в новостях"    0.563      0.354

    The greeting scored HIGHER than the weakest genuine query, so no
    embedding threshold could separate them, while the lexical score
    separated them perfectly. Lexical now gates; embeddings only rank.

    This test asserts the gate applies REGARDLESS of whether an embedding
    model happens to be installed -- the environment difference that hid
    the bug for two iterations.
    """
    isolated_config.use_embeddings = True     # ask for embeddings if available
    kb = KnowledgeBase(isolated_config)
    kb.add("Задача: какие последние новости про ИИ\nОтвет: РИА Новости 18 июня.",
           source="llm", confidence=0.55, status="unverified")

    for unrelated in ["Привет, Мана. Я твой создатель, Алексей.",
                      "напиши функцию сортировки",
                      "Который час?"]:
        assert kb.search(unrelated, top_k=3, min_confidence=0.3) == [], (
            f"{unrelated!r} got through the gate; if an embedding model is installed "
            f"this means the lexical gate was bypassed again")

    assert kb.search("какие последние новости про ИИ", top_k=3, min_confidence=0.3), (
        "the gate must not reject genuinely relevant queries")
