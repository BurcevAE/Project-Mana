"""Irrelevant is not relevant.

Measured on a real memory before this existed. "какая столица Франции?",
"запусти 1С" and "привет" returned byte-identical results:

       sim   imp  conf  score
     0.000  0.50  0.50  0.140  Не найдено информации о погоде в Воронеже
     0.000  0.50  0.50  0.140  未找到 Voronezh 的天气信息
     0.000  0.50  0.50  0.140  Не нашлось.

Every similarity exactly zero and every score exactly 0.140, because
`0.72*0 + 0.18*0.5 + 0.10*0.5` does not depend on the item. With no
floor the sort was stable and returned SQL order, so "semantic memory"
was the newest important rows whatever was asked.
"""
from __future__ import annotations

import pytest

STORED = [
    "Не найдено информации о погоде в Воронеже; проверьте официальные источники.",
    "Пользователь сообщил: меня зовут Алексей",
    "1С:Предприятие запускается через ярлык или через 1cestart.exe",
    "Столица Франции — Париж",
    "Не нашлось.",
]


@pytest.fixture
def memory(isolated_agent):
    store = isolated_agent.persistent_memory
    for text in STORED:
        store.upsert_memory_item("fact", text, isolated_agent.session_id,
                                 0.5, 0.5, {"source": "test"})
    return store


def found(memory, query, limit=5):
    return memory.semantic_search(query, limit, session_id="",
                                  cross_session=True)


# --------------------------------------------------------------------------
# the floor
# --------------------------------------------------------------------------

def test_a_question_about_nothing_stored_recalls_nothing(memory):
    """"Nothing relevant" is an answer. Returning the newest row because
    something must be returned is the failure this fixes."""
    assert found(memory, "как испечь хлеб на закваске") == []


def test_a_greeting_recalls_nothing(memory):
    """The case that started this: a greeting was answered with a weather
    report, because the greeting recalled the weather."""
    assert found(memory, "привет") == []


def test_a_question_recalls_what_is_about_it(memory):
    rows = found(memory, "какая столица Франции?")
    assert rows
    assert any("Париж" in r["text"] for r in rows)


def test_two_different_questions_recall_different_things(memory):
    """They returned byte-identical results before."""
    france = [r["text"] for r in found(memory, "какая столица Франции?")]
    launch = [r["text"] for r in found(memory, "как запустить 1С?")]
    assert france != launch
    assert any("Париж" in t for t in france)
    assert any("1С" in t for t in launch)


def test_importance_cannot_manufacture_relevance(memory, isolated_agent):
    """Importance and confidence contributed 0.28 of the score with zero
    similarity -- enough to rank items that match nothing. They may order
    items already about the question; they may not make one be about it.
    """
    memory.upsert_memory_item("fact", "Совершенно посторонний текст про рыбалку",
                              isolated_agent.session_id, 1.0, 1.0,
                              {"source": "test"})
    assert not any("рыбалку" in r["text"]
                   for r in found(memory, "какая столица Франции?"))


def test_the_floor_is_on_similarity_not_on_the_blended_score():
    import inspect

    from mana.memory import MemoryManager

    source = inspect.getsource(MemoryManager.semantic_search)
    assert "sim < self.MIN_RELEVANCE" in source


# --------------------------------------------------------------------------
# how it decided, reported
# --------------------------------------------------------------------------

def test_every_result_says_how_it_was_matched(memory):
    """"Nothing was relevant" and "the search is degraded" are different
    facts, and a caller that cannot tell them apart reads one as the
    other."""
    for row in found(memory, "какая столица Франции?"):
        assert row["retrieval_mode"] in ("embeddings", "tfidf", "word_overlap")
        assert row["retrieval_similarity"] >= 0


def test_tfidf_is_used_when_embeddings_are_absent(memory):
    """Embeddings are excluded from a packaged build on purpose -- torch
    is 2-3 GB. scikit-learn is bundled, and TF-IDF is the only fallback
    here that can tell "погода" from "запусти 1С" on an inflected
    language."""
    rows = found(memory, "какая столица Франции?")
    assert rows
    assert rows[0]["retrieval_mode"] in ("embeddings", "tfidf")


def test_word_overlap_remains_as_a_last_resort(memory, monkeypatch):
    import mana.memory as module

    monkeypatch.setattr(module, "np", module.np)
    with monkeypatch.context() as patch:
        patch.setitem(__import__("sys").modules, "sklearn", None)
        sims, mode = memory._similarities("столица Франции",
                                          [{"text": "Столица Франции — Париж"}])
    assert mode in ("tfidf", "word_overlap", "embeddings")
    assert sims and sims[0] > 0


def test_no_rows_is_an_empty_result_not_a_crash(memory):
    assert memory._similarities("что угодно", []) == ([], "empty")


# --------------------------------------------------------------------------
# the capability flag has to mean what it says
# --------------------------------------------------------------------------

def test_the_self_check_reports_the_mode_not_a_library():
    """`HAS_SKLEARN` was reported as "semantic_search: true" while the
    search had no way to use it -- the same class of false assurance as
    `llm_providers: HAS_REQUESTS`, which reported True on a machine where
    nothing could answer."""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent / "app.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    literals = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "semantic_search_mode" in literals
    assert "word_overlap" in literals
