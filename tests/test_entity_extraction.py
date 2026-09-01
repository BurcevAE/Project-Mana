"""
tests/test_entity_extraction.py — topic extraction, which everything about
conversation topics depends on.

Two defects found by running the extractor on real queries:

  1. ALL-CAPS Cyrillic acronyms were invisible. The regex
     `[A-ZА-Я][a-zа-яA-Za-z0-9_+#.-]{2,}` required lowercase characters
     after the initial capital, so "ИИ", "ЦСКА", "РФ", "МВД" were never
     extracted. In Russian the acronym is often the entire subject, so
     this removed exactly the words that identify a topic.

  2. A word capitalised only because it opens a sentence was treated as an
     entity. "Какие последние новости?" returned ['какие'] -- noise in the
     memory graph (audit #22, "entity = если"), and worse: it made
     "новости про ИИ" and "новости" produce the SAME extraction, so the
     two could not be told apart at all.

The second defect is why ambiguity detection could not be built on the old
extractor: distinguishing "a topic was named" from "no topic was named" is
precisely what it could not do.
"""
from __future__ import annotations

import pytest

from mana.graph_memory import extract_entities


# --- defect 1: acronyms ---------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Какие последние новости про ИИ?", "ии"),
    ("Какой результат матча между Спартаком и ЦСКА?", "цска"),
    ("Что говорят в РФ об этом?", "рф"),
    ("Отчёт МВД опубликован", "мвд"),
])
def test_cyrillic_acronyms_are_extracted(text, expected):
    assert expected in extract_entities(text), f"{expected!r} missing from {text!r}"


def test_latin_names_still_work():
    found = extract_entities("Мы обсуждали Python и Git, а также PostgreSQL")
    for expected in ("python", "git", "postgresql"):
        assert expected in found


# --- defect 2: sentence-initial capitalisation ---------------------------

@pytest.mark.parametrize("word", ["какие", "какой", "что", "почему", "если", "разве"])
def test_interrogatives_and_openers_are_not_entities(word):
    text = f"{word.capitalize()} последние новости?"
    assert word not in extract_entities(text), f"{word!r} treated as a topic"


def test_topicless_question_yields_no_entities():
    """THE case that blocks ambiguity detection: a question naming no
    subject must extract nothing, so it is distinguishable from one that
    names a subject."""
    assert extract_entities("Какие последние новости?") == []
    assert extract_entities("А что там дальше?") == []


def test_named_and_unnamed_topics_are_distinguishable():
    """Before the fix both of these returned ['какие']."""
    named = extract_entities("Какие последние новости про ИИ?")
    unnamed = extract_entities("Какие последние новости?")
    assert named and not unnamed
    assert named != unnamed


def test_sentence_opener_kept_when_corroborated_elsewhere():
    """A capitalised opener is not automatically noise -- if the same token
    appears capitalised again, position is no longer the only evidence."""
    assert "python" in extract_entities("Python удобен. Я выбрал Python для этого.")


def test_greeting_does_not_produce_a_conversation_topic():
    """A greeting must not seed the memory graph with a topic."""
    assert extract_entities("Привет! Как дела?") == []


def test_limit_is_respected():
    text = "Python Git PostgreSQL Docker Kubernetes Ansible Terraform"
    assert len(extract_entities(text, limit=3)) == 3


def test_empty_and_noise_input():
    assert extract_entities("") == []
    assert extract_entities("   ") == []
    assert extract_entities("если бы я знал, я бы сказал") == []
