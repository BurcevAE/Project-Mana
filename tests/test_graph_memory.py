"""
tests/test_graph_memory.py — distillation heuristics and multi-hop graph
traversal. Pure computation + SQLite; no LLM/web needed.
"""
from __future__ import annotations

from mana import graph_memory as gm
from mana.memory import MemoryManager


def test_strip_filler_removes_known_filler_phrases():
    raw = "Ну, короче, я типа думаю что, собственно говоря, Git — система контроля версий."
    out = gm.strip_filler(raw)
    assert "короче" not in out.lower()
    assert "собственно говоря" not in out.lower()
    assert "git" in out.lower()


def test_extractive_distill_caps_length_and_keeps_content():
    long_text = (
        "Ну вот смотри, короче говоря, я тут подумал, в общем-то это как бы важно: "
        "Python 3.12 добавил улучшенную обработку ошибок. Кстати говоря, это довольно круто. "
        "Так сказать, короче, вот и всё что я хотел сказать про это."
    )
    out = gm.extractive_distill(long_text, max_chars=120)
    assert len(out) <= 120
    assert "python 3.12" in out.lower()


def test_extract_entities_finds_capitalized_terms():
    ents = gm.extract_entities("Мы обсуждали Python и Git, а также PostgreSQL для эволюции MANA.")
    assert "python" in ents
    assert "git" in ents


def test_record_turn_creates_node_and_follows_edge(isolated_config):
    mm = MemoryManager(isolated_config)
    store = gm.GraphMemoryStore(mm)
    n1 = store.record_turn("s1", "Расскажи про Git", "Git — система контроля версий.")
    n2 = store.record_turn("s1", "А как дела?", "Всё хорошо.")
    assert n1 and n2
    stats = store.stats("s1")
    assert stats["turn_nodes"] == 2
    assert stats["edges"] >= 1  # at least the FOLLOWS edge n1 -> n2


def test_multihop_traversal_finds_node_missed_by_direct_semantic_search(isolated_config):
    """The core claim behind graph memory: a node sharing an entity with
    the query, but with low lexical overlap, should still surface via
    graph traversal even when it does NOT make the direct semantic seed
    list. This mirrors the exact test used during development."""
    mm = MemoryManager(isolated_config)
    store = gm.GraphMemoryStore(mm)
    session = "s1"

    n_git = store.record_turn(session, "Расскажи про контроль версий",
                               "Git — распределённая система контроля версий с полной историей коммитов.")
    distractors = [
        ("Какая сегодня погода?", "Сегодня переменная облачность и лёгкий ветер."),
        ("Что приготовить?", "Можно сделать ризотто с грибами и пармезаном."),
        ("Кто выиграл матч?", "Победу одержала гостевая команда со счётом два один."),
        ("Посоветуй книгу", "Стоит прочитать роман о путешествии через несколько поколений."),
        ("Какой фильм посмотреть?", "Стоит посмотреть новую драму про музыкантов."),
    ]
    for u, a in distractors:
        store.record_turn(session, u, a)
    store.record_turn(session, "А что с бранчами в Git?",
                       "Ветки в Git позволяют параллельно разрабатывать фичи без конфликтов.")

    # Only 2 direct semantic seeds, worded to overlap the LATER git turn, not the earlier one.
    seeds_only = mm.semantic_search("параллельно фичи разработка", limit=2, session_id=session, cross_session=True)
    seed_ids = [r["id"] for r in seeds_only]
    assert n_git not in seed_ids, "sanity check: the earlier git turn should not be a direct seed here"

    context, trace = store.graph_context(session, "параллельно фичи разработка",
                                          depth=2, limit=5, seed_limit=2, recency_backbone=0)
    assert n_git in trace["visited"], "the earlier git turn should be reachable via the shared 'git' entity"
    assert n_git in trace["used"], "and should make it into the final ranked context with a realistic limit"


def test_episode_rollup_triggers_after_threshold_not_before(isolated_config):
    mm = MemoryManager(isolated_config)
    store = gm.GraphMemoryStore(mm)
    session = "s1"
    for i in range(4):
        store.record_turn(session, f"вопрос {i}", f"ответ {i}")
    assert store.maybe_rollup_episode(session, every_n_turns=5) is None, "must not roll up before threshold"
    store.record_turn(session, "вопрос 4", "ответ 4")
    episode_id = store.maybe_rollup_episode(session, every_n_turns=5)
    assert episode_id, "must roll up once the threshold is reached"
    assert store.maybe_rollup_episode(session, every_n_turns=5) is None, "must not roll up again with no new turns"
