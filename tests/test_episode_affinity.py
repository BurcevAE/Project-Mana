"""
tests/test_episode_affinity.py — the scorer that episode-based context
would rest on. Measured offline first; NOT wired into the agent.

Benchmark result on real logged dialogues (7 turns, 3 episodes, 5
follow-up queries), scripts/benchmark_episode_affinity.py:

    scorer      R@1   R@2  ties   contamination   loss
    words      0.80  0.80  0.20           0.00    0.00
    stems      1.00  1.00  0.00           0.00    0.00
    ngrams     0.80  1.00  0.00           0.25    0.14   (floor=0)

Two findings worth keeping:

  * A per-turn floor INSIDE the chosen episode only caused loss. At
    floor=0.05 stems lost 29% of the turns it should have kept and
    prevented no contamination, because episode selection was already
    correct. The episode is the filter; a second filter inside it bought
    nothing at this scale. (Scale caveat below.)

  * Word-level overlap fails on Russian inflection and aggregation does
    NOT rescue it -- I predicted it would and the measurement said
    otherwise. Its apparent R@1 of 1.00 in the first run was an artefact:
    every episode tied at 0.0 and list order happened to pick correctly.
    The benchmark now refuses to count a tie as a hit, which is the more
    important fix -- a metric that rewards luck is worse than no metric.
"""
from __future__ import annotations

import pytest

from mana.episode_affinity import (Episode, episode_affinity, rank_episodes,
                                    score_ngrams, score_stems, score_words, stem)

NEWS = Episode("A", ["Привет, Мана. Какие есть последние новости про ИИ?",
                     "Но 28 мая это очень давно, ты знаешь какая сейчас дата?",
                     "То есть РИА новости не публиковала новости свежее чем 28 мая?"])
DATES = Episode("B", ["Сколько прошло дней с 29 мая?",
                      "126 дней, думаешь это самые свежие новости?"])
FOOTBALL = Episode("C", ["Скажи как прошел матч ЦСКА Локомотив? какой был счет?",
                         "Как сыграли Зенит и ЦСКА? Какой был счет?"])
ALL = [NEWS, DATES, FOOTBALL]


# --- the observed contamination case -------------------------------------

def test_the_126_days_turn_does_not_win_a_football_query():
    """The concrete failure this whole line of work came from: a football
    question answered with "матч, который мог состояться 126 дней назад"."""
    ranked = rank_episodes("Узнай когда был последний матч и с каким счетом завершился", ALL)
    assert ranked[0][0] == "C", ranked
    dates_score = dict(ranked)["B"]
    football_score = dict(ranked)["C"]
    assert football_score > dates_score


@pytest.mark.parametrize("query,expected", [
    ("Узнай когда был последний матч и с каким счетом завершился", "C"),
    ("что там с новостями", "A"),
    ("а какой был счёт", "C"),
    ("сколько это дней получилось", "B"),
    ("что там про ИИ говорили", "A"),
])
def test_top_ranked_episode_is_correct(query, expected):
    assert rank_episodes(query, ALL, "stems")[0][0] == expected


# --- inflection ----------------------------------------------------------

def test_stemming_collapses_russian_inflection():
    assert stem("новостями") == stem("новости"), (stem("новостями"), stem("новости"))


def test_word_overlap_fails_on_inflection_at_turn_level():
    """Documents WHY stemming was added -- and why the turn-level
    counterexample did not disqualify word overlap at episode level."""
    assert score_words("что там с новостями", "Какие есть последние новости про ИИ?") == 0.0
    assert score_stems("что там с новостями", "Какие есть последние новости про ИИ?") > 0.0


def test_word_overlap_does_not_recover_at_episode_level_either():
    """Corrects a claim I made and then measured: I expected aggregation
    over several turns to rescue word overlap from inflection. It does
    not -- "новостями" matches no token in the news episode at all. Word
    overlap only scored R@1 1.00 in an early benchmark run because every
    episode tied at 0.0 and list order picked the right one. Counting
    that tie as a hit was the measurement bug; with ties excluded, word
    overlap scores 0.80 with 20% undecided."""
    assert episode_affinity("что там с новостями", NEWS, "words") == 0.0
    assert episode_affinity("что там с новостями", NEWS, "stems") > 0.0


# --- the design property: no hard boundary -------------------------------

def test_a_turn_can_relate_to_several_episodes():
    """"126 дней, думаешь это самые свежие новости?" is arithmetic ABOUT
    news freshness. A partition would have to pick one; affinity does not,
    which is the point of scoring instead of segmenting."""
    turn = "126 дней, думаешь это самые свежие новости?"
    assert score_stems(turn, NEWS.turns[0]) > 0.0
    assert score_stems(turn, DATES.turns[0]) > 0.0


def test_ranking_makes_no_decision():
    ranked = rank_episodes("что там", ALL)
    assert len(ranked) == len(ALL)
    assert all(isinstance(score, float) for _, score in ranked)


def test_affinity_uses_best_turns_not_the_mean():
    """A long episode must not dilute one clearly relevant turn."""
    padded = Episode("A", NEWS.turns + ["погода", "ужин", "кино", "музыка", "спорт"])
    assert episode_affinity("что там с новостями", padded, "stems") > 0.0


def test_empty_episode_scores_zero():
    assert episode_affinity("что угодно", Episode("X", [])) == 0.0


def test_unrelated_query_scores_low_everywhere():
    ranked = rank_episodes("напиши функцию сортировки на python", ALL, "stems")
    assert all(score < 0.1 for _, score in ranked), ranked


def test_not_wired_into_the_agent_yet():
    """Guard the sequencing decision: this is measured offline before it
    can drop half a conversation in the live context path."""
    import pathlib
    ctx = (pathlib.Path(__file__).resolve().parent.parent
           / "mana" / "agent_parts" / "context.py").read_text(encoding="utf-8")
    assert "episode_affinity" not in ctx, (
        "episode_affinity reached _build_context -- if that is intended, "
        "update this test and report contamination/loss on real logs first")
