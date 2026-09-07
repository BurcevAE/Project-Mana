"""Practice, on an oracle that was proved first.

The load-bearing test is `test_practice_refuses_an_unverified_oracle`.
Without it the whole verification step in `acquire.py` is decoration: a
corpus generated on a subtly wrong rules engine is wrong in a way nothing
downstream can detect, so the refusal has to be a refusal and not a
warning.
"""
from __future__ import annotations

import json

import pytest

from mana import acquire
from mana.cognition import chess_arena as arena
from mana.cognition.chess_arena import (Corpus, Game, Labelled, Player,
                                        RandomPlayer, SearchPlayer, Unverified)

acquire.ensure_importable()
pytest.importorskip("chess", reason="оракул не приобретён")


@pytest.fixture(scope="module")
def chess():
    return arena.oracle()


# --------------------------------------------------------------------------
# the refusal that makes the acquisition real
# --------------------------------------------------------------------------

def test_practice_refuses_an_unverified_oracle(monkeypatch):
    refused = acquire.Verification(
        "chess_rules", "chess", acquire.REFUSED,
        (acquire.CheckResult("взятие на проходе", False, True, False, "правила"),))
    monkeypatch.setattr(acquire, "verify", lambda name, deep=False: refused)

    with pytest.raises(Unverified) as raised:
        arena.play(RandomPlayer(), RandomPlayer())
    assert "ОТКЛОНЁН" in str(raised.value)


def test_an_absent_oracle_is_also_a_refusal(monkeypatch):
    absent = acquire.Verification("chess_rules", "", acquire.ABSENT,
                                  note="не установлен")
    monkeypatch.setattr(acquire, "verify", lambda name, deep=False: absent)
    with pytest.raises(Unverified):
        arena.self_play(RandomPlayer(), 1)


def test_verified_but_vanished_is_a_refusal(monkeypatch):
    passing = acquire.Verification("chess_rules", "chess", acquire.VERIFIED)
    monkeypatch.setattr(acquire, "verify", lambda name, deep=False: passing)
    monkeypatch.setattr(acquire, "installed", lambda name: None)
    with pytest.raises(Unverified):
        arena.oracle()


# --------------------------------------------------------------------------
# games actually finish
# --------------------------------------------------------------------------

def test_a_game_reaches_a_real_ending(chess):
    game = arena.play(RandomPlayer(), RandomPlayer(), seed=1, chess=chess)
    assert game.result in ("1-0", "0-1", "1/2-1/2")
    assert game.plies > 0
    assert game.reason


def test_draw_claims_are_asked_for(chess):
    """Without repetition and the fifty-move rule two shallow searches
    shuffle pieces to the ply cap, and the game is recorded as a draw for
    the wrong reason."""
    import inspect

    source = inspect.getsource(arena.play)
    assert "claim_draw=True" in source


def test_the_ply_cap_bounds_the_worst_case(chess):
    game = arena.play(RandomPlayer(), RandomPlayer(), seed=3, max_plies=20,
                      chess=chess)
    assert game.plies <= 20


def test_the_same_seed_gives_the_same_game(chess):
    first = arena.play(RandomPlayer(), RandomPlayer(), seed=42, chess=chess)
    second = arena.play(RandomPlayer(), RandomPlayer(), seed=42, chess=chess)
    assert first.result == second.result
    assert first.plies == second.plies
    assert first.sampled == second.sampled


# --------------------------------------------------------------------------
# the search is not just producing legal noise
# --------------------------------------------------------------------------

def test_the_material_player_beats_random(chess):
    """A sanity check on the search itself. Measured over 20 games it
    scored 0.975; a small run here only has to show it is not a coin."""
    played = arena.match(SearchPlayer(depth=2, name="material"),
                         RandomPlayer(), 6, seed=500, chess=chess)
    score = arena.score_of(played, "material")
    assert score["games"] == 6
    assert score["score"] > 0.8


def test_a_forced_mate_is_taken(chess):
    """Mate scored by depth, so a win is finished rather than wandered
    around in."""
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")   # Ra8#
    move = SearchPlayer(depth=2).choose(board, __import__("random").Random(0))
    board.push(move)
    assert board.is_checkmate()


def test_colours_alternate_in_a_match(chess):
    played = arena.match(SearchPlayer(depth=1, name="a"),
                         RandomPlayer("b"), 4, seed=9, chess=chess)
    assert [g.white for g in played] == ["a", "b", "a", "b"]


# --------------------------------------------------------------------------
# positions are not independent observations
# --------------------------------------------------------------------------

def test_only_a_few_positions_are_kept_per_game(chess):
    game = arena.play(RandomPlayer(), RandomPlayer(), seed=11, chess=chess)
    assert 0 < len(game.sampled) <= arena.POSITIONS_PER_GAME
    assert len(set(game.sampled)) == len(game.sampled)


def test_the_opening_is_not_sampled():
    """Openings are shared by every game and say nothing about the
    players; the last plies are decided already and label themselves."""
    kept = arena._sample_indices(120, 4)
    assert min(kept) >= arena.SKIP_OPENING_PLIES
    assert max(kept) < 120 - arena.SKIP_ENDING_PLIES


def test_a_game_too_short_to_sample_yields_nothing():
    assert arena._sample_indices(10, 4) == []


def test_every_position_carries_the_game_it_came_from(chess):
    """Losing that is how correlated positions get counted as independent
    observations."""
    played = arena.self_play(RandomPlayer(), 3, seed=21, chess=chess)
    rows = arena.labelled(played)
    assert rows
    assert {r.game_id for r in rows} == {0, 1, 2}
    for row in rows:
        assert row.label == played[row.game_id].score


def test_the_corpus_reports_games_and_positions_separately(tmp_path, chess):
    """Stated together on purpose: the position count is not a count of
    independent examples and must never be read as one."""
    corpus = Corpus(tmp_path / "games.jsonl")
    corpus.append(arena.self_play(RandomPlayer(), 4, seed=31, chess=chess))
    stats = corpus.stats()
    assert stats["games"] == 4
    assert stats["independent_units"] == stats["games"]
    assert stats["positions"] > stats["games"]


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def test_a_player_against_itself_is_not_scored():
    """Folding it in as a draw would drag every score towards 0.5 and
    quietly flatter a weak player."""
    games = [Game("a", "a", "1-0", "checkmate", 40, 0.1, 0)]
    score = arena.score_of(games, "a")
    assert score["games"] == 0
    assert score["self_play_skipped"] == 1


def test_scoring_counts_from_both_colours():
    games = [Game("a", "b", "1-0", "checkmate", 40, 0.1, 0),
             Game("b", "a", "1-0", "checkmate", 40, 0.1, 1),
             Game("a", "b", "1/2-1/2", "stalemate", 40, 0.1, 2)]
    score = arena.score_of(games, "a")
    assert (score["wins"], score["draws"], score["losses"]) == (1, 1, 1)
    assert score["score"] == 0.5


# --------------------------------------------------------------------------
# the corpus on disk
# --------------------------------------------------------------------------

def test_games_survive_a_round_trip(tmp_path, chess):
    corpus = Corpus(tmp_path / "games.jsonl")
    played = arena.self_play(RandomPlayer(), 3, seed=41, chess=chess)
    assert corpus.append(played) == 3
    read_back = corpus.games()
    assert [g.as_dict() for g in read_back] == [g.as_dict() for g in played]


def test_the_corpus_accumulates_across_runs(tmp_path, chess):
    corpus = Corpus(tmp_path / "games.jsonl")
    corpus.append(arena.self_play(RandomPlayer(), 2, seed=1, chess=chess))
    corpus.append(arena.self_play(RandomPlayer(), 2, seed=50, chess=chess))
    assert corpus.stats()["games"] == 4


def test_a_damaged_line_costs_games_not_a_crash(tmp_path, chess):
    path = tmp_path / "games.jsonl"
    corpus = Corpus(path)
    corpus.append(arena.self_play(RandomPlayer(), 1, seed=61, chess=chess))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{ not json at all\n")
    corpus.append(arena.self_play(RandomPlayer(), 1, seed=62, chess=chess))
    assert corpus.stats()["games"] == 2


def test_an_unwritable_corpus_does_not_raise(tmp_path, chess):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    corpus = Corpus(blocker / "sub" / "games.jsonl")
    assert corpus.append(arena.self_play(RandomPlayer(), 1, seed=71,
                                         chess=chess)) == 0


def test_the_module_says_the_baseline_is_a_floor():
    """It is the control arm a learned evaluation has to beat, not a
    hand-written solution smuggled in as a starting point."""
    import re

    doc = re.sub(r"\s+", " ", arena.__doc__ or "")
    assert "a floor, not a fix" in doc
