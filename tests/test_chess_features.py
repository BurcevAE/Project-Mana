"""What the board said, with nothing added.

The point of this module is what it refuses to do, so most of these tests
are about absences: no threshold, no band, no field named for a
consequence, no ranking. `chess_findings` measures a vocabulary a person
wrote; the strongest entry in it was mined by hand from three games and
then rejected on a hundred and sixty. This layer exists so the next class
can be found rather than guessed, and it can only do that if it contains
no guesses.
"""
from __future__ import annotations

import pytest

chess = pytest.importorskip("chess", reason="оракул не приобретён")

from mana.cognition import chess_features as feat


def _game(moves, judged=None, thoughts=None, us="white", status="mate",
          winner="white", source="local"):
    return {"game": "g1", "us": us, "source": source, "level": 0,
            "initial_fen": "startpos", "status": status, "winner": winner,
            "moves": list(moves), "judged": list(judged or []),
            "thoughts": list(thoughts or [])}


SCHOLARS = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]


# --------------------------------------------------------------------------
# facts, and only facts
# --------------------------------------------------------------------------

def test_nothing_here_names_a_conclusion():
    """A feature named for its supposed consequence is a conclusion
    wearing a feature's clothes, and measuring one proves only that
    somebody already believed it."""
    # The names, not the prose: the docstring lists the words it forbids,
    # and a scan of the text trips on the warning itself. What must stay
    # clean is what a search will read -- the fields and the public names.
    fields = set(feat.Move("g", 1).as_dict())
    public = {name for name in dir(feat) if not name.startswith("_")}
    for smuggled in ("blunder", "mistake", "bad", "good", "risky",
                     "overextend", "weak", "strong", "error", "quality"):
        assert not any(smuggled in name.lower() for name in fields), smuggled
        assert not any(smuggled in name.lower() for name in public), smuggled


def test_no_thresholds_or_bands_appear():
    """A band is already an interpretation. `chess_findings` has
    SMALL_MARGIN and LARGE_MARGIN because a person decided where the
    interesting line was; this layer decides nothing."""
    import inspect

    for line in inspect.getsource(feat).splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or '"""' in stripped:
            continue
        for banded in (">= 25", "> 300", "< 25", "_MARGIN", "_FLOOR", "_CEIL"):
            assert banded not in stripped, line


def test_every_move_reports_the_position_it_was_made_in():
    moves = feat.observe(_game(SCHOLARS))
    assert [row.ply for row in moves] == [1, 3, 5, 7]      # white's moves
    first = moves[0]
    assert first.legal_moves == 20 and first.material == 0
    assert first.our_pieces == 16 and first.their_pieces == 16
    assert first.piece == "P" and first.is_capture is False
    assert moves[-1].is_capture is True and moves[-1].captured == "P"
    assert moves[-1].gives_check is True


def test_a_forced_position_is_recorded_as_a_count_not_a_verdict():
    """`legal_moves = 1` is a property of a position. Whether being there
    is good, bad, or caused by the moves before it is not this layer's
    to say -- and saying it is how a person's guess becomes a finding."""
    # King h1, black queen g2: h2 and g1 are covered, so the only legal
    # move is taking the queen. (My first attempt was stalemate, which has
    # no legal move at all -- a different thing entirely.)
    board = chess.Board("7k/8/8/8/8/8/6q1/7K w - - 0 1")
    row = feat.observe({"game": "g", "us": "white", "source": "local",
                        "initial_fen": board.fen(), "status": "draw",
                        "winner": "", "moves": ["h1g2"], "judged": [],
                        "thoughts": []})[0]
    assert row.legal_moves == 1
    assert not hasattr(row, "forced")
    assert "forced" not in row.as_dict()


def test_the_side_decides_what_material_means():
    black = _game(SCHOLARS, us="black", winner="white")
    moves = feat.observe(black)
    assert [row.ply for row in moves] == [2, 4, 6]        # black's moves
    assert moves[-1].material < 0 or moves[-1].material == 0


def test_the_outcome_is_from_manas_side():
    assert feat.observe(_game(SCHOLARS, winner="white"))[0].outcome == feat.WON
    assert feat.observe(_game(SCHOLARS, winner="black"))[0].outcome == feat.LOST
    assert feat.observe(_game(SCHOLARS, winner=""))[0].outcome == feat.DRAWN
    unfinished = feat.observe(_game(SCHOLARS, status="started", winner=""))
    assert unfinished[0].outcome == feat.UNFINISHED


# --------------------------------------------------------------------------
# horizons
# --------------------------------------------------------------------------

def _judged(plies, scores):
    return [{"ply": ply, "score_after": score, "score_before": score,
             "loss": 0.0, "judged_by": "stockfish"}
            for ply, score in zip(plies, scores)]


def test_later_evaluations_come_from_scores_already_recorded():
    """Centipawn loss answers "was this worse than the best move" and
    nothing else. It cannot see a move that is fine now and ruinous in six
    plies -- which is the mistake a two-ply search makes by construction."""
    game = _game(SCHOLARS, judged=_judged([1, 3, 5, 7], [10, 40, -60, -900]))
    moves = feat.observe(game)
    assert moves[0].later[1] == 30.0                      # 40 - 10
    assert moves[0].later[3] == -910.0                    # -900 - 10
    assert moves[1].later[1] == -100.0


def test_a_horizon_past_the_end_is_absent_not_zero():
    """An unmeasured horizon is not a horizon at which nothing changed --
    the rule this project states everywhere else."""
    game = _game(SCHOLARS, judged=_judged([1, 3, 5, 7], [10, 40, -60, -900]))
    moves = feat.observe(game)
    assert moves[-1].later[1] is None
    assert moves[-1].later[3] is None
    assert moves[0].as_dict()["after_10"] is None


def test_an_unjudged_move_has_no_horizons():
    moves = feat.observe(_game(SCHOLARS))
    assert all(row.loss is None for row in moves)
    assert all(row.later[n] is None for row in moves for n in feat.HORIZONS)


# --------------------------------------------------------------------------
# derived, never stored
# --------------------------------------------------------------------------

def test_observing_writes_nothing_into_the_record():
    """A new observable applies to every game ever played the moment it
    is written, and a feature cannot drift out of agreement with the game
    it describes."""
    game = _game(SCHOLARS, judged=_judged([1, 3], [10, 40]))
    before = dict(game)
    feat.observe(game)
    assert game == before
    assert "features" not in game


def test_a_damaged_game_does_not_lose_the_others():
    good = _game(SCHOLARS)
    broken = _game(["not-a-move", "e7e5"])
    moves = feat.observe_all([good, broken, good])
    assert len({row.game for row in moves}) == 1
    assert len(moves) == 8                                # both good games


def test_the_summary_carries_its_denominator_and_ranks_nothing():
    """Sorting observables by anything is the first step of deciding
    which one matters, and that decision is not this layer's."""
    game = _game(SCHOLARS, judged=_judged([1, 3, 5, 7], [10, 40, -60, -900]))
    text = feat.describe(feat.observe(game))
    assert "ходов MANA: 4" in text and "судимых: 4" in text
    assert "известно" in text and "из 4" in text
    assert "самый" not in text and "лучш" not in text


def test_no_observations_is_an_answer():
    assert feat.describe([]) == "наблюдений нет"
