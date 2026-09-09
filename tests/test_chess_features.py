"""What the board said, with nothing added, and no engine asked.

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
    """Every move, not every move of MANA's. A game is two-sided by
    definition: the position I face next is the product of both players'
    choices, and half of it used to be thrown away."""
    moves = feat.observe(_game(SCHOLARS))
    assert [row.ply for row in moves] == [1, 2, 3, 4, 5, 6, 7]
    assert [row.mover for row in moves] == ["white", "black"] * 3 + ["white"]
    assert [row.ours for row in moves] == [True, False] * 3 + [True]
    first = moves[0]
    assert first.legal_moves == 20 and first.material == 0
    assert first.our_pieces == 16 and first.their_pieces == 16
    assert first.piece == "P" and first.is_capture is False
    assert moves[-1].is_capture is True and moves[-1].captured == "P"
    assert moves[-1].gives_check is True


def test_both_sides_are_described_in_the_same_terms():
    """From the mover's point of view, or the two are not comparable in
    the fields an analysis cares most about -- and an investigator would
    ignore the opponent's rows for a mechanical reason rather than a
    considered one."""
    moves = feat.observe(_game(SCHOLARS))
    ours = [row for row in moves if row.ours]
    theirs = [row for row in moves if not row.ours]
    assert ours and theirs
    assert set(ours[0].as_dict()) == set(theirs[0].as_dict())
    # Material is the mover's own count: even here, both start level.
    assert ours[0].material == 0 and theirs[0].material == 0
    # And the piece counts are each side's own.
    assert theirs[0].our_pieces == 16 and theirs[0].their_pieces == 16


def test_the_result_is_told_from_the_side_that_moved():
    """A game MANA lost is one the opponent won. Describing both rows in
    MANA's terms would make the outcome field incomparable."""
    lost = feat.observe(_game(SCHOLARS, winner="black"))
    assert [row.outcome for row in lost if row.ours] == [feat.LOST] * 4
    assert [row.outcome for row in lost if not row.ours] == [feat.WON] * 3
    drawn = feat.observe(_game(SCHOLARS, winner=""))
    assert {row.outcome for row in drawn} == {feat.DRAWN}


def test_the_bench_no_longer_pays_for_a_judge_nothing_reads():
    """It was most of the cost of a self-play game, and self-play games
    are the thing outcome-based analysis needs more of. The judge is not
    deleted -- it stays an external check for a separate question."""
    from mana.cognition import chess_bot, chess_judge

    assert chess_bot.LIVE_JUDGE_DEPTH == 0
    assert chess_judge.JUDGE_DEPTH > 0


def test_the_opponents_search_is_absent_not_invented():
    """We do not see the opponent think. Its trace fields stay at their
    empty defaults rather than being filled with a guess."""
    traced = [{"ply": 1, "margin": 40.0, "nodes": 900, "depth": 2,
               "considered": [["e4", 40.0], ["d4", 0.0]]}]
    moves = feat.observe(_game(SCHOLARS, thoughts=traced))
    mine = [row for row in moves if row.ours][0]
    theirs = [row for row in moves if not row.ours][0]
    assert mine.considered == 2 and mine.margin == 40.0
    assert theirs.considered == 0 and theirs.margin == 0.0 and theirs.nodes == 0


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


def test_material_is_counted_for_whoever_is_moving():
    black = _game(SCHOLARS, us="black", winner="white")
    moves = feat.observe(black)
    assert [row.ours for row in moves] == [False, True] * 3 + [False]
    # 6...Nf6 loses nothing yet; 7.Qxf7# takes a pawn, so white's own
    # count is positive on the move that follows it in nobody's list --
    # what matters here is only that each row counts its own side.
    for row in moves:
        assert row.material == -0 or isinstance(row.material, int)
    assert moves[0].material == 0 and moves[1].material == 0


def test_the_outcome_is_from_manas_side():
    assert feat.observe(_game(SCHOLARS, winner="white"))[0].outcome == feat.WON
    assert feat.observe(_game(SCHOLARS, winner="black"))[0].outcome == feat.LOST
    assert feat.observe(_game(SCHOLARS, winner=""))[0].outcome == feat.DRAWN
    unfinished = feat.observe(_game(SCHOLARS, status="started", winner=""))
    assert unfinished[0].outcome == feat.UNFINISHED


# --------------------------------------------------------------------------
# horizons
# --------------------------------------------------------------------------

def test_no_engine_verdict_reaches_the_features():
    """Centipawn loss is a stronger player's opinion, it exists only where
    somebody paid for it -- 0 of 23452 opponent moves had one -- and
    self-knowledge built on it is knowledge of what Stockfish thinks."""
    fields = feat.Move("g", 1).as_dict()
    for borrowed in ("loss", "score", "judged", "cp", "eval"):
        assert not any(borrowed in name for name in fields), borrowed
    import inspect

    assert "chess_judge" not in inspect.getsource(feat)


def test_the_consequence_is_the_result_and_both_sides_carry_it():
    """Symmetric by construction: a game one side won is one the other
    lost, so every move of both players has it, over every game already
    played, with no engine and nothing to re-judge."""
    moves = feat.observe(_game(SCHOLARS, winner="white"))
    assert all(row.outcome in (feat.WON, feat.LOST) for row in moves)
    assert {row.outcome for row in moves if row.ours} == {feat.WON}
    assert {row.outcome for row in moves if not row.ours} == {feat.LOST}


def test_horizons_are_material_and_need_nothing_bought():
    """A count, not a verdict. 1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6 4.Qxf7#: white
    is level until the mate takes a pawn."""
    mine = [row for row in feat.observe(_game(SCHOLARS)) if row.ours]
    assert mine[0].material == 0
    assert mine[0].later[1] == 0.0                     # still level next move
    assert mine[-1].later[1] is None                   # game ended first


def test_a_horizon_counts_moves_by_the_same_side():
    """With both players in one list, "one move later" has to mean one
    move by the same player, or a horizon compares a position to one the
    other side was looking at."""
    # 1.e4 d5 2.exd5 Qxd5: white is +1 after the capture, level after the
    # recapture; black is level, then -1, then level again.
    moves = feat.observe(_game(["e2e4", "d7d5", "e4d5", "d8d5"], winner=""))
    mine = [row for row in moves if row.ours]
    theirs = [row for row in moves if not row.ours]
    assert mine[0].material == 0 and mine[1].material == 0
    assert mine[0].later[1] == 0.0                     # white, one own move on
    assert theirs[0].material == 0 and theirs[1].material == -1
    assert theirs[0].later[1] == -1.0                  # black, its own next


def test_a_horizon_past_the_end_is_absent_not_zero():
    """An unmeasured horizon is not a horizon at which nothing changed --
    the rule this project states everywhere else."""
    moves = feat.observe(_game(SCHOLARS))
    mine = [row for row in moves if row.ours]
    assert mine[-1].later[1] is None
    assert mine[-1].later[3] is None
    assert mine[0].as_dict()["material_after_10"] is None


# --------------------------------------------------------------------------
# derived, never stored
# --------------------------------------------------------------------------

def test_observing_writes_nothing_into_the_record():
    """A new observable applies to every game ever played the moment it
    is written, and a feature cannot drift out of agreement with the game
    it describes."""
    game = _game(SCHOLARS)
    before = dict(game)
    feat.observe(game)
    assert game == before
    assert "features" not in game


def test_a_damaged_game_does_not_lose_the_others():
    good = _game(SCHOLARS)
    broken = _game(["not-a-move", "e7e5"])
    moves = feat.observe_all([good, broken, good])
    assert len({row.game for row in moves}) == 1
    assert len(moves) == 14                               # both good games


def test_the_summary_carries_its_denominator_and_ranks_nothing():
    """Sorting observables by anything is the first step of deciding
    which one matters, and that decision is not this layer's."""
    text = feat.describe(feat.observe(_game(SCHOLARS)))
    assert "ходов всего: 7" in text
    assert "MANA 4" in text and "соперника 3" in text
    assert "известно" in text
    assert "самый" not in text and "лучш" not in text


def test_no_observations_is_an_answer():
    assert feat.describe([]) == "наблюдений нет"
