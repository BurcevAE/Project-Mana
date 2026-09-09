"""An outside judge, and a player that says why.

The judge decides nothing. Every test here is about that line: Stockfish
is asked what a move cost after it was made, and is never asked what to
play. The moment an engine picks the move, the experiment stops being
about MANA and becomes a measurement of the engine.

The trace tests are the other half. A decision nobody can look inside is
one nobody can diagnose -- said twice already in this project, about
`onec_launch` reporting a launch it never watched and about `classify`
returning a kind with no way to ask which feature decided.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

chess = pytest.importorskip("chess", reason="оракул не приобретён")

from mana.cognition import chess_arena as arena
from mana.cognition import chess_judge as judge

engine = pytest.mark.skipif(not judge.available(),
                            reason="Stockfish не установлен")


# --------------------------------------------------------------------------
# the judge is never the brain
# --------------------------------------------------------------------------

def test_the_judge_module_never_picks_a_move():
    """The whole point. An engine that chooses would make every result a
    fact about the engine."""
    import inspect

    source = inspect.getsource(judge)
    for choosing in ("engine.play(", ".play(board", "def choose"):
        assert choosing not in source


def test_the_player_is_manas_own_search():
    player = arena.SearchPlayer(depth=2)
    move = player.choose(chess.Board(), random.Random(1))
    assert move in chess.Board().legal_moves
    assert player.nodes > 0          # it searched; nothing was asked


@engine
def test_a_blunder_costs_more_than_a_reasonable_move():
    """Scholar's mate: 6...Nf6 walks into mate and 6...g6 does not."""
    with judge.Judge(depth=8) as judging:
        board = chess.Board()
        for san in ("e4", "e5", "Qh5", "Nc6", "Bc4"):
            board.push_san(san)
        walked_in = judging.judge_move(board, board.parse_san("Nf6"))
        defended = judging.judge_move(board, board.parse_san("g6"))

    assert walked_in.blunder is True
    assert defended.loss < judge.MISTAKE
    assert walked_in.loss > defended.loss


@engine
def test_a_mate_cannot_set_the_average_by_itself():
    """A forced mate converts to ten thousand centipawns. One of those in
    a hundred moves would make the mean a report about whether the game
    ended in mate."""
    with judge.Judge(depth=8) as judging:
        board = chess.Board()
        for san in ("e4", "e5", "Qh5", "Nc6", "Bc4"):
            board.push_san(san)
        walked_in = judging.judge_move(board, board.parse_san("Nf6"))
    assert walked_in.loss == judge.MAX_LOSS


@engine
def test_every_judgement_says_who_made_it():
    with judge.Judge(depth=6) as judging:
        row = judging.judge_move(chess.Board(), chess.Board().parse_san("e4"))
    assert row.judged_by == judge.BY_ENGINE
    assert row.as_dict()["judged_by"] == judge.BY_ENGINE


def test_without_an_engine_it_falls_back_and_says_so():
    """An absent engine is a normal state with a working fallback, and a
    number whose source cannot be told apart is one nobody can compare."""
    judging = judge.Judge(path=None)
    judging.path = None
    assert judging.kind == judge.BY_MATERIAL

    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
    row = judging.judge_move(board, board.parse_san("Qh4"))
    assert row.judged_by == judge.BY_MATERIAL
    assert row.as_dict()["judged_by"] == judge.BY_MATERIAL


def test_the_material_judge_sees_a_pawn_given_away():
    """One reply deep is all it sees, and that is stated rather than
    stretched: after 1.e4 e5, playing d4 hands a pawn to exd4 and Nf3
    hands over nothing.

    My first attempt at this test used 1.Nf3 e5 2.Nxe5 as the blunder,
    which is not one -- it wins a pawn. The judge was right and the test
    was wrong.
    """
    judging = judge.Judge(path=None)
    judging.path = None
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    gives_a_pawn = judging.judge_move(board, board.parse_san("d4"))
    keeps_it = judging.judge_move(board, board.parse_san("Nf3"))
    assert gives_a_pawn.loss >= 100
    assert keeps_it.loss == 0.0


# --------------------------------------------------------------------------
# the player says why
# --------------------------------------------------------------------------

def test_a_traced_move_carries_what_was_considered():
    player = arena.SearchPlayer(depth=2, trace=True)
    board = chess.Board()
    move = player.choose(board, random.Random(3))

    thought = player.thoughts[-1]
    assert thought.chosen == board.san(move)
    assert len(thought.considered) == board.legal_moves.count()
    assert thought.nodes > 0
    assert thought.considered == sorted(thought.considered,
                                        key=lambda row: -row[1])


def test_the_margin_says_whether_it_was_a_decision():
    """A move ahead by two centipawns is a tie the evaluation happened to
    win. Measured on the baseline: 77% of moves had no margin at all, and
    those cost 195 against 151 for the rest."""
    player = arena.SearchPlayer(depth=2, trace=True)
    player.choose(chess.Board(), random.Random(5))
    thought = player.thoughts[-1]
    # Material counting at depth two in the opening: everything ties.
    assert thought.margin == 0.0
    assert thought.close_call is True


def test_tracing_is_off_unless_asked_for():
    """The scores are computed either way; keeping them for five hundred
    games would hold several million rows for nobody."""
    player = arena.SearchPlayer(depth=2)
    player.choose(chess.Board(), random.Random(1))
    assert player.thoughts == []


def test_a_trace_survives_being_written_down():
    player = arena.SearchPlayer(depth=2, trace=True)
    player.choose(chess.Board(), random.Random(2))
    row = player.thoughts[-1].as_dict()
    assert set(row) >= {"ply", "fen", "chosen", "considered", "margin",
                        "close_call", "depth", "nodes"}
    assert row["fen"] == chess.Board().fen()


def test_the_summary_reports_its_denominator():
    """"нашли 23 ошибки" means nothing without how many moves were looked
    at -- the same rule `invariants.summarise` states for turns."""
    rows = [judge.Judged(ply=1, fen="", move="e4", loss=0.0),
            judge.Judged(ply=2, fen="", move="Qh5", loss=400.0)]
    summary = judge.summarise(rows)
    assert summary["moves"] == 2
    assert summary["blunders"] == 1 and summary["mistakes"] == 1
    assert summary["mean_loss"] == 200.0


# --------------------------------------------------------------------------
# where the engine is looked for
# --------------------------------------------------------------------------

def test_an_override_cannot_hide_an_installed_engine(monkeypatch):
    """`shared_data_root()` was added so an acquired binary is found
    however MANA was started, and then written to honour MANA_DATA_DIR --
    which is the thing that makes it move with the launch. Set the
    variable and both roots collapse, bringing back the failure that
    judged three real games by material."""
    from mana import paths

    monkeypatch.setenv("MANA_DATA_DIR", str(Path.cwd()))
    roots = judge.searched()
    assert paths.platform_data_root() / judge.ENGINE_DIRNAME in roots


def test_the_places_looked_in_are_listed_without_repeats(monkeypatch, tmp_path):
    monkeypatch.setenv("MANA_DATA_DIR", str(tmp_path))
    roots = judge.searched()
    assert len(roots) == len(set(roots))
    assert tmp_path / judge.ENGINE_DIRNAME in roots


def test_an_explicit_path_wins_over_the_search(monkeypatch, tmp_path):
    """An escape hatch for a search that is wrong for a reason nobody has
    found yet: the same command found the engine from one shell and not
    from another on the same machine."""
    named = tmp_path / "my-stockfish.exe"
    named.write_bytes(b"")
    monkeypatch.setenv(judge.ENGINE_ENV, str(named))
    assert judge.engine_path() == named
    assert judge.available() is True


def test_a_named_path_that_is_not_there_is_reported_not_ignored(monkeypatch, tmp_path):
    monkeypatch.setenv(judge.ENGINE_ENV, str(tmp_path / "absent.exe"))
    rows = judge.diagnose()
    assert rows[0]["why"].endswith("но файла нет")


def test_each_place_says_what_was_wrong_with_it(monkeypatch, tmp_path):
    """"Не найден" plus a list of paths leaves the reader to guess which
    of four things happened, and the four have different remedies."""
    a_file = tmp_path / "not-a-dir"
    a_file.write_text("", encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(judge, "searched",
                        lambda: [tmp_path / "absent", a_file, empty])
    reasons = [row["why"] for row in judge.diagnose()]
    assert "каталога нет" in reasons[0]
    assert "это не каталог" in reasons[1]
    assert "файлов stockfish* нет" in reasons[2]


def test_a_directory_that_cannot_be_read_does_not_stop_the_search(monkeypatch, tmp_path):
    """A different account, a policy, a drive that went away -- normal
    states here, and none of them may hide a working engine elsewhere."""
    engine = tmp_path / "good"
    engine.mkdir()
    (engine / "stockfish.exe").write_bytes(b"")

    class _Refusing(type(tmp_path)):
        def is_dir(self):
            raise PermissionError(13, "refused")

    monkeypatch.setattr(judge, "searched",
                        lambda: [_Refusing(tmp_path / "denied"), engine])
    assert judge.engine_path() == engine / "stockfish.exe"


def test_a_missing_engine_says_where_it_looked():
    """"Stockfish не найден" names nothing that can be checked, and it was
    read three times in one day without telling anyone where to look."""
    from mana.cognition import chess_bot

    import mana.cognition.chess_judge as judging
    original = judging.engine_path
    judging.engine_path = lambda: None
    try:
        note = chess_bot.judge_note(8)
    finally:
        judging.engine_path = original
    assert "искала:" in note
    assert judge.ENGINE_DIRNAME.replace("/", os.sep) in note
    assert judge.ENGINE_ENV in note              # and how to answer it
