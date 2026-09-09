"""From a finding to a change, and to its test.

The step that separates an agent from an analyst. A finding says a
property goes with losing; this asks whether changing it changes
anything, and answers by playing.

Most of these tests guard the two places where this could quietly become
a person choosing the answer: the direction must come from the
measurement rather than from anybody's view of what pawn moves mean, and
a property that cannot be computed for a move before it is played must be
refused with the reason rather than approximated.
"""
from __future__ import annotations

import random

import pytest

chess = pytest.importorskip("chess", reason="оракул не приобретён")

from mana.cognition import chess_action as act
from mana.cognition import chess_outcome as outcome
from mana.cognition import findings as ledger_mod
from mana.core.gates import ACCEPTED, MIN_PAIRED_TRIALS, NOT_EVALUATED, REJECTED


def _finding(prop, share, verdict=ACCEPTED, window=outcome.WHOLE):
    return ledger_mod.Finding(
        question=outcome.QUESTION,
        approach={"property": prop, "what": prop, "window": window,
                  "needs_trace": False},
        verdict=verdict,
        measurement=ledger_mod.measurement_of(
            trials=MIN_PAIRED_TRIALS + 10, interval=(share - 0.1, share + 0.1),
            null=0.5, effect=share),
        conditions={"worlds": ["local"], "games": 100})


# --------------------------------------------------------------------------
# nobody says what to change
# --------------------------------------------------------------------------

def test_the_direction_comes_from_the_measurement():
    """Not from anybody's view of what the property means. Where the side
    with more of it lost, the change prefers less of it."""
    more, _ = act.propose([_finding("captures", 0.91)])
    less, _ = act.propose([_finding("pawn_moves", 0.35)])
    assert more[0].direction == act.MORE and "больше" in more[0].describe()
    assert less[0].direction == act.LESS and "меньше" in less[0].describe()


def test_only_accepted_findings_become_changes():
    """A rejected correlation is a question that was answered no, and a
    change built on one would be an experiment nobody had a reason to
    run."""
    changes, refused = act.propose([
        _finding("captures", 0.51, verdict=REJECTED),
        _finding("king_moves", 0.5, verdict=NOT_EVALUATED)])
    assert changes == [] and refused == []


def test_a_change_carries_the_finding_it_came_from():
    """A change nobody can trace back to the measurement that suggested
    it is a change somebody made up."""
    source = _finding("captures", 0.91)
    change, _ = act.propose([source])
    assert change[0].from_finding == source.finding_id
    assert change[0].as_dict()["from_finding"] == source.finding_id


def test_the_same_action_from_two_windows_is_one_experiment():
    """Both windows usually agree, and running it twice would spend the
    budget on the same question and count one answer as two."""
    changes, _ = act.propose([_finding("captures", 0.91, window=outcome.WHOLE),
                              _finding("captures", 0.76, window=outcome.EARLY)])
    assert len(changes) == 1
    assert len(changes[0].also_from) == 1


# --------------------------------------------------------------------------
# executability, run rather than reasoned about
# --------------------------------------------------------------------------

def test_a_property_that_cannot_be_computed_before_the_move_is_refused():
    """How many legal moves a side will have depends on the opponent's
    reply. A refusal is a result: it names a gap between what MANA can
    measure about itself and what it can act on."""
    changes, refused = act.propose([_finding("legal_moves", 0.92),
                                    _finding("material_gained_3", 0.90),
                                    _finding("margin", 0.80)])
    assert changes == []
    assert {row["property"] for row in refused} == {
        "legal_moves", "material_gained_3", "margin"}
    for row in refused:
        assert row["why"] and row["finding"]


def test_every_scorer_actually_runs_on_a_board():
    """Run, not reasoned about: a candidate that looked executable and
    was not has cost this project a discovery budget before."""
    board = chess.Board()
    for name in act.SCORERS:
        can, why = act.executable(name)
        assert can, f"{name}: {why}"
        for move in list(board.legal_moves)[:5]:
            assert isinstance(act.SCORERS[name](board, move), float)


def test_a_scorer_leaves_the_board_as_it_found_it():
    """Two of them push a move to look at what follows."""
    board = chess.Board()
    before = board.fen()
    for name in act.SCORERS:
        act.SCORERS[name](board, next(iter(board.legal_moves)))
    assert board.fen() == before


# --------------------------------------------------------------------------
# the change, applied
# --------------------------------------------------------------------------

def test_the_tie_break_picks_by_the_property():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("d5")
    tied = list(board.legal_moves)
    taking = act.Change("captures", act.MORE)
    avoiding = act.Change("captures", act.LESS)
    rng = random.Random(0)
    assert board.is_capture(taking.choose(board, tied, rng))
    assert not board.is_capture(avoiding.choose(board, tied, rng))


def test_the_search_is_not_overridden_where_it_decided():
    """A tie-break replaces a coin toss, never a judgement. Where the
    search had one best move, the change must not touch it."""
    from mana.cognition.chess_arena import SearchPlayer

    board = chess.Board()
    board.push_san("e4")
    board.push_san("d5")
    player = SearchPlayer(depth=2, trace=True)
    tuned = act.Tuned(player, act.Change("king_moves", act.MORE))
    move = tuned.choose(board, random.Random(3))
    top = max(score for _, score in player._root_scores)
    tied = [san for san, score in player._root_scores if score == top]
    if len(tied) == 1:
        assert board.san(move) == tied[0]
    else:
        assert board.san(move) in tied


def test_the_changed_player_differs_in_one_thing_only():
    from mana.cognition.chess_arena import SearchPlayer

    player = SearchPlayer(depth=2, trace=True)
    tuned = act.Tuned(player, act.Change("captures", act.MORE))
    assert tuned.depth == player.depth
    assert tuned.evaluate is player.evaluate


# --------------------------------------------------------------------------
# the experiment
# --------------------------------------------------------------------------

def test_colours_alternate_so_the_first_move_is_not_what_is_measured():
    seen = []
    result = act.duel(act.Change("captures", act.MORE), games=4, depth=1,
                      seed=5, on_game=lambda number, said: seen.append(said))
    assert result.games == 4
    assert result.changed_won + result.unchanged_won + result.drawn == 4
    assert len(seen) == 4


def test_a_drawn_game_answers_neither_way():
    result = act.Duel(change=act.Change("captures", act.MORE),
                      games=40, changed_won=18, unchanged_won=12, drawn=10)
    assert result.decided == 30
    assert act.measure(result)["trials"] == 30


def test_a_change_that_does_nothing_is_rejected_not_left_open():
    result = act.Duel(change=act.Change("captures", act.MORE),
                      games=80, changed_won=35, unchanged_won=35, drawn=10)
    measurement = act.measure(result)
    assert act.verdict_for(measurement) == REJECTED
    assert "исход не изменился" in act._note(result.change, measurement)


def test_a_change_that_wins_is_a_causal_claim():
    result = act.Duel(change=act.Change("captures", act.MORE),
                      games=80, changed_won=60, unchanged_won=10, drawn=10)
    measurement = act.measure(result)
    assert act.verdict_for(measurement) == ACCEPTED
    assert measurement["interval"][0] > act.NO_EFFECT
    assert "выигрывает чаще" in act._note(result.change, measurement)


def test_too_few_decided_games_is_not_evaluated():
    result = act.Duel(change=act.Change("captures", act.MORE),
                      games=40, changed_won=10, unchanged_won=5, drawn=25)
    assert act.verdict_for(act.measure(result)) == NOT_EVALUATED


def test_the_experiment_reaches_the_ledger_with_its_provenance(tmp_path):
    book = ledger_mod.Ledger(path=tmp_path / "findings.jsonl")
    source = _finding("captures", 0.91)
    change = act.propose([source])[0][0]
    result = act.Duel(change=change, games=80, changed_won=60,
                      unchanged_won=10, drawn=10)
    written = act.record(change, result, depth=2, ledger=book)
    assert written.question == act.QUESTION
    assert written.approach["from_finding"] == source.finding_id
    assert written.conditions["opponent"] == "тот же игрок без изменения"
    assert len(book.findings()) == 1


def test_the_correction_counts_the_experiments_run_together():
    result = act.Duel(change=act.Change("captures", act.MORE),
                      games=80, changed_won=60, unchanged_won=10, drawn=10)
    alone = act.measure(result, questions=1)
    among = act.measure(result, questions=6)
    assert among["alpha"] < alone["alpha"]
    width = lambda m: m["interval"][1] - m["interval"][0]
    assert width(among) > width(alone)


# --------------------------------------------------------------------------
# what a lever can move, asked before games are spent
# --------------------------------------------------------------------------

def _ties(count=3):
    """Positions where several moves are tied, built by hand."""
    board = chess.Board()
    board.push_san("e4")
    board.push_san("d5")
    all_san = [board.san(m) for m in board.legal_moves]
    return [(board.copy(), all_san) for _ in range(count)]


def test_a_lever_that_cannot_change_a_move_is_measured_as_such():
    """Three of the first six experiments were run on levers that could
    not change a move. `pieces` differed in none of four hundred ties, and
    seventy games bought a verdict about the shuffle."""
    board = chess.Board()                       # no promotions available
    ties = [(board.copy(), [board.san(m) for m in board.legal_moves])]
    inert = act.reach(act.Change("promotions", act.MORE), ties)
    assert inert["varies"] == 0 and inert["share"] == 0.0

    lively = act.reach(act.Change("pawn_moves", act.MORE), ties)
    assert lively["varies"] == 1 and lively["share"] == 1.0


def test_an_inert_lever_is_not_evaluated_rather_than_rejected():
    """REJECTED reads as "the finding was tested and failed". What failed
    was the experiment's ability to differ from doing nothing, and
    telling those apart is what the gates in this project are for."""
    change = act.Change("pieces", act.MORE)
    result = act.Duel(change=change, games=70, changed_won=30,
                      unchanged_won=30, drawn=10)
    finding = act.record(change, result, depth=2,
                         reached={"share": 0.0, "ties": 400, "varies": 0},
                         ledger=_nowhere())
    assert finding.verdict == NOT_EVALUATED
    assert "не двигает ничего" in finding.note
    assert "жребии" in finding.note or "жребий" in finding.note


def test_reach_travels_with_a_measured_experiment():
    change = act.Change("pawn_moves", act.LESS)
    result = act.Duel(change=change, games=70, changed_won=40,
                      unchanged_won=20, drawn=10)
    finding = act.record(change, result, depth=2,
                         reached={"share": 0.6, "ties": 400, "varies": 240},
                         ledger=_nowhere())
    assert finding.measurement["reach"] == 0.6
    assert "рычаг работал в 60%" in finding.note


def test_two_levers_that_pick_the_same_move_are_one_experiment():
    """`material`, `pieces` and `captures` disagree with each other in
    nought to one per cent of ties: three experiments, one answer."""
    ties = _ties()
    same = act.disagreement(act.Change("captures", act.MORE),
                            act.Change("captures", act.MORE), ties)
    assert same == 0.0
    apart = act.disagreement(act.Change("pawn_moves", act.MORE),
                             act.Change("pawn_moves", act.LESS), ties)
    assert apart > 0.0


def test_ties_are_taken_from_games_actually_played():
    """The question is what a lever would do in the positions this player
    reaches, and a position sampled from nowhere answers a different
    one."""
    from mana.cognition.chess_arena import SearchPlayer

    game = {"moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]}
    ties = act.ties_in([game], SearchPlayer(depth=1, trace=True), limit=5)
    assert ties and all(len(tied) > 1 for _, tied in ties)


def _nowhere():
    """A ledger that keeps nothing: these tests are about the verdict."""
    class _Void:
        def record(self, finding):
            return True

    return _Void()
