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
    # Provenance is a condition, not part of the question: the
    # observational finding's own id contains the number of games it was
    # computed on, so putting it in the approach renamed the experiment
    # every cycle and no ledger could say it had been run.
    assert written.conditions["from_finding"] == source.finding_id
    assert written.approach == change.identity()
    assert "from_finding" not in written.approach
    assert written.conditions["opponent"] == "тот же игрок без изменения"


    assert len(book.findings()) == 1


def test_the_same_intervention_keeps_its_name_as_the_corpus_grows():
    """The identity of an experiment must not drift with the evidence
    that suggested it, or nothing can say the experiment has been run."""
    early = act.Change("pawn_moves", act.LESS, from_finding="obs-when-small")
    later = act.Change("pawn_moves", act.LESS, from_finding="obs-when-large")
    assert early.identity() == later.identity()
    assert early.as_dict() != later.as_dict()      # display still differs


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


# --------------------------------------------------------------------------
# composition: new levers out of the ones already derived
# --------------------------------------------------------------------------

def _ties_for(count=12):
    """Real positions with several moves rated equal, from real games."""
    import chess

    from mana.cognition import chess_action
    from mana.cognition.chess_arena import SearchPlayer

    board = chess.Board()
    player = SearchPlayer(depth=1, trace=True)
    out = []
    import random as _random

    while len(out) < count and not board.is_game_over() and board.ply() < 120:
        player.choose(board, _random.Random(1))
        scores = getattr(player, "_root_scores", None) or []
        if scores:
            top = max(score for _, score in scores)
            tied = [san for san, score in scores if score == top]
            if len(tied) > 1:
                out.append((board.copy(), tied))
        board.push(_random.Random(board.ply()).choice(list(board.legal_moves)))
    return out


def test_a_composite_keeps_its_primary_ordering_and_breaks_the_rest():
    """`A then B` is A wherever A speaks, and B only where A ties."""
    import chess

    from mana.cognition import chess_action

    board = chess.Board("rnbqkbnr/ppp2ppp/8/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3")
    tied = [move for move in board.legal_moves]
    pawns = chess_action.Change("pawn_moves", chess_action.MORE)
    both = chess_action.Change("pawn_moves", chess_action.MORE,
                               then=(("captures", chess_action.MORE),))
    import random

    picked_first = pawns.choose(board, tied, random.Random(3))
    picked_both = both.choose(board, tied, random.Random(3))
    assert board.piece_at(picked_first.from_square).piece_type == chess.PAWN
    assert board.piece_at(picked_both.from_square).piece_type == chess.PAWN
    assert board.is_capture(picked_both)          # the second key decided


def test_a_primitive_identity_did_not_change_when_composition_arrived():
    """Every refutation in the ledger was written against this dict. A
    new key in it would silently reopen five answered questions."""
    from mana.cognition import chess_action

    change = chess_action.Change("pawn_moves", chess_action.LESS,
                                 from_finding="obs-1")
    assert change.identity() == {
        "property": "pawn_moves", "direction": -1,
        "what": "среди равных ходов выбирать тот, у которого "
                "«pawn_moves» меньше"}


def test_a_composite_has_its_own_identity():
    from mana.cognition import chess_action

    first = chess_action.Change("pawn_moves", chess_action.LESS)
    both = chess_action.Change("pawn_moves", chess_action.LESS,
                               then=(("captures", chess_action.MORE),))
    assert both.identity() != first.identity()
    assert both.identity()["then"] == [["captures", 1]]


def test_reach_of_a_composite_counts_the_second_key_too():
    """An inert property is not inert as a second key: it is asked only
    where the first one ties, which is a different question."""
    import chess

    from mana.cognition import chess_action

    board = chess.Board()
    tied = list(board.legal_moves)
    ties = [(board, tied)]
    inert = chess_action.Change("promotions", chess_action.MORE)
    assert chess_action.reach(inert, ties)["varies"] == 0
    composed = chess_action.Change("pawn_moves", chess_action.MORE,
                                   then=(("promotions", chess_action.MORE),))
    assert chess_action.reach(composed, ties)["varies"] == 1


def _made_ties():
    """Two positions where a pawn capture, a piece capture and a quiet
    pawn move are all rated the same. Constructed rather than sampled:
    the question is what composition does when a second key has
    something to say, and an opening walk rarely offers one."""
    import chess

    board = chess.Board(
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 0 3")
    moves = lambda names: [chess.Move.from_uci(uci) for uci in names]
    return [(board, moves(["e4d5", "c3d5", "d2d4"])),
            (board, moves(["e4d5", "c3d5", "g1f3", "d2d4"]))]


def test_compose_refuses_a_part_wearing_a_new_name():
    """When the second key never varies inside the first key's best set,
    the composite plays exactly as its primary does. Measured on real
    ties: this is the common case, not a corner one."""
    from mana.cognition import chess_action

    ties = _ties_for(16)
    parts = [chess_action.Change("pawn_moves", chess_action.LESS,
                                 from_finding="o1"),
             chess_action.Change("promotions", chess_action.MORE,
                                 from_finding="o2")]
    for change, reached in chess_action.compose(parts, ties):
        assert reached["differs"] > 0
    inert = chess_action.Change("promotions", chess_action.MORE,
                                then=(("pawn_moves", chess_action.LESS),))
    primary = chess_action.Change("pawn_moves", chess_action.LESS)
    # promotions never varies here, so "promotions then pawn_moves" is
    # pawn_moves. The generator must not sell that as a new lever.
    assert chess_action.narrows(inert, primary, ties) == 0.0
    assert not [made for made, _ in chess_action.compose(parts, ties)
                if made.name() == inert.name()]


def test_novelty_is_the_set_of_moves_not_the_shuffle_s_pick():
    """Two levers can leave different sets and be handed the same move by
    the same seed. `disagreement` reports that as agreement, which would
    throw away a real new lever."""
    from mana.cognition import chess_action

    ties = _made_ties()
    part = chess_action.Change("pawn_moves", chess_action.MORE)
    made = chess_action.Change("pawn_moves", chess_action.MORE,
                               then=(("captures", chess_action.MORE),))
    assert chess_action.disagreement(made, part, ties) == 0.0
    assert chess_action.narrows(made, part, ties) > 0.0


def test_compose_produces_levers_that_reach_and_differ():
    from mana.cognition import chess_action

    ties = _made_ties()
    parts = [chess_action.Change("pawn_moves", chess_action.MORE,
                                 from_finding="o1"),
             chess_action.Change("captures", chess_action.MORE,
                                 from_finding="o2")]
    made = chess_action.compose(parts, ties)
    assert made, "composition produced nothing at all"
    names = [change.name() for change, _ in made]
    assert "pawn_moves+1×captures+1" in names
    for change, reached in made:
        assert reached["varies"] > 0                    # can move a move
        assert reached["differs"] > 0                   # is not its parts
        assert change.then
        assert change.from_finding and change.also_from  # provenance kept


def test_a_composed_change_is_playable_by_the_wrapper():
    """The point is behaviour, not a new row: the composite has to drive
    a real player."""
    import random

    import chess

    from mana.cognition import chess_action
    from mana.cognition.chess_arena import SearchPlayer

    change = chess_action.Change("captures", chess_action.MORE,
                                 then=(("checks_given", chess_action.MORE),))
    player = chess_action.Tuned(SearchPlayer(depth=1, trace=True), change)
    board = chess.Board()
    move = player.choose(board, random.Random(0))
    assert move in board.legal_moves
    assert "captures+1×checks_given+1" in player.name


# --------------------------------------------------------------------------
# a new kind of action, made from the record rather than declared
# --------------------------------------------------------------------------

def _decided(winner, moves, times=6):
    return [{"source": "local", "initial_fen": "startpos", "winner": winner,
             "moves": list(moves)} for _ in range(times)]


def test_learn_values_the_winners_actions_above_the_losers():
    """A: white won every game, so white's moves score above 0.5 and
    black's below -- from outcomes alone, with nothing about chess."""
    from mana.cognition import chess_action

    made = chess_action.learn(_decided("white", ["e2e4", "e7e5", "g1f3"]))
    assert made is not None and made.property == chess_action.LEARNED
    assert made.table["e2e4"] > 0.5 and made.table["g1f3"] > 0.5
    assert made.table["e7e5"] < 0.5
    assert made.from_finding.startswith("record:6:")       # provenance


def test_learn_ignores_draws_and_actions_seen_too_rarely():
    from mana.cognition import chess_action

    assert chess_action.learn(_decided("", ["e2e4", "e7e5"])) is None
    made = chess_action.learn(_decided("black", ["d2d4", "d7d5"])
                              + _decided("white", ["c2c4"], times=2))
    assert "c2c4" not in made.table                  # seen twice: neutral
    assert made.table["d7d5"] > 0.5 > made.table["d2d4"]


def test_a_learned_change_is_one_question_however_often_it_is_refit():
    """The table is a condition of the answer, not the question: a refit
    on a longer record must not rename the experiment, or a refuted
    action would come back every time the record grew."""
    from mana.cognition import chess_action

    first = chess_action.learn(_decided("white", ["e2e4", "e7e5"]))
    second = chess_action.learn(_decided("black", ["e2e4", "e7e5"], times=9))
    assert first.table != second.table
    assert first.identity() == second.identity()
    assert "then" not in first.identity()             # not a composite
    assert first.identity()["property"] not in chess_action.SCORERS


def test_a_learned_change_moves_a_move_on_a_prepared_position():
    """C: two moves the search rated equal; the record prefers one."""
    import random

    import chess

    from mana.cognition import chess_action

    board = chess.Board()
    tied = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    made = chess_action.Change(chess_action.LEARNED, chess_action.MORE,
                               table={"e2e4": 0.3, "d2d4": 0.8})
    assert chess_action.reach(made, [(board, tied)])["varies"] == 1
    for seed in range(5):
        assert made.choose(board, tied, random.Random(seed)).uci() == "d2d4"

    blank = chess_action.Change(chess_action.LEARNED, chess_action.MORE)
    assert not chess_action.known(blank)           # no table, no action
    assert chess_action.reach(blank, [(board, tied)])["varies"] == 0


def test_a_learned_change_goes_through_the_duel_into_the_ledger():
    """D: four real games, then the ordinary recorder. Too few to decide
    anything -- the point is the lifecycle, not the verdict."""
    from mana.cognition import chess_action, findings

    made = chess_action.learn(_decided("white", ["e2e4", "e7e5", "g1f3",
                                                 "b8c6", "d2d4"]))
    result = chess_action.duel(made, games=4, depth=1, seed=3)
    assert result.games == 4
    book = findings.Ledger()
    finding = chess_action.record(made, result, depth=1, ledger=book,
                                  reached={"share": 0.5, "ties": 2,
                                           "varies": 1})
    assert finding.verdict == "NOT_EVALUATED"            # 4 games, honestly
    assert finding.approach == made.identity()
    assert finding.conditions["learned_table"] == \
        chess_action.table_digest(made.table)
    assert any(row.finding_id == finding.finding_id for row in book.latest())
