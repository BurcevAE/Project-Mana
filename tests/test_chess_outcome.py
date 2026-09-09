"""Which properties go with losing, asked of the result and nothing else.

No engine and no thresholds. A game has two sides, one had more of a
property than the other, and one of them won -- so the question is a
comparison inside a game and never a cutoff. In self-play the opponent is
the same player, so everything the two sides share cancels exactly, and
what is left is what differed between the winner and the loser.

The tests that matter most are the ones about what would quietly make a
result meaningless: an opponent with no search trace turning "considered
more moves" into "was MANA", a whole-game window letting the result
explain the property, and a draw counted as half an answer.
"""
from __future__ import annotations

import pytest

chess = pytest.importorskip("chess", reason="оракул не приобретён")

from mana.cognition import chess_outcome as oc
from mana.core.gates import ACCEPTED, MIN_PAIRED_TRIALS, NOT_EVALUATED, REJECTED


def _sides(number, ours, theirs, we_won, source="local", traced=True):
    """One reduced game, given directly rather than replayed."""
    return oc.Sides(
        game=f"g{number}", source=source, we_won=we_won, both_traced=traced,
        ours={(name, window): value for name, value in ours.items()
              for window in oc.WINDOWS},
        theirs={(name, window): value for name, value in theirs.items()
                for window in oc.WINDOWS})


def _run(count, higher_wins, prop="captures"):
    """`higher_wins` games where the side with more of it won, the rest
    where it lost."""
    out = []
    for i in range(count):
        wins = i < higher_wins
        out.append(_sides(i, {prop: 2.0}, {prop: 1.0}, we_won=wins))
    return out


# --------------------------------------------------------------------------
# the pairing
# --------------------------------------------------------------------------

def test_a_lopsided_property_is_accepted():
    games = _run(MIN_PAIRED_TRIALS + 20, MIN_PAIRED_TRIALS + 18)
    out = [f for f in oc.look(games, record=False)
           if f.approach["property"] == "captures"
           and f.approach["window"] == oc.WHOLE]
    assert len(out) == 1 and out[0].verdict == ACCEPTED
    assert out[0].measurement["interval"][0] > oc.NO_CONNECTION
    assert "чаще выигрывает" in out[0].note


def test_a_property_that_loses_is_a_finding_too():
    """A property whose higher side loses is as much a result as one
    whose higher side wins, and collapsing the two into "no effect" would
    throw away the half that points at a mistake."""
    games = _run(MIN_PAIRED_TRIALS + 20, 2)
    out = [f for f in oc.look(games, record=False)
           if f.approach["property"] == "captures"
           and f.approach["window"] == oc.WHOLE][0]
    assert out.verdict == ACCEPTED
    assert out.measurement["interval"][1] < oc.NO_CONNECTION
    assert "чаще проигрывает" in out.note


def test_a_coin_toss_is_rejected_not_left_open():
    half = MIN_PAIRED_TRIALS + 20
    out = [f for f in oc.look(_run(half, half // 2), record=False)
           if f.approach["property"] == "captures"
           and f.approach["window"] == oc.WHOLE][0]
    assert out.verdict == REJECTED and "не связано" in out.note


def test_too_few_games_is_not_evaluated():
    out = oc.look(_run(5, 5), record=False)
    assert {f.verdict for f in out} == {NOT_EVALUATED}
    assert "Не измерено — это не ноль" in out[0].note


# --------------------------------------------------------------------------
# what would make a result meaningless
# --------------------------------------------------------------------------

def test_a_draw_answers_neither_way_and_is_absent():
    games = _run(MIN_PAIRED_TRIALS, MIN_PAIRED_TRIALS)
    games += [_sides(99, {"captures": 2.0}, {"captures": 1.0}, we_won=None)]
    rows = oc.pairs(games, _prop("captures"))
    assert len(rows) == MIN_PAIRED_TRIALS
    assert "g99" not in [game for game, _ in rows]


def test_a_tie_on_the_property_answers_neither_way():
    games = _run(MIN_PAIRED_TRIALS, MIN_PAIRED_TRIALS)
    games += [_sides(98, {"captures": 1.0}, {"captures": 1.0}, we_won=True)]
    assert len(oc.pairs(games, _prop("captures"))) == MIN_PAIRED_TRIALS


def test_a_property_undefined_for_a_side_is_absent_not_a_half():
    games = _run(MIN_PAIRED_TRIALS, MIN_PAIRED_TRIALS)
    games += [_sides(97, {"captures": None}, {"captures": 1.0}, we_won=True)]
    assert len(oc.pairs(games, _prop("captures"))) == MIN_PAIRED_TRIALS


def test_a_search_property_is_not_measured_where_only_one_side_has_it():
    """On Lichess the opponent has no trace, so "the side that considered
    more moves" would be "the side that was MANA" -- MANA's win rate
    wearing a property's name."""
    untraced = [_sides(i, {"considered": 30.0}, {"considered": 0.0},
                       we_won=True, source="lichess", traced=False)
                for i in range(MIN_PAIRED_TRIALS + 10)]
    rows = oc.pairs(untraced, _prop("considered"))
    assert rows == []
    out = [f for f in oc.look(untraced, record=False)
           if f.approach["property"] == "considered"][0]
    assert out.verdict == NOT_EVALUATED
    assert out.approach["needs_trace"] is True


def test_a_search_property_is_measured_where_both_sides_have_one():
    traced = [_sides(i, {"considered": 30.0}, {"considered": 10.0},
                     we_won=(i % 4 != 0), traced=True)
              for i in range(MIN_PAIRED_TRIALS + 20)]
    assert len(oc.pairs(traced, _prop("considered"))) == MIN_PAIRED_TRIALS + 20


def _prop(name):
    return next(p for p in oc.PROPERTIES if p.name == name)


# --------------------------------------------------------------------------
# the two windows
# --------------------------------------------------------------------------

def test_every_property_is_asked_in_both_windows():
    """Measured over a whole game, "the side with more material won" is
    close to a restatement of winning. The early window is where the
    property still precedes the result."""
    out = oc.look(_run(5, 5), record=False)
    assert len(out) == len(oc.PROPERTIES) * len(oc.WINDOWS)
    for prop in oc.PROPERTIES:
        windows = {f.approach["window"] for f in out
                   if f.approach["property"] == prop.name}
        assert windows == set(oc.WINDOWS)


def test_the_early_window_is_the_first_half_of_a_sides_own_moves():
    from mana.cognition import chess_features as feat

    rows = [feat.Move("g", ply, legal_moves=ply) for ply in range(1, 11)]
    assert [r.legal_moves for r in oc._window(rows, oc.EARLY)] == [1, 2, 3, 4, 5]
    assert len(oc._window(rows, oc.WHOLE)) == 10
    assert len(oc._window(rows[:1], oc.EARLY)) == 1        # never empty


def test_the_correction_counts_properties_times_windows():
    measurement = oc.measure(_run(MIN_PAIRED_TRIALS + 5, 20),
                             _prop("captures"))
    assert measurement["questions_asked"] == len(oc.PROPERTIES) * len(oc.WINDOWS)
    assert measurement["alpha"] < oc.ALPHA


# --------------------------------------------------------------------------
# what it says, and what it refuses to say
# --------------------------------------------------------------------------

def test_nothing_here_is_called_a_law():
    """A law is earned by changing something and measuring the change.
    Calling a correlation one would be the most expensive lie this system
    could tell itself."""
    import inspect

    source = inspect.getsource(oc)
    for claimed in ("Law", "LAW", "закон "):
        body = source.split('"""', 2)[-1]        # past the module docstring
        assert claimed not in body


def test_only_what_moved_is_reported():
    """A verdict repeated after every game is noise; a verdict that
    changed is the only thing worth interrupting for."""
    out = oc.look(_run(MIN_PAIRED_TRIALS + 20, MIN_PAIRED_TRIALS + 18),
                  record=False)
    first = oc.changes({}, out)
    assert first and all("новое" in line for line in first)
    assert oc.changes(oc.state(out), out) == []


def test_a_changed_verdict_says_what_it_was():
    weak = oc.look(_run(MIN_PAIRED_TRIALS + 20, (MIN_PAIRED_TRIALS + 20) // 2),
                   record=False)
    strong = oc.look(_run(MIN_PAIRED_TRIALS + 20, MIN_PAIRED_TRIALS + 19),
                     record=False)
    moved = [line for line in oc.changes(oc.state(weak), strong)
             if "captures" in line]
    assert moved and f"{REJECTED} → {ACCEPTED}" in moved[0]


def test_the_worlds_are_never_pooled():
    games = (_run(MIN_PAIRED_TRIALS + 5, MIN_PAIRED_TRIALS + 4)
             + [_sides(100 + i, {"captures": 2.0}, {"captures": 1.0},
                       we_won=False, source="lichess")
                for i in range(MIN_PAIRED_TRIALS + 5)])
    out = [f for f in oc.look(games, record=False)
           if f.approach["property"] == "captures"
           and f.approach["window"] == oc.WHOLE]
    assert len(out) == 2
    assert {f.conditions["worlds"][0] for f in out} == {"local", "lichess"}
    assert out[0].finding_id != out[1].finding_id


def test_the_answers_reach_the_ledger_and_are_found_again(tmp_path):
    from mana.cognition import findings as ledger_mod

    book = ledger_mod.Ledger(path=tmp_path / "findings.jsonl")
    out = oc.look(_run(MIN_PAIRED_TRIALS + 20, MIN_PAIRED_TRIALS + 18),
                  ledger=book)
    assert len(book.findings()) == len(out)
    oc.look(_run(MIN_PAIRED_TRIALS + 20, MIN_PAIRED_TRIALS + 18), ledger=book)
    assert len(book.findings()) == len(out)          # a repeat is not a point
