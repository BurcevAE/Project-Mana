"""A causal verdict becomes a version of the player.

One test per clause of the contract, because the clauses only work
together: evidence without provenance is a change nobody can trace,
provenance without rollback is a change nobody can undo, and rollback
without corpus separation leaves a record that means nothing either way.
"""
from __future__ import annotations

import pytest

chess = pytest.importorskip("chess", reason="оракул не приобретён")

from mana.cognition import chess_action as act
from mana.cognition import chess_version as ver
from mana.core.gates import ACCEPTED, MIN_PAIRED_TRIALS, NOT_EVALUATED, REJECTED


def _evidence(prop="pawn_moves", direction=act.LESS, verdict=ACCEPTED,
              reach=0.6, trials=MIN_PAIRED_TRIALS + 5, **kw):
    row = {"change": act.Change(prop, direction, from_finding="causal-1"),
           "causal_finding": "causal-1",
           "observational_finding": "obs-1",
           "reach": reach, "trials": trials, "verdict": verdict,
           "effect": 0.68}
    row.update(kw)
    return row


# --------------------------------------------------------------------------
# 1. a change is data
# --------------------------------------------------------------------------

def test_an_adoption_is_data_and_nothing_edits_the_player():
    """A change that lives in source cannot be reverted by a running
    system and cannot be told from one somebody made by hand."""
    import inspect

    source = inspect.getsource(ver)
    for editing in ("write_text(\"\"\"", "chess_arena.py", "PIECE_VALUE ="):
        assert editing not in source
    ver.adopt(_evidence())
    assert ver.path().exists()
    assert ver.version() == 1


# --------------------------------------------------------------------------
# 2. evidence named in advance
# --------------------------------------------------------------------------

def test_a_rejected_verdict_is_never_adopted():
    with pytest.raises(ver.Refused) as raised:
        ver.adopt(_evidence(verdict=REJECTED))
    assert REJECTED in str(raised.value) and ACCEPTED in str(raised.value)
    assert ver.version() == 0


def test_an_inert_lever_is_never_adopted():
    """Its experiment was about the shuffle, not about the finding."""
    with pytest.raises(ver.Refused) as raised:
        ver.adopt(_evidence(reach=0.0))
    assert "не двигает" in str(raised.value)


def test_too_few_decided_games_is_never_adopted():
    with pytest.raises(ver.Refused) as raised:
        ver.adopt(_evidence(trials=MIN_PAIRED_TRIALS - 1))
    assert str(MIN_PAIRED_TRIALS) in str(raised.value)


def test_missing_provenance_is_refused_by_name():
    """"не хватает свидетельства" with nothing after it is a message
    nobody can act on."""
    row = _evidence()
    row["causal_finding"] = ""
    with pytest.raises(ver.Refused) as raised:
        ver.adopt(row)
    assert "causal_finding" in str(raised.value)


def test_the_same_change_is_not_adopted_twice():
    ver.adopt(_evidence())
    with pytest.raises(ver.Refused) as raised:
        ver.adopt(_evidence())
    assert "уже в силе" in str(raised.value)


def test_checking_says_nothing_when_the_contract_is_met():
    assert ver.check(_evidence()) == ""


# --------------------------------------------------------------------------
# 3. provenance travels
# --------------------------------------------------------------------------

def test_an_adoption_carries_both_halves_of_its_provenance():
    """The correlation that suggested the change and the duel that tested
    it. A change nobody can trace back is one somebody made up."""
    adopted = ver.adopt(_evidence())
    assert adopted.causal_finding == "causal-1"
    assert adopted.observational_finding == "obs-1"
    assert adopted.reach == 0.6 and adopted.trials >= MIN_PAIRED_TRIALS
    assert adopted.as_dict()["effect"] == 0.68


# --------------------------------------------------------------------------
# 4. rollback, and history is not erased
# --------------------------------------------------------------------------

def test_reverting_takes_the_change_out_of_force_and_keeps_the_record():
    """A history that erases its mistakes cannot be used to check whether
    they repeat."""
    adopted = ver.adopt(_evidence())
    assert ver.version() == 1
    ver.revert(adopted, "не воспроизвелось на новом опыте")
    assert ver.version() == 0
    assert len(ver.history()) == 1
    assert ver.history()[0].state == ver.REVERTED
    assert "не воспроизвелось" in ver.history()[0].note


def test_the_composition_names_itself_because_a_count_is_not_enough():
    """Reverting one change and adopting another leaves the count the
    same and the player different."""
    first = ver.adopt(_evidence("pawn_moves", act.LESS))
    one = ver.fingerprint()
    ver.revert(first, "проверка")
    ver.adopt(_evidence("king_moves", act.LESS))
    assert ver.version() == 1
    assert ver.fingerprint() != one


def test_a_fresh_installation_is_version_zero():
    assert ver.version() == 0 and ver.in_force() == []
    assert ver.fingerprint() == "v0-base"
    assert "изменений нет" in ver.describe()


# --------------------------------------------------------------------------
# 5. the corpus never mixes versions
# --------------------------------------------------------------------------

def test_findings_are_computed_inside_a_version_never_across():
    """A game is evidence about the player that played it. A finding
    computed across an adoption is about neither."""
    from mana.cognition import chess_bot, chess_outcome

    def side(number, version, won):
        return chess_outcome.Sides(
            game=f"g{number}", source=chess_bot.LOCAL, player_version=version,
            we_won=won, both_traced=True,
            ours={("captures", w): 2.0 for w in chess_outcome.WINDOWS},
            theirs={("captures", w): 1.0 for w in chess_outcome.WINDOWS})

    games = ([side(i, 0, True) for i in range(MIN_PAIRED_TRIALS + 5)]
             + [side(100 + i, 1, False) for i in range(MIN_PAIRED_TRIALS + 5)])
    out = [f for f in chess_outcome.look(games, record=False)
           if f.approach["property"] == "captures"
           and f.approach["window"] == chess_outcome.WHOLE]
    assert len(out) == 2                       # one per version, not pooled
    assert {f.conditions["player_versions"][0] for f in out} == {0, 1}
    assert out[0].finding_id != out[1].finding_id


def test_a_game_record_says_which_player_played_it():
    from mana.cognition import chess_bot

    row = chess_bot.Seat(game_id="g", player_version=2,
                         player_composition="v2-abcd").as_dict()
    assert row["player_version"] == 2 and row["player_composition"] == "v2-abcd"
    assert chess_bot.Seat(game_id="g").as_dict()["player_version"] == 0


# --------------------------------------------------------------------------
# 6. provisional until confirmed on new experience
# --------------------------------------------------------------------------

def test_an_adoption_starts_provisional():
    """The duel that justified it ran before the change existed, and a
    result that holds only on the evidence which suggested it is the
    oldest failure this project guards against."""
    assert ver.adopt(_evidence()).state == ver.PROVISIONAL


def test_confirming_needs_a_full_re_test_on_new_games():
    adopted = ver.adopt(_evidence())
    with pytest.raises(ver.Refused) as raised:
        ver.confirm(adopted, trials=4, effect=0.7)
    assert str(ver.CONFIRM_GAMES) in str(raised.value)
    assert ver.in_force()[0].state == ver.PROVISIONAL

    ver.confirm(adopted, trials=ver.CONFIRM_GAMES, effect=0.68,
                finding_id="fresh-1")
    assert ver.in_force()[0].state == ver.CONFIRMED
    assert "fresh-1" in ver.in_force()[0].note


# --------------------------------------------------------------------------
# the player it composes
# --------------------------------------------------------------------------

def test_the_player_is_the_base_plus_what_is_in_force():
    from mana.cognition.chess_arena import SearchPlayer

    base = SearchPlayer(depth=2, trace=True)
    assert ver.player(base) is base            # version zero changes nothing

    ver.adopt(_evidence("pawn_moves", act.LESS))
    tuned = ver.player(base)
    assert tuned is not base
    assert tuned.depth == base.depth           # one change, nothing else
    assert tuned.change.property == "pawn_moves"


def test_reverting_gives_back_exactly_what_was_there_before():
    from mana.cognition.chess_arena import SearchPlayer

    base = SearchPlayer(depth=2, trace=True)
    adopted = ver.adopt(_evidence())
    assert ver.player(base) is not base
    ver.revert(adopted, "откат")
    assert ver.player(base) is base


def test_nothing_calls_this_yet():
    """Wiring it into the bench makes the ladder measure a different
    player, which resets what the ladder has measured. A separate
    decision, and not this module's to take."""
    import subprocess
    import sys

    found = subprocess.run(
        [sys.executable, "-c",
         "import pathlib,sys;"
         "hits=[p for p in pathlib.Path('mana').rglob('*.py')"
         " if 'chess_version' in p.read_text(encoding='utf-8')"
         " and p.name not in ('chess_version.py','version.py')];"
         "print(len(hits))"],
        capture_output=True, text=True)
    assert found.stdout.strip() == "0", found.stdout
