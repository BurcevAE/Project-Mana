"""The games become findings, or say why not.

Two things are checked here and they matter in opposite directions. The
path has to be able to reach ACCEPTED -- a reader that can only ever say
"not measured" is the same failure as no reader at all, dressed as
caution. And it has to refuse when the evidence is thin, on the record
that actually exists rather than on a convenient one.

The rest is about the ways this kind of reader lies: counting moves as
independent observations, pooling two worlds, sorting by effect and
reporting the winner, and answering ten questions at one confidence.
"""
from __future__ import annotations

import pytest

from mana.cognition import chess_bot
from mana.cognition import chess_findings as reader
from mana.cognition import findings as ledger_mod
from mana.core.gates import ACCEPTED, MIN_PAIRED_TRIALS, NOT_EVALUATED, REJECTED


def _game(number, expensive, cheap, source=chess_bot.LIVE):
    """One game: `expensive` losses on captures, `cheap` on everything else."""
    thoughts, judged = [], []
    ply = 1
    for loss in expensive:
        thoughts.append({"ply": ply, "margin": 100.0, "close_call": False,
                         "forced": False, "considered": [["Nxe5", 100.0],
                                                         ["Nf3", 0.0]]})
        judged.append({"ply": ply, "move": "Nxe5", "loss": float(loss),
                       "mistake": loss >= 100, "blunder": loss >= 300,
                       "judged_by": "stockfish"})
        ply += 2
    for loss in cheap:
        thoughts.append({"ply": ply, "margin": 0.0, "close_call": True,
                         "forced": False, "considered": [["Nf3", 0.0],
                                                         ["Nc3", 0.0]]})
        judged.append({"ply": ply, "move": "Nf3", "loss": float(loss),
                       "mistake": loss >= 100, "blunder": loss >= 300,
                       "judged_by": "stockfish"})
        ply += 2
    return {"game": f"g{number}", "source": source, "us": "white",
            "status": "draw", "winner": "", "initial_fen": "startpos",
            "moves": [], "thoughts": thoughts, "judged": judged}


def _many(count, expensive=(600, 700), cheap=(0, 10), source=chess_bot.LIVE):
    """Games that differ from each other, because identical ones have no
    spread and a zero-width interval proves nothing about the interval."""
    out = []
    for i in range(count):
        wobble = (i % 7) * 20 - 60
        out.append(_game(i, tuple(max(0.0, v + wobble) for v in expensive),
                         tuple(max(0.0, v + wobble / 2) for v in cheap), source))
    return out


# --------------------------------------------------------------------------
# it can reach an answer
# --------------------------------------------------------------------------

def test_a_real_difference_over_enough_games_is_accepted():
    """A reader that can only ever say "not measured" is no reader at
    all, wearing caution as a costume."""
    out = reader.look(games=_many(MIN_PAIRED_TRIALS + 2), record=False)
    captures = [f for f in out if f.approach["partition"] == "capture"]
    assert len(captures) == 1
    finding = captures[0]
    assert finding.verdict == ACCEPTED
    assert finding.measurement["interval"][1] < 0     # worse than the rest
    assert finding.measurement["trials"] == MIN_PAIRED_TRIALS + 2
    assert finding.failure.failure == ledger_mod.WORSE


def test_no_difference_over_enough_games_is_rejected():
    """"We measured it and it is not so" closes a question. "We could not
    measure it" does not, and the two must not come back the same."""
    out = reader.look(games=_many(MIN_PAIRED_TRIALS + 2,
                                  expensive=(100, 100), cheap=(100, 100)),
                      record=False)
    captures = [f for f in out if f.approach["partition"] == "capture"][0]
    assert captures.verdict == REJECTED
    low, high = captures.measurement["interval"]
    assert low <= 0 <= high


def test_too_few_games_is_not_evaluated_however_many_moves_there_are():
    """Eighty-five moves from three games are three observations. They
    share an opponent, an opening and every earlier mistake."""
    out = reader.look(games=_many(3, expensive=(900,) * 20, cheap=(0,) * 20),
                      record=False)
    for finding in out:
        assert finding.verdict == NOT_EVALUATED
        assert finding.measurement["trials"] <= 3
        assert finding.failure.failure == ledger_mod.NOT_MEASURED
    assert "Не измерено — это не ноль" in out[0].note


# --------------------------------------------------------------------------
# the ways a reader like this lies
# --------------------------------------------------------------------------

def test_a_game_without_both_sides_contributes_nothing():
    """Absent, not zero. A game with no capture in it is not evidence
    that captures cost the same as everything else."""
    games = _many(4) + [_game(99, expensive=(), cheap=(0, 0, 0, 0))]
    rows = reader.pairs(games, dict((p.name, p) for p in reader.PARTITIONS)["capture"])
    assert len(rows) == 4
    assert "g99" not in [row[0] for row in rows]


def test_one_move_against_one_move_is_not_a_comparison():
    thin = [_game(0, expensive=(600,), cheap=(0,))]
    part = dict((p.name, p) for p in reader.PARTITIONS)["capture"]
    assert reader.pairs(thin, part) == []


def test_the_two_worlds_are_never_pooled():
    """A game against itself has an opponent that shares MANA's
    evaluation and its blind spots; a game against a stranger does not."""
    games = (_many(MIN_PAIRED_TRIALS + 1, source=chess_bot.LIVE)
             + _many(MIN_PAIRED_TRIALS + 1, source=chess_bot.LOCAL))
    out = reader.look(games=games, record=False)
    captures = [f for f in out if f.approach["partition"] == "capture"]
    assert len(captures) == 2                        # one per world
    worlds = [f.conditions["worlds"] for f in captures]
    assert worlds == [[chess_bot.LIVE], [chess_bot.LOCAL]]   # sorted by name
    for finding in captures:
        assert finding.measurement["trials"] == MIN_PAIRED_TRIALS + 1
    # And they are separate rows in the ledger, not one overwriting the other.
    assert captures[0].finding_id != captures[1].finding_id


def test_every_declared_class_is_answered_every_time():
    """Measuring ten and reporting the one that came out is how a
    coincidence becomes a finding."""
    out = reader.look(games=_many(5), record=False)
    assert [f.approach["partition"] for f in out] == [p.name for p in reader.PARTITIONS]


def test_the_interval_is_widened_for_the_number_of_questions():
    """Ten questions at one confidence produce half a false answer each
    pass, and the correction belongs where the number is known."""
    part = dict((p.name, p) for p in reader.PARTITIONS)["capture"]
    games = _many(MIN_PAIRED_TRIALS + 2)
    alone = reader.measure(games, part, questions=1)
    corrected = reader.measure(games, part, questions=len(reader.PARTITIONS))
    assert corrected["alpha"] < alone["alpha"]
    width = lambda m: m["interval"][1] - m["interval"][0]
    assert width(corrected) > width(alone)
    assert corrected["questions_asked"] == len(reader.PARTITIONS)


def test_a_class_taken_out_of_the_data_says_so():
    """`capture_moderate` came from looking at three real games. A class
    mined from the data it is measured on is guaranteed to look real."""
    mined = [p for p in reader.PARTITIONS if p.mined]
    assert [p.name for p in mined] == ["capture_moderate"]
    out = reader.look(games=_many(3), record=False)
    row = [f for f in out if f.approach["partition"] == "capture_moderate"][0]
    assert row.approach["mined_from_data"] is True
    assert "выведен из данных" in reader.describe(out)


def test_the_judge_travels_with_the_finding():
    """A loss judged by the material fallback and one judged by Stockfish
    are not the same quantity, and three real games were judged by the
    wrong one without anything noticing."""
    games = _many(3)
    for game in games:
        for row in game["judged"]:
            row["judged_by"] = "material"
    out = reader.look(games=games, record=False)
    assert out[0].conditions["judged_by"] == ["material"]
    assert out[0].conditions["games"] == 3
    assert out[0].conditions["questions_asked"] == len(reader.PARTITIONS)


def test_findings_are_returned_in_the_order_asked_not_by_effect():
    """Sorting by effect is the reading habit that turns a table of
    measurements into a headline."""
    out = reader.look(games=_many(MIN_PAIRED_TRIALS + 2), record=False)
    names = [f.approach["partition"] for f in out]
    assert names.index("forced") < names.index("capture")


# --------------------------------------------------------------------------
# it reaches the ledger a machine reads
# --------------------------------------------------------------------------

def test_the_answers_land_in_the_ledger_and_are_found_again(tmp_path):
    """The whole point. Until now no chess result reached a store any
    machine could read, which is this project's oldest failure."""
    book = ledger_mod.Ledger(path=tmp_path / "findings.jsonl")
    out = reader.look(games=_many(MIN_PAIRED_TRIALS + 2), ledger=book)
    assert len(book.findings()) == len(out)

    part = [p for p in reader.PARTITIONS if p.name == "capture"][0]
    known = book.already_tried(
        reader.QUESTION,
        {"partition": part.name, "what": part.what,
         "mined_from_data": part.mined},
        reader.conditions(_many(MIN_PAIRED_TRIALS + 2)))
    assert known is not None and known["match"] == "exact"
    assert known["finding"]["verdict"] == ACCEPTED


def test_the_same_class_under_new_conditions_is_a_new_point(tmp_path):
    """A result measured on thirty-two games says something about
    thirty-two games. More games is a new point in a series, not an
    overwrite."""
    book = ledger_mod.Ledger(path=tmp_path / "findings.jsonl")
    reader.look(games=_many(MIN_PAIRED_TRIALS + 2), ledger=book)
    reader.look(games=_many(MIN_PAIRED_TRIALS + 9), ledger=book)
    captures = [f for f in book.findings()
                if f.approach["partition"] == "capture"]
    assert len({f.finding_id for f in captures}) == 2
    assert len({f.approach_id for f in captures}) == 1     # one series


def test_the_same_games_measured_twice_are_not_two_observations(tmp_path):
    """`already_tried` is asked before writing, not after. Repeating the
    statistics command appended nine identical rows a time, which is how
    a bounded ledger loses real history to repeats."""
    book = ledger_mod.Ledger(path=tmp_path / "findings.jsonl")
    games = _many(MIN_PAIRED_TRIALS + 2)
    reader.look(games=games, ledger=book)
    first = len(book.findings())
    reader.look(games=games, ledger=book)
    reader.look(games=games, ledger=book)
    assert len(book.findings()) == first

    # But more games is a new point, and still lands.
    reader.look(games=_many(MIN_PAIRED_TRIALS + 9), ledger=book)
    assert len(book.findings()) > first


def test_the_same_game_twice_is_one_observation():
    """A repeated game is never evidence: this reader counts games as
    independent observations. A self-play seed that did not vary wrote
    three byte-identical games into the record in three cooldowns, and
    three copies of one game would move a NOT_EVALUATED towards a verdict
    on nothing."""
    one = _game(1, (600, 700), (0, 10))
    one["moves"] = ["e2e4", "e7e5", "g1f3"]
    twin = dict(one, game="different-id")
    other = dict(one, game="other", moves=["d2d4", "d7d5"])

    kept, dropped = reader.without_repeats([one, twin, other])
    assert dropped == 1
    assert [row["game"] for row in kept] == ["g1", "other"]


def test_a_repeat_cannot_inflate_a_measurement():
    games = _many(MIN_PAIRED_TRIALS + 2)
    for i, game in enumerate(games):
        game["moves"] = ["e2e4", "e7e5"] if i % 2 else ["d2d4", "d7d5"]
    out = reader.look(games=games, record=False)
    # Thirty-two games, two distinct move sequences: two observations.
    assert out[0].measurement["trials"] <= 2
    assert out[0].verdict == NOT_EVALUATED


def test_games_without_moves_are_not_treated_as_copies_of_each_other():
    """The synthetic records in these tests carry no move list, and
    collapsing them all into one would silently empty every measurement."""
    kept, dropped = reader.without_repeats(_many(5))
    assert dropped == 0 and len(kept) == 5
