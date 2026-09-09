"""The ladder: play Stockfish until it stops you.

The rule is the thing under test, and it is tested without a network. A
bench whose promotion rule can only be checked by playing on Lichess is
one whose rule is checked by nobody.

    10 побед подряд      → следующий уровень
    30 побед подряд на 8 → конец
    ничья или поражение  → серия с нуля

The draw is the clause most likely to be argued with, so it has its own
test: a streak of consecutive wins is what was asked for, and treating a
draw as neutral would let a ladder climb without ever winning ten of
anything.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

chess = pytest.importorskip("chess", reason="оракул не приобретён")

from mana.cognition import chess_bench as bench_mod
from mana.cognition import chess_bot
from mana.net import lichess


def _ladder(**kw):
    return bench_mod.Ladder(**kw)


def _win(ladder, times=1):
    out = None
    for _ in range(times):
        out = ladder.note(True)
    return out


# --------------------------------------------------------------------------
# the promotion rule
# --------------------------------------------------------------------------

def test_ten_in_a_row_promotes():
    ladder = _ladder()
    assert _win(ladder, bench_mod.WINS_TO_ADVANCE - 1) == bench_mod.CLIMBING
    assert ladder.level == 1 and ladder.streak == 9
    assert _win(ladder) == bench_mod.ADVANCED
    assert ladder.level == 2 and ladder.streak == 0


def test_a_loss_resets_the_streak_and_keeps_the_level():
    ladder = _ladder()
    _win(ladder, 9)
    assert ladder.note(False) == bench_mod.BROKEN
    assert ladder.streak == 0 and ladder.level == 1


def test_a_draw_is_not_a_win():
    """The clause most likely to be argued with. Counting a draw as
    neutral would let a ladder climb without winning ten of anything."""
    ladder = _ladder()
    _win(ladder, 9)
    assert ladder.note(False, drawn=True) == bench_mod.BROKEN
    assert ladder.streak == 0


def test_the_top_level_needs_thirty_and_then_ends():
    ladder = _ladder(level=bench_mod.LAST_LEVEL)
    assert ladder.needed() == bench_mod.WINS_TO_FINISH
    assert _win(ladder, bench_mod.WINS_TO_FINISH - 1) == bench_mod.CLIMBING
    assert ladder.done is False
    assert _win(ladder) == bench_mod.FINISHED
    assert ladder.done is True
    assert ladder.level == bench_mod.LAST_LEVEL       # nothing above it


def test_the_ladder_never_climbs_past_the_top():
    ladder = _ladder(level=bench_mod.LAST_LEVEL)
    _win(ladder, bench_mod.WINS_TO_FINISH)
    assert ladder.level == bench_mod.LAST_LEVEL


def test_every_level_keeps_its_own_denominator():
    """The stall level is the number this bench exists to produce, and it
    is unreadable without how many games were played to reach it."""
    ladder = _ladder()
    _win(ladder, bench_mod.WINS_TO_ADVANCE)           # clears level 1
    ladder.note(False)
    ladder.note(False)
    assert ladder.by_level["1"] == {"games": 10, "wins": 10}
    assert ladder.by_level["2"] == {"games": 2, "wins": 0}
    assert ladder.best_streak["1"] == 10
    assert ladder.games == 12 and ladder.wins == 10
    text = bench_mod.summarise(ladder)
    assert "уровень 1: 10 из 10" in text and "уровень 2: 0 из 2" in text


def test_the_summary_says_where_it_stopped():
    ladder = _ladder()
    _win(ladder, 3)
    ladder.note(False)
    assert "остановилась: 1" in bench_mod.summarise(ladder)
    assert "лучшая серия 3 из нужных 10" in bench_mod.summarise(ladder)


# --------------------------------------------------------------------------
# stopping and resuming
# --------------------------------------------------------------------------

def test_the_ladder_survives_an_interruption(tmp_path, monkeypatch):
    """A bench interrupted at level 3 and restarted at level 1 would
    spend its first hour re-measuring what it already measured, and those
    numbers would be indistinguishable from new ones."""
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")
    ladder = _ladder()
    _win(ladder, bench_mod.WINS_TO_ADVANCE)
    _win(ladder, 4)
    ladder.save()

    back = bench_mod.Ladder.load()
    assert back.level == 2 and back.streak == 4
    assert back.by_level["1"]["wins"] == 10


def test_a_damaged_state_file_starts_over_rather_than_crashing(tmp_path, monkeypatch):
    path = tmp_path / "bench.json"
    path.write_text("{не json", encoding="utf-8")
    monkeypatch.setattr(bench_mod, "state_path", lambda: path)
    assert bench_mod.Ladder.load().level == bench_mod.FIRST_LEVEL


def test_no_state_file_is_a_fresh_ladder(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "none.json")
    fresh = bench_mod.Ladder.load()
    assert fresh.level == bench_mod.FIRST_LEVEL and fresh.games == 0


# --------------------------------------------------------------------------
# the loop, with the network replaced
# --------------------------------------------------------------------------

class _Bot:
    """A bot that finishes a game whenever the bench asks for one."""

    def __init__(self, results):
        self.results = list(results)
        self.finished = []
        self.seats = {}
        self.judge_depth = 0
        self.stopped = False

    def run(self, games=0):
        while not self.stopped:
            time.sleep(0.01)

    def stop(self):
        self.stopped = True

    def deliver(self):
        if not self.results:
            self.stopped = True
            return
        winner = self.results.pop(0)
        seat = chess_bot.Seat(game_id=f"g{len(self.finished)}", us=True,
                              source=chess_bot.LIVE, level=1)
        seat.status = "mate" if winner else "draw"
        seat.winner = "white" if winner is True else ("black" if winner is False else "")
        self.finished.append(seat)


class _Client:
    def __init__(self, bot):
        self.bot = bot
        self.levels = []

    def challenge_ai(self, level=1, **kw):
        self.levels.append(level)
        threading.Timer(0.02, self.bot.deliver).start()
        return {"id": "x"}


def _bench(results, tmp_path, monkeypatch):
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")
    bot = _Bot(results)
    return bench_mod.Bench(client=_Client(bot), bot=bot, ladder=_ladder(), gap=0.0)


def test_the_bench_challenges_the_level_it_is_on(tmp_path, monkeypatch):
    bench = _bench([True] * (bench_mod.WINS_TO_ADVANCE + 1), tmp_path, monkeypatch)
    bench.run(limit=bench_mod.WINS_TO_ADVANCE + 1)
    assert bench.client.levels[:10] == [1] * 10        # ten at level one
    assert bench.client.levels[10] == 2                # then level two
    assert bench.ladder.level == 2


def test_a_defeat_keeps_it_where_it_is(tmp_path, monkeypatch):
    bench = _bench([True, True, False, True], tmp_path, monkeypatch)
    bench.run(limit=4)
    assert bench.client.levels == [1, 1, 1, 1]
    assert bench.ladder.level == 1 and bench.ladder.streak == 1


def test_stopping_ends_the_run_and_keeps_the_state(tmp_path, monkeypatch):
    """Ctrl+C is one of the two ways this ends, and the run it stops has
    to leave a readable state behind."""
    bench = _bench([True] * 40, tmp_path, monkeypatch)
    threading.Timer(0.15, bench.stop).start()
    ladder = bench.run()
    assert ladder.games >= 1
    assert bench_mod.Ladder.load().games == ladder.games


def test_a_refused_challenge_does_not_break_the_ladder(tmp_path, monkeypatch):
    """Lichess refusing is a normal outcome with a message."""
    bench = _bench([True], tmp_path, monkeypatch)

    def refuse(level=1, **kw):
        raise lichess.LichessError("429")

    bench.client.challenge_ai = refuse
    # `_sleep` waits on the stop event, not on time.sleep -- patching the
    # module clock left this test waiting the real backoff, three minutes
    # of a suite that has to stay fast enough to be run.
    monkeypatch.setattr(bench, "_sleep", lambda seconds: None)
    bench.run(limit=2)
    assert bench.ladder.games == 0 and bench.ladder.level == 1


def test_nothing_about_the_player_changes_between_games():
    """The ladder is a measuring instrument. Changing the thing being
    measured while it measures is how a bench stops being one."""
    import inspect

    source = inspect.getsource(bench_mod)
    for adapting in ("adopt(", "install(", "PLAY_DEPTH =", "depth="):
        assert adapting not in source.split("class Bench")[1]


# --------------------------------------------------------------------------
# what the rate limit exposed
# --------------------------------------------------------------------------

def test_a_game_without_a_result_is_not_a_draw(tmp_path, monkeypatch):
    """It is no evidence at all. Folding one in as a draw broke a real
    streak -- one such game is in the record, status "started" after 19
    moves. "Unmeasured is not zero" is the rule everywhere else here."""
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")
    bench = bench_mod.Bench(client=object(), bot=_Bot([]), ladder=_ladder())
    _win(bench.ladder, 4)

    unfinished = chess_bot.Seat(game_id="g", us=True, source=chess_bot.LIVE)
    unfinished.status = "started"
    unfinished.winner = ""
    bench._fold(unfinished)
    assert bench.ladder.streak == 4          # untouched
    assert bench.ladder.games == 4           # and not counted

    real_draw = chess_bot.Seat(game_id="g2", us=True, source=chess_bot.LIVE)
    real_draw.status = "draw"
    real_draw.winner = ""
    bench._fold(real_draw)
    assert bench.ladder.streak == 0 and bench.ladder.games == 5


def test_the_wait_after_a_refusal_grows(tmp_path, monkeypatch):
    """Ten seconds and two requests per attempt is answering rate
    limiting by producing more of what caused it."""
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")
    bot = _Bot([])
    bench = bench_mod.Bench(client=_Client(bot), bot=bot, ladder=_ladder(), gap=0.0)

    def refuse(level=1, **kw):
        raise lichess.LichessError("429")

    bench.client.challenge_ai = refuse
    waited = []
    monkeypatch.setattr(bench, "_sleep", lambda s: waited.append(s))
    for _ in range(4):
        bench._one_game()
    assert waited == [60.0, 120.0, 240.0, 480.0]
    assert max(waited) <= bench_mod.REFUSED_WAIT_MAX


def test_the_wait_is_capped_and_a_success_clears_it(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")
    bot = _Bot([True])
    bench = bench_mod.Bench(client=_Client(bot), bot=bot, ladder=_ladder(), gap=0.0)
    bench._refused = 20
    monkeypatch.setattr(bench, "_sleep", lambda s: None)

    def refuse(level=1, **kw):
        raise lichess.LichessError("429")

    real = bench.client.challenge_ai
    bench.client.challenge_ai = refuse
    waited = []
    monkeypatch.setattr(bench, "_sleep", lambda s: waited.append(s))
    bench._one_game()
    assert waited == [bench_mod.REFUSED_WAIT_MAX]

    bench.client.challenge_ai = real
    bench._one_game()
    assert bench._refused == 0               # a game started; the wait resets


def test_a_long_backoff_still_answers_ctrl_c(tmp_path, monkeypatch):
    """A fifteen-minute wait that ignores a stop is a bench nobody can
    stop, which is one of the two ways this was asked to end."""
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")
    bench = bench_mod.Bench(client=object(), bot=_Bot([]), ladder=_ladder())
    bench.stop()
    started = time.time()
    bench._sleep(30.0)
    assert time.time() - started < 1.0


def test_asking_for_the_console_twice_does_not_double_the_output():
    """One event became two lines, which reads as two processes running
    rather than as one sink registered twice."""
    from mana import events

    events.install_console_sink()
    events.install_console_sink()
    events.install_console_sink()
    assert sum(1 for sink in events.BUS.sinks()
               if sink is events.console_sink) == 1
