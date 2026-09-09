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
from pathlib import Path

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


_Client_base = _Client


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

    source = inspect.getsource(bench_mod).split("class Bench")[1]
    # Not a banned word -- an invariant. The bench may read the past with
    # an unmodified player (that is how reach is measured), and it may run
    # a modified one inside an experiment. What it may never do is put a
    # modified player into a game the ladder counts, or adopt anything.
    for adapting in ("adopt(", "install(", "policy.adopt", "Tuned("):
        assert adapting not in source
    # And the modified player is built only inside the duel -- twice
    # there, once per colour, so that the change is not itself a colour
    # advantage.
    from mana.cognition import chess_action

    whole = inspect.getsource(chess_action)
    inside = inspect.getsource(chess_action.duel)
    assert inside.count("Tuned(") == whole.count("Tuned(") == 2


def test_a_duel_game_can_never_be_counted_as_an_observation():
    """A duel is played by a modified player and the findings are about
    the unmodified one. Without this boundary a hundred experiments would
    turn the corpus into a blend of players that nothing afterwards could
    separate."""
    from mana.cognition import chess_outcome

    assert chess_bot.DUEL not in chess_bot.OBSERVATIONAL
    duel_game = chess_outcome.Sides(game="d1", source=chess_bot.DUEL,
                                    we_won=True)
    real = chess_outcome.Sides(game="g1", source=chess_bot.LOCAL, we_won=True)
    out = chess_outcome.look([duel_game, real], record=False)
    assert {f.conditions["worlds"][0] for f in out} == {chess_bot.LOCAL}


def test_the_ladder_plays_with_the_unmodified_player(tmp_path, monkeypatch):
    """The experiment lives in the cooldown; the games the ladder counts
    are played by the player the ladder is measuring."""
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")
    bot = _Bot([True])
    bench = bench_mod.Bench(client=_Client(bot), bot=bot, ladder=_ladder(),
                            gap=0.0)
    from mana.cognition import chess_action

    made = []
    monkeypatch.setattr(chess_action, "Tuned",
                        lambda player, change: made.append(change) or player)
    bench._one_game()
    assert made == []                       # no modified player in a real game


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
    """Not a sleep: a held state. Lichess warns that the limits are
    composite, so a second refusal after a full minute is not an error to
    retry through -- it is a longer wait."""
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")
    bot = _Bot([])
    bench = bench_mod.Bench(client=_Client(bot), bot=bot, ladder=_ladder(), gap=0.0)

    held = []
    bench.client.hold = lambda seconds: held.append(seconds) or seconds
    bench.client.cooldown_left = lambda: 0.0

    def refuse(level=1, **kw):
        raise lichess.RateLimited("429", 60.0)

    bench.client.challenge_ai = refuse
    for _ in range(4):
        bench._one_game()
    assert held == [60.0, 120.0, 240.0, 480.0]
    assert max(held) <= bench_mod.REFUSED_WAIT_MAX


def test_the_wait_is_capped_and_a_success_clears_it(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")
    bot = _Bot([True])
    bench = bench_mod.Bench(client=_Client(bot), bot=bot, ladder=_ladder(), gap=0.0)
    bench._refused = 20
    held = []
    bench.client.hold = lambda seconds: held.append(seconds) or seconds
    bench.client.cooldown_left = lambda: 0.0

    real = bench.client.challenge_ai

    def refuse(level=1, **kw):
        raise lichess.RateLimited("429", 60.0)

    bench.client.challenge_ai = refuse
    bench._one_game()
    assert held == [bench_mod.REFUSED_WAIT_MAX]

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


# --------------------------------------------------------------------------
# the cooldown is worked through, not slept through
# --------------------------------------------------------------------------

def test_a_cooldown_is_spent_on_the_record_not_on_sleeping(tmp_path, monkeypatch):
    """Lichess is a slow external sensor, not a lock on MANA."""
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")
    bot = _Bot([True])
    bench = bench_mod.Bench(client=_Client(bot), bot=bot, ladder=_ladder(), gap=0.0)
    bench.client.cooldown_left = lambda: 30.0
    bench.client.hold = lambda seconds: seconds

    # Counted, not timed: a wall-clock stop made this flake under load,
    # and what is being asserted is that the wait is worked through, not
    # how fast the machine is.
    thought = []

    def working(budget=0.0):
        thought.append(1)
        if len(thought) >= 3:
            bench.stop()
        return "думала"

    monkeypatch.setattr(bench, "think", working)
    monkeypatch.setattr(bench_mod, "TICK", 0.01)
    bench.run()
    assert len(thought) >= 3                  # it worked while it waited
    assert bench.client.levels == []           # and asked Lichess nothing


def test_the_remaining_time_is_reported_without_asking_lichess(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")
    bot = _Bot([])
    bench = bench_mod.Bench(client=_Client(bot), bot=bot, ladder=_ladder(), gap=0.0)
    monkeypatch.setattr(bench, "think", lambda budget=0.0: "читала записи")

    from mana import events

    said = []
    sink = events.subscribe(lambda e: said.append((e.text, e.data.get("chess"))))
    try:
        monkeypatch.setattr(bench_mod, "TICK", 0.01)
        bench._while_waiting(37.0)
    finally:
        events.unsubscribe(sink)
    text, payload = said[-1]
    assert "охлаждение" in text and "37" in text and "читала записи" in text
    assert payload["cooldown"] == 37.0 and payload["doing"] == "читала записи"


def test_thinking_never_touches_the_network(tmp_path, monkeypatch):
    """The cooldown is the one time nothing may be asked of Lichess, and
    exactly when there is most to do locally."""
    import inspect

    body = inspect.getsource(bench_mod.Bench.think)
    for reaching in ("self.client", "challenge", "requests", "api."):
        assert reaching not in body


def test_thinking_rejudges_the_games_the_fallback_judged(tmp_path, monkeypatch):
    """Games judged by the fallback are worth less than nothing until
    they are re-judged: their losses are on a different scale."""
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")
    bot = _Bot([])
    bench = bench_mod.Bench(client=_Client(bot), bot=bot, ladder=_ladder(), gap=0.0)

    from mana.cognition import chess_bot as bot_mod
    from mana.cognition import chess_judge

    calls = {}

    def fake_rejudge(depth=0, only="", limit=0, stale_only=True, on_game=None):
        calls.update({"depth": depth, "limit": limit, "stale_only": stale_only})
        return {"judged": 2, "games": 5}

    monkeypatch.setattr(bot_mod, "recorded", lambda: [
        {"game": "a", "judged": [{"judged_by": "material"}], "thoughts": []},
        {"game": "b", "judged": [{"judged_by": "stockfish"}], "thoughts": []}])
    monkeypatch.setattr(bot_mod, "rejudge", fake_rejudge)
    monkeypatch.setattr(chess_judge, "engine_path", lambda: Path("stockfish"))

    said = bench.think()
    assert "пересудила 2" in said
    assert calls["limit"] == bench_mod.REJUDGE_PER_TICK and calls["stale_only"] is True


def test_with_a_clean_record_it_reads_instead(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")
    bot = _Bot([])
    bench = bench_mod.Bench(client=_Client(bot), bot=bot, ladder=_ladder(), gap=0.0)

    from mana.cognition import chess_bot as bot_mod

    monkeypatch.setattr(bot_mod, "recorded", lambda: [
        {"game": "b", "source": "lichess", "judged": [{"judged_by": "stockfish"}],
         "thoughts": []}])
    said = bench.think()
    assert "прочитала 1" in said and "не хватает" in said


def test_an_empty_record_is_an_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")
    bot = _Bot([])
    bench = bench_mod.Bench(client=_Client(bot), bot=bot, ladder=_ladder(), gap=0.0)

    from mana.cognition import chess_bot as bot_mod

    monkeypatch.setattr(bot_mod, "recorded", lambda: [])
    assert "записей пока нет" in bench.think()


# --------------------------------------------------------------------------
# the wait outlives the process
# --------------------------------------------------------------------------

def test_a_cooldown_survives_a_restart(tmp_path, monkeypatch):
    """A cooldown that lives only in a process is not a cooldown.
    Restarting reset the timer and the streak, so every restart asked
    Lichess again at once and began the escalation afresh -- the very
    behaviour the escalation exists to prevent, performed by the person
    trying to avoid it."""
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")

    held = []

    class _Client(_Client_base):
        def hold(self, seconds):
            held.append(seconds)
            return seconds

        def cooldown_left(self):
            return 0.0

    bot = _Bot([])
    first = bench_mod.Bench(client=_Client(bot), bot=bot, ladder=_ladder(), gap=0.0)

    def refuse(level=1, **kw):
        raise lichess.RateLimited("429", 60.0)

    first.client.challenge_ai = refuse
    first._one_game()
    assert first.ladder.refused == 1
    assert first.ladder.cooldown_left() > 0

    # A new process reads the same file.
    again = bench_mod.Ladder.load()
    assert again.refused == 1
    assert again.cooldown_left() > 0

    bot2 = _Bot([])
    second = bench_mod.Bench(client=_Client(bot2), bot=bot2, ladder=again, gap=0.0)
    assert second._refused == 1                  # the streak continues
    assert held[-1] > 0                          # and the wait was re-applied

    second.client.challenge_ai = refuse
    second._one_game()
    assert second.ladder.refused == 2            # 60 -> 120, not 60 again


def test_a_started_game_clears_the_saved_wait(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")
    bot = _Bot([True])
    bench = bench_mod.Bench(client=_Client(bot), bot=bot, ladder=_ladder(), gap=0.0)
    bench.ladder.next_request_at = 9e9
    bench.ladder.refused = 4
    bench._one_game()
    assert bench.ladder.next_request_at == 0.0 and bench.ladder.refused == 0


def test_the_summary_says_when_lichess_may_be_asked_again():
    """There was no way to find out when it was safe to start, short of
    starting."""
    ladder = _ladder()
    ladder.next_request_at = time.time() + 90
    ladder.refused = 2
    text = bench_mod.summarise(ladder)
    assert "следующий запрос не раньше" in text and "отказов подряд 2" in text
    assert "перезапуск не сбрасывает" in text

    ladder.next_request_at = 0.0
    assert "можно запускать" in bench_mod.summarise(ladder)


# --------------------------------------------------------------------------
# the wait produces evidence
# --------------------------------------------------------------------------

def test_a_long_enough_wait_is_spent_playing(tmp_path, monkeypatch):
    """Re-reading an unchanged corpus is not work: the same games cannot
    say anything on the second pass. What is short is games."""
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")
    bot = _Bot([])
    bench = bench_mod.Bench(client=_Client(bot), bot=bot, ladder=_ladder(), gap=0.0)

    from mana.cognition import chess_bot as bot_mod

    asked = {}

    def fake_play(games=1, judge_depth=0, pause=0.0, quiet=False, stop=None, **kw):
        asked.update({"games": games, "quiet": quiet, "pause": pause})
        seat = chess_bot.Seat(game_id="local-x", source=chess_bot.LOCAL)
        seat.thoughts = [{"close_call": True}, {"close_call": False}]
        seat.judged = [{"loss": 0.0}, {"loss": 400.0, "blunder": True}]
        return [seat]

    monkeypatch.setattr(bot_mod, "recorded", lambda: [])
    monkeypatch.setattr(bot_mod, "play_locally", fake_play)
    said = bench.think(budget=60.0)
    assert "сыграла с собой" in said and "2 ходов" in said
    assert asked == {"games": 1, "quiet": True, "pause": 0.0}


def test_a_short_wait_does_not_start_a_game(tmp_path, monkeypatch):
    """Starting one with ten seconds to go would hold the next challenge
    back for work that had no deadline."""
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")
    bot = _Bot([])
    bench = bench_mod.Bench(client=_Client(bot), bot=bot, ladder=_ladder(), gap=0.0)

    from mana.cognition import chess_bot as bot_mod

    monkeypatch.setattr(bot_mod, "recorded", lambda: [])
    monkeypatch.setattr(bot_mod, "play_locally",
                        lambda **kw: pytest.fail("не должна была играть"))
    assert bench.think(budget=5.0) == "записей пока нет"


def test_the_same_answer_is_not_repeated_every_tick(tmp_path, monkeypatch):
    """Twelve identical lines is noise pretending to be progress."""
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")
    bot = _Bot([])
    bench = bench_mod.Bench(client=_Client(bot), bot=bot, ladder=_ladder(), gap=0.0)
    monkeypatch.setattr(bench, "think", lambda budget=0.0: "то же самое")
    monkeypatch.setattr(bench_mod, "TICK", 0.01)

    from mana import events

    spoken = []
    sink = events.subscribe(lambda e: spoken.append(e.text))
    try:
        bench._while_waiting(30.0)
        bench._while_waiting(25.0)
        bench._while_waiting(20.0)
    finally:
        events.unsubscribe(sink)
    assert len([text for text in spoken if text]) == 1
    assert len(spoken) == 3               # the window still gets every frame


def test_a_quiet_self_play_game_does_not_flood_the_console():
    """Eighty move lines per game would bury the one line that matters."""
    from mana import events

    text = []
    sink = events.subscribe(lambda e: text.append(e.text))
    try:
        chess_bot.play_locally(games=1, depth=1, judge_depth=0, pause=0.0,
                               record=False, quiet=True, max_plies=30)
    finally:
        events.unsubscribe(sink)
    spoken = [line for line in text if line.strip()]
    # Nothing at all: the judge is off on purpose, which is the normal
    # state now, and a quiet game says nothing to the console. Only the
    # fallback -- a judge asked for and quietly replaced -- still speaks.
    assert spoken == []
    assert len(text) > 10                      # the window still got the moves


def test_an_event_with_no_text_prints_nothing():
    from mana import events

    printed = []
    original = events.write_console
    events.write_console = lambda line: printed.append(line)
    try:
        events.console_sink(events.Event(kind=events.STATUS, text=""))
        events.console_sink(events.Event(kind=events.STATUS, text="есть что сказать"))
    finally:
        events.write_console = original
    assert printed == ["есть что сказать"]


def test_two_self_play_games_in_a_row_are_not_the_same_game():
    """The seed used to be the index inside the batch, so a batch of one
    produced game zero every time -- three cooldowns wrote three
    byte-identical games into a record that counts games."""
    first = chess_bot.play_locally(games=1, depth=1, judge_depth=0, pause=0.0,
                                   record=False, quiet=True, max_plies=30)[0]
    second = chess_bot.play_locally(games=1, depth=1, judge_depth=0, pause=0.0,
                                    record=False, quiet=True, max_plies=30)[0]
    assert first.seed != second.seed
    assert first.moves != second.moves


def test_a_recorded_seed_replays_the_same_game():
    """Varying the seed is only safe because the seed is written down: a
    game that cannot be reproduced is one whose bugs cannot be examined
    after the fact, which is how the repeated games were found at all."""
    first = chess_bot.play_locally(games=1, depth=1, judge_depth=0, pause=0.0,
                                   record=False, quiet=True, max_plies=30)[0]
    again = chess_bot.play_locally(games=1, depth=1, judge_depth=0, pause=0.0,
                                   record=False, quiet=True, seed=first.seed,
                                   max_plies=30)[0]
    assert again.moves == first.moves
    assert first.as_dict()["seed"] == first.seed


def test_an_unjudged_game_reports_no_count_rather_than_zero():
    """"зевков 0" for a game no judge looked at is unmeasured dressed as
    measured. It went out in a status line the moment the judge was
    switched off, which is how it was found."""
    seat = chess_bot.Seat(game_id="g", source=chess_bot.LOCAL)
    seat.thoughts = [{"close_call": True}, {"close_call": False}]
    seat.judged = []
    row = chess_bot.summarise(seat)
    assert row["blunders"] is None and row["mistakes"] is None
    assert row["mean_loss"] is None
    assert row["moves"] == 2 and row["close_share"] == 0.5

    seat.judged = [{"loss": 400.0, "mistake": True, "blunder": True},
                   {"loss": 0.0, "mistake": False, "blunder": False}]
    counted = chess_bot.summarise(seat)
    assert counted["blunders"] == 1 and counted["mean_loss"] == 200.0


def test_a_cooldown_tick_is_spent_not_slept_through(tmp_path, monkeypatch):
    """A self-play game costs half a second now that no engine is asked;
    waiting five seconds between them would spend a fifteen-minute
    cooldown on a fraction of the games it could hold."""
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")
    bot = _Bot([])
    bench = bench_mod.Bench(client=_Client(bot), bot=bot, ladder=_ladder(), gap=0.0)
    bench.client.cooldown_left = lambda: 300.0
    bench.client.hold = lambda seconds: seconds

    done = []
    monkeypatch.setattr(bench, "think",
                        lambda budget=0.0: done.append(1) or f"партия {len(done)}")
    monkeypatch.setattr(bench_mod, "TICK", 0.2)
    bench._while_waiting(300.0)
    assert len(done) > 1                    # more than one unit inside a tick


def test_a_cooldown_does_not_kill_the_run_it_exists_to_fill(tmp_path, monkeypatch):
    """`Bot.run()` opens with an account check, and a cooldown set by a
    refused challenge refuses that too. Treating it as fatal stopped the
    bench, which then could not even play by itself -- the network half
    silencing the half that needs no network."""
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")

    class _Refusing(_Bot):
        def run(self, games=0):
            raise lichess.RateLimited("охлаждение", 30.0)

    bot = _Refusing([])
    bench = bench_mod.Bench(client=_Client(bot), bot=bot, ladder=_ladder(),
                            gap=0.0)
    bench.client.cooldown_left = lambda: 0.0
    monkeypatch.setattr(bench_mod, "LISTEN_RETRY", 0.01)
    threading.Timer(0.3, bench.stop).start()
    bench._listen()
    assert bench._stop.is_set()               # only because we stopped it


def test_a_broken_stream_does_not_stop_the_local_half(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")

    class _Broken(_Bot):
        def run(self, games=0):
            raise RuntimeError("сеть отвалилась")

    bot = _Broken([])
    bench = bench_mod.Bench(client=_Client(bot), bot=bot, ladder=_ladder(),
                            gap=0.0)
    bench.client.cooldown_left = lambda: 0.0
    monkeypatch.setattr(bench_mod, "LISTEN_RETRY", 0.001)
    monkeypatch.setattr(bench_mod, "LISTEN_GIVE_UP", 3)
    bench._listen()
    # It gave up on the stream and did not stop the bench.
    assert not bench._stop.is_set()


def test_a_set_stop_is_why_self_play_returned_nothing(tmp_path, monkeypatch):
    """The failure that produced "прочитала" where it should have said
    "сыграла": play_locally checks the stop flag on its first line."""
    played = chess_bot.play_locally(games=1, depth=1, judge_depth=0,
                                    pause=0.0, record=False, quiet=True,
                                    stop=threading.Event(), max_plies=30)
    assert played                              # a clear flag plays

    stopped = threading.Event()
    stopped.set()
    assert chess_bot.play_locally(games=1, depth=1, judge_depth=0, pause=0.0,
                                  record=False, quiet=True, stop=stopped,
                                  max_plies=30) == []


def test_the_ladder_says_which_player_it_measures(tmp_path, monkeypatch):
    """Requirement eight: an instrument whose subject was swapped without
    a word is worse than one that stopped."""
    from mana.cognition import chess_version

    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")
    bot = _Bot([])
    bench = bench_mod.Bench(client=_Client(bot), bot=bot, ladder=_ladder(),
                            gap=0.0)
    _win(bench.ladder, 5)
    assert bench.ladder.measures == "v0-base"

    monkeypatch.setattr(chess_version, "confirmed_fingerprint",
                        lambda: "v1-abcd")
    said = []
    from mana import events

    sink = events.subscribe(lambda e: said.append(e.text))
    try:
        bench._check_subject()
    finally:
        events.unsubscribe(sink)
    assert bench.ladder.wins == 0 and bench.ladder.games == 0
    assert bench.ladder.measures == "v1-abcd"
    assert any("состав игрока сменился" in line for line in said)


def test_the_ladder_is_not_reset_while_the_baseline_holds(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_mod, "state_path", lambda: tmp_path / "bench.json")
    bot = _Bot([])
    bench = bench_mod.Bench(client=_Client(bot), bot=bot, ladder=_ladder(),
                            gap=0.0)
    _win(bench.ladder, 5)
    bench._check_subject()
    assert bench.ladder.wins == 5
