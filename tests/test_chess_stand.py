"""
The chess stand, reachable from the window: «Мана, поиграй в шахматы на
стенде» starts the ladder in the background, «останови стенд» stops it, and
the reply says what actually happened. No network: Lichess and the bench
are replaced; the ladder rule itself is `chess_bench`'s, tested there.
"""
from __future__ import annotations

import threading

import pytest

from mana.apps import chess_stand
from mana.apps import intent as app_intent
from mana.cognition import chess_bench, chess_watch
from mana.net import lichess


@pytest.mark.parametrize("text", ["Мана, поиграй в шахматы на стенде",
                                  "поиграй в шахматы на стенде",
                                  "Мана запусти шахматный стенд"])
def test_the_sentence_starts_the_stand(text):
    found = app_intent.match(text)
    assert found is not None and found.action == "start_chess_stand"
    assert found.tool == "chess_stand_start" and found.params == {"from_level": 1}


@pytest.mark.parametrize("text", ["останови стенд", "Мана, останови стенд",
                                  "стоп стенд", "заверши стенд"])
def test_the_sentence_stops_the_stand(text):
    found = app_intent.match(text)
    assert found is not None and found.action == "stop_chess_stand"


@pytest.mark.parametrize("text", ["как запустить шахматный стенд?",
                                  "не останавливай стенд",
                                  "поиграй в шахматы",
                                  "что такое стенд"])
def test_talking_about_the_stand_starts_nothing(text):
    assert app_intent.match(text) is None


class FakeBench:
    def __init__(self, client=None, ladder=None, **kwargs):
        self.ladder = ladder
        self.stopped = threading.Event()

    def run(self, limit=0):
        self.stopped.wait(10)
        return self.ladder

    def stop(self):
        self.stopped.set()


class FakeLichess:
    def __init__(self, bot=True):
        self.bot = bot

    def describe(self):
        return {"bot": self.bot, "token": True, "user": "mana-bot", "env": "test",
                "error": "" if self.bot else "аккаунт не BOT"}


@pytest.fixture
def quiet(monkeypatch, tmp_path):
    monkeypatch.setattr(chess_bench, "Bench", FakeBench)
    monkeypatch.setattr(chess_bench, "state_path", lambda: tmp_path / "bench.json")
    monkeypatch.setattr(chess_watch, "start", lambda port: (_ for _ in ()).throw(OSError("busy")))
    monkeypatch.setattr(lichess, "Lichess", lambda: FakeLichess(bot=True))
    yield
    chess_stand.stop()


def test_the_stand_runs_in_the_background_and_stops_on_request(quiet):
    started = chess_stand.start()
    assert started["ok"] and not started["already"] and started["level"] == 1
    assert chess_stand.running()
    again = chess_stand.start()
    assert again["ok"] and again["already"]
    stopped = chess_stand.stop()
    assert stopped["stopped"] and stopped["summary"]
    assert not chess_stand.running()
    assert chess_stand.stop()["stopped"] is False


def test_without_a_bot_account_it_refuses_with_the_reason(quiet, monkeypatch):
    monkeypatch.setattr(lichess, "Lichess", lambda: FakeLichess(bot=False))
    refused = chess_stand.start()
    assert not refused["ok"] and "не BOT" in refused["error"]
    assert not chess_stand.running()


def test_a_pause_lichess_imposed_is_kept_when_starting_from_level_one(quiet):
    saved = chess_bench.Ladder(level=4, next_request_at=12345.0, refused=3)
    saved.save()
    chess_stand.start()
    ladder = chess_stand._run.bench.ladder
    assert ladder.level == 1 and ladder.next_request_at == 12345.0 and ladder.refused == 3


def test_the_agent_answers_from_what_happened(isolated_agent, quiet):
    result = isolated_agent.solve_task("Мана, поиграй в шахматы на стенде")
    assert result["plan"]["kind"] == "app_action"
    assert result["answer"].startswith("Стенд запущен")
    assert "с уровня 1" in result["answer"] and "останови стенд" in result["answer"]
    stopped = isolated_agent.solve_task("останови стенд")
    assert stopped["answer"].startswith("Стенд остановлен")
