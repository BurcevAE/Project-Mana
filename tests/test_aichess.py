"""AI Chess Arena, transport only.

The two things worth guarding are the same two the Lichess client needed
and one this world adds. The client must not grade a move, the key must
never travel back out, and this environment's house rule -- games end
after seventy moves on piece count -- must be attached to every record,
because a result from here is not comparable with a result from a world
that ends games the way chess does.
"""
from __future__ import annotations

import json

import pytest

from mana.net import aichess


class _Response:
    def __init__(self, status=200, payload=None, lines=()):
        self.status_code = status
        self._payload = payload if payload is not None else {"ok": True}
        self.text = json.dumps(self._payload)
        self._lines = list(lines)

    def json(self):
        return self._payload

    def iter_lines(self):
        return iter(self._lines)


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.responses.pop(0)


ACTIVE = {
    "game_id": "xK7m", "your_color": "white", "your_turn": True,
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "moves": [], "legal_moves": ["e2e4", "d2d4"], "opponent": "openai/gpt-4o",
    "white_time": 300.0, "black_time": 300.0, "status": "in_progress"}


# --------------------------------------------------------------------------
# the world describes, it does not grade
# --------------------------------------------------------------------------

def test_the_client_never_grades_a_move():
    """The same line `chess_judge` holds: MANA learns from what happened,
    not from what a stronger player thought of it."""
    import inspect

    source = inspect.getsource(aichess)
    for grading in ("centipawn", "cp_loss", "blunder"):
        assert grading not in source.lower()


def test_a_seat_is_what_the_server_said():
    seat = aichess.seat_of(ACTIVE)
    assert seat.game_id == "xK7m" and seat.us is True and seat.our_turn is True
    assert seat.legal_moves == ["e2e4", "d2d4"]
    assert seat.opponent == "openai/gpt-4o"
    assert seat.over is False


def test_a_finished_game_says_so():
    seat = aichess.seat_of(dict(ACTIVE, status="finished",
                                result_reason="checkmate"))
    assert seat.over is True and seat.result_reason == "checkmate"


def test_no_game_is_an_answer():
    assert aichess.seat_of({}) is None


# --------------------------------------------------------------------------
# the house rule travels
# --------------------------------------------------------------------------

def test_every_record_carries_the_environment_and_its_ending_rule():
    """Games here end after seventy moves on piece count. That is not a
    chess rule, so a result from here is not comparable with one from a
    world that ends games the way chess does."""
    row = aichess.seat_of(ACTIVE).as_dict()
    assert row["environment"] == aichess.ENVIRONMENT
    assert row["termination_rule"] == aichess.TERMINATION
    assert "70" in aichess.TERMINATION


# --------------------------------------------------------------------------
# the key
# --------------------------------------------------------------------------

def test_no_key_is_a_state_with_a_remedy(monkeypatch):
    monkeypatch.delenv(aichess.TOKEN_ENV, raising=False)
    monkeypatch.setattr(aichess, "token", lambda: "")
    state = aichess.Arena(bearer="").describe()
    assert state["token"] is False and aichess.TOKEN_ENV in state["note"]
    assert "--aichess ИМЯ" in state["note"]


def test_describing_never_reports_the_key():
    arena = aichess.Arena(bearer="secret-key-value",
                          session=_Session([_Response(payload={"in_queue": True,
                                                               "active_game": None})]))
    assert "secret-key-value" not in json.dumps(arena.describe(),
                                                ensure_ascii=False)


def test_a_request_without_a_key_is_refused_before_the_network(monkeypatch):
    monkeypatch.setattr(aichess, "token", lambda: "")
    with pytest.raises(aichess.NoToken):
        aichess.Arena(bearer="").activity()


def test_registration_needs_no_key_because_it_makes_one():
    session = _Session([_Response(payload={"id": "a", "token": "tok_x"})])
    out = aichess.Arena(bearer="", session=session).register("MyBot")
    assert out["token"] == "tok_x"
    method, url, kwargs = session.calls[0]
    assert method == "POST" and url.endswith("/api/v1/players")
    assert kwargs["json"] == {"name": "MyBot", "type": "agent"}
    assert "Authorization" not in kwargs["headers"]


def test_a_key_that_cannot_be_sent_is_refused_before_it_is_stored():
    result = aichess.save_token("tok_кириллица")
    assert result["ok"] is False and "скопировалось" in result["error"]


# --------------------------------------------------------------------------
# the endpoints
# --------------------------------------------------------------------------

def test_an_unsupported_time_control_is_refused_here_not_there():
    arena = aichess.Arena(bearer="x", session=_Session([]))
    with pytest.raises(aichess.ArenaError):
        arena.join_queue(45)
    assert arena.session.calls == []


def test_a_named_model_can_be_asked_for_directly():
    """A hundred games against one model, then another, is how a
    principle is shown to survive a change of opponent."""
    session = _Session([_Response(payload={"game_id": "g1"})])
    aichess.Arena(bearer="x", session=session).challenge_model(
        "openai/gpt-4o", time_control=300)
    method, url, kwargs = session.calls[0]
    assert url.endswith("/api/v1/games")
    assert kwargs["json"]["opponent_id"] == "openai/gpt-4o"


def test_a_rejected_key_says_so_plainly():
    arena = aichess.Arena(bearer="x", session=_Session([_Response(status=401)]))
    with pytest.raises(aichess.ArenaError) as raised:
        arena.activity()
    assert "401" in str(raised.value)


def test_only_one_request_goes_out_at_a_time():
    import inspect

    assert "_one_at_a_time" in inspect.getsource(aichess.Arena._request)


# --------------------------------------------------------------------------
# the stream
# --------------------------------------------------------------------------

def test_the_stream_reads_events_and_skips_frame_separators():
    lines = [b"event: state",
             b'data: {"game_id": "xK7m"}',
             b"",
             b"event: ping",
             b"data: {}",
             b"",
             b"event: game_over",
             b'data: {"result_reason": "checkmate"}']
    arena = aichess.Arena(bearer="x", session=_Session([_Response(lines=lines)]))
    got = list(arena.stream("xK7m"))
    assert [row["event"] for row in got] == ["state", "ping", "game_over"]
    assert got[0]["game_id"] == "xK7m"


def test_a_half_line_does_not_end_the_stream():
    lines = [b"data: {broken", b'data: {"ok": 1}']
    arena = aichess.Arena(bearer="x", session=_Session([_Response(lines=lines)]))
    assert [row["ok"] for row in arena.stream("g")] == [1]
