"""Playing in a world MANA did not make up.

The tests that matter most here are the two refusals. `chess_judge` is
kept from ever picking a move; this module is kept from ever upgrading the
account, because that call is irreversible and belongs to the person whose
account it is. Both are checked by reading the source, so they survive
edits by someone who does not know why the rule exists.

Everything else is protocol detail that has already gone wrong somewhere:
a stream reader that treats a keep-alive as an ending, a client that
answers rate limiting with more requests, a board held in this process
disagreeing with the board Lichess believes in.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest

chess = pytest.importorskip("chess", reason="оракул не приобретён")

from mana.cognition import chess_bot, chess_watch
from mana.net import lichess


# --------------------------------------------------------------------------
# the account is not ours to change
# --------------------------------------------------------------------------

def test_the_client_cannot_upgrade_the_account():
    """Irreversible, and about a person's account rather than about MANA."""
    import inspect

    source = inspect.getsource(lichess)
    # The path appears three times on purpose -- the constant, the guard
    # that refuses it, and the command a person runs themselves. What
    # must not exist is a line that both names it and asks for it.
    asking = ("_request(", ".post(", ".request(", "session.get(")
    for line in source.splitlines():
        if "upgrade" not in line.lower():
            continue
        assert not any(verb in line for verb in asking), line
    assert "raise LichessError" in source.split("if path == UPGRADE_PATH:")[1][:400]


def test_building_the_upgrade_path_by_hand_is_refused():
    client = lichess.Lichess(bearer="x")
    with pytest.raises(lichess.LichessError) as raised:
        client._url(lichess.UPGRADE_PATH)
    assert "необратим" in str(raised.value)
    assert "curl" in str(raised.value)          # the remedy travels with it


def test_the_upgrade_command_is_something_a_person_can_run():
    """Runnable in the shell they are actually in.

    `curl` in PowerShell is an alias for Invoke-WebRequest, which cannot
    take a string for -Headers: the plain form is a parse error, not a
    request. And the placeholder carries no angle brackets, because
    substituting into "Bearer <ВАШ_ТОКЕН>" tends to take the word Bearer
    with it and produce a 401 that looks like a bad token.
    """
    text = lichess.upgrade_command()
    assert "POST" in text and lichess.UPGRADE_PATH in text
    assert "curl.exe" in text
    assert "Bearer ВАШ_ТОКЕН" in text and "<" not in text


# --------------------------------------------------------------------------
# the token
# --------------------------------------------------------------------------

def test_the_environment_wins_over_the_stored_token(monkeypatch):
    monkeypatch.setenv(lichess.TOKEN_ENV, "from-env")
    assert lichess.token() == "from-env"


def test_no_token_is_a_state_with_a_remedy(monkeypatch):
    monkeypatch.delenv(lichess.TOKEN_ENV, raising=False)
    monkeypatch.setattr(lichess, "token", lambda: "")
    state = lichess.Lichess(bearer="").describe()
    assert state["token"] is False
    assert lichess.TOKEN_ENV in state["note"]


def test_describing_the_account_never_reports_the_token():
    client = lichess.Lichess(bearer="secret-token-value")
    client.account = lambda: {"username": "manabot", "title": "BOT"}
    state = client.describe()
    assert "secret-token-value" not in json.dumps(state, ensure_ascii=False)
    assert state["bot"] is True and state["user"] == "manabot"


def test_an_account_that_is_not_a_bot_says_what_to_do():
    client = lichess.Lichess(bearer="x")
    client.account = lambda: {"username": "someone", "title": ""}
    state = client.describe()
    assert state["bot"] is False
    assert "необратимо" in state["note"] and "curl" in state["note"]


def test_a_missing_token_refuses_before_any_request(monkeypatch):
    monkeypatch.setattr(lichess, "token", lambda: "")
    with pytest.raises(lichess.NoToken):
        lichess.Lichess(bearer="")._request("GET", "/api/account")


# --------------------------------------------------------------------------
# the transport
# --------------------------------------------------------------------------

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
        self.calls: List[tuple] = []
        self.headers: Dict[str, str] = {}

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.responses.pop(0)


def test_a_keep_alive_line_is_not_the_end_of_the_stream():
    """Lichess sends an empty line every few seconds. A reader that stops
    on one disconnects constantly and looks like a network fault."""
    lines = [b"", json.dumps({"type": "challenge"}).encode(), b"",
             b"{half", json.dumps({"type": "gameStart"}).encode()]
    client = lichess.Lichess(bearer="x", session=_Session([_Response(lines=lines)]))
    got = list(client._stream("/api/stream/event"))
    assert [row["type"] for row in got] == ["challenge", "gameStart"]


def test_rate_limiting_is_waited_out_once_not_retried_in_a_loop(monkeypatch):
    slept: List[float] = []
    monkeypatch.setattr(lichess.time, "sleep", lambda s: slept.append(s))
    session = _Session([_Response(status=429), _Response(payload={"ok": 1})])
    client = lichess.Lichess(bearer="x", session=session)
    assert client._request("GET", "/api/account") == {"ok": 1}
    assert len(session.calls) == 2                  # one retry, not a loop
    assert slept and slept[0] > 0


def test_a_rejected_token_names_the_scopes_it_needs():
    client = lichess.Lichess(bearer="x", session=_Session([_Response(status=401)]))
    with pytest.raises(lichess.LichessError) as raised:
        client._request("GET", "/api/account")
    for scope in lichess.SCOPES:
        assert scope in str(raised.value)


# --------------------------------------------------------------------------
# what MANA agrees to play
# --------------------------------------------------------------------------

def test_a_refusal_carries_its_reason():
    """A decline nobody can read is indistinguishable from a bug."""
    policy = chess_bot.Policy()
    accept, why = policy.verdict(lichess.Challenge(id="1", by="x",
                                                   variant="atomic"))
    assert accept is False and "atomic" in why


def test_rated_games_are_off_until_someone_turns_them_on():
    rated = lichess.Challenge(id="1", by="x", rated=True, speed="rapid")
    assert chess_bot.Policy().verdict(rated)[0] is False
    assert chess_bot.Policy(rated=True).verdict(rated)[0] is True


def test_a_normal_challenge_is_accepted():
    ok = lichess.Challenge(id="1", by="x", speed="rapid")
    accept, why = chess_bot.Policy().verdict(ok)
    assert accept is True and why


# --------------------------------------------------------------------------
# one game, without a network
# --------------------------------------------------------------------------

class _Fake:
    """A Lichess that plays 1.e4 e5 and then stops."""

    def __init__(self, frames):
        self.frames = frames
        self.sent: List[str] = []
        self.accepted: List[str] = []
        self.declined: List[str] = []

    def account(self):
        return {"username": "manabot", "title": "BOT"}

    def stream_game(self, game_id) -> Iterator[Dict[str, Any]]:
        return iter(self.frames)

    def move(self, game_id, uci, offering_draw=False):
        self.sent.append(uci)
        return {"ok": True}

    def accept(self, cid):
        self.accepted.append(cid)
        return {"ok": True}

    def decline(self, cid, reason="generic"):
        self.declined.append(cid)
        return {"ok": True}


def _bot(frames, **kw):
    bot = chess_bot.Bot(client=_Fake(frames), judge_depth=0, record=False, **kw)
    bot.username = "manabot"
    return bot


FULL = {"type": "gameFull", "id": "g1", "initialFen": "startpos",
        "white": {"id": "manabot", "name": "manabot"},
        "black": {"id": "rival", "name": "rival"},
        "state": {"type": "gameState", "moves": "", "status": "started"}}


def test_it_moves_when_it_is_its_turn_and_not_otherwise():
    frames = [FULL,
              {"type": "gameState", "moves": "e2e4 e7e5", "status": "started"},
              {"type": "gameState", "moves": "e2e4 e7e5 g1f3 b8c6",
               "status": "mate", "winner": "white"}]
    bot = _bot(frames)
    bot._play(lichess.GameStart(id="g1"))
    seat = bot.finished[0]
    assert len(bot.client.sent) == 2          # one for each of its turns
    for uci in bot.client.sent:
        assert len(uci) >= 4
    assert seat.status == "mate" and seat.winner == "white"


def test_every_move_it_makes_is_recorded_with_what_it_considered():
    bot = _bot([FULL])
    bot._play(lichess.GameStart(id="g1"))
    seat = bot.finished[0]
    assert len(seat.thoughts) == 1
    thought = seat.thoughts[0]
    assert thought["considered"] and thought["chosen"]
    assert len(thought["considered"]) == 20   # every legal first move
    assert "margin" in thought and "close_call" in thought


def test_the_side_it_plays_comes_from_the_game_not_from_a_guess():
    black = dict(FULL, white={"id": "rival", "name": "rival"},
                 black={"id": "manabot", "name": "manabot"})
    bot = _bot([black])
    bot._play(lichess.GameStart(id="g1"))
    seat = bot.finished[0]
    assert seat.us is False and seat.opponent == "rival"
    assert bot.client.sent == []              # white has not moved yet


def test_the_position_is_rebuilt_from_the_stream_not_kept_in_memory():
    """After a reconnect the board this process holds and the board
    Lichess believes in can differ, and theirs is the one that decides
    the game."""
    seat = chess_bot.Seat(game_id="g", moves=["e2e4", "e7e5", "g1f3"])
    board = seat.board()
    assert board.fen().startswith(
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b")


def test_a_finished_game_is_not_played_into():
    frames = [dict(FULL, state={"moves": "", "status": "aborted"})]
    bot = _bot(frames)
    bot._play(lichess.GameStart(id="g1"))
    assert bot.client.sent == []
    assert bot.finished[0].status == "aborted"


def test_a_challenge_is_refused_while_a_game_is_running():
    bot = _bot([])
    bot.seats["busy"] = chess_bot.Seat(game_id="busy")
    bot._on_challenge(lichess.Challenge(id="c1", by="rival", speed="rapid"))
    assert bot.client.declined == ["c1"] and bot.client.accepted == []


def test_our_own_challenge_coming_back_is_ignored():
    bot = _bot([])
    bot._on_challenge(lichess.Challenge(id="c1", by="manabot", speed="rapid"))
    assert bot.client.declined == [] and bot.client.accepted == []


def test_playing_requires_the_account_to_be_a_bot():
    bot = _bot([])
    bot.client.account = lambda: {"username": "someone", "title": ""}
    with pytest.raises(lichess.LichessError) as raised:
        bot.run()
    assert "не бот" in str(raised.value) and "необратимо" in str(raised.value)


def test_the_summary_of_a_game_reports_its_denominator():
    seat = chess_bot.Seat(game_id="g")
    seat.thoughts = [{"close_call": True}, {"close_call": False}]
    seat.judged = [{"loss": 0.0, "mistake": False, "blunder": False},
                   {"loss": 400.0, "mistake": True, "blunder": True}]
    row = chess_bot.summarise(seat)
    assert row["moves"] == 2 and row["judged"] == 2
    assert row["blunders"] == 1 and row["mean_loss"] == 200.0
    assert row["close_share"] == 0.5


def test_nothing_in_the_bot_adapts_between_games():
    """Stage 0 on purpose: a baseline collected by a player that changes
    while it is measured is not a baseline."""
    import inspect

    source = inspect.getsource(chess_bot)
    for learning in ("adopt(", "policy.adopt", "install(", "propose("):
        assert learning not in source


# --------------------------------------------------------------------------
# the window
# --------------------------------------------------------------------------

def test_the_window_listens_on_loopback_only():
    """It serves an unauthenticated page reporting what MANA is doing."""
    with pytest.raises(ValueError):
        chess_watch.start(port=0, host="0.0.0.0")


def test_the_window_forwards_only_chess_events():
    from mana import events

    before = len(chess_watch.STATE.frames)
    chess_watch._sink(events.Event(kind="status", text="не про шахматы"))
    chess_watch._sink(events.Event(kind="status", text="",
                                   data={"chess": {"kind": "move"}}))
    assert len(chess_watch.STATE.frames) == before + 1
    assert chess_watch.STATE.frames[-1] == {"kind": "move"}


def test_a_window_opened_mid_game_is_not_blank():
    chess_watch.STATE.frames = [{"kind": "position", "fen": "x"}]
    sink = chess_watch.STATE.subscribe()
    try:
        assert sink.get_nowait() == {"kind": "position", "fen": "x"}
    finally:
        chess_watch.STATE.unsubscribe(sink)


def test_the_page_needs_nothing_from_a_network():
    """The moment worth watching is often the one where the connection is
    what is failing."""
    for fetched in ("cdn", "http://", "https://fonts", "<script src"):
        assert fetched not in chess_watch.PAGE.replace(
            "https://lichess.org", "")


def test_a_second_window_cannot_take_a_port_that_is_already_serving():
    """Found by running it. `HTTPServer` sets `allow_reuse_address`, and
    on Windows that lets a second process bind a port another one is
    already listening on: both sit in LISTENING, the first keeps taking
    the connections, and the new window shows the previous game. For a
    window whose whole job is to show what is happening now, quietly
    showing something else is the worst available failure.
    """
    first = chess_watch.start(0)
    try:
        port = first.server_address[1]
        with pytest.raises(OSError):
            chess_watch.start(port)
    finally:
        first.shutdown()
        first.server_close()


def test_the_command_stops_before_playing_when_the_account_is_not_a_bot(capsys):
    """Nothing is played, and the irreversible step is printed rather
    than taken."""
    from mana import cli

    class _Client:
        def describe(self):
            return {"token": True, "env": lichess.TOKEN_ENV, "user": "someone",
                    "bot": False, "note": "…" + lichess.upgrade_command()}

    original = lichess.Lichess
    lichess.Lichess = lambda *a, **k: _Client()
    try:
        assert cli._lichess(1) == 1
    finally:
        lichess.Lichess = original
    out = capsys.readouterr().out
    assert "curl" in out and "бот: нет" in out


# --------------------------------------------------------------------------
# what actually gets pasted
# --------------------------------------------------------------------------

def test_invisible_characters_are_removed_from_a_token():
    """Found by running it. PowerShell piped what looked like an empty
    string and delivered a byte-order mark; it survived `.strip()`, was
    taken for a token, reached an Authorization header and came back as a
    latin-1 codec error four frames inside urllib3. A token copied from a
    web page with a zero-width space in it does the same.
    """
    assert lichess.clean("\ufeff") == ""
    assert lichess.clean(" \ufefflip_abc123\u200b ") == "lip_abc123"


def test_a_token_that_cannot_be_sent_is_refused_with_a_reason():
    assert lichess.usable("lip_abc123") == ""
    wrong = lichess.usable("lip_абв")
    assert "скопировалось лишнее" in wrong
    assert "codec" not in wrong                 # not urllib3's message


def test_an_unsendable_token_is_never_stored(monkeypatch):
    """Refused before the credential store is touched, so a bad paste
    does not become a saved state that fails on every later run."""
    keyring = pytest.importorskip("keyring")
    stored = []
    monkeypatch.setattr(keyring, "set_password",
                        lambda *a: stored.append(a))
    monkeypatch.delenv(lichess.TOKEN_ENV, raising=False)

    result = lichess.save_token("lip_абв")
    assert result["ok"] is False and "скопировалось" in result["error"]
    assert stored == []
    assert lichess.TOKEN_ENV not in __import__("os").environ

    assert lichess.save_token("lip_abc123")["ok"] is True
    assert stored == [("MANA", lichess.TOKEN_ENV, "lip_abc123")]


def test_a_broken_token_is_reported_as_a_state_not_a_traceback():
    state = lichess.Lichess(bearer="lip_абв").describe()
    assert "скопировалось лишнее" in state["error"]
    assert "user" not in state                  # nothing was asked of Lichess


def test_the_request_layer_refuses_a_token_it_cannot_put_in_a_header():
    client = lichess.Lichess(bearer="lip_абв", session=_Session([]))
    with pytest.raises(lichess.LichessError) as raised:
        client._request("GET", "/api/account")
    assert "скопировалось лишнее" in str(raised.value)


def test_a_secret_read_from_a_pipe_says_it_was_not_typed(monkeypatch, capsys):
    """A prompt nobody can see, blocking on input that will never come,
    is the failure this branch exists to avoid."""
    import io

    from mana import cli

    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("lip_abc123\n"))
    assert cli._read_secret("токен: ") == "lip_abc123"
    assert "не с клавиатуры" in capsys.readouterr().out


# --------------------------------------------------------------------------
# a game to watch without an account
# --------------------------------------------------------------------------

def test_playing_locally_needs_no_client_and_no_account():
    """The window was reachable only through the live command, which
    stops before opening it when the account is not yet a bot -- so the
    board, the reasoning panel and the judge could not be seen at all."""
    played = chess_bot.play_locally(games=1, depth=1, judge_depth=0,
                                    pause=0.0, record=False)
    assert len(played) == 1
    seat = played[0]
    assert seat.source == chess_bot.LOCAL
    assert seat.moves and seat.thoughts
    assert seat.status in ("mate", "draw", "stopped")


def test_a_local_game_emits_the_frames_the_live_bot_emits():
    """A window that shows one world differently from the other is one
    you cannot compare two runs in."""
    from mana import events

    seen = []
    sink = events.subscribe(lambda e: seen.append((e.data or {}).get("chess")))
    try:
        chess_bot.play_locally(games=1, depth=1, judge_depth=0, pause=0.0,
                               record=False)
    finally:
        events.unsubscribe(sink)
    frames = [f for f in seen if f]
    assert frames
    keys = {"kind", "fen", "us", "turn", "ply", "last", "last_uci",
            "thought", "judged", "summary", "source"}
    assert keys <= set(frames[-1])
    assert {f["kind"] for f in frames} <= {"ready", "position", "move", "over"}
    # The run says which judge is about to work, before it works. Three
    # real games were judged by the fallback and nothing said so out loud.
    assert frames[0]["kind"] == "ready" and frames[0]["judge"]


def test_only_the_judged_side_becomes_a_log_entry():
    """The log is decisions with a margin and a cost beside each. An
    entry with neither is a blank row that reads as a defect."""
    from mana import events

    moves = []
    sink = events.subscribe(
        lambda e: moves.append((e.data or {}).get("chess") or {}))
    try:
        chess_bot.play_locally(games=1, depth=1, judge_depth=0, pause=0.0,
                               record=False)
    finally:
        events.unsubscribe(sink)
    for frame in [f for f in moves if f.get("kind") == "move"]:
        assert frame["thought"], frame["last"]


def test_a_local_game_has_no_lichess_url():
    seat = chess_bot.Seat(game_id="local-1", source=chess_bot.LOCAL)
    assert seat.url == ""
    assert chess_bot.Seat(game_id="abc").url.endswith("/abc")


# --------------------------------------------------------------------------
# what the record adds up to
# --------------------------------------------------------------------------

def test_the_two_worlds_are_counted_apart():
    """A game against itself has an opponent sharing its evaluation and
    its blind spots; a game against a stranger does not. An average over
    both is a number about nothing."""
    rows = [
        {"source": chess_bot.LOCAL, "status": "draw", "winner": "",
         "thoughts": [{"close_call": True}, {"close_call": False}],
         "judged": [{"loss": 0.0}, {"loss": 400.0, "mistake": True,
                                    "blunder": True}]},
        {"source": chess_bot.LIVE, "status": "mate", "winner": "black",
         "thoughts": [{"close_call": False}],
         "judged": [{"loss": 100.0, "mistake": True}]}]
    figures = chess_bot.stats(rows)
    assert figures["games"] == 2
    local = figures["by_source"][chess_bot.LOCAL]
    live = figures["by_source"][chess_bot.LIVE]
    assert local["mean_loss"] == 200.0 and live["mean_loss"] == 100.0
    assert local["close_share"] == 0.5 and live["close_share"] == 0.0
    assert local["blunders"] == 1 and live["blunders"] == 0
    assert live["results"] == {"mate black": 1}


def test_every_count_carries_its_denominator():
    """"23 зевка" is not a fact until it says out of how many moves."""
    figures = chess_bot.stats([
        {"source": chess_bot.LIVE, "thoughts": [{"close_call": False}],
         "judged": [{"loss": 0.0}]}])
    row = figures["by_source"][chess_bot.LIVE]
    assert row["judged"] == 1 and row["moves"] == 1 and row["games"] == 1


def test_a_half_written_line_does_not_lose_the_rest(tmp_path, monkeypatch):
    """A run killed mid-write leaves one broken line, and losing every
    game recorded before it would be the wrong response to that."""
    path = tmp_path / "games.jsonl"
    path.write_text('{"source": "local", "moves": []}\n{"source": "liv\n'
                    '{"source": "lichess", "moves": []}\n', encoding="utf-8")
    monkeypatch.setattr(chess_bot, "games_path", lambda: path)
    rows = chess_bot.recorded()
    assert len(rows) == 2
    assert chess_bot.stats(rows)["games"] == 2


def test_no_record_is_an_empty_answer_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(chess_bot, "games_path", lambda: tmp_path / "nope.jsonl")
    assert chess_bot.recorded() == []
    assert chess_bot.stats()["games"] == 0


# --------------------------------------------------------------------------
# judging the record again, without playing it again
# --------------------------------------------------------------------------

def _recorded(tmp_path, monkeypatch, rows):
    import json

    path = tmp_path / "games.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                            for r in rows), encoding="utf-8")
    monkeypatch.setattr(chess_bot, "games_path", lambda: path)
    return path


def _played(name="g1", source=chess_bot.LIVE):
    """1.e4 e5 2.Nf3 Nc6, judged badly on purpose."""
    return {"game": name, "source": source, "us": "white",
            "initial_fen": "startpos", "status": "draw", "winner": "",
            "moves": ["e2e4", "e7e5", "g1f3", "b8c6"],
            "thoughts": [{"ply": 1, "margin": 0.0, "close_call": True,
                          "forced": False, "considered": [["e4", 0.0]]}],
            "judged": [{"ply": 1, "move": "e4", "loss": 0.0,
                        "judged_by": "material"}]}


def test_rejudging_replays_nothing_and_changes_only_the_verdict(tmp_path, monkeypatch):
    """`chess_judge` promised this in its docstring and provided no way
    to do it; it has been needed twice, both times because a run had
    judged with the material fallback."""
    if not __import__("mana.cognition.chess_judge", fromlist=["x"]).available():
        pytest.skip("Stockfish не установлен")
    path = _recorded(tmp_path, monkeypatch, [_played()])
    out = chess_bot.rejudge(depth=6)
    assert out["judged"] == 1 and not out.get("error")

    rows = chess_bot.recorded()
    assert rows[0]["moves"] == ["e2e4", "e7e5", "g1f3", "b8c6"]   # untouched
    assert rows[0]["judged_again_at_depth"] == 6
    assert {j["judged_by"] for j in rows[0]["judged"]} == {"stockfish"}
    assert len(rows[0]["judged"]) == 2          # both of white's moves
    assert Path(out["backup"]).exists()


def test_a_game_that_finishes_mid_rejudge_is_not_lost(tmp_path, monkeypatch):
    """The bot appends to the same file, and a whole-file rewrite would
    drop whatever arrived while this ran."""
    import json

    path = _recorded(tmp_path, monkeypatch, [_played("old")])

    class _Arriving:
        """A judge that appends a new game the first time it is used."""
        kind = "stockfish"

        def __init__(self, *a, **k):
            self.first = True

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def judge_move(self, board, move, ply=0):
            if self.first:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(_played("arrived"),
                                            ensure_ascii=False) + "\n")
                self.first = False
            from mana.cognition.chess_judge import Judged

            return Judged(ply=ply, fen="", move=board.san(move), loss=0.0,
                          judged_by="stockfish")

    from mana.cognition import chess_judge

    monkeypatch.setattr(chess_judge, "Judge", _Arriving)
    monkeypatch.setattr(chess_judge, "engine_path", lambda: Path("stockfish"))
    out = chess_bot.rejudge(depth=6)
    assert out["kept_arrivals"] == 1
    assert {r["game"] for r in chess_bot.recorded()} == {"old", "arrived"}


def test_rejudging_without_an_engine_refuses_instead_of_guessing(tmp_path, monkeypatch):
    """Silently falling back to material is what produced two records
    that had to be judged again."""
    _recorded(tmp_path, monkeypatch, [_played()])
    from mana.cognition import chess_judge

    monkeypatch.setattr(chess_judge, "engine_path", lambda: None)
    out = chess_bot.rejudge(depth=6)
    assert out["judged"] == 0 and "Stockfish не найден" in out["error"]
    assert chess_bot.recorded()[0]["judged"][0]["judged_by"] == "material"


def test_rejudging_an_empty_record_is_an_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(chess_bot, "games_path", lambda: tmp_path / "none.jsonl")
    assert chess_bot.rejudge(depth=6)["games"] == 0
