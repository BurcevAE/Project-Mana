"""
mana.net.aichess — AI Chess Arena, transport only.

Why this world and not another
-------------------------------
Lichess gave MANA strangers and Stockfish levels. This gives it a
population: matchmaking near its own rating, named language models as
fixed opponents, and humans in the same pool. That is what a question
about transfer needs -- a principle that survives one opponent and not
another is a principle about that opponent.

    очередь          соперник примерно своего уровня, меняющийся
    opponent_id      названная модель, сто партий против одной и той же
    is_public        человек в той же среде

What this module refuses to be
-------------------------------
A judge. The server returns the state of the world -- position, legal
moves, clocks, whose turn, who the opponent is, how the game ended -- and
nothing that grades a move. That is the same line `chess_judge` holds and
the reason the outcome-based analysis can stay as it is: MANA learns from
what happened, not from what a stronger player thought of it.

The rule this environment does not share with chess
-----------------------------------------------------
Games end after seventy moves and the winner is decided on piece count.
That is a house rule, not a chess rule, and a result from here is
therefore not comparable with a result from Lichess without saying so.
`ENVIRONMENT` and `TERMINATION` travel with every game record for exactly
that reason -- pooling two environments that end games differently is the
same error as pooling two worlds, one layer down.

Credentials
-----------
`POST /players` returns a permanent key. It goes to Windows Credential
Manager or the environment, never to a file and never to the repository --
the rule every key in this project follows. Registration itself is not
done from here: it creates an identity on somebody else's service under a
name the user chooses, so `register()` exists but the command that calls
it is theirs to run.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from .. import events

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

BASE = "https://aichess.co"

#: Where the key lives. Filled from keyring before anything is built,
#: exactly as the Lichess token is.
TOKEN_ENV = "MANA_AICHESS_TOKEN"
KEYRING_SERVICE = "MANA"

#: Named in every record: this environment does not end games the way
#: chess does, so its results are not comparable with another world's
#: without the name of the rule beside them.
ENVIRONMENT = "aichess"
TERMINATION = "aichess_70_move_piece_count"

#: Time controls the server accepts, in seconds.
TIME_CONTROLS = (60, 180, 300, 600, 900, 1800)

#: Documented polling interval. Used only as a fallback: the game itself
#: comes over the stream, and polling a game that has a stream is asking
#: a question the server already offered to answer.
POLL_SECONDS = 1.5

#: How long a stream may say nothing before it is treated as dead. The
#: server sends `ping`, so silence much longer than that is a dead socket
#: rather than a quiet game.
STREAM_TIMEOUT = 30.0


class ArenaError(RuntimeError):
    """The server refused or failed."""


class NoToken(ArenaError):
    """No key on this machine. A state with a remedy, not a malfunction."""


def token() -> str:
    """The key, from the environment or the OS credential store."""
    from .lichess import clean

    found = clean(os.environ.get(TOKEN_ENV, ""))
    if found:
        return found
    try:
        import keyring

        stored = keyring.get_password(KEYRING_SERVICE, TOKEN_ENV)
    except Exception:
        return ""
    return clean(stored or "")


def save_token(value: str) -> Dict[str, Any]:
    """Store the key. Never returned, logged, or written to a file."""
    from .lichess import clean, usable

    value = clean(value or "")
    if value:
        wrong = usable(value)
        if wrong:
            return {"ok": False, "error": wrong}
    try:
        import keyring

        if value:
            keyring.set_password(KEYRING_SERVICE, TOKEN_ENV, value)
        else:
            try:
                keyring.delete_password(KEYRING_SERVICE, TOKEN_ENV)
            except Exception:
                pass
    except Exception as exc:
        return {"ok": False, "error": f"хранилище недоступно: {type(exc).__name__}: {exc}"}
    if value:
        os.environ[TOKEN_ENV] = value
    else:
        os.environ.pop(TOKEN_ENV, None)
    return {"ok": True, "stored": bool(value), "env": TOKEN_ENV}


@dataclass
class Seat:
    """The world's own description of a game in progress.

    Every field is what the server said. Nothing here is derived and
    nothing is graded: `legal_moves` arrives only when it is our turn,
    which is the server telling us what is permitted rather than us
    re-deriving it and risking a different answer.
    """
    game_id: str
    us: bool = True                       # True = white, chess's convention
    our_turn: bool = False
    fen: str = ""
    moves: List[str] = field(default_factory=list)
    legal_moves: List[str] = field(default_factory=list)
    opponent: str = ""
    white_time: float = 0.0
    black_time: float = 0.0
    status: str = ""
    result_reason: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def over(self) -> bool:
        return bool(self.status) and self.status != "in_progress"

    def as_dict(self) -> Dict[str, Any]:
        return {"game": self.game_id, "environment": ENVIRONMENT,
                "termination_rule": TERMINATION,
                "us": "white" if self.us else "black",
                "opponent": self.opponent, "moves": list(self.moves),
                "status": self.status, "result_reason": self.result_reason,
                "white_time": self.white_time, "black_time": self.black_time}


def seat_of(row: Dict[str, Any]) -> Optional[Seat]:
    """Read an `active_game` block, or None when there is no game."""
    if not row:
        return None
    return Seat(
        game_id=str(row.get("game_id", "")),
        us=str(row.get("your_color", "white")) == "white",
        our_turn=bool(row.get("your_turn")),
        fen=str(row.get("fen", "")),
        moves=[str(move) for move in row.get("moves", [])],
        legal_moves=[str(move) for move in row.get("legal_moves", [])],
        opponent=str(row.get("opponent", "")),
        white_time=float(row.get("white_time", 0.0) or 0.0),
        black_time=float(row.get("black_time", 0.0) or 0.0),
        status=str(row.get("status", "")),
        result_reason=str(row.get("result_reason", "")),
        raw=dict(row))


class Arena:
    """The Agent API. Decides nothing about how to play.

    Every method is one documented endpoint. Keeping the decisions out of
    the transport is what let the Lichess bot be tested without a network,
    and it is what will let one brain face both worlds.
    """

    def __init__(self, bearer: str = "", base: str = BASE,
                 session: Any = None) -> None:
        self.base = base.rstrip("/")
        self.bearer = bearer or token()
        self._session = session
        self._one_at_a_time = threading.Lock()

    @property
    def session(self) -> Any:
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": "MANA/bot (https://github.com/BurcevAE/Project-Mana)"})
        return self._session

    def _headers(self, need_token: bool = True) -> Dict[str, str]:
        if not need_token:
            return {}
        if not self.bearer:
            raise NoToken(
                f"нет ключа: задайте {TOKEN_ENV} или сохраните его в "
                "диспетчере учётных данных Windows")
        return {"Authorization": f"Bearer {self.bearer}"}

    def _request(self, method: str, path: str, need_token: bool = True,
                 **kwargs: Any) -> Any:
        headers = dict(self._headers(need_token))
        headers.update(kwargs.pop("headers", {}))
        with self._one_at_a_time:
            response = self.session.request(method, f"{self.base}{path}",
                                            headers=headers, timeout=30,
                                            **kwargs)
        if response.status_code == 401:
            raise ArenaError("ключ не принят (401)")
        if response.status_code >= 400:
            raise ArenaError(f"{method} {path} → {response.status_code}: "
                             f"{response.text[:200]}")
        try:
            return response.json()
        except ValueError:
            return {"ok": True}

    # ---------- identity ----------

    def register(self, name: str) -> Dict[str, Any]:
        """Create the agent's identity. Called by a command the user runs.

        Not done automatically: it puts a name on somebody else's service
        and returns a permanent key, and both are the user's to decide.
        """
        return self._request("POST", "/api/v1/players", need_token=False,
                             json={"name": name, "type": "agent"})

    # ---------- finding a game ----------

    def join_queue(self, time_control: int = 300) -> Dict[str, Any]:
        if int(time_control) not in TIME_CONTROLS:
            raise ArenaError(f"контроль {time_control} не из {TIME_CONTROLS}")
        return self._request("POST", "/api/v1/queue/join",
                             json={"time_control": int(time_control)})

    def leave_queue(self) -> Dict[str, Any]:
        return self._request("POST", "/api/v1/queue/leave")

    def challenge_model(self, opponent_id: str, time_control: int = 300,
                        colour: str = "random") -> Dict[str, Any]:
        """A named model as a fixed opponent.

        The reason this matters more than the queue: a hundred games
        against one model, then a hundred against another, is how a
        principle is shown to survive a change of opponent -- or not to.
        """
        return self._request("POST", "/api/v1/games",
                             json={"opponent_id": opponent_id,
                                   "color": colour,
                                   "time_control": int(time_control)})

    def open_game(self, time_control: int = 300,
                  colour: str = "random") -> Dict[str, Any]:
        return self._request("POST", "/api/v1/games",
                             json={"color": colour, "is_public": True,
                                   "time_control": int(time_control)})

    # ---------- playing ----------

    def activity(self) -> Dict[str, Any]:
        return self._request("GET", "/api/v1/activity")

    def seat(self) -> Optional[Seat]:
        """The game in progress, as the server describes it."""
        return seat_of(self.activity().get("active_game") or {})

    def game(self, game_id: str) -> Optional[Seat]:
        row = self._request("GET", f"/api/v1/games/{game_id}")
        return seat_of(row.get("state") or row)

    def move(self, game_id: str, move: str) -> Dict[str, Any]:
        """Send a move, in UCI or SAN -- the server takes either."""
        return self._request("POST", f"/api/v1/games/{game_id}/moves",
                             json={"move": move})

    def resign(self, game_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/api/v1/games/{game_id}/resign")

    def stream(self, game_id: str) -> Iterator[Dict[str, Any]]:
        """Server-sent events for one game.

        Preferred over polling: the server offered to tell us, and asking
        it every second and a half for something it will push is a way of
        spending its patience. Polling stays as the fallback when this
        drops.
        """
        response = self.session.get(
            f"{self.base}/api/v1/games/{game_id}/stream",
            headers=self._headers(), stream=True,
            timeout=(10, STREAM_TIMEOUT))
        if response.status_code >= 400:
            raise ArenaError(f"поток {game_id} → {response.status_code}")
        kind = ""
        for raw in response.iter_lines():
            line = raw.decode("utf-8") if raw else ""
            if not line:
                continue                   # frame separator, not an ending
            if line.startswith("event:"):
                kind = line.split(":", 1)[1].strip()
                continue
            if not line.startswith("data:"):
                continue
            body = line.split(":", 1)[1].strip()
            try:
                payload = json.loads(body) if body else {}
            except ValueError:
                continue
            payload.setdefault("event", kind or "state")
            yield payload

    def leaderboard(self, kind: str = "agent", limit: int = 20) -> Any:
        return self._request("GET", "/api/v1/leaderboard",
                             params={"type": kind, "limit": int(limit)})

    def close(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def describe(self) -> Dict[str, Any]:
        """What this machine can do here. Never reports the key itself."""
        out: Dict[str, Any] = {"token": bool(self.bearer), "env": TOKEN_ENV,
                               "environment": ENVIRONMENT,
                               "termination_rule": TERMINATION}
        if not self.bearer:
            out["note"] = (f"нет ключа: зарегистрируйтесь один раз "
                           f"(--aichess ИМЯ) или задайте {TOKEN_ENV}")
            return out
        try:
            row = self.activity()
        except ArenaError as exc:
            out["error"] = str(exc)
            return out
        out["in_queue"] = bool(row.get("in_queue"))
        seat = seat_of(row.get("active_game") or {})
        out["game"] = seat.game_id if seat else ""
        return out
