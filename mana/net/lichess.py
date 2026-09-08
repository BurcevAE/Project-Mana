"""
mana.net.lichess — the Bot API, as a world MANA acts in.

What this module is for
-----------------------
`chess_arena` gave MANA an opponent it made up. This gives it a world it
did not: opponents it has never seen, clocks that run while it thinks,
positions no generator of ours produced, and a result nobody here can
argue with. Everything the chess programme wants to ask -- does a class of
errors survive contact with strangers, does a correction transfer -- needs
data from outside the process that would like the answer to come out well.

The one thing this module will not do
--------------------------------------
Upgrade the account. `POST /api/bot/account/upgrade` is irreversible:
Lichess never converts a BOT account back, and the account can never play
from the website again. That is a decision about a person's account, and
the fact that MANA holds a token which *could* make it is exactly why the
call is not in here. `upgrade_command()` prints what the user runs
themselves; a test reads this module's source to keep the call out of it,
the same way `chess_judge` is kept from ever picking a move.

Credentials
-----------
The token comes from the environment, and `mana_desktop` puts it there
from Windows Credential Manager via keyring -- the rule every API key in
this project follows. Never in a file, never in the repository, never in
a log line. `describe()` reports whether a token was found and never what
it is.

Rate limits are part of the protocol, not an error
---------------------------------------------------
Lichess answers 429 when a client asks too fast, and the documented
response is to wait a minute rather than to retry immediately. A retry
loop that ignores it gets the token blocked, so `_request` waits once and
says so through the event bus. Streams are a different shape: they are
long-lived NDJSON connections that send an empty line every few seconds
to keep the socket alive, and a reader that treats a blank line as the
end of the stream disconnects every six seconds.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from .. import events

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

BASE = "https://lichess.org"

#: Where the token lives. `mana_desktop` fills this from keyring before
#: anything is constructed, exactly as it does for the LLM provider keys.
TOKEN_ENV = "MANA_LICHESS_TOKEN"

#: The service name under which keyring stores MANA's secrets.
KEYRING_SERVICE = "MANA"

#: Scopes a bot needs. `bot:play` is the one the play endpoints check;
#: `challenge:write` is what lets MANA start games against Lichess's own
#: Stockfish levels instead of waiting for a human to challenge it.
SCOPES = ("bot:play", "challenge:write")

#: The irreversible call. Named so it can be recognised and refused, and
#: deliberately never passed to `_request`.
UPGRADE_PATH = "/api/bot/account/upgrade"

#: Documented wait after a 429. Lichess asks for a minute; asking again
#: sooner is how a token gets blocked.
RATE_LIMIT_WAIT = 60.0

#: How long a stream may say nothing before it is treated as dead. The
#: server sends a blank keep-alive line every few seconds, so silence for
#: much longer than that means the connection is gone, not idle.
STREAM_TIMEOUT = 20.0


class LichessError(RuntimeError):
    """Lichess refused or failed. Distinct from a missing token, which is
    a fact about this machine rather than about the server."""


class NoToken(LichessError):
    """No token on this machine. Not a malfunction -- a state with a
    remedy, and the remedy is named in the message."""


def token() -> str:
    """The token, from the environment or the OS credential store.

    In that order, so a token set for one run wins over the stored one
    without anything being overwritten. Returns "" rather than raising:
    "is there a token" is a question with an answer, and the callers that
    need one raise their own refusal with the remedy attached.
    """
    found = os.environ.get(TOKEN_ENV, "").strip()
    if found:
        return found
    try:
        import keyring

        stored = keyring.get_password(KEYRING_SERVICE, TOKEN_ENV)
    except Exception:
        return ""
    return (stored or "").strip()


def save_token(value: str) -> Dict[str, Any]:
    """Put a token into the credential store, and into this process.

    The value is never returned, logged or written to a file. Empty
    removes it.
    """
    value = (value or "").strip()
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


def upgrade_command() -> str:
    """What the user runs to turn their account into a bot.

    Not run from here, and not runnable from here. Returned as text so
    the person whose account it is makes the decision with the command in
    front of them.
    """
    return ('curl -X POST https://lichess.org/api/bot/account/upgrade '
            '-H "Authorization: Bearer <ВАШ_ТОКЕН>"')


@dataclass
class Challenge:
    """An invitation, with enough to decide on it without a second call."""
    id: str
    by: str
    rated: bool = False
    variant: str = "standard"
    speed: str = ""
    #: The colour MANA would play, as Lichess states it: white/black/random.
    colour: str = "random"
    raw: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        kind = "рейтинговая" if self.rated else "тренировочная"
        return f"{self.by} зовёт: {self.variant}/{self.speed}, {kind}"


@dataclass
class GameStart:
    id: str
    colour: str = ""
    opponent: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


class Lichess:
    """The Bot API. One session, because it is mostly long-lived streams.

    Every method is a single documented endpoint. Nothing here decides
    what to play or whether to accept -- that belongs to the caller, and
    keeping the decisions out of the transport is what lets the bot loop
    be tested without a network.
    """

    def __init__(self, bearer: str = "", base: str = BASE,
                 session: Any = None) -> None:
        self.base = base.rstrip("/")
        self.bearer = bearer or token()
        self._session = session
        self._blocked_until = 0.0

    # ---------- transport ----------

    @property
    def session(self) -> Any:
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self.bearer}",
                "User-Agent": "MANA/bot (https://github.com/BurcevAE/Project-Mana)"})
        return self._session

    def _url(self, path: str) -> str:
        if path == UPGRADE_PATH:
            # Unreachable through the public methods; here so that a
            # future caller that constructs the path by hand still cannot
            # make the account irreversible through this client.
            raise LichessError(
                "перевод аккаунта в бота необратим и не делается отсюда; "
                f"выполните сами: {upgrade_command()}")
        return f"{self.base}{path}"

    def _require_token(self) -> None:
        if not self.bearer:
            raise NoToken(
                f"нет токена: задайте {TOKEN_ENV} в окружении или сохраните "
                "его в диспетчере учётных данных Windows")

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """One call, with the documented pause after a 429.

        Waits once and retries once. A loop here would be a client that
        answers rate limiting by producing more of what caused it.
        """
        self._require_token()
        url = self._url(path)
        for attempt in (1, 2):
            wait = self._blocked_until - time.time()
            if wait > 0:
                time.sleep(min(wait, RATE_LIMIT_WAIT))
            response = self.session.request(method, url, timeout=30, **kwargs)
            if response.status_code == 429 and attempt == 1:
                self._blocked_until = time.time() + RATE_LIMIT_WAIT
                events.emit(events.WARNING,
                            f"Lichess просит подождать {RATE_LIMIT_WAIT:.0f}с "
                            f"({method} {path})", lichess={"rate_limited": path})
                continue
            if response.status_code == 401:
                raise LichessError(
                    "Lichess не принял токен (401). Проверьте, что у токена "
                    f"есть права {', '.join(SCOPES)} и что аккаунт — бот")
            if response.status_code >= 400:
                raise LichessError(f"{method} {path} → {response.status_code}: "
                                   f"{response.text[:200]}")
            try:
                return response.json()
            except ValueError:
                return {"ok": True}
        raise LichessError(f"{method} {path}: ограничение частоты не снялось")

    def _stream(self, path: str) -> Iterator[Dict[str, Any]]:
        """NDJSON, one object per line.

        Blank lines are the server's keep-alive and are skipped rather
        than treated as the end -- a reader that stops on one disconnects
        every few seconds and looks like a network fault.
        """
        self._require_token()
        response = self.session.get(self._url(path), stream=True,
                                    timeout=(10, STREAM_TIMEOUT))
        if response.status_code >= 400:
            raise LichessError(f"поток {path} → {response.status_code}: "
                               f"{response.text[:200]}")
        for line in response.iter_lines():
            if not line:
                continue                      # keep-alive, not an ending
            try:
                yield json.loads(line.decode("utf-8"))
            except ValueError:
                continue                      # a half line; the next one is whole

    # ---------- account ----------

    def account(self) -> Dict[str, Any]:
        return self._request("GET", "/api/account")

    def is_bot(self) -> bool:
        return str(self.account().get("title", "")).upper() == "BOT"

    def describe(self) -> Dict[str, Any]:
        """What this machine can do with Lichess right now.

        Answers without playing anything, and never reports the token
        itself -- only whether one was found and what it turned out to be
        good for. An absent token is reported as a state with a remedy,
        not as a failure.
        """
        out: Dict[str, Any] = {"token": bool(self.bearer), "env": TOKEN_ENV}
        if not self.bearer:
            out["note"] = (f"нет токена: задайте {TOKEN_ENV} или сохраните "
                           "его в диспетчере учётных данных")
            return out
        try:
            me = self.account()
        except LichessError as exc:
            out["error"] = str(exc)
            return out
        out["user"] = me.get("username", "")
        out["bot"] = str(me.get("title", "")).upper() == "BOT"
        out["url"] = f"{self.base}/@/{out['user']}" if out["user"] else ""
        if not out["bot"]:
            out["note"] = ("аккаунт ещё не бот. Это делается один раз и "
                           "необратимо, поэтому командой от вас: "
                           + upgrade_command())
        return out

    # ---------- challenges ----------

    def challenges(self) -> List[Challenge]:
        data = self._request("GET", "/api/challenge")
        return [_challenge(row) for row in data.get("in", [])]

    def accept(self, challenge_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/api/challenge/{challenge_id}/accept")

    def decline(self, challenge_id: str, reason: str = "generic") -> Dict[str, Any]:
        return self._request("POST", f"/api/challenge/{challenge_id}/decline",
                             data={"reason": reason})

    def challenge_ai(self, level: int = 1, clock_limit: int = 300,
                     clock_increment: int = 3, colour: str = "random",
                     fen: str = "") -> Dict[str, Any]:
        """Start a game against Lichess's own Stockfish at a fixed level.

        The reason this exists: a bot with no challengers plays nothing,
        and an experiment that needs three hundred games cannot wait for
        strangers. Levels are a ladder with known steps, which makes
        "the opponent got stronger" something the record states rather
        than something the result has to be corrected for.

        Note the two Stockfishes are different things and never meet:
        this one is an opponent on Lichess's side, `chess_judge` is a
        local judge that never picks a move.
        """
        body = {"level": max(1, min(8, int(level))), "color": colour}
        if fen:
            body["fen"] = fen
        if clock_limit:
            body["clock.limit"] = int(clock_limit)
            body["clock.increment"] = int(clock_increment)
        else:
            body["days"] = 1
        return self._request("POST", "/api/challenge/ai", data=body)

    # ---------- games ----------

    def stream_events(self) -> Iterator[Dict[str, Any]]:
        """Challenges and game starts, as they happen."""
        return self._stream("/api/stream/event")

    def stream_game(self, game_id: str) -> Iterator[Dict[str, Any]]:
        """One game: a `gameFull`, then a `gameState` per move."""
        return self._stream(f"/api/bot/game/stream/{game_id}")

    def move(self, game_id: str, uci: str, offering_draw: bool = False) -> Dict[str, Any]:
        path = f"/api/bot/game/{game_id}/move/{uci}"
        if offering_draw:
            path += "?offeringDraw=true"
        return self._request("POST", path)

    def resign(self, game_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/api/bot/game/{game_id}/resign")

    def chat(self, game_id: str, text: str, room: str = "player") -> Dict[str, Any]:
        return self._request("POST", f"/api/bot/game/{game_id}/chat",
                             data={"room": room, "text": text[:140]})

    def close(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None


def _challenge(row: Dict[str, Any]) -> Challenge:
    return Challenge(
        id=str(row.get("id", "")),
        by=str((row.get("challenger") or {}).get("name", "?")),
        rated=bool(row.get("rated")),
        variant=str((row.get("variant") or {}).get("key", "standard")),
        speed=str(row.get("speed", "")),
        colour=str(row.get("color", "random")),
        raw=row)


def game_start(row: Dict[str, Any]) -> GameStart:
    game = row.get("game") or row
    return GameStart(id=str(game.get("id", "")),
                     colour=str(game.get("color", "")),
                     opponent=str((game.get("opponent") or {}).get("username", "")),
                     raw=game)
