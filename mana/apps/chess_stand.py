"""
mana.apps.chess_stand — the chess ladder, started and stopped by asking.

"Мана, поиграй в шахматы на стенде" starts the same bench `mana.cmd --bench`
runs (`cognition/chess_bench.py`), in the background, from level 1:

    10 побед подряд       -> следующий уровень
    30 побед подряд на 8  -> конец
    ничья серию сбрасывает
    «останови стенд»      -> конец в любой момент

While Lichess holds a pause after refusing a game, the bench plays itself
and runs its experiments -- it does that on its own; nothing here adds to
it. The observation window serves on 127.0.0.1 only.

Nothing about the ladder is new here. What is new is that the window can
reach it, and that the reply to "поиграй" is built from what actually
started -- or from why it could not -- never from the request.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from .. import events

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: How long `stop` waits for the run to wind down before answering.
STOP_WAIT = 30.0

_lock = threading.Lock()
_run: Optional["_Stand"] = None
_last_summary = ""


class _Stand:
    def __init__(self, bench: Any, thread: threading.Thread, httpd: Any, url: str,
                 account: str) -> None:
        self.bench = bench
        self.thread = thread
        self.httpd = httpd
        self.url = url
        self.account = account


def running() -> bool:
    return _run is not None and _run.thread.is_alive()


def _close_window(httpd: Any) -> None:
    if httpd is None:
        return
    try:
        httpd.shutdown()
        httpd.server_close()
    except Exception:
        pass


def _play(stand: "_Stand") -> None:
    global _last_summary
    from ..cognition import chess_bench
    try:
        stand.bench.run()
    except Exception as exc:
        events.emit(events.WARNING, f"стенд остановился с ошибкой: {exc}")
    finally:
        try:
            _last_summary = chess_bench.summarise(stand.bench.ladder)
        except Exception:
            _last_summary = ""
        _close_window(stand.httpd)


def start(from_level: int = 1) -> Dict[str, Any]:
    """Start the ladder in the background; refuse, with the reason, when it
    cannot run. A second start while it runs reports the run in progress."""
    global _run
    from ..cognition import chess_bench, chess_watch
    from ..net import lichess

    with _lock:
        if running():
            return {"ok": True, "already": True, "status": _run.bench.ladder.describe(),
                    "url": _run.url, "account": _run.account}
        client = lichess.Lichess()
        state = client.describe()
        if not state.get("bot"):
            why = (state.get("error") or state.get("note")
                   or ("нет токена Lichess — ввести: mana.cmd --lichess-token"
                       if not state.get("token") else "аккаунт Lichess не BOT"))
            return {"ok": False, "error": f"стенд не запущен: {why}"}
        # From level 1, as asked -- but a pause Lichess imposed is kept: a
        # fresh ladder that forgot it would ask again at once and start the
        # escalation over.
        saved = chess_bench.Ladder.load()
        level = max(chess_bench.FIRST_LEVEL, min(chess_bench.LAST_LEVEL, int(from_level)))
        ladder = chess_bench.Ladder(level=level, next_request_at=saved.next_request_at,
                                    refused=saved.refused)
        httpd, url = None, ""
        try:
            httpd = chess_watch.start(chess_watch.DEFAULT_PORT)
            url = chess_watch.url(httpd)
        except OSError:
            pass                        # plays without a window rather than not at all
        bench = chess_bench.Bench(client=client, ladder=ladder)
        stand = _Stand(bench, threading.Thread(target=lambda: _play(stand),
                                               name="MANA-stand", daemon=True),
                       httpd, url, str(state.get("user") or ""))
        _run = stand
        stand.thread.start()
        return {"ok": True, "already": False, "level": ladder.level, "url": url,
                "account": stand.account,
                "rule": (f"{chess_bench.WINS_TO_ADVANCE} побед подряд — следующий уровень; "
                         f"{chess_bench.WINS_TO_FINISH} подряд на {chess_bench.LAST_LEVEL} — "
                         f"конец; ничья серию сбрасывает")}


def stop() -> Dict[str, Any]:
    """Stop the ladder and say where it stood."""
    from ..cognition import chess_bench

    with _lock:
        stand = _run
    if stand is None or not stand.thread.is_alive():
        return {"ok": True, "stopped": False,
                "summary": _last_summary or "стенд не запущен"}
    stand.bench.stop()
    stand.thread.join(timeout=STOP_WAIT)
    summary = _last_summary if not stand.thread.is_alive() else ""
    return {"ok": True, "stopped": True,
            "summary": summary or chess_bench.summarise(stand.bench.ladder)}
