"""
mana_desktop.server — a local HTTP surface for the window.

Built on `http.server` rather than FastAPI/uvicorn on purpose. The window
talks to one process on loopback; the whole API is nine routes and one
event stream. Pulling in starlette, uvicorn and an async runtime would add
~15 MB to the installer and an event loop to reason about, in exchange for
routing sugar this does not need. The agent's own concurrency is threads
(brain pool, decomposition, evolution) -- a threading server matches it.

Server-sent events, not WebSocket: the traffic is one-directional (the
agent narrating; the window posting the occasional command over plain
POST), SSE is a dozen lines over the standard library, and it reconnects
by itself.

Binding: 127.0.0.1 on a port the OS picks. Never 0.0.0.0 -- this exposes
memory contents and lets the caller spend API quota, and a desktop app has
no reason to be reachable from the network.
"""
from __future__ import annotations

import json
import queue
import threading
import traceback
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from mana import events, paths

WEB_DIR = Path(__file__).resolve().parent / "web"


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class _Handler(BaseHTTPRequestHandler):
    server_version = "MANA"
    session = None            # set on the server instance below

    # ---------- plumbing ----------

    def log_message(self, *_args: Any) -> None:
        """Silence the default stderr access log: in a windowed build
        stderr may not exist, and every request here is one the app made of
        itself."""

    def _send(self, code: int, payload: Any, content_type: str = "application/json") -> None:
        if content_type == "application/json":
            body = json.dumps(_json_safe(payload), ensure_ascii=False).encode("utf-8")
        else:
            body = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type + ("; charset=utf-8" if "json" in content_type or "text" in content_type else ""))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def _body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    # ---------- routes ----------

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        route = urlparse(self.path).path
        try:
            if route in ("/", "/index.html"):
                return self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html")
            if route == "/api/state":
                return self._send(200, self.session.state())
            if route == "/api/brains":
                return self._send(200, self.session.brains())
            if route == "/api/evolution":
                return self._send(200, self.session.evolution())
            if route == "/api/code-history":
                return self._send(200, {"entries": self.session.code_history()})
            if route == "/api/paths":
                return self._send(200, paths.status())
            if route == "/api/events":
                return self._stream_events()
            return self._send(404, {"error": "not found"})
        except Exception as exc:
            self._send(500, {"error": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exc()})

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        body = self._body()
        try:
            if route == "/api/ask":
                return self._send(200, self.session.ask(str(body.get("text", ""))))
            if route == "/api/cancel":
                return self._send(200, self.session.cancel())
            if route == "/api/memory-search":
                return self._send(200, self.session.memory_search(str(body.get("query", ""))))
            if route == "/api/evolution/start":
                return self._send(200, self.session.start_evolution(int(body.get("cycles", 1))))
            if route == "/api/evolution/stop":
                return self._send(200, self.session.stop_evolution())
            if route == "/api/key":
                return self._send(200, save_api_key(str(body.get("env", "")), str(body.get("value", ""))))
            return self._send(404, {"error": "not found"})
        except Exception as exc:
            self._send(500, {"error": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exc()})

    # ---------- event stream ----------

    def _stream_events(self) -> None:
        """One SSE connection per open window.

        A bounded queue on purpose: evolution with verbose logging emits
        faster than a browser renders, and an unbounded buffer would grow
        until the process died. Dropping the oldest line is the right
        trade for a live log -- what matters is the tail.
        """
        outbox: "queue.Queue[Any]" = queue.Queue(maxsize=1000)

        def sink(event: Any) -> None:
            try:
                outbox.put_nowait(event)
            except queue.Full:
                try:
                    outbox.get_nowait()
                    outbox.put_nowait(event)
                except Exception:
                    pass

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        events.subscribe(sink, replay=True)
        try:
            while True:
                try:
                    event = outbox.get(timeout=15.0)
                    payload = json.dumps({"kind": event.kind, "text": event.text,
                                          "data": _json_safe(event.data), "ts": event.ts},
                                         ensure_ascii=False)
                    chunk = f"data: {payload}\n\n"
                except queue.Empty:
                    chunk = ": keep-alive\n\n"   # comment frame; keeps proxies and the browser happy
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
            pass                                  # window closed or reloaded
        finally:
            events.unsubscribe(sink)


def save_api_key(env_var: str, value: str) -> Dict[str, Any]:
    """Store a brain's API key in the OS credential store.

    Asking a desktop user to set GEMINI_API_KEY in System Properties is not
    an application. Keys go to Windows Credential Manager via `keyring`;
    the environment variable is set for this process too, so the brain
    becomes usable without a restart. It is never written to a file.
    """
    env_var = env_var.strip()
    if not env_var:
        return {"ok": False, "error": "не указана переменная"}
    import os
    try:
        import keyring
        if value:
            keyring.set_password("MANA", env_var, value)
        else:
            try:
                keyring.delete_password("MANA", env_var)
            except Exception:
                pass
    except Exception as exc:
        return {"ok": False, "error": f"хранилище недоступно: {type(exc).__name__}: {exc}"}
    if value:
        os.environ[env_var] = value
    else:
        os.environ.pop(env_var, None)
    return {"ok": True, "env": env_var, "note": "перезапустите приложение, чтобы мозг попал в пул"}


def load_saved_keys() -> Dict[str, bool]:
    """Pull stored keys into the environment before the agent is built.

    BrainPool reads `api_key_env` at construction time, so this has to run
    first -- otherwise a key the user saved last session would not exist
    for the pool that is about to be created.
    """
    import os
    from mana.brains import default_catalog
    from mana.config import Config

    loaded: Dict[str, bool] = {}
    try:
        import keyring
    except Exception:
        return loaded
    for spec in default_catalog(Config()):
        if not spec.api_key_env or os.environ.get(spec.api_key_env):
            continue
        try:
            stored = keyring.get_password("MANA", spec.api_key_env)
        except Exception:
            stored = None
        if stored:
            os.environ[spec.api_key_env] = stored
            loaded[spec.api_key_env] = True
    return loaded


def start_server(session: Any, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    handler = type("_BoundHandler", (_Handler,), {"session": session})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, name="MANA-HTTP", daemon=True).start()
    return httpd
