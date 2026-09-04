"""
serve_ui.py — run the desktop UI without the window.

The interface is a local page, so it can be developed and tested in an
ordinary browser: same server, same session, same event stream, no
WebView2 and no pywebview. Useful for anyone iterating on web/index.html,
and it is how the UI was verified before the windowed build existed.

    python serve_ui.py [--port 8765]
"""
from __future__ import annotations

import argparse
import sys
import time
import webbrowser

from mana import events
from mana.cli import build_config, build_parser
from mana_desktop.server import load_saved_keys, start_server
from mana_desktop.session import AgentSession


def main() -> int:
    parser = argparse.ArgumentParser(description="MANA UI without the window")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-browser", action="store_true",
                        help="не открывать браузер, только поднять сервер")
    opts = parser.parse_args()

    events.install_console_sink()
    load_saved_keys()

    args = build_parser().parse_args(["--no-llm"] if opts.no_llm else [])
    config = build_config(args)
    config.local_exec_enabled = True

    session = AgentSession(config)
    httpd = start_server(session, port=opts.port)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"

    # Printing a URL and then sleeping forever looks exactly like a hang:
    # nothing opens, nothing else is printed, and the prompt never comes
    # back. Reported from a live run as "the browser never opened and it
    # said nothing". Open it, and say plainly that the silence afterwards
    # is the server running rather than a stall.
    if not opts.no_browser:
        try:
            webbrowser.open(url)
        except Exception as exc:                       # pragma: no cover
            print(f"не смог открыть браузер ({exc}); откройте адрес вручную")

    print("")
    print(f"MANA UI: {url}")
    print("Сервер работает. Это не зависание — терминал занят, пока UI открыт.")
    print("Ctrl+C чтобы остановить.")
    print("")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("остановлено")
    finally:
        httpd.shutdown()
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
