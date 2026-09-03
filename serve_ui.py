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

from mana import events
from mana.cli import build_config, build_parser
from mana_desktop.server import load_saved_keys, start_server
from mana_desktop.session import AgentSession


def main() -> int:
    parser = argparse.ArgumentParser(description="MANA UI without the window")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-llm", action="store_true")
    opts = parser.parse_args()

    events.install_console_sink()
    load_saved_keys()

    args = build_parser().parse_args(["--no-llm"] if opts.no_llm else [])
    config = build_config(args)
    config.local_exec_enabled = True

    session = AgentSession(config)
    httpd = start_server(session, port=opts.port)
    print(f"UI: http://127.0.0.1:{httpd.server_address[1]}/")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
