"""
mana_desktop.window — the desktop shell.

pywebview over the system WebView2 (already present on Windows 11), so the
window itself costs about 8 MB rather than the ~60 MB of a bundled Qt or
the ~120 MB of Chromium. The UI is a local page served by
`mana_desktop.server`; this module only opens a window onto it and holds
the process alive.

Single instance is enforced with a named mutex, not a lock file: MANA's
state is one SQLite database plus a state pickle, and two processes
writing to both is corruption, not contention. A lock file would survive a
crash and lock the user out of their own app; a kernel mutex is released
when the process dies, however it dies.
"""
from __future__ import annotations

import ctypes
import sys
import threading
from typing import Any, Optional

from mana import events, paths

APP_TITLE = "MANA"
_MUTEX_NAME = "Global\\MANA_SingleInstance_A17B"


def _acquire_single_instance() -> Optional[Any]:
    """Return the mutex handle, or None if another MANA already runs.

    Windows only; on other platforms this is a no-op that always allows
    the launch, because the packaged application is a Windows target and
    inventing a portable lock here would be untested code on every path
    that matters.
    """
    if sys.platform != "win32":
        return object()
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.CreateMutexW(None, False, _MUTEX_NAME)
        ERROR_ALREADY_EXISTS = 183
        if kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
            return None
        return handle
    except Exception:
        # A failure to check must not prevent the app from starting: the
        # worst case of allowing a second instance is a locked database
        # error the user can see, which beats an app that refuses to open.
        return object()


def run_window() -> int:
    import webview

    if _acquire_single_instance() is None:
        try:
            ctypes.windll.user32.MessageBoxW(
                None, "MANA уже запущена.", APP_TITLE, 0x40)
        except Exception:
            print("MANA уже запущена.")
        return 0

    from mana.cli import build_config, build_parser
    from mana_desktop.server import load_saved_keys, start_server
    from mana_desktop.session import AgentSession

    # Keys first: BrainPool reads api_key_env when it is constructed, so a
    # key saved in a previous session has to be in the environment before
    # the agent exists, not after.
    restored = load_saved_keys()
    if restored:
        events.emit(events.STATUS, f"Загружены сохранённые ключи: {', '.join(sorted(restored))}")

    args = build_parser().parse_args([])
    config = build_config(args)
    config.local_exec_enabled = True     # the sandbox ships with the app

    session = AgentSession(config)
    # MANA_UI_PORT pins the port instead of letting the OS choose. Two
    # uses: opening the same interface in a normal browser next to the
    # window, and checking a packaged build from a script -- with an
    # OS-assigned port there is no way to ask a --windowed build whether
    # its server actually came up.
    import os as _os
    try:
        pinned = int(_os.environ.get("MANA_UI_PORT", "0"))
    except ValueError:
        pinned = 0
    httpd = start_server(session, port=pinned)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    events.emit(events.STATUS, f"Интерфейс на {url}")

    window = webview.create_window(APP_TITLE, url, width=1120, height=780,
                                   min_size=(760, 560))

    def _on_closed() -> None:
        session.close()

    window.events.closed += _on_closed
    try:
        webview.start()
    finally:
        try:
            httpd.shutdown()
        except Exception:
            pass
        session.close()
    return 0
