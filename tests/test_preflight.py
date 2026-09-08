"""
tests/test_preflight.py — what the window needs before it tries to open.

Written against a specific failure: on Windows 10 without the WebView2
Runtime, `webview.start()` raises "no suitable GUI toolkit" deep inside
pywebview, and a windowed PyInstaller build has no console to carry it.
The user double-clicks and sees nothing at all -- there is nothing to
search for and no reason to suspect a missing component rather than a
broken program.
"""
from __future__ import annotations

import sys

import pytest

from mana_desktop import preflight


def test_this_machine_is_described_without_guessing():
    """Whatever the answer, it is a measurement: a version string or an
    empty one, never an assumption from the Windows version."""
    status = preflight.status()
    assert isinstance(status["webview2"], str)
    assert isinstance(status["can_open_window"], bool)
    assert status["can_open_window"] is (not status["problems"])


def test_a_missing_runtime_is_detected_and_explained(monkeypatch):
    monkeypatch.setattr(preflight, "WEBVIEW2_KEYS",
                        (("HKLM", r"SOFTWARE\NoSuchKeyAnywhere"),))
    assert preflight.webview2_version() == ""
    problems = preflight.check()
    assert any("WebView2" in p["what"] for p in problems)


def test_every_problem_carries_what_to_do_about_it(monkeypatch):
    """A problem reported without a remedy is one the person cannot act
    on, and they conclude the program is broken rather than incomplete."""
    monkeypatch.setattr(preflight, "WEBVIEW2_KEYS",
                        (("HKLM", r"SOFTWARE\NoSuchKeyAnywhere"),))
    monkeypatch.setattr(preflight, "windows_build", lambda: 9600)
    problems = preflight.check()
    assert len(problems) == 2                     # old Windows, and no runtime
    for problem in problems:
        assert problem["what"] and problem["detail"] and problem["fix"]


def test_the_remedy_names_where_to_get_it(monkeypatch):
    monkeypatch.setattr(preflight, "WEBVIEW2_KEYS",
                        (("HKLM", r"SOFTWARE\NoSuchKeyAnywhere"),))
    fix = preflight.check()[0]["fix"]
    assert preflight.DOWNLOAD in fix
    # And the way to work without it at all, which is not obvious.
    assert "--cli" in fix


def test_all_three_registry_locations_are_looked_at():
    """The machine install lands under WOW6432Node even on 64-bit, and a
    per-user install writes only to HKCU. Checking one of them was how
    this would have reported "missing" on a machine that had it."""
    hives = [hive for hive, _ in preflight.WEBVIEW2_KEYS]
    paths = [path for _, path in preflight.WEBVIEW2_KEYS]
    assert hives.count("HKLM") == 2 and hives.count("HKCU") == 1
    assert any("WOW6432Node" in p for p in paths)


def test_a_placeholder_version_counts_as_absent(monkeypatch):
    """EdgeUpdate leaves pv=0.0.0.0 behind after a removal, and treating
    that as installed would send the user to a window that never opens."""
    if sys.platform != "win32":
        pytest.skip("реестр только на Windows")

    class FakeKey:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import winreg
    monkeypatch.setattr(winreg, "OpenKey", lambda *a, **k: FakeKey())
    monkeypatch.setattr(winreg, "QueryValueEx", lambda *a: ("0.0.0.0", 1))
    assert preflight.webview2_version() == ""


def test_blocking_reports_before_it_returns(monkeypatch):
    """Printed AND shown in a dialog: a console build prints, a windowed
    one launched from a shortcut has nowhere to print and only the dialog
    reaches anybody."""
    monkeypatch.setattr(preflight, "check",
                        lambda: [{"what": "нет", "detail": "d", "fix": "f"}])
    shown = []
    if sys.platform == "win32":
        import ctypes
        monkeypatch.setattr(ctypes.windll.user32, "MessageBoxW",
                            lambda *a: shown.append(a) or 1)
    assert preflight.block_with_reason() is True
    if sys.platform == "win32":
        assert shown, "диалог не показан — окно молча не откроется"


def test_nothing_is_blocked_when_the_machine_is_fine(monkeypatch):
    monkeypatch.setattr(preflight, "check", lambda: [])
    assert preflight.block_with_reason() is False


def test_the_window_checks_before_importing_the_backend():
    """Order matters: importing pywebview first would produce the same
    cryptic failure this exists to replace."""
    import inspect

    from mana_desktop import window

    source = inspect.getsource(window.run_window)
    assert source.index("block_with_reason") < source.index("import webview")
