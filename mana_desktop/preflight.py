"""
mana_desktop.preflight — what the window needs before it tries to open.

Why this exists
---------------
The window is drawn by WebView2. Windows 11 ships it; Windows 10 does not
guarantee it. Without it `webview.start()` raises "no suitable GUI
toolkit" -- and a windowed PyInstaller build has no console, so the
executable exits and the user sees **nothing at all**. A double-click that
produces silence is the worst failure mode an application has, because
there is nothing to search for and no reason to suspect a missing
component rather than a broken program.

The same defect was fixed once already, in the installer's "Диагностика"
shortcut: a windowed build with nowhere to print did nothing visible. The
answer is the same one -- when there is no console, say it in a dialog.

Scope, deliberately narrow
---------------------------
Only what actually stops the window from opening, and only checks whose
failure has an action attached. A preflight that lists everything it can
think of trains people to click past it, and the one entry that mattered
goes past with the rest.

WebView2 is missing on a Windows 10 machine: real, common, one download
away. That is worth a dialog. A missing font is not.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: The Evergreen WebView2 Runtime's update client id. Stable across
#: versions; the runtime writes `pv` under it when installed.
WEBVIEW2_GUID = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"

#: Three places it can be, and all three must be looked at: the machine
#: install lands under WOW6432Node even on 64-bit, and a per-user install
#: writes to HKCU only. Checking one of them was how this check would
#: have reported "missing" on a machine that had it.
WEBVIEW2_KEYS = (
    ("HKLM", rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_GUID}"),
    ("HKLM", rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_GUID}"),
    ("HKCU", rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_GUID}"),
)

DOWNLOAD = "https://developer.microsoft.com/microsoft-edge/webview2/"

#: Windows 10 1607. Below it neither Python 3.13+ nor the runtime is
#: supported, so a failure there is not something MANA can work around.
MIN_BUILD = 14393


def webview2_version() -> str:
    """The installed WebView2 Runtime version, or "" if there is none."""
    if sys.platform != "win32":
        return ""
    try:
        import winreg
    except Exception:                                   # pragma: no cover
        return ""
    hives = {"HKLM": winreg.HKEY_LOCAL_MACHINE, "HKCU": winreg.HKEY_CURRENT_USER}
    for hive_name, path in WEBVIEW2_KEYS:
        try:
            with winreg.OpenKey(hives[hive_name], path) as key:
                version, _ = winreg.QueryValueEx(key, "pv")
                if version and str(version) != "0.0.0.0":
                    return str(version)
        except OSError:
            continue
    return ""


def windows_build() -> int:
    if sys.platform != "win32":
        return 0
    try:
        return int(sys.getwindowsversion().build)
    except Exception:                                   # pragma: no cover
        return 0


def check() -> List[Dict[str, str]]:
    """Problems that stop the window opening. Empty list means go ahead.

    Each entry carries what is wrong AND what to do about it. A problem
    reported without a remedy is a problem the person cannot act on, and
    they will conclude the program is broken rather than incomplete.
    """
    problems: List[Dict[str, str]] = []
    if sys.platform != "win32":
        return problems

    build = windows_build()
    if build and build < MIN_BUILD:
        problems.append({
            "what": "слишком старая версия Windows",
            "detail": f"сборка {build}, нужна не ниже {MIN_BUILD} (Windows 10 1607)",
            "fix": "обновите Windows; обойти это MANA не может",
        })

    if not webview2_version():
        problems.append({
            "what": "не установлен WebView2 Runtime",
            "detail": ("им рисуется окно MANA. В Windows 11 он встроен, "
                       "в Windows 10 — нет"),
            "fix": (f"скачайте «Evergreen Standalone Installer» — {DOWNLOAD}\n"
                    "Либо запустите MANA.exe --cli, она работает и без окна"),
        })

    return problems


def describe(problems: Optional[List[Dict[str, str]]] = None) -> str:
    problems = check() if problems is None else problems
    if not problems:
        return "всё на месте"
    lines = []
    for problem in problems:
        lines.append(f"{problem['what']}: {problem['detail']}")
        lines.append(f"  что делать: {problem['fix']}")
    return "\n".join(lines)


def block_with_reason() -> bool:
    """Report the problems where a person will see them. True: do not start.

    Printed AND shown in a dialog, because the two cases need different
    channels and there is no way to satisfy both with one. A console build
    prints; a windowed build launched from a shortcut has nowhere to print
    and only the dialog reaches anybody.
    """
    problems = check()
    if not problems:
        return False

    text = describe(problems)
    try:
        print(f"MANA не может открыть окно.\n\n{text}")
    except Exception:
        pass

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                None, f"MANA не может открыть окно.\n\n{text}",
                "MANA", 0x10)
        except Exception:
            pass
    return True


def status() -> Dict[str, Any]:
    """For --self-check and the Система tab."""
    problems = check()
    return {
        "webview2": webview2_version() or "",
        "windows_build": windows_build(),
        "can_open_window": not problems,
        "problems": problems,
    }
