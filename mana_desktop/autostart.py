"""
mana_desktop.autostart — run MANA when the user logs in.

One registry value under HKCU\\...\\CurrentVersion\\Run, deliberately the
same value name the installer writes, so the checkbox in the installer and
the switch in the window are two views of one setting rather than two
settings that disagree. A user who ticks the box at install time and later
turns it off in the window expects it to be off.

HKCU and not HKLM: the install is per-user (see installer/mana.iss for
why it has to be), so a machine-wide autostart entry would survive an
uninstall performed by a different account and point at a program that no
longer exists.

Reading is separated from writing on purpose. `status()` reports what is
actually registered, including the case where something else wrote the
value -- an older install in another directory, most likely. Showing "on"
for an entry pointing somewhere else would be a lie of exactly the kind
this project keeps finding in its own instrumentation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

#: Matches the ValueName in installer/mana.iss. Changing one without the
#: other produces two independent autostart entries.
VALUE_NAME = "MANA"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def supported() -> bool:
    return sys.platform == "win32"


def command() -> str:
    """The command line that would be registered for this installation."""
    if getattr(sys, "frozen", False):
        return f'"{Path(sys.executable)}"'
    # A source checkout. Registered honestly rather than refused: the
    # author runs MANA from source, and `status()` shows the exact string
    # so there is no doubt about which copy would start.
    launcher = Path(sys.executable).with_name("pythonw.exe")
    if not launcher.exists():
        launcher = Path(sys.executable)
    app = Path(__file__).resolve().parent.parent / "app.py"
    return f'"{launcher}" "{app}"'


def _read() -> str:
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            value, _ = winreg.QueryValueEx(key, VALUE_NAME)
            return str(value)
    except FileNotFoundError:
        return ""
    except OSError:
        return ""


def status() -> Dict[str, Any]:
    if not supported():
        return {"supported": False, "enabled": False, "registered": "",
                "note": "автозапуск реализован только для Windows"}
    registered = _read()
    mine = command()
    note = ""
    if registered and registered != mine:
        # Not an error, and not silently overwritten either: the user gets
        # told which copy of MANA is actually set to start.
        note = "в автозапуске записана другая копия MANA"
    return {"supported": True, "enabled": bool(registered),
            "registered": registered, "command": mine, "note": note}


def enable() -> Dict[str, Any]:
    if not supported():
        return {"ok": False, "error": "только Windows"}
    import winreg
    try:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, command())
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, **status()}


def disable() -> Dict[str, Any]:
    if not supported():
        return {"ok": False, "error": "только Windows"}
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0,
                            winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, VALUE_NAME)
    except FileNotFoundError:
        pass                      # already off; the desired state is the state
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, **status()}
