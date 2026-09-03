"""
mana.paths — where MANA's files live, in every way it can be launched.

Why this module exists
----------------------
MANA had exactly one notion of location: "relative to the current working
directory". That is correct for `python mana_run.py` run from the project
folder and wrong for every other way of starting it. Three separate
failures come from it, and all three are silent:

  1. **State follows the shell, not the app.** `memory_db_path` defaults to
     "mana_memory/mana_memory.sqlite3" (config.py). Launched from a desktop
     shortcut, the working directory is wherever Windows felt like -- so
     MANA opens an empty database, greets you as a stranger, and looks like
     it lost its memory. It didn't; it is looking in a different folder.

  2. **The sandbox stops being a sandbox.** LocalVerifier ran candidate
     code with `sys.executable`. In a frozen build `sys.executable` is
     MANA.exe, not python.exe, so "verify this code" would have re-launched
     the whole agent. See `sandbox_python()`.

  3. **Self-patching writes into a temp folder Windows then deletes.**
     code_evolution.apply_patch() rewrites real source files. Under
     PyInstaller's onefile mode the package lives in a per-run extraction
     directory that is destroyed at exit: the patch applies, the changelog
     records it, and the change evaporates. See `package_is_writable()`.

Design
------
Three roots, deliberately distinct, because they have different lifetimes
and different permissions:

  * `install_root()` -- where the program is. Read-mostly. Under a frozen
    build this is the folder containing the .exe; in development it is the
    repository.
  * `data_root()`    -- where the user's state is. Always writable, per
    user, survives reinstalls: %LOCALAPPDATA%\\MANA on Windows,
    ~/.local/share/mana elsewhere.
  * `package_root()` -- where the `mana` package's .py files are. This is
    the one the agent may rewrite, and the only one where being read-only
    is a feature-losing condition rather than a crash.

Nothing here changes behaviour for `python mana_run.py` in a checkout: the
relative-path defaults keep resolving against the working directory unless
MANA is frozen or MANA_DATA_DIR is set. The desktop app opts in; the
existing CLI and the whole test suite do not have to.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Set by a packaged launcher (or a test) to override the data location.
DATA_DIR_ENV = "MANA_DATA_DIR"

_APP_DIR_NAME = "MANA"


def is_frozen() -> bool:
    """True when running from a PyInstaller (or similar) bundle."""
    return bool(getattr(sys, "frozen", False))


def install_root() -> Path:
    """Folder the program was installed/checked out into.

    Frozen: the directory holding the executable -- NOT `sys._MEIPASS`,
    which in onefile mode is a temporary extraction directory. Anything
    read from `sys._MEIPASS` disappears at exit, which is precisely the
    trap `package_is_writable()` exists to detect.
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def bundle_root() -> Path:
    """Where PyInstaller unpacked bundled data, or the install root.

    Use this for read-only assets shipped with the app (icons, the web UI).
    Never use it for anything MANA writes.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass) if meipass else install_root()


def data_root() -> Path:
    """Per-user writable directory for memory, state, caches and logs.

    Order of precedence: MANA_DATA_DIR, then the platform location, and --
    for a plain development run -- the current working directory, so a
    checkout keeps behaving exactly as it did before this module existed.
    """
    override = os.environ.get(DATA_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if not is_frozen():
        return Path.cwd()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
        return root / _APP_DIR_NAME
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / _APP_DIR_NAME.lower()


def resolve_data_path(path: str | os.PathLike) -> Path:
    """Make a Config path absolute against `data_root()`.

    An absolute path is returned untouched -- that is what the CLI flags
    and tests/conftest.py supply, and rewriting those would break both.
    """
    p = Path(path).expanduser()
    return p if p.is_absolute() else (data_root() / p)


def package_root() -> Path:
    """Directory holding the `mana` package's source files."""
    return Path(__file__).resolve().parent


def package_is_writable() -> bool:
    """Can the agent actually rewrite its own source here?

    Checked by touching a file rather than by reading permission bits:
    Windows ACLs, read-only installs under Program Files and PyInstaller's
    temporary extraction directory all fail in different ways that a
    stat() would not agree on.
    """
    root = package_root()
    probe = root / ".mana_write_probe"
    try:
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def package_is_ephemeral() -> bool:
    """True when the package lives in a directory that will be deleted.

    PyInstaller's onefile mode extracts into `sys._MEIPASS` and removes it
    at exit. Source there is writable -- which makes this the dangerous
    case rather than the obvious one: self-patching appears to succeed and
    is gone by the next launch.

    The discriminator is *where* `_MEIPASS` points, not whether it exists:
    onedir sets it too (to the application's own folder, which is
    permanent), so testing for its presence flagged a perfectly good
    onedir build as ephemeral -- caught by running --self-check on the
    first packaged build. Only an extraction directory outside the
    executable's own folder is temporary.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return False
    extraction = Path(meipass).resolve()
    app_dir = Path(sys.executable).resolve().parent
    if extraction == app_dir:
        return False                      # onedir with --contents-directory .
    try:
        extraction.relative_to(app_dir)
        return False                      # onedir with the default _internal/
    except ValueError:
        pass
    try:
        package_root().relative_to(extraction)
        return True                       # onefile: unpacked to %TEMP%
    except ValueError:
        return False


def sandbox_python() -> str:
    """Interpreter that LocalVerifier should spawn for candidate code.

    In a frozen build `sys.executable` is the application itself, so using
    it would re-launch MANA instead of running the snippet. A packaged
    build ships an embeddable CPython next to the executable; this finds
    it. If it is missing, the caller gets a path that does not exist and
    LocalVerifier reports a normal execution failure -- which is the
    correct outcome, because silently running the wrong binary is worse
    than a visible "sandbox unavailable".
    """
    override = os.environ.get("MANA_SANDBOX_PYTHON", "").strip()
    if override:
        return override
    if not is_frozen():
        return sys.executable
    exe = "python.exe" if os.name == "nt" else "python3"
    for candidate in (install_root() / "python" / exe,
                      install_root() / "python" / "bin" / exe):
        if candidate.exists():
            return str(candidate)
    return str(install_root() / "python" / exe)


def sandbox_python_available() -> bool:
    return Path(sandbox_python()).exists()


def logs_dir() -> Path:
    d = data_root() / "logs"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def status() -> dict:
    """Everything about location in one inspectable dict -- surfaced by
    `--paths-status` and by the desktop app's diagnostics screen, because
    "where is it looking?" is the first question every one of the failures
    in this module's docstring produces."""
    return {
        "frozen": is_frozen(),
        "install_root": str(install_root()),
        "bundle_root": str(bundle_root()),
        "data_root": str(data_root()),
        "package_root": str(package_root()),
        "package_writable": package_is_writable(),
        "package_ephemeral": package_is_ephemeral(),
        "self_patching_possible": package_is_writable() and not package_is_ephemeral(),
        "sandbox_python": sandbox_python(),
        "sandbox_python_available": sandbox_python_available(),
        "data_dir_override": os.environ.get(DATA_DIR_ENV, ""),
    }
