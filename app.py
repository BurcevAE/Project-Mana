"""
app.py — entry point for the packaged MANA application.

The executable is a launcher, not the agent. The `mana` package ships
beside it as ordinary .py files (PyInstaller `--add-data "mana;mana"`,
never frozen into the bundle), for one reason: `code_evolution` rewrites
those files, and code compiled into a bundle cannot be rewritten. Freezing
the package would not raise an error -- it would make self-improvement
silently stop working, which is worse.

So this file does three things before anything else runs:

  1. puts the shipped `mana/` directory at the front of sys.path, so the
     patchable copy is the one that gets imported;
  2. reports whether self-patching and the sandbox interpreter are
     actually available in this installation, instead of letting the
     agent discover it mid-cycle;
  3. hands off to the CLI (--console builds and any command-line use) or
     to the window.

Run modes:
    MANA.exe                 -> window
    MANA.exe --cli <args>    -> exactly the old command line
    MANA.exe --self-check    -> verify a packaged build can still do the
                                three things freezing usually breaks
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Imported for its side effect on PyInstaller's dependency analysis, not
# for anything it defines: `mana` is excluded from that analysis (see
# build_exe.py), so the standard-library modules it needs would otherwise
# never be collected. Removing this line breaks the packaged build only --
# a source run is unaffected, which is exactly why it is easy to delete by
# accident.
import packaging_deps  # noqa: F401


def _bootstrap_package_path() -> Path:
    """Import `mana` from the shipped source directory, not from a bundle.

    In a frozen build PyInstaller would happily import a bundled copy of
    the package; that copy is read-only and, in onefile mode, temporary.
    Putting the shipped directory first makes "the code MANA runs" and
    "the code MANA can patch" the same files -- the invariant the whole
    self-improvement design rests on.
    """
    here = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) \
        else Path(__file__).resolve().parent
    package = here / "mana"
    if package.is_dir() and str(here) not in sys.path[:1]:
        sys.path.insert(0, str(here))
    return here


APP_DIR = _bootstrap_package_path()


def self_check() -> int:
    """Prove a packaged build kept the capabilities freezing usually eats.

    This is the milestone worth running first after any packaging change:
    if these three pass in the .exe, the risky part of the migration is
    done and everything after it is ordinary application work.
    """
    import json
    from mana import code_evolution, paths
    from mana.config import Config
    from mana.verifier import LocalVerifier

    report = {"paths": paths.status(), "checks": {}}

    cfg = Config()
    cfg.ensure_dirs()
    report["checks"]["state_paths_absolute"] = all(
        Path(getattr(cfg, f)).is_absolute() for f in Config.STATE_PATH_FIELDS)

    cfg.local_exec_enabled = True
    sandbox = LocalVerifier(cfg).verify_code("print(6 * 7)")
    report["checks"]["sandbox_runs_code"] = bool(sandbox.get("ok") and "42" in sandbox.get("stdout", ""))
    report["sandbox"] = {k: sandbox.get(k) for k in ("ok", "error", "stdout", "sandbox_missing")}

    patching = code_evolution.self_patching_available()
    report["checks"]["self_patching_possible"] = bool(patching.get("ok"))
    report["self_patching"] = patching

    arithmetic = LocalVerifier(cfg).verify_expression("17*23")
    report["checks"]["arithmetic_verifier"] = bool(arithmetic.get("ok") and arithmetic.get("value") == 391)

    ok = all(report["checks"].values())
    report["ok"] = ok
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


def main() -> int:
    argv = sys.argv[1:]

    if argv and argv[0] == "--self-check":
        return self_check()

    if argv and argv[0] == "--cli":
        sys.argv = [sys.argv[0]] + argv[1:]
        from mana.cli import main as cli_main
        return cli_main()

    # No arguments: the window. Falls back to the CLI's own help rather
    # than crashing when the GUI dependency is absent, so a console build
    # without pywebview is still a usable program.
    try:
        from mana_desktop.window import run_window
    except Exception as exc:
        print(f"Оконный режим недоступен: {exc}")
        print("Запустите с --cli для командной строки или --self-check для диагностики.")
        return 2
    return run_window()


if __name__ == "__main__":
    raise SystemExit(main())
