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

import ctypes
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

    # What the build actually carries. Until this was here, a package
    # built in an environment missing ddgs passed every check above and
    # shipped with web search permanently unavailable -- the failure was
    # visible only to a user who asked MANA to search the web and got
    # told it could not.
    #
    # The bar is deliberately different in the two cases. A frozen build
    # either bundled a declared package or lost it, so any absence is a
    # packaging defect. A source checkout is a developer's machine, where
    # the manifest describes what the INSTALLER carries and not what the
    # checkout must have; there, only the packages MANA cannot start
    # without are a failure.
    import packaging_deps
    found = packaging_deps.resolve()
    report["dependencies"] = {f.spec.package: (f.version if f.present else None)
                              for f in found}
    report["deliberately_absent"] = [name for name, _ in
                                     packaging_deps.DELIBERATELY_ABSENT]
    report["checks"]["required_dependencies_present"] = not [
        f for f in found if not f.present and f.spec.required]

    # The flags mana actually branches on, which are not the same claim as
    # "the package imports". `import sklearn` succeeding says nothing
    # about `from sklearn.feature_extraction.text import TfidfVectorizer`
    # -- and TfidfVectorizer is what knowledge.py needs, so a bundle that
    # collected the top-level package and none of its submodules would
    # pass a package check and still have no semantic search.
    from mana import optional_deps
    capabilities = {
        "semantic_search": optional_deps.HAS_SKLEARN,
        "web_search": optional_deps.HAS_WEB,
        "pdf_reading": optional_deps.HAS_FITZ,
        "hardware_detection": optional_deps.HAS_PSUTIL,
        "llm_providers": optional_deps.HAS_REQUESTS,
    }
    report["capabilities"] = capabilities

    if getattr(sys, "frozen", False):
        # A frozen build either bundled a declared package or lost it, so
        # either absence is a packaging defect. A source checkout is a
        # developer's machine, where this manifest describes what the
        # INSTALLER carries and not what the checkout must have.
        report["checks"]["bundled_dependencies_present"] = not [
            f for f in found if not f.present]
        report["checks"]["bundled_capabilities_work"] = all(capabilities.values())

    ok = all(report["checks"].values())
    report["ok"] = ok
    _deliver_self_check(report, ok)
    return 0 if ok else 1


def _deliver_self_check(report: dict, ok: bool) -> None:
    """Print the report, and make sure it lands somewhere when nothing reads
    stdout.

    The installer puts a "Диагностика MANA" shortcut in the Start menu
    pointing at --self-check. A windowed build launched from a shortcut has
    no console: the JSON goes to a handle nobody is reading, the process
    exits, and from the user's side the shortcut did nothing at all. So the
    report is also written to a file beside the agent's data, and a message
    box says whether it passed and where to look.

    stdout still gets it when there is a stdout -- redirecting to a file
    from a terminal keeps working exactly as before.
    """
    import json
    text = json.dumps(report, ensure_ascii=False, indent=2)
    try:
        print(text)
        printed = True
    except Exception:
        # A windowed build can have sys.stdout set to None, in which case
        # print raises rather than discarding.
        printed = False

    from mana import paths
    written = ""
    try:
        target = Path(paths.data_root()) / "self_check.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        written = str(target)
    except Exception:
        pass

    # Exactly one condition: the report could not be printed. Two cleverer
    # rules were tried and both were wrong. isatty() is false for
    # `MANA.exe --self-check > out.json`, which is a script and must not
    # get a modal dialog. GetConsoleWindow() is zero under MinTTY, where a
    # person IS watching -- that version blocked this very session on a
    # dialog nobody asked for. A windowed build launched from a shortcut
    # has sys.stdout set to None, print raises, and that is the whole of
    # the case this exists for.
    if printed or sys.platform != "win32":
        return
    failed = [name for name, value in report["checks"].items() if not value]
    summary = "Все проверки пройдены." if ok else "Не пройдено: " + ", ".join(failed)
    if written:
        summary += f"\n\nПодробности: {written}"
    try:
        ctypes.windll.user32.MessageBoxW(
            None, summary, "Диагностика MANA", 0x40 if ok else 0x10)
    except Exception:
        pass


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
