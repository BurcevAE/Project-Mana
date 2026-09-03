"""
build_exe.py — package MANA as a Windows application.

Two decisions are baked in here and both are load-bearing:

  * **--onedir, never --onefile.** Onefile extracts everything to a temp
    directory that Windows deletes at exit. `code_evolution.apply_patch`
    would write a real patch into that directory, record it in the
    changelog, and lose it -- leaving a self-improvement log that
    disagrees with the code actually running. `paths.package_is_ephemeral`
    detects it and refuses, but the right answer is to not build that way.

  * **`mana/` ships as data, not as a frozen module.** Compiled-in code
    cannot be rewritten, and rewriting its own source is the thing this
    agent exists to do. app.py puts the shipped directory first on
    sys.path so the imported copy and the patchable copy are the same
    files.

The heavy optional dependencies are excluded on purpose: torch and
sentence-transformers add 2-3 GB for a semantic search that already has a
working TF-IDF fallback (see mana/optional_deps.py, which flags every
capability instead of assuming it). A user who wants them installs them
into the shipped python/ directory; the app notices and uses them.

    python build_exe.py            # console build, for --self-check
    python build_exe.py --windowed # the app
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"
EMBED = ROOT / "python-embed"

EXCLUDED = [
    "torch", "sentence_transformers", "faster_whisper", "sounddevice", "pyttsx3",
    "fitz", "sklearn", "scipy", "matplotlib", "PIL", "pandas", "IPython",
    "pytest", "PyInstaller",
]

#: pywebview reaches its backend through late imports PyInstaller's
#: analyser cannot see, so the Windows one has to be named explicitly.
#: Without this the packaged app raises "no suitable GUI toolkit" on a
#: machine where the source version works perfectly.
HIDDEN = ["webview.platforms.edgechromium", "webview.platforms.winforms",
          "clr_loader", "keyring.backends.Windows"]


def build(windowed: bool) -> int:
    if not EMBED.is_dir():
        print(f"нет каталога {EMBED.name}/ — распакуйте туда embeddable-дистрибутив Python.")
        print("Без него песочница (--run-code, self-improve-code) в собранном приложении не работает.")
        return 1

    for path in (DIST, BUILD):
        shutil.rmtree(path, ignore_errors=True)

    cmd = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
        "--name", "MANA",
        # Both directories are DATA. mana/ must stay as .py files so the
        # agent can patch itself; python/ is the sandbox interpreter that
        # replaces sys.executable in a frozen build.
        "--add-data", f"{ROOT / 'mana'}{';'}mana",
        "--add-data", f"{EMBED}{';'}python",
        # The window shell and its page. Unlike mana/, this one IS frozen
        # into the bundle: nothing rewrites it at runtime, so there is no
        # reason to ship it as loose files. web/ still has to be declared
        # as data -- PyInstaller collects .py, never .html.
        "--add-data", f"{ROOT / 'mana_desktop' / 'web'}{';'}mana_desktop/web",
        # Flat layout: PyInstaller 6 puts collected files under
        # _internal/ by default, which would put mana/ and python/ one
        # level below where paths.install_root() looks for them. Keeping
        # the contents beside the executable makes the folder layout the
        # same thing the code assumes.
        "--contents-directory", ".",
        # PyInstaller would otherwise follow the imports in mana/ and
        # freeze a second, read-only copy of the package into the bundle --
        # which app.py's sys.path order would shadow, but which would also
        # double the size and confuse anyone reading the output.
        "--exclude-module", "mana",
        "--console" if not windowed else "--windowed",
    ]
    for module in EXCLUDED:
        cmd += ["--exclude-module", module]
    if windowed:
        for module in HIDDEN:
            cmd += ["--hidden-import", module]
    cmd.append(str(ROOT / "app.py"))

    print("$", " ".join(cmd[:6]), "...")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        return result.returncode

    app_dir = DIST / "MANA"
    total = sum(f.stat().st_size for f in app_dir.rglob("*") if f.is_file())
    print(f"\nГотово: {app_dir}")
    print(f"Размер: {total / 1e6:.1f} МБ")
    print(f"Проверка: {app_dir / 'MANA.exe'} --self-check")
    return 0


if __name__ == "__main__":
    raise SystemExit(build("--windowed" in sys.argv))
