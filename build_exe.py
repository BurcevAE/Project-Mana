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

#: Kept OUT of the bundle, each for a reason stated in
#: packaging_deps.DELIBERATELY_ABSENT or because nothing in mana imports
#: it. sklearn, scipy and fitz used to be on this list and should not have
#: been: sklearn carries the learned router and the TF-IDF search that the
#: "there is a fallback" argument was appealing to, scipy is what sklearn
#: imports, and fitz is how a PDF gets read. Excluding them did not save a
#: user from a 3 GB download; it removed three working features from every
#: packaged build.
EXCLUDED = [
    "torch", "sentence_transformers", "faster_whisper", "sounddevice", "pyttsx3",
    "matplotlib", "PIL", "pandas", "IPython", "pytest", "PyInstaller",
]

#: pywebview reaches its backend through late imports PyInstaller's
#: analyser cannot see, so the Windows one has to be named explicitly.
#: Without this the packaged app raises "no suitable GUI toolkit" on a
#: machine where the source version works perfectly.
HIDDEN = ["webview.platforms.edgechromium", "webview.platforms.winforms",
          "clr_loader", "keyring.backends.Windows",
          # ddgs picks its HTTP backend at call time, so the analyser sees
          # no import and the packaged build fails on the first search.
          "ddgs", "ddgs.engines"]

#: Packages whose submodules are pulled in wholesale.
COLLECTED = ["sklearn", "ddgs"]


def build(windowed: bool, allow_missing: bool = False,
          sign_with: str = "") -> int:
    if not EMBED.is_dir():
        # Naming the file and the place: the previous message said what
        # was missing but not where it comes from, which turns a
        # two-minute prerequisite into a search.
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(f"нет каталога {EMBED.name}/ — в нём должен лежать embeddable-дистрибутив Python.")
        print(f"  1. скачайте «Windows embeddable package (64-bit)» для Python {version}:")
        print(f"     https://www.python.org/downloads/windows/")
        print(f"  2. распакуйте архив в {EMBED}")
        print(f"  3. запустите сборку снова")
        print("Без него песочница (--run-code, self-improve-code) в собранном приложении не работает.")
        return 1

    # The build environment is part of the build. A try/except around an
    # import is the right thing at runtime and the wrong thing here: it
    # let a machine without ddgs installed produce a package that printed
    # "Готово", weighed the expected number of megabytes, passed
    # --self-check, and had no web search.
    import packaging_deps
    found = packaging_deps.resolve()
    print("Зависимости, которые пакет обязан нести:")
    print(packaging_deps.report(found))
    absent = [f for f in found if not f.present]
    if absent and not allow_missing:
        print()
        print("Сборка остановлена: этих пакетов нет в окружении сборки,")
        print("и без них собранное приложение потеряет перечисленное выше.")
        print()
        print("  " + sys.executable + " -m pip install " +
              " ".join(f.spec.package for f in absent))
        print()
        print("Если это осознанное решение: python build_exe.py --allow-missing")
        return 1
    if absent:
        print()
        print("--allow-missing: собираю без " +
              ", ".join(f.spec.package for f in absent))
    print()
    print("Сознательно не входит в пакет:")
    for name, why in packaging_deps.DELIBERATELY_ABSENT:
        print(f"  {name}: {why}")
    print()

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
    for module in HIDDEN:
        cmd += ["--hidden-import", module]
    # Collected wholesale rather than left to the analyser: both reach
    # large parts of themselves through lazy imports, and the failure mode
    # is not a build error but a ModuleNotFoundError inside a feature the
    # user tries months later.
    for package in COLLECTED:
        cmd += ["--collect-submodules", package]
    cmd.append(str(ROOT / "app.py"))

    print("$", " ".join(cmd[:6]), "...")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        return result.returncode

    app_dir = DIST / "MANA"
    total = sum(f.stat().st_size for f in app_dir.rglob("*") if f.is_file())

    # Shipped beside the executable so that "what is in this build?"
    # has an answer which does not require running it. Nothing reads
    # this file at runtime; a person reading a bug report does.
    import json
    from datetime import datetime, timezone
    (app_dir / "build_manifest.json").write_text(json.dumps({
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "windowed": windowed,
        "bundled": {f.spec.package: f.version for f in found if f.present},
        "missing": [f.spec.package for f in found if not f.present],
        "deliberately_absent": [
            {"what": name, "why": why}
            for name, why in packaging_deps.DELIBERATELY_ABSENT],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    if sign_with:
        code = sign(app_dir / "MANA.exe", sign_with)
        if code != 0:
            return code

    print(f"\nГотово: {app_dir}")
    print(f"Размер: {total / 1e6:.1f} МБ")
    print(f"Проверка: {app_dir / 'MANA.exe'} --self-check")
    return 0


def sign(target: Path, certificate: str) -> int:
    """Authenticode-sign one file with signtool.

    What a signature buys, stated plainly because it is routinely
    oversold: it proves the file has not been altered since it left
    the named signer, and it puts a publisher name in the UAC prompt
    in place of "Неизвестный издатель". It does NOT by itself stop
    SmartScreen warning about a fresh download -- that clears once the
    certificate has accumulated reputation, and immediately only with
    an EV certificate. A self-signed certificate buys neither: it is
    untrusted on every machine except the one that created it, so it
    is useful for testing this code path and for nothing else.

    Unsigned is a legitimate choice for a tool its author runs on
    their own machines. It is a bad surprise for anyone else, which is
    why the installer states which of the two a given build is.
    """
    tool = shutil.which("signtool") or shutil.which("signtool.exe")
    if not tool:
        print("signtool не найден. Он входит в Windows SDK; добавьте в PATH")
        print("  C:/Program Files (x86)/Windows Kits/10/bin/<версия>/x64")
        return 1
    cmd = [tool, "sign", "/fd", "SHA256",
           # RFC-3161 timestamp. Without it every signed copy stops
           # verifying on the day the certificate expires, including
           # the ones already installed on other people's machines.
           "/tr", "http://timestamp.digicert.com", "/td", "SHA256"]
    cmd += (["/f", certificate] if certificate.lower().endswith((".pfx", ".p12"))
            else ["/n", certificate])
    cmd.append(str(target))
    print("$", " ".join(cmd[:4]), "...", target.name)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    argv = sys.argv[1:]
    cert = ""
    if "--sign" in argv:
        index = argv.index("--sign")
        cert = argv[index + 1] if index + 1 < len(argv) else ""
        if not cert:
            print("--sign требует .pfx-файл или имя субъекта сертификата")
            raise SystemExit(2)
    raise SystemExit(build("--windowed" in argv,
                           allow_missing="--allow-missing" in argv,
                           sign_with=cert))
