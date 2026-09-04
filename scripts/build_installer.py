#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_installer.py — compile installer/mana.iss into a setup .exe.

Runs three checks before invoking Inno Setup, each of which corresponds to
a way a packaged build has actually gone out wrong:

  1. dist\\MANA exists and holds MANA.exe -- otherwise the compiler
     happily produces an installer for an empty directory.
  2. build_manifest.json lists no missing packages. build_exe.py already
     refuses to build with a dependency absent, but --allow-missing can
     override it, and an installer is the wrong place for that override
     to pass unnoticed.
  3. The build is the windowed one. A console build opens a black
     terminal behind the window on every launch; it is the right thing to
     ship to yourself and the wrong thing to hand anyone else.

Usage:
    python build_exe.py --windowed
    python scripts/build_installer.py
    python scripts/build_installer.py --sign "путь\\к\\cert.pfx"
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DIST = ROOT / "dist"
APP_DIR = DIST / "MANA"
SCRIPT = ROOT / "installer" / "mana.iss"

#: Where Inno Setup lands. The LocalAppData entry is first because it is
#: what `winget install JRSoftware.InnoSetup` produces -- winget installs
#: per-user by default, so the machine-wide paths are the less likely
#: ones, not the other way round. Searched only after PATH, so a
#: deliberately chosen copy still wins.
def _candidates() -> "list[Path]":
    import os
    roots = [os.environ.get("LOCALAPPDATA", ""),
             os.environ.get("ProgramFiles(x86)", ""),
             os.environ.get("ProgramFiles", "")]
    out = []
    for index, root in enumerate(roots):
        if not root:
            continue
        base = Path(root) / ("Programs" if index == 0 else "")
        # Inno Setup 7 exists; the directory name carries the major
        # version, so both are looked for rather than one hard-coded.
        for version in ("Inno Setup 6", "Inno Setup 7"):
            out.append(base / version / "ISCC.exe")
    return out


def find_compiler() -> str:
    found = shutil.which("ISCC") or shutil.which("ISCC.exe")
    if found:
        return found
    for candidate in _candidates():
        if candidate.is_file():
            return str(candidate)
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sign", default="",
                        help=".pfx-файл или имя субъекта сертификата")
    args = parser.parse_args()

    if not (APP_DIR / "MANA.exe").is_file():
        print(f"нет {APP_DIR / 'MANA.exe'}")
        print("  сначала: python build_exe.py --windowed")
        return 1

    manifest_path = APP_DIR / "build_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"не читается {manifest_path}: {exc}")
        print("  пересоберите: python build_exe.py --windowed")
        return 1

    if manifest.get("missing"):
        print("В собранном пакете нет: " + ", ".join(manifest["missing"]))
        print("Это следы --allow-missing. Установщик с такой сборкой раздаст")
        print("приложение без этих возможностей и ничего об этом не скажет.")
        return 1

    if not manifest.get("windowed"):
        print("Собрана консольная версия: при каждом запуске за окном будет")
        print("открываться чёрное окно терминала.")
        print("  пересоберите: python build_exe.py --windowed")
        return 1

    compiler = find_compiler()
    if not compiler:
        print("не найден ISCC.exe — компилятор Inno Setup 6.")
        print("  скачать: https://jrsoftware.org/isdl.php")
        print("  winget install --id JRSoftware.InnoSetup -e")
        print("  искали здесь:")
        for candidate in _candidates():
            print(f"    {candidate}")
        return 1

    from mana.version import PRODUCT_VERSION

    # Signed before packaging, not after: the installer embeds a copy of
    # MANA.exe, so signing the .exe afterwards would leave the copy inside
    # the installer unsigned. The setup file itself is signed below.
    if args.sign:
        from build_exe import sign
        code = sign(APP_DIR / "MANA.exe", args.sign)
        if code != 0:
            return code

    total = sum(f.stat().st_size for f in APP_DIR.rglob("*") if f.is_file())
    print(f"источник: {APP_DIR}  ({total / 1e6:.0f} МБ до сжатия)")
    print(f"в пакете: {', '.join(sorted(manifest.get('bundled', {})))}")
    print(f"версия:   {PRODUCT_VERSION}")
    print()

    command = [compiler, f"/DMyAppVersion={PRODUCT_VERSION}", str(SCRIPT)]
    print("$", " ".join(command))
    result = subprocess.run(command, cwd=str(ROOT))
    if result.returncode != 0:
        return result.returncode

    setup = DIST / f"MANA-{PRODUCT_VERSION}-setup.exe"
    if args.sign and setup.is_file():
        from build_exe import sign
        code = sign(setup, args.sign)
        if code != 0:
            return code

    print()
    if setup.is_file():
        print(f"Готово: {setup}")
        print(f"Размер: {setup.stat().st_size / 1e6:.1f} МБ")
    if not args.sign:
        # Said plainly rather than left to be discovered by whoever runs
        # the file: an unsigned installer produces a SmartScreen warning
        # naming an unknown publisher, and a user who was not warned about
        # that reasonably reads it as the program being unsafe.
        print()
        print("Установщик не подписан: Windows покажет предупреждение")
        print("SmartScreen и «Неизвестный издатель». Для подписи нужен")
        print("сертификат Authenticode: python scripts/build_installer.py "
              "--sign cert.pfx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
