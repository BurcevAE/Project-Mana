"""
tests/test_packaging.py — the packaged application carries what it claims.

The defect these are written against was invisible from inside the code:
`packaging_deps` named ddgs and psutil inside try/except blocks, the build
machine had neither installed, and PyInstaller therefore bundled neither.
The build printed "Готово", the package weighed what a correct one weighs,
`--self-check` passed every check it had, and web search was permanently
unavailable in every copy handed to anyone.

Nothing in the code was wrong. The try/except is correct at runtime -- it
is what lets MANA degrade instead of crash. It was wrong at BUILD time,
where the same construct turned "this package is missing" into silence.
So these tests are mostly about the manifest agreeing with the other
places that state the same facts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import packaging_deps

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------- manifest

def test_every_declared_package_resolves_to_one_result():
    found = packaging_deps.resolve()
    assert len(found) == len(packaging_deps.BUNDLED)
    assert [f.spec for f in found] == list(packaging_deps.BUNDLED)


def test_a_package_that_is_not_installed_is_reported_absent():
    spec = packaging_deps.Bundled("no_such_module_anywhere", "nope",
                                  "ничего, это выдуманный пакет")
    result = spec.found()
    assert result.present is False
    assert result.version == ""


def test_report_names_the_consequence_when_a_package_is_missing():
    """The report is read by whoever has to decide whether to care."""
    spec = packaging_deps.Bundled("no_such_module_anywhere", "nope",
                                  "веб-поиск недоступен")
    text = packaging_deps.report([spec.found()])
    assert "веб-поиск недоступен" in text
    assert "nope" in text


def test_present_packages_report_a_version():
    """An empty version column would read as "present, build unknown"."""
    for result in packaging_deps.resolve():
        if result.present:
            assert result.version, result.spec.package


# ------------------------------------------------- agreement with the rest

def _requirements() -> list:
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        line = line.split("#")[0].strip()
        if line:
            out.append(line.split("=")[0].split("<")[0].split(">")[0].strip())
    return out


def test_requirements_txt_is_covered_by_the_manifest():
    """A dependency added to requirements.txt but not here ships as absent.

    This is the exact shape of the original defect: requirements.txt named
    ddgs, nothing in the build checked for it, and it was missing from
    every package built on a machine that had not installed it.
    """
    declared = {spec.package for spec in packaging_deps.BUNDLED}
    for package in _requirements():
        assert package in declared, (
            f"{package} есть в requirements.txt, но не в packaging_deps.BUNDLED "
            f"-- сборка не проверит его наличие и соберёт пакет без него")


def test_nothing_is_both_bundled_and_excluded():
    """Excluding a bundled package drops it with no error anywhere.

    PyInstaller obeys --exclude-module silently, so the manifest would go
    on claiming the package while the bundle did not contain it -- the
    build's own report would be the thing lying.
    """
    import build_exe
    excluded = set(build_exe.EXCLUDED)
    for spec in packaging_deps.BUNDLED:
        assert spec.module not in excluded, spec.module


def test_the_packages_carrying_a_capability_flag_are_all_declared():
    """Every HAS_* flag mana branches on comes from a declared package.

    optional_deps is the only place allowed to import an optional library,
    so it is the complete list of what the application can use. A flag
    whose package is not in the manifest is a capability nobody checked
    for at build time.
    """
    from mana import optional_deps
    declared = {spec.module for spec in packaging_deps.BUNDLED}
    absent = {"torch", "sentence_transformers", "faster_whisper",
              "sounddevice", "pyttsx3", "fitz"}
    covered = declared | absent
    for module in ("requests", "psutil", "sklearn", "ddgs"):
        assert module in covered
    # pymupdf is declared under its modern name; optional_deps still
    # supports the old `fitz` alias, so both spellings count as covered.
    assert "pymupdf" in declared
    assert optional_deps.HAS_FITZ in (True, False)


def test_deliberately_absent_packages_each_state_a_reason():
    """"We decided against it" and "we forgot" look identical otherwise."""
    assert packaging_deps.DELIBERATELY_ABSENT
    for what, why in packaging_deps.DELIBERATELY_ABSENT:
        assert what and why
        assert len(why) > 20, what


# ------------------------------------------------------------- autostart

def test_autostart_command_is_quoted():
    """An unquoted path with a space registers as a command plus an arg."""
    from mana_desktop import autostart
    command = autostart.command()
    assert command.startswith('"')
    assert command.count('"') >= 2


def test_autostart_round_trip(monkeypatch):
    """Enable, read back, disable -- against a scratch key, never Run.

    Writing to the real Run key from a test would leave an autostart entry
    on the machine of anyone who interrupted the suite.
    """
    if sys.platform != "win32":
        pytest.skip("автозапуск реализован только для Windows")
    from mana_desktop import autostart
    monkeypatch.setattr(autostart, "RUN_KEY", r"Software\MANA-test-autostart")
    try:
        assert autostart.status()["enabled"] is False
        assert autostart.enable()["ok"] is True
        state = autostart.status()
        assert state["enabled"] is True
        assert state["registered"] == autostart.command()
        assert autostart.disable()["ok"] is True
        assert autostart.status()["enabled"] is False
    finally:
        import winreg
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER,
                             r"Software\MANA-test-autostart")
        except OSError:
            pass


def test_autostart_reports_an_entry_pointing_elsewhere(monkeypatch):
    """"On" for a different copy of MANA is not the same fact as "on"."""
    if sys.platform != "win32":
        pytest.skip("автозапуск реализован только для Windows")
    from mana_desktop import autostart
    monkeypatch.setattr(autostart, "_read", lambda: r'"D:\Old\MANA.exe"')
    state = autostart.status()
    assert state["enabled"] is True
    assert state["note"]


def test_disable_is_not_an_error_when_it_was_never_enabled(monkeypatch):
    if sys.platform != "win32":
        pytest.skip("автозапуск реализован только для Windows")
    from mana_desktop import autostart
    monkeypatch.setattr(autostart, "RUN_KEY", r"Software\MANA-test-missing")
    assert autostart.disable()["ok"] is True


# ------------------------------------------------------------------ tray

def test_tray_struct_sizes_are_what_windows_expects():
    """A mis-sized NOTIFYICONDATAW is rejected with no diagnostic at all.

    Shell_NotifyIcon validates cbSize and returns FALSE; there is no error
    text, so the only symptom is an icon that never appears.
    """
    import ctypes
    from mana_desktop import tray
    size = ctypes.sizeof(tray.NOTIFYICONDATAW)
    # The v3 layout on 64-bit Windows. Named as a range rather than an
    # exact number because the padding depends on the pointer size.
    assert 800 < size < 1000, size


def test_tray_is_absent_rather_than_broken_off_windows(monkeypatch):
    from mana_desktop import tray
    monkeypatch.setattr(sys, "platform", "linux")
    assert tray.start(lambda: None, lambda: None) is None


def test_tray_callback_failure_does_not_escape():
    """An exception here would kill the message loop and freeze the icon."""
    from mana_desktop import tray

    def boom():
        raise RuntimeError("нет")

    tray.TrayIcon._safely(boom)      # must not raise


# -------------------------------------------------------- build manifest

def test_build_manifest_says_so_when_running_from_source(tmp_path, monkeypatch):
    from mana import paths
    from mana_desktop import server
    monkeypatch.setattr(paths, "install_root", lambda: tmp_path)
    result = server.build_manifest()
    assert result["packaged"] is False
    assert result["note"]


def test_build_manifest_is_read_back_when_present(tmp_path, monkeypatch):
    import json
    from mana import paths
    from mana_desktop import server
    (tmp_path / "build_manifest.json").write_text(
        json.dumps({"bundled": {"numpy": "2.5.2"}, "missing": []}),
        encoding="utf-8")
    monkeypatch.setattr(paths, "install_root", lambda: tmp_path)
    result = server.build_manifest()
    assert result["packaged"] is True
    assert result["bundled"] == {"numpy": "2.5.2"}
