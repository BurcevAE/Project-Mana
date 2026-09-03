"""
tests/test_desktop.py — the assumptions that break when MANA stops being
a script run from its own folder.

Every failure guarded here is silent by nature: MANA keeps running and
produces wrong behaviour rather than an error. That is why they are worth
tests even though none of them can happen in a development checkout --
they happen in the packaged application, where nobody is watching a
console.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from mana import events, paths


# ---------------------------------------------------------------------------
# where things live
# ---------------------------------------------------------------------------

def test_absolute_paths_are_never_rewritten(tmp_path):
    """The CLI flags and tests/conftest.py both pass absolute paths.
    Anchoring those to the data root would silently relocate a user's
    explicitly chosen database."""
    absolute = tmp_path / "explicit" / "memory.sqlite3"
    assert paths.resolve_data_path(absolute) == absolute


def test_relative_paths_follow_the_data_root(monkeypatch, tmp_path):
    """Config's defaults are relative ("mana_memory/mana_memory.sqlite3").
    Started from a shortcut, the working directory is arbitrary -- which is
    how MANA would open an empty database and look like it lost its
    memory."""
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "state"))
    resolved = paths.resolve_data_path("mana_memory/mana_memory.sqlite3")
    assert resolved == tmp_path / "state" / "mana_memory" / "mana_memory.sqlite3"
    assert resolved.is_absolute()


def test_development_run_keeps_resolving_against_the_cwd(monkeypatch, tmp_path):
    """The migration must not change `python mana_run.py` in a checkout."""
    monkeypatch.delenv(paths.DATA_DIR_ENV, raising=False)
    monkeypatch.setattr(paths, "is_frozen", lambda: False)
    monkeypatch.chdir(tmp_path)
    assert paths.resolve_data_path("mana_v3_4_state.pkl") == tmp_path / "mana_v3_4_state.pkl"


def test_frozen_run_uses_localappdata(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.DATA_DIR_ENV, raising=False)
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    root = paths.data_root()
    assert root.name == "MANA"
    assert str(tmp_path) in str(root)


def test_ensure_dirs_settles_every_state_path(monkeypatch, tmp_path):
    """Resolution is written back onto Config, so MemoryManager,
    ExperienceDB and LocalVerifier all see one absolute path instead of
    each re-deriving it from the working directory."""
    from mana.config import Config
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "state"))
    cfg = Config()
    cfg.ensure_dirs()
    for field in Config.STATE_PATH_FIELDS:
        value = getattr(cfg, field)
        assert Path(value).is_absolute(), f"{field} stayed relative: {value!r}"
        assert str(tmp_path) in value, f"{field} did not follow the data root: {value!r}"


def test_state_path_fields_covers_every_written_path():
    """Regression guard for the list itself: the old ensure_dirs() loop
    omitted local_exec_workdir, so the sandbox directory was created
    somewhere else entirely."""
    from mana.config import Config
    assert "local_exec_workdir" in Config.STATE_PATH_FIELDS
    assert "memory_root" in Config.STATE_PATH_FIELDS


# ---------------------------------------------------------------------------
# the sandbox interpreter
# ---------------------------------------------------------------------------

def test_sandbox_python_is_the_interpreter_in_a_checkout(monkeypatch):
    monkeypatch.delenv("MANA_SANDBOX_PYTHON", raising=False)
    monkeypatch.setattr(paths, "is_frozen", lambda: False)
    assert paths.sandbox_python() == sys.executable


def test_frozen_sandbox_never_points_at_the_application(monkeypatch, tmp_path):
    """The bug this exists for: with `sys.executable` in a frozen build,
    "verify this code" would have launched MANA.exe again and reported
    whatever the agent printed as the result of the snippet."""
    monkeypatch.delenv("MANA_SANDBOX_PYTHON", raising=False)
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setattr(paths, "install_root", lambda: tmp_path)
    chosen = paths.sandbox_python()
    assert Path(chosen) != Path(sys.executable)
    assert "python" in Path(chosen).name.lower()


def test_verifier_refuses_when_the_sandbox_interpreter_is_missing(isolated_config, monkeypatch, tmp_path):
    """A missing sandbox must look like a missing sandbox. Reporting a pass
    would make every generated test vacuously succeed."""
    from mana.verifier import LocalVerifier
    monkeypatch.setenv("MANA_SANDBOX_PYTHON", str(tmp_path / "nope" / "python.exe"))
    isolated_config.local_exec_enabled = True
    result = LocalVerifier(isolated_config).verify_code("print(1)")
    assert result["ok"] is False
    assert result.get("sandbox_missing") is True
    assert "not found" in result["error"]


def test_verifier_still_runs_normally_with_a_real_interpreter(isolated_config, monkeypatch):
    monkeypatch.delenv("MANA_SANDBOX_PYTHON", raising=False)
    from mana.verifier import LocalVerifier
    isolated_config.local_exec_enabled = True
    result = LocalVerifier(isolated_config).verify_code("print('ok')")
    assert result["ok"] is True
    assert "ok" in result["stdout"]


# ---------------------------------------------------------------------------
# self-patching under a bundle
# ---------------------------------------------------------------------------

def test_self_patching_is_refused_from_a_temporary_extraction_dir(monkeypatch):
    """PyInstaller's onefile mode extracts the package to a directory it
    deletes at exit. Writing there succeeds, the changelog records it, and
    the change is gone by the next launch -- a self-improvement log that
    disagrees with the running code."""
    from mana import code_evolution
    monkeypatch.setattr(paths, "package_is_ephemeral", lambda: True)
    verdict = code_evolution.self_patching_available()
    assert verdict["ok"] is False
    assert "onefile" in verdict["reason"]


def test_self_patching_is_refused_when_the_install_is_read_only(monkeypatch):
    from mana import code_evolution
    monkeypatch.setattr(paths, "package_is_ephemeral", lambda: False)
    monkeypatch.setattr(paths, "package_is_writable", lambda: False)
    verdict = code_evolution.self_patching_available()
    assert verdict["ok"] is False
    assert "not writable" in verdict["reason"]


def test_apply_patch_refuses_rather_than_recording_a_lie(monkeypatch):
    """The check is repeated inside apply_patch, not only at startup: an
    install can lose write access between launch and acceptance, and the
    changelog must never claim a change that did not land."""
    from mana import code_evolution
    monkeypatch.setattr(paths, "package_is_ephemeral", lambda: True)
    result = code_evolution.apply_patch(
        "local_fallback", "def _local_fallback(task: str) -> str:\n    return 'x'\n",
        evaluation={}, decision={"accepted": True}, instruction="test")
    assert result["applied"] is False
    assert "onefile" in result["reason"]


def test_paths_status_answers_can_mana_still_improve_itself():
    status = paths.status()
    assert status["self_patching_possible"] == (
        status["package_writable"] and not status["package_ephemeral"])
    assert "sandbox_python" in status


# ---------------------------------------------------------------------------
# the event bus
# ---------------------------------------------------------------------------

def test_emit_survives_a_sink_that_raises():
    """An output channel that can kill the agent is worse than one that
    drops a line: every call site here is reporting on work that already
    finished."""
    bus = events.EventBus()
    seen = []

    def broken(_event):
        raise RuntimeError("sink exploded")

    bus.subscribe(broken)
    bus.subscribe(seen.append)
    bus.emit(events.STATUS, "первая строка")
    bus.emit(events.STATUS, "вторая строка")
    assert [e.text for e in seen] == ["первая строка", "вторая строка"]


def test_a_broken_sink_is_dropped_not_retried():
    bus = events.EventBus()
    calls = []

    def broken(_event):
        calls.append(1)
        raise RuntimeError("boom")

    bus.subscribe(broken)
    for _ in range(5):
        bus.emit(events.STATUS, "x")
    assert len(calls) == 1, "a failing sink must be removed, not called every time"


def test_late_subscriber_can_replay_the_startup_banner():
    """The desktop window opens after the agent has constructed itself."""
    bus = events.EventBus()
    bus.emit(events.BANNER, "MANA v5.10.1")
    seen = []
    bus.subscribe(seen.append, replay=True)
    assert [e.text for e in seen] == ["MANA v5.10.1"]


def test_write_console_survives_an_encoding_it_cannot_handle(monkeypatch, capsys):
    """The observed crash: a warning containing an emoji raised
    UnicodeEncodeError under cp1251, so a stale state file became a fatal
    error inside the handler reporting it."""
    class NarrowStream:
        encoding = "cp1251"

        def __init__(self):
            self.written = []
            self.strict = True

        def write(self, text):
            if self.strict and any(ord(c) > 0x400 and ord(c) < 0x410 or ord(c) > 0x2000 for c in text):
                self.strict = False
                raise UnicodeEncodeError("charmap", text, 0, 1, "cannot encode")
            self.written.append(text)

        def flush(self):
            pass

    stream = NarrowStream()
    monkeypatch.setattr(sys, "stdout", stream)
    events.write_console("⚠️ Не удалось восстановить state")
    assert stream.written, "the fallback path must still produce output"


def test_write_console_is_silent_when_there_is_no_stdout(monkeypatch):
    """A --windowed build has no console at all."""
    monkeypatch.setattr(sys, "stdout", None)
    events.write_console("что-нибудь")   # must not raise


def test_agent_startup_emits_a_banner_event_instead_of_printing(isolated_config, capsys):
    """15 print() calls became one event carrying the same facts as data --
    which is what lets the window render a status panel rather than parse
    lines back out of a text blob."""
    from mana import ManaAgent
    seen = []
    sink = events.subscribe(seen.append)
    try:
        agent = ManaAgent(isolated_config)
    finally:
        events.unsubscribe(sink)
    try:
        banners = [e for e in seen if e.kind == events.BANNER]
        assert len(banners) == 1
        assert "MANA v" in banners[0].text
        assert banners[0].data["session"] == agent.session_id
        assert "paths" in banners[0].data
    finally:
        try:
            agent.persistent_memory.close()
            agent.experience.close()
        except Exception:
            pass


def test_onedir_build_is_not_mistaken_for_a_temporary_extraction(monkeypatch, tmp_path):
    """Caught by running --self-check on the first packaged build: onedir
    sets sys._MEIPASS too (to the app's own folder), so testing merely for
    its presence declared a perfectly permanent install ephemeral and
    disabled self-improvement in exactly the build that supports it."""
    app_dir = tmp_path / "MANA"
    (app_dir / "mana").mkdir(parents=True)
    monkeypatch.setattr(sys, "_MEIPASS", str(app_dir), raising=False)
    monkeypatch.setattr(sys, "executable", str(app_dir / "MANA.exe"), raising=False)
    monkeypatch.setattr(paths, "package_root", lambda: app_dir / "mana")
    assert paths.package_is_ephemeral() is False


def test_onedir_default_internal_layout_is_also_permanent(monkeypatch, tmp_path):
    app_dir = tmp_path / "MANA"
    internal = app_dir / "_internal"
    (internal / "mana").mkdir(parents=True)
    monkeypatch.setattr(sys, "_MEIPASS", str(internal), raising=False)
    monkeypatch.setattr(sys, "executable", str(app_dir / "MANA.exe"), raising=False)
    monkeypatch.setattr(paths, "package_root", lambda: internal / "mana")
    assert paths.package_is_ephemeral() is False


def test_onefile_extraction_outside_the_app_folder_is_ephemeral(monkeypatch, tmp_path):
    """The real onefile shape: an extraction dir under %TEMP%, nowhere
    near the .exe."""
    app_dir = tmp_path / "Downloads"
    extraction = tmp_path / "Temp" / "_MEI123456"
    (extraction / "mana").mkdir(parents=True)
    app_dir.mkdir(parents=True)
    monkeypatch.setattr(sys, "_MEIPASS", str(extraction), raising=False)
    monkeypatch.setattr(sys, "executable", str(app_dir / "MANA.exe"), raising=False)
    monkeypatch.setattr(paths, "package_root", lambda: extraction / "mana")
    assert paths.package_is_ephemeral() is True
