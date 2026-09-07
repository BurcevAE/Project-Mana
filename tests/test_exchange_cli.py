"""
tests/test_exchange_cli.py — the exchange has a command line, and one.

Written against a gap found while checking two real installations: the
packaged MANA exposed only --self-check, --cli and the window, so an
installation that was not also a source checkout could not export or
import anything. The federation machinery existed and nothing on a user's
machine could reach it -- the same "built and connected to nothing"
pattern this project keeps finding, one level out, at distribution.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mana.cognition import exchange


@pytest.fixture
def instance(tmp_path, monkeypatch):
    """An installation with its own state directory and its own key."""
    from mana import paths
    from mana.core import identity

    monkeypatch.setattr(paths, "data_root", lambda: tmp_path)
    monkeypatch.setenv(identity.INSTANCE_ENV, "cli-test")
    identity.reset_cache()
    return tmp_path


def test_the_queue_lands_beside_the_agent_state(instance):
    assert exchange.default_queue_path().parent == instance


def test_show_runs_on_an_empty_queue(instance, capsys):
    """The first thing anybody types, on a machine that has done nothing
    yet. It must report emptiness rather than fail on it."""
    assert exchange.command_line([]) == 0
    out = capsys.readouterr().out
    assert "экземпляр" in out
    assert "воспроизведений нет" in out


def test_export_then_import_round_trips(instance, tmp_path, capsys):
    queue = exchange.Queue(exchange.default_queue_path())
    record = queue.record_hypothesis(
        "create_program_template",
        {"name": "probe", "steps": ["OBSERVE", "ANSWER"]})
    queue.record_report(record["hypothesis_id"], "ACCEPTED", trials=40)

    bundle = tmp_path / "пакет.json"
    assert exchange.command_line(["export", str(bundle)]) == 0
    assert bundle.is_file()

    raw = json.loads(bundle.read_text(encoding="utf-8"))
    assert len(raw["hypotheses"]) == 1
    assert len(raw["reports"]) == 1
    # Signed on the way out, so the receiver can check who it came from.
    assert raw["reports"][0]["signature"]


def test_export_without_a_filename_refuses(instance, capsys):
    assert exchange.command_line(["export"]) == 2
    assert "нужно имя файла" in capsys.readouterr().out


def test_importing_a_foreign_file_reports_rather_than_crashes(
        instance, tmp_path, capsys):
    alien = tmp_path / "чужое.json"
    alien.write_text(json.dumps({"format": "что-то другое"}), encoding="utf-8")
    assert exchange.command_line(["import", str(alien)]) == 1
    assert "не принят" in capsys.readouterr().out


def test_the_import_says_verdicts_are_not_adopted(instance, tmp_path, capsys):
    """The sentence matters as much as the code.

    Someone importing a bundle needs to know that a foreign ACCEPTED did
    not just become true here, or they will read the queue as a set of
    conclusions.
    """
    bundle = tmp_path / "b.json"
    exchange.export_bundle(bundle, [exchange.Hypothesis(
        "create_program_template", {"name": "t", "steps": ["OBSERVE", "ANSWER"]})])
    exchange.command_line(["import", str(bundle)])
    out = capsys.readouterr().out
    assert "НЕ приняты как истина" in out


def test_an_unknown_command_is_refused_with_the_list(instance, capsys):
    assert exchange.command_line(["выгрузи-всё"]) == 2
    assert "export" in capsys.readouterr().out


def test_the_packaged_app_reaches_it(instance):
    """The gap this closes: an installed MANA had no route to the
    exchange at all."""
    import inspect

    import app

    source = inspect.getsource(app.main)
    assert "--exchange" in source
    assert "command_line" in source


def test_the_script_is_a_wrapper_and_not_a_second_implementation():
    """Two implementations of one thing is how they come to disagree, and
    this repository already carries a test saying so about another pair."""
    source = Path("scripts/run_exchange.py").read_text(encoding="utf-8")
    assert "command_line" in source
    # The display logic must live in the module, not here.
    assert "что где воспроизвелось" not in source
    assert "replication(" not in source
