"""
tests/test_console.py — what this program prints has to be readable.

Written against a defect a user hit: the packaged application printed its
Russian messages as cp1251 into a console already set to 65001, so
`--peer show` came out as question marks. The public key it prints was the
whole reason for running it, and it could not be read.
"""
from __future__ import annotations

import io
import sys

from mana import console


class Stream(io.StringIO):
    """A stream that records how it was reconfigured."""

    def __init__(self, encoding: str) -> None:
        super().__init__()
        self._encoding = encoding
        self.calls = []

    @property
    def encoding(self) -> str:
        return self._encoding

    def reconfigure(self, **kwargs):
        self.calls.append(kwargs)
        self._encoding = kwargs.get("encoding", self._encoding)


def test_a_locale_stream_is_switched_to_utf8(monkeypatch):
    out, err = Stream("cp1251"), Stream("cp1251")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    assert console.speak_utf8() is True
    assert out.calls[0]["encoding"] == "utf-8"
    # replace, not strict: a diagnostic that cannot print is worse than
    # one printed imperfectly.
    assert out.calls[0]["errors"] == "replace"
    assert err.calls


def test_a_stream_already_in_utf8_is_left_alone(monkeypatch):
    out = Stream("UTF-8")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", Stream("utf8"))
    assert console.speak_utf8() is False
    assert out.calls == []


def test_a_windowed_build_without_streams_does_not_crash(monkeypatch):
    """A windowed PyInstaller build can have sys.stdout set to None, and
    failing to start over an encoding is not a trade worth making."""
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    assert console.speak_utf8() is False


def test_every_entry_point_calls_it():
    """One implementation, and every route to it. Three copies of a
    two-line fix is how one gets missed and the bug returns on whichever
    path nobody tested."""
    from pathlib import Path
    for name in ("app.py", "mana_run.py", "serve_ui.py"):
        source = Path(name).read_text(encoding="utf-8")
        assert "speak_utf8" in source, name
