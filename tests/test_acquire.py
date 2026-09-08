"""Acquiring an oracle, and refusing one that cannot be trusted.

The load-bearing test is `test_a_subtly_wrong_engine_is_refused`. A
verification suite that only ever passes is worth nothing, and a rules
engine that is quietly incorrect is a **lying oracle**: every measurement
built on it is poisoned and nothing downstream can tell. So the suite is
tested against an engine that is wrong in exactly the way a real one
would be -- one special rule, silently missing.
"""
from __future__ import annotations

import sys

import pytest

from mana import acquire
from mana.acquire import (VERIFIED, REFUSED, ABSENT, CheckResult, Capability,
                          Provider, ConsentRequired, verify_chess_rules)

# The provider lives where acquisitions land, not on the default import
# path -- so the suite has to look there before deciding it is absent.
# Without this the most important test here would silently skip on any
# machine that had actually acquired the capability.
acquire.ensure_importable()
chess = pytest.importorskip("chess", reason="провайдер не приобретён")


# --------------------------------------------------------------------------
# the truth must not come from the thing being tested
# --------------------------------------------------------------------------

def test_the_published_numbers_are_literals_in_the_module():
    """If these were computed by the provider, the suite would be asking
    the library whether it agrees with itself."""
    assert acquire.PERFT_START == {1: 20, 2: 400, 3: 8902, 4: 197281}
    assert acquire.PERFT_KIWIPETE == {1: 48, 2: 2039, 3: 97862}


def test_perft_is_counted_here_not_asked_for():
    """python-chess ships its own perft. Asking a provider for the number
    it is being graded on is grading its own homework."""
    import inspect

    source = inspect.getsource(acquire)
    # It walks `legal_moves` itself and never asks the provider for a
    # node count. (`_perft` is recursive, so its own name appears -- what
    # must not appear is a call to somebody else's.)
    assert "for move in board.legal_moves" in source
    assert ".perft(" not in source


def test_a_real_provider_passes_every_check():
    checks = verify_chess_rules(chess, deep=True)
    assert checks
    failed = [c.name for c in checks if not c.ok]
    assert failed == []


def test_the_checks_are_not_vacuous():
    """A suite that passes without doing anything is worse than none.
    perft(4) is 197281 nodes; it cannot come back instantly or empty."""
    checks = {c.name: c for c in verify_chess_rules(chess, deep=True)}
    assert checks["perft(4) из начальной позиции"].got == 197281
    assert checks["perft(3) Kiwipete"].got == 97862


# --------------------------------------------------------------------------
# and it must catch an engine that is wrong
# --------------------------------------------------------------------------

class _NoEnPassant:
    """A chess module that is right about everything except one rule.

    This is what a broken provider actually looks like: not obviously
    unusable, just quietly missing a special case. It passes a casual
    look and poisons every game played on it.
    """

    def __init__(self) -> None:
        self.Board = self._make_board()

    @staticmethod
    def _make_board():
        real = chess.Board

        class Board(real):
            @property
            def legal_moves(self):
                return [m for m in super().legal_moves
                        if not self.is_en_passant(m)]

        return Board


class _CallsStalemateMate:
    """Scores a draw as a win. The difference between 0.5 and 1.0 on
    every such game, invisible in the result."""

    def __init__(self) -> None:
        real = chess.Board

        class Board(real):
            def is_stalemate(self):
                return False

            def is_checkmate(self):
                return real.is_checkmate(self) or real.is_stalemate(self)

        self.Board = Board


def test_a_subtly_wrong_engine_is_refused():
    checks = verify_chess_rules(_NoEnPassant(), deep=False)
    failed = {c.name for c in checks if not c.ok}
    assert "взятие на проходе легально" in failed
    # And it shows up in the node counts too, which is why the published
    # tables are the first thing checked.
    assert any("Kiwipete" in name for name in failed)


def test_confusing_stalemate_with_mate_is_caught():
    checks = verify_chess_rules(_CallsStalemateMate(), deep=False)
    failed = {c.name for c in checks if not c.ok}
    assert "пат — это пат, а не мат" in failed


def test_a_refused_provider_is_not_trusted(monkeypatch):
    broken = Capability("broken_thing", "проверка, которая должна упасть",
                        (Provider("nothing", "nothing", "ничего"),),
                        lambda module, deep=False: [
                            CheckResult("заведомо неверно", False, 1, 2, "истина")],
                        "истина откуда-то ещё")
    monkeypatch.setitem(acquire._BY_NAME, "broken_thing", broken)
    monkeypatch.setattr(acquire, "_import", lambda name: object())

    checked = acquire.verify("broken_thing")
    assert checked.status == REFUSED
    assert checked.trusted is False
    assert "ОТКЛОНЁН" in checked.describe()
    assert "нельзя" in checked.describe()


def test_a_suite_that_cannot_run_is_a_refusal_not_a_pass(monkeypatch):
    """Refused rather than retried with fewer checks: a TypeError inside
    the suite would otherwise be swallowed and the provider graded by a
    second, quieter run."""
    def explodes(module, deep=False):
        raise TypeError("the suite itself is broken")

    broken = Capability("explodes", "падает", (Provider("x", "x", "x"),),
                        explodes, "неважно")
    monkeypatch.setitem(acquire._BY_NAME, "explodes", broken)
    monkeypatch.setattr(acquire, "_import", lambda name: object())

    checked = acquire.verify("explodes")
    assert checked.status == REFUSED
    assert checked.failures()


# --------------------------------------------------------------------------
# three states, kept apart
# --------------------------------------------------------------------------

def test_absent_is_not_refused(monkeypatch):
    """"We checked and it is wrong" and "we could not check" are
    different facts."""
    monkeypatch.setattr(acquire, "_import", lambda name: None)
    checked = acquire.verify("chess_rules")
    assert checked.status == ABSENT
    assert checked.trusted is False
    assert checked.checks == ()


def test_an_undeclared_capability_is_absent_not_installed():
    checked = acquire.verify("something_nobody_declared")
    assert checked.status == ABSENT
    assert "не объявлена" in checked.note


# --------------------------------------------------------------------------
# consent, and the allowlist
# --------------------------------------------------------------------------

def test_installing_without_consent_raises():
    """`pip install` executes code from the package. A caller that forgot
    to ask is a bug to fix, not a condition to handle quietly."""
    with pytest.raises(ConsentRequired):
        acquire.install("chess_rules")


def test_only_declared_capabilities_can_be_acquired():
    """A system that installs whatever it finds is not self-improving but
    self-compromising."""
    result = acquire.install("some_package_from_the_internet", consented=True)
    assert result["ok"] is False
    assert "не объявлена" in result["error"]


def test_nothing_installs_on_its_own_initiative():
    import inspect

    source = inspect.getsource(acquire)
    # The only place a subprocess is started is inside `install`, and
    # `install` cannot be reached without consent.
    assert source.count("subprocess.Popen") == 1
    assert 'raise ConsentRequired' in source


def test_acquired_packages_land_outside_the_application(monkeypatch, tmp_path):
    """A frozen build's folder is replaced wholesale by the next installer
    run, which would silently undo every acquisition."""
    monkeypatch.setenv("MANA_DATA_DIR", str(tmp_path))
    import importlib
    from mana import paths
    importlib.reload(paths)
    assert acquire.packages_dir().parent == tmp_path.resolve()


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def test_status_says_where_the_truth_came_from():
    """A verification is only worth as much as the independence of what
    it compared against, so the source travels with it."""
    reported = acquire.status("chess_rules")
    assert reported["known"] is True
    assert "perft" in reported["truth_source"]


def test_the_module_states_where_openness_stops():
    """MANA can verify a provider against ground truth it was given; it
    cannot invent trustworthy ground truth for a domain it does not know.
    If that caveat leaves the module, this fails."""
    import re

    # Whitespace-normalised: the sentence is wrapped across lines in the
    # source and a literal match would fail on the line break.
    doc = re.sub(r"\s+", " ", acquire.__doc__ or "")
    assert "cannot invent trustworthy ground truth for a domain it does not know" in doc
