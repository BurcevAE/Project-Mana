"""
mana.acquire — gaining a capability MANA did not have, and proving it works.

The distinction this module exists to draw
-------------------------------------------
"Найти правила игры в интернете и запомнить их" sounds like acquiring a
capability and is not one. Rules arrive as **text**, and text cannot say
whether Nf3 is legal in a position. You cannot play by it, learn from it,
or measure against it.

What can be acquired is an **executable oracle**: something that answers
exactly, so everything downstream -- practice, learning, the acceptance
gates -- has ground truth to stand on.

Verification is the whole point
--------------------------------
A library MANA found and installed is nobody. Before anything is learned
on it, it is checked against truth **stated here, in this file, as
literal constants** -- never computed by the thing being tested. A suite
that asks a library "are you right?" and believes the answer proves
nothing.

For chess those constants are published facts: the perft node counts from
the initial position (20, 400, 8902, 197281) and from the standard
"Kiwipete" position, which exists precisely because it exercises
castling, en passant and promotion -- the rules a subtly wrong engine
gets wrong.

That matters more than it looks. A rules engine that is quietly incorrect
is a **lying oracle**: every measurement below it is poisoned and nothing
downstream can tell. It is worse than having no oracle at all, because
the failure is invisible. So a provider that fails a check is REFUSED and
not used, rather than used with a warning.

Three states, as everywhere here
---------------------------------
    VERIFIED   installed, and every check against known truth passed
    REFUSED    installed, and a check failed -- do not use it
    ABSENT     not installed; nothing was checked, nothing is claimed

"We checked and it is wrong" and "we could not check" are different facts.

The allowlist, and where openness stops
----------------------------------------
Only capabilities declared in `CAPABILITIES` can be acquired. `pip
install` executes arbitrary code at install time, so a system that
installs whatever it finds is not self-improving but self-compromising.
Consent is required and explicit, in the same shape as the Ollama flow:
MANA says what and why, a person agrees, and only then does anything run.

This is a deliberate ceiling and it is worth naming plainly. **MANA can
verify a provider against ground truth it was given; it cannot invent
trustworthy ground truth for a domain it does not know.** For chess the
truth is published and independent. For an arbitrary new domain, somebody
has to supply it -- and a verification suite written by the same process
that chose the library is circular. That is the honest boundary of
"self-developing" as this module implements it.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

VERIFIED = "VERIFIED"
REFUSED = "REFUSED"
ABSENT = "ABSENT"

#: Where acquired packages go. Not into the application directory: a
#: frozen build has no pip and its folder is replaced wholesale by the
#: next installer run, which would silently undo every acquisition.
PACKAGES_DIRNAME = "packages"

#: An install that has not finished in this long has gone wrong. Chosen
#: long enough for a slow network and short enough that a hung process
#: does not look like a working one.
INSTALL_TIMEOUT = 600


@dataclass(frozen=True)
class CheckResult:
    """One comparison against truth that did not come from the provider."""
    name: str
    ok: bool
    expected: Any
    got: Any
    truth_from: str = ""
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "ok": self.ok,
                "expected": self.expected, "got": self.got,
                "truth_from": self.truth_from, "detail": self.detail}


@dataclass(frozen=True)
class Provider:
    """A package that might supply a capability."""
    package: str                 # what pip installs
    module: str                  # what Python imports
    why: str                     # what it gives, in one line
    pure_python: bool = True     # a wheel with C extensions may not fit here

    def as_dict(self) -> Dict[str, Any]:
        return {"package": self.package, "module": self.module,
                "why": self.why, "pure_python": self.pure_python}


@dataclass(frozen=True)
class Capability:
    """Something MANA wants to be able to do, and how to know it can."""
    name: str
    what: str
    providers: Tuple[Provider, ...]
    #: Signature is fixed: `verify(module, deep: bool)`. Fixed rather than
    #: sniffed, so a suite that raises a TypeError of its own is reported
    #: as a refusal instead of being quietly re-run with fewer checks.
    verify: Callable[..., List[CheckResult]]
    truth_source: str

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "what": self.what,
                "providers": [p.as_dict() for p in self.providers],
                "truth_source": self.truth_source}


@dataclass(frozen=True)
class Verification:
    """What was checked, and whether the provider may be trusted."""
    capability: str
    provider: str
    status: str
    checks: Tuple[CheckResult, ...] = ()
    elapsed: float = 0.0
    note: str = ""

    @property
    def trusted(self) -> bool:
        return self.status == VERIFIED

    def failures(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.ok]

    def as_dict(self) -> Dict[str, Any]:
        return {"capability": self.capability, "provider": self.provider,
                "status": self.status, "trusted": self.trusted,
                "elapsed": round(self.elapsed, 2), "note": self.note,
                "checks": [c.as_dict() for c in self.checks]}

    def describe(self) -> str:
        if self.status == ABSENT:
            return f"{self.capability}: не установлено ({self.note})"
        if self.status == REFUSED:
            bad = ", ".join(c.name for c in self.failures())
            return (f"{self.capability}: ОТКЛОНЁН — {self.provider} не прошёл "
                    f"проверку ({bad}). Учиться на нём нельзя.")
        return (f"{self.capability}: проверен — {self.provider}, "
                f"{len(self.checks)} проверок против известной истины, "
                f"{self.elapsed:.1f}с")


# --------------------------------------------------------------------------
# where acquired packages live
# --------------------------------------------------------------------------

def packages_dir() -> Path:
    from .paths import data_root
    return Path(data_root()) / PACKAGES_DIRNAME


def ensure_importable() -> Path:
    """Put the acquisition directory on the import path.

    Called before every import attempt rather than once at startup, so a
    package installed while MANA is running is reachable without a
    restart -- the same reasoning as re-probing Ollama after a download.
    """
    target = packages_dir()
    text = str(target)
    if text not in sys.path:
        sys.path.insert(0, text)
    return target


def _import(module: str) -> Any:
    ensure_importable()
    try:
        return importlib.import_module(module)
    except ImportError:
        importlib.invalidate_caches()
        try:
            return importlib.import_module(module)
        except ImportError:
            return None


# --------------------------------------------------------------------------
# chess: the verification suite
#
# Every number below is a published fact about the rules of chess. None of
# it is produced by the library under test. If a provider disagrees with
# any of it, the provider is wrong.
# --------------------------------------------------------------------------

#: Node counts from the initial position, the standard perft table. These
#: are the most reproduced numbers in computer chess and are independent
#: of any implementation.
PERFT_START = {1: 20, 2: 400, 3: 8902, 4: 197281}

#: The "Kiwipete" position and its perft counts. It exists precisely
#: because it exercises castling both sides, en passant and promotion --
#: the rules a subtly wrong engine gets wrong while passing the initial
#: position.
KIWIPETE_FEN = ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/"
                "R3K2R w KQkq - 0 1")
PERFT_KIWIPETE = {1: 48, 2: 2039, 3: 97862}

#: Fool's mate: 1.f3 e5 2.g4 Qh4#. Checkmate, not merely check.
FOOLS_MATE_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"

#: Black to move, not in check, and no legal move: stalemate, not mate.
#: Telling these two apart is where a wrong engine scores a draw as a win.
STALEMATE_FEN = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"

#: White pawn e5, black pawn just played d7-d5. exd6 en passant must be
#: legal, and it must disappear if it is not taken immediately.
EN_PASSANT_FEN = "8/8/8/3pP3/8/8/8/K6k w - d6 0 1"

#: A white pawn on the seventh with an empty square ahead promotes, and a
#: promotion is four moves (queen, rook, bishop, knight), not one.
PROMOTION_FEN = "8/4P3/8/8/8/8/8/K6k w - - 0 1"


def _perft(board: Any, depth: int) -> int:
    """Count leaf nodes. Written here so the count is ours, not theirs.

    A provider that shipped its own `perft` and was asked for it would be
    grading its own homework; only move generation and legality are taken
    from the library, which is exactly what is under test.
    """
    if depth <= 0:
        return 1
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += _perft(board, depth - 1)
        board.pop()
    return total


def verify_chess_rules(chess: Any, deep: bool = False) -> List[CheckResult]:
    """Check a chess library against facts it did not supply."""
    results: List[CheckResult] = []

    def note(name: str, expected: Any, got: Any, source: str,
             detail: str = "") -> None:
        results.append(CheckResult(name, expected == got, expected, got,
                                   source, detail))

    depths = (1, 2, 3, 4) if deep else (1, 2, 3)
    board = chess.Board()
    for depth in depths:
        note(f"perft({depth}) из начальной позиции", PERFT_START[depth],
             _perft(board, depth), "опубликованная таблица perft")

    kiwi_depths = (1, 2, 3) if deep else (1, 2)
    board = chess.Board(KIWIPETE_FEN)
    for depth in kiwi_depths:
        note(f"perft({depth}) Kiwipete", PERFT_KIWIPETE[depth],
             _perft(board, depth), "опубликованная таблица perft",
             "рокировка, взятие на проходе, превращение")

    board = chess.Board(FOOLS_MATE_FEN)
    note("детский мат — это мат", True, bool(board.is_checkmate()),
         "правила шахмат", "1.f3 e5 2.g4 Ф:h4#")

    board = chess.Board(STALEMATE_FEN)
    note("пат — это пат, а не мат", (True, False),
         (bool(board.is_stalemate()), bool(board.is_checkmate())),
         "правила шахмат", "путаница здесь превращает ничью в победу")

    board = chess.Board(EN_PASSANT_FEN)
    legal = {board.san(m) for m in board.legal_moves}
    note("взятие на проходе легально", True, "exd6" in legal,
         "правила шахмат", f"легальные ходы: {sorted(legal)}")

    board = chess.Board(PROMOTION_FEN)
    promotions = [m for m in board.legal_moves if m.promotion]
    note("превращение даёт четыре хода, а не один", 4, len(promotions),
         "правила шахмат", "ферзь, ладья, слон, конь")

    return results


CAPABILITIES: Tuple[Capability, ...] = (
    Capability(
        name="chess_rules",
        what="точные правила шахмат: легальные ходы, мат, пат, ничья",
        providers=(Provider("chess", "chess",
                            "python-chess: правила, FEN, PGN, UCI",
                            pure_python=True),),
        verify=verify_chess_rules,
        truth_source="опубликованные таблицы perft и правила ФИДЕ"),
)

_BY_NAME = {c.name: c for c in CAPABILITIES}


def capability(name: str) -> Optional[Capability]:
    return _BY_NAME.get(name)


# --------------------------------------------------------------------------
# acquiring
# --------------------------------------------------------------------------

class ConsentRequired(PermissionError):
    """Installing was attempted without a person agreeing to it.

    Raised rather than returned: `pip install` runs arbitrary code from
    the package at install time, so a caller that forgot to ask is a bug
    to fix, not a condition to handle quietly.
    """


def installed(name: str) -> Optional[Any]:
    """The imported provider module, or None if none is installed."""
    cap = capability(name)
    if cap is None:
        return None
    for provider in cap.providers:
        module = _import(provider.module)
        if module is not None:
            return module
    return None


def verify(name: str, deep: bool = False) -> Verification:
    """Check an installed provider against truth it did not supply.

    Cheap enough to run before every use. The default depth stops at
    perft(3); `deep` adds perft(4) and the third Kiwipete level, which
    together are about twenty times the work.
    """
    cap = capability(name)
    if cap is None:
        return Verification(name, "", ABSENT, note="способность не объявлена")

    for provider in cap.providers:
        module = _import(provider.module)
        if module is None:
            continue
        started = time.perf_counter()
        try:
            checks = tuple(cap.verify(module, deep=deep))
        except Exception as exc:
            # Deliberately not retried without `deep`: a TypeError raised
            # inside the suite would be swallowed by that retry and the
            # provider would be graded by a second, quieter run. A suite
            # that cannot run is a refusal, not a reason to run less of it.
            return Verification(
                name, provider.package, REFUSED,
                (CheckResult("проверка не отработала", False, "проверка проходит",
                             f"{type(exc).__name__}: {exc}", cap.truth_source),),
                time.perf_counter() - started,
                "провайдер не смог ответить на проверку")
        status = VERIFIED if all(c.ok for c in checks) else REFUSED
        return Verification(name, provider.package, status, checks,
                            time.perf_counter() - started)

    return Verification(name, "", ABSENT,
                        note="ни один провайдер не установлен")


def install(name: str, consented: bool = False,
            on_line: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Install a declared provider, with consent, and verify it.

    Never on MANA's own initiative. `pip install` executes code from the
    package, so consent is the boundary between self-improving and
    self-compromising, and it is passed in rather than assumed.

    Installing is not acquiring. What comes back says whether the
    capability may be used, and a provider that installs cleanly and then
    fails a check is REFUSED.
    """
    if not consented:
        raise ConsentRequired(
            f"установка «{name}» требует явного согласия человека")

    cap = capability(name)
    if cap is None:
        return {"ok": False, "error": f"способность «{name}» не объявлена"}

    from .paths import sandbox_python
    target = packages_dir()
    target.mkdir(parents=True, exist_ok=True)

    provider = cap.providers[0]
    command = [sandbox_python(), "-m", "pip", "install",
               "--no-input", "--disable-pip-version-check",
               "--target", str(target), provider.package]
    if provider.pure_python:
        # A wheel only. Building from source runs setup.py, which is the
        # same arbitrary-code problem one layer down.
        command.insert(-1, "--only-binary")
        command.insert(-1, ":all:")

    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace")
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    lines: List[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if not line:
            continue
        lines.append(line)
        if on_line is not None:
            try:
                on_line(line)
            except Exception:
                pass
    code = process.wait(timeout=INSTALL_TIMEOUT)
    if code != 0:
        return {"ok": False, "error": f"pip вернул {code}",
                "output": lines[-12:]}

    checked = verify(name)
    return {"ok": checked.trusted, "verification": checked.as_dict(),
            "describe": checked.describe(), "package": provider.package,
            "installed_into": str(target)}


def status(name: str) -> Dict[str, Any]:
    """Everything about one capability, for showing a person."""
    cap = capability(name)
    if cap is None:
        return {"name": name, "known": False}
    checked = verify(name)
    return {"name": name, "known": True, "what": cap.what,
            "truth_source": cap.truth_source,
            "providers": [p.as_dict() for p in cap.providers],
            "verification": checked.as_dict(),
            "describe": checked.describe()}


def status_all() -> List[Dict[str, Any]]:
    return [status(cap.name) for cap in CAPABILITIES]
