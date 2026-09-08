"""
mana.apps — driving the applications installed on this machine.

Why this is a separate layer
----------------------------
Until now MANA reasoned and wrote files inside its own directory. Here it
reaches other people's programs and other people's data, and the
consequences are of a different kind: a mistake in reasoning costs a wrong
answer, a mistake here costs a corrupted document or the wrong record
posted into an accounting database.

So the layer is built around three rules rather than around convenience.

**Absence is visible.** `available()` lists every capability and, when one
is missing, names exactly what is missing. That is the direct lesson of
phase 22, where a try/except around an import turned "this package is not
here" into silence and the packaged application quietly shipped without
web search. A tool that is not there must say so, not be absent quietly.

**Refuse rather than guess.** No program, no pretending: `AppUnavailable`
with the reason. The same rule the algorithmic brains follow -- answer
exactly or decline.

**Writing is separated from reading.** Reads execute immediately. Writes
are *proposed*: `propose_*` returns a description of what would change
together with the current value, and only `confirm(request_id)` executes
it. The idiom is taken from the cognitive genome's `propose`/`apply` with
claim-id matching, because it already exists in this project and has
already been exercised.

Why writes work that way and not behind an "allow writes" flag
---------------------------------------------------------------
The agent rewrites its own source, its sandbox has a documented bypass,
and phase 22.1 found it capable of answering a greeting with a number
lifted out of unrelated recalled memory. An executor like that must not be
handed write permission "for the session": what gets confirmed has to be
the operation, not the intention.
"""
from __future__ import annotations

import dataclasses
import importlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"


class AppUnavailable(RuntimeError):
    """The program, or the binding for it, is not on this machine.

    A distinct type rather than a bare RuntimeError: "not installed" is
    not a malfunction, it is a fact about the environment, and callers
    have to be able to tell the two apart. Exactly the distinction
    `BrainRefusal` draws for brains.
    """


@dataclasses.dataclass(frozen=True)
class Capability:
    """One capability and whatever it stands on."""
    name: str
    what: str                      # what it lets MANA do, shown to the user
    modules: tuple = ()            # python packages it cannot work without
    executable: str = ""           # program it cannot work without
    com_id: str = ""               # COM object it cannot work without

    def missing(self) -> List[str]:
        """What is absent. An empty list means the capability is there."""
        gaps = []
        for module in self.modules:
            try:
                importlib.import_module(module)
            except Exception:
                gaps.append(f"пакет {module}")
        if self.executable and not find_executable(self.executable):
            gaps.append(f"программа {self.executable}")
        if self.com_id and not com_registered(self.com_id):
            gaps.append(f"COM-объект {self.com_id}")
        return gaps


def find_executable(name: str) -> str:
    """Path to the program, or "".

    PATH is not the only place to look: Notepad++ installs into Program
    Files and does not put itself on PATH, so the machine directories are
    checked too. The order means a deliberately chosen copy on PATH still
    wins over the default installation.
    """
    found = shutil.which(name)
    if found:
        return found
    stem = Path(name).stem
    roots = [os.environ.get("ProgramFiles", ""),
             os.environ.get("ProgramFiles(x86)", ""),
             os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")]
    for root in roots:
        if not root:
            continue
        candidate = Path(root) / stem / name
        if candidate.is_file():
            return str(candidate)
    # 1C does not follow the "<Program Files>/<name>/<name>.exe" shape:
    # the launcher sits in 1cv8/common and each client in 1cv8/<version>/bin.
    if name.lower() == "1cestart.exe":
        from .onec_launch import executables
        return executables().get("starter", "")
    return ""


def com_registered(prog_id: str) -> bool:
    """Whether a COM object is registered.

    Answered from the registry rather than by creating the object.
    Creating the 1C connector starts a fair amount of machinery and, on
    this machine, leaves a "Win32 exception releasing IUnknown" behind;
    none of that is needed to answer "does it exist at all".
    """
    if sys.platform != "win32":
        return False
    try:
        import winreg
    except Exception:                                  # pragma: no cover
        return False
    for hive, path in ((winreg.HKEY_CLASSES_ROOT, prog_id),
                       (winreg.HKEY_CURRENT_USER, rf"SOFTWARE\Classes\{prog_id}")):
        try:
            with winreg.OpenKey(hive, path):
                return True
        except OSError:
            continue
    return False


#: Everything this layer can do, and what each thing stands on.
CAPABILITIES = (
    Capability("docx", "чтение и запись .docx без установленного Word",
               modules=("docx",)),
    Capability("xlsx", "чтение и запись .xlsx без установленного Excel",
               modules=("openpyxl",)),
    # The engine is deliberately not named here: `chess_judge` looks for
    # it in MANA's data directory rather than on PATH, and its absence is
    # a degraded judge with a working fallback rather than a missing
    # capability. What this line is about is whether MANA can play at all.
    Capability("chess", "играть в шахматы: свой поиск, судья и партии на lichess",
               modules=("chess", "requests")),
    Capability("editor", "открыть файл в Notepad++ на нужной строке",
               executable="notepad++.exe"),
    # pythoncom is named alongside win32com because the COM thread calls
    # CoInitialize directly, and it is a DLL-backed extension PyInstaller
    # places in pywin32_system32/ rather than a plain module. Present in
    # the package and importable from it are different claims, and phase
    # 22 is what happens when the second one goes unchecked.
    Capability("onec", "запросы к 1С:Предприятие 8 и запись с подтверждением",
               modules=("win32com.client", "pythoncom"),
               com_id="V83.COMConnector"),
    # Separate from `onec` because the requirements differ: starting the
    # client needs the executables and nothing else, while talking to the
    # data needs pywin32 and a registered COM connector. Merging them
    # would make "запусти 1С" unavailable on a machine where only the COM
    # part is missing -- a refusal with the wrong reason attached.
    Capability("onec_client", "запуск 1С:Предприятия и создание баз",
               executable="1cestart.exe"),
)


def available() -> Dict[str, Dict[str, Any]]:
    """What works on this machine, and what exactly is missing elsewhere."""
    out: Dict[str, Dict[str, Any]] = {}
    for capability in CAPABILITIES:
        gaps = capability.missing()
        out[capability.name] = {"what": capability.what,
                                "available": not gaps,
                                "missing": gaps}
    return out


def require(name: str) -> None:
    """Assert a capability is present, or refuse with a usable reason."""
    for capability in CAPABILITIES:
        if capability.name != name:
            continue
        gaps = capability.missing()
        if gaps:
            raise AppUnavailable(
                f"{capability.what} — недоступно: не хватает {', '.join(gaps)}")
        return
    raise AppUnavailable(f"неизвестная возможность: {name}")


def describe() -> str:
    """The human-readable summary the window shows."""
    lines = []
    for name, info in available().items():
        mark = "  есть " if info["available"] else "  НЕТ  "
        tail = info["what"] if info["available"] else \
            f"{info['what']} — нет: {', '.join(info['missing'])}"
        lines.append(f"{mark} {name:8s} {tail}")
    return "\n".join(lines)
