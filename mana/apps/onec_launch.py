"""
mana.apps.onec_launch — starting the 1C client, not talking to its data.

Separate from `onec.py` on purpose. That module opens a COM connection and
reads or writes records; this one launches the application a person then
uses. They share nothing but the vendor: different executables, different
failure modes, different risks.

Three things a person actually asks for
---------------------------------------
1. "запусти 1С" -- open the base list and let me choose. `1cestart.exe`
   with no arguments does exactly this, and the choosing happens in 1C's
   own dialog, where it belongs.
2. "запусти базу X под логином Y" -- straight into one base.
3. "создай базу" -- make a new infobase and open the Configurator on it.

The password problem, measured rather than assumed
---------------------------------------------------
1C takes credentials as command-line arguments (`/N` and `/P`). Windows
lets **any process on the machine read another process's command line
without administrator rights** -- verified here with
`Get-CimInstance Win32_Process`, which printed the argument back verbatim.
So a password passed this way is exposed to anything running as this user
for as long as 1C is open.

There is no documented way to feed the GUI client a password over stdin,
so the choice is: pass it and accept that, or omit it and let 1C prompt.
Both are supported, the password is never written to a log or a file, and
every launch that carries one returns a `warning` saying so. Hiding the
trade-off would have been the easy option and the wrong one -- a user who
is not told cannot decide.

Reporting what happened, not what was asked for
------------------------------------------------
Every launch here comes back with an `mana.outcome.Outcome`: the goal in
the person's words, the state the launch was supposed to produce, and the
state actually found on the machine afterwards. The verdict is derived
from the last two. See that module for why there are three verdicts and
why an axis nobody could look at is never counted as one that came out
right.

Creating a base asks instead of guessing
-----------------------------------------
`create_base` refuses without an explicit path. That refusal *is* the
question: the agent relays it, the person answers with a location, and the
base is created there. Defaulting to some directory of our choosing would
put a database somewhere nobody chose, and the person would find it later
by accident.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import AppUnavailable
from ..outcome import Outcome

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.1"

#: Where 1C keeps the list of bases the launcher shows.
IBASES = Path(os.environ.get("APPDATA", "")) / "1C" / "1CEStart" / "ibases.v8i"

#: Roots that hold an installation. `common` holds the launcher, and each
#: version gets its own `bin` beside it.
INSTALL_ROOTS = (r"C:\Program Files\1cv8", r"C:\Program Files (x86)\1cv8")


def _install_root() -> Optional[Path]:
    for candidate in INSTALL_ROOTS:
        path = Path(candidate)
        if path.is_dir():
            return path
    return None


def _versions(root: Path) -> List[Path]:
    """Installed version directories, newest first.

    Sorted by the numeric parts rather than as strings, because "8.3.9"
    sorts after "8.3.27" alphabetically and picking the older client to
    open a newer base is a failure with a confusing message.
    """
    def key(path: Path):
        return [int(part) if part.isdigit() else 0
                for part in path.name.split(".")]
    found = [p for p in root.iterdir()
             if p.is_dir() and re.match(r"^\d+\.\d+\.\d+", p.name)]
    return sorted(found, key=key, reverse=True)


def executables() -> Dict[str, str]:
    """The 1C programs available here: launcher, thick client, thin client."""
    root = _install_root()
    out: Dict[str, str] = {}
    if root is None:
        return out
    launcher = root / "common" / "1cestart.exe"
    if launcher.is_file():
        out["starter"] = str(launcher)
    for version in _versions(root):
        for name, exe in (("client", "1cv8.exe"), ("thin", "1cv8c.exe")):
            candidate = version / "bin" / exe
            if name not in out and candidate.is_file():
                out[name] = str(candidate)
        if "client" in out and "thin" in out:
            break
    out["version"] = _versions(root)[0].name if _versions(root) else ""
    return out


def _require(kind: str) -> str:
    found = executables().get(kind, "")
    if not found:
        raise AppUnavailable(
            f"не найден {kind} 1С:Предприятия; искали в "
            f"{', '.join(INSTALL_ROOTS)}")
    return found


# --------------------------------------------------------------- the base list


def bases() -> List[Dict[str, Any]]:
    """The bases 1C's own launcher would offer.

    Read from ibases.v8i rather than reconstructed: this is the list the
    person already recognises, in the order they already see it. The file
    is UTF-8 with a BOM and its section headers are the base names.
    """
    if not IBASES.is_file():
        return []
    text = IBASES.read_text(encoding="utf-8-sig", errors="replace")
    out: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            current = {"name": line[1:-1], "connect": "", "kind": "",
                       "location": ""}
            out.append(current)
        elif current is not None and line.startswith("Connect="):
            connect = line[len("Connect="):].strip()
            current["connect"] = connect
            server = re.search(r'Srvr="([^"]+)"', connect)
            reference = re.search(r'Ref="([^"]+)"', connect)
            file_path = re.search(r'File="([^"]+)"', connect)
            if server and reference:
                current["kind"] = "клиент-сервер"
                current["location"] = f"{server.group(1)} / {reference.group(1)}"
            elif file_path:
                current["kind"] = "файловая"
                current["location"] = file_path.group(1)
    return out


# ------------------------------------------------------------------ launching


def _command_line(parts) -> str:
    """A Windows command line 1C will actually parse.

    subprocess builds this itself from a list, and for 1C it builds it
    wrong: a connection string is `File="C:\\path";` -- quotes and all --
    and list2cmdline escapes those inner quotes as \\" , which 1C rejects
    with "Неверные или отсутствующие параметры соединения". Measured:
    passing the argument list works only while no path contains a space,
    and every real base here lives under C:\\1С\\Базы\\... with spaces in
    every segment. So the line is assembled here and handed to
    CreateProcess verbatim.

    Inside a quoted Windows argument a literal quote is written twice,
    which is what turns `File=""C:\\path"";` on the way in into
    `File="C:\\path";` on the way out.
    """
    return " ".join(parts)


def _quoted(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _connection(target) -> str:
    """The File= connection string, quoted the way 1C expects."""
    return _quoted(f'File="{target}";')


#: The image the base-selection window actually runs as. 1cestart.exe is
#: a shim that picks a version, starts this and exits.
SELECTOR_IMAGE = "1cv8s.exe"


def _pids(image: str) -> set:
    """PIDs of a running image, via tasklist. Empty set on any trouble."""
    try:
        listing = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {image}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, errors="replace", timeout=15)
    except Exception:
        return set()
    found = set()
    for line in listing.stdout.splitlines():
        fields = [f.strip('"') for f in line.split('","')]
        if len(fields) > 1 and fields[1].isdigit():
            found.add(int(fields[1]))
    return found


#: What 1C prints in the title bar of the Configurator, lower-cased. No
#: Предприятие window carries this word, which is what makes it usable as
#: evidence in one direction: finding it when Предприятие was asked for is
#: a real contradiction. Not finding it says nothing, because the
#: Предприятие window and the password dialog are both titled
#: "1С:Предприятие" and telling them apart by title is not possible.
DESIGNER_MARKER = "конфигуратор"


def _window_title(pid: int) -> str:
    """The window title of one process, or "" when it has none yet.

    `tasklist /V` prints it as the last CSV field -- verified on this
    machine rather than taken from documentation. A process with no window
    is reported by tasklist as "N/A", which is not a title and comes back
    as "".

    Only the title. The command line, which is where /P puts the password,
    is never read: this module refuses to write that anywhere, and reading
    it back into a returned dict would undo that in one line.
    """
    try:
        listing = subprocess.run(
            ["tasklist", "/FI", f"PID eq {int(pid)}", "/V", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, errors="replace", timeout=15)
    except Exception:
        return ""
    for line in listing.stdout.splitlines():
        fields = [f.strip('"') for f in line.split('","')]
        if len(fields) > 1 and fields[1].strip() == str(int(pid)):
            title = fields[-1].strip()
            return "" if title.upper() == "N/A" else title
    return ""


def _observe(process, base: str, settle: float) -> Dict[str, Dict[str, Any]]:
    """Look at what the launch left behind. Reports only what is there.

    Waits for a window title to appear rather than for a fixed period,
    and stops early if the process exits -- which is the case worth
    catching, and the one the old code reported as a successful launch.
    """
    observed: Dict[str, Any] = {}
    evidence: Dict[str, Any] = {"pid": process.pid}

    title = ""
    deadline = time.time() + max(0.5, settle)
    while True:
        code = process.poll()
        if code is not None:
            # It started and then stopped. Nothing else is worth reading:
            # there is no window to look at and the exit code is the fact.
            observed["running"] = False
            evidence["exit_code"] = code
            return {"observed": observed, "evidence": evidence,
                    "note": "процесс 1С завершился сразу после запуска"}
        title = _window_title(process.pid)
        if title or time.time() >= deadline:
            break
        time.sleep(0.5)

    observed["running"] = process.poll() is None
    evidence["window"] = title

    low = title.lower()
    if base and base.strip().lower() in low:
        observed["base"] = base
    if DESIGNER_MARKER in low:
        observed["mode"] = "конфигуратор"

    note = "" if title else ("окно ещё не появилось: 1С может спрашивать "
                             "пароль или всё ещё открываться")
    return {"observed": observed, "evidence": evidence, "note": note}


def launch_selector(settle: float = 4.0) -> Dict[str, Any]:
    """Open the base list and let the person choose. Scenario 1.

    Nothing is chosen here on their behalf, which is the whole point of
    the request: 1C's own dialog is where a person picks a base, and it
    already knows how to do that.

    The pid needs explaining, because the obvious one is wrong.
    `1cestart.exe` is a shim: it works out which version to use, starts
    1cv8s.exe and exits, all inside about a second. The first version of
    this function returned the shim's pid, which was already dead by the
    time anyone looked -- a caller checking whether the launch worked
    would have concluded it failed while the window sat on screen. So the
    window process is located by comparing the running 1cv8s.exe processes
    before and after, and when it cannot be found that is reported rather
    than papered over with a number that means nothing.
    """
    starter = _require("starter")
    before = _pids(SELECTOR_IMAGE)
    shim = subprocess.Popen(_command_line([_quoted(starter)]))

    window_pid = None
    deadline = time.time() + max(0.5, settle)
    while time.time() < deadline:
        appeared = _pids(SELECTOR_IMAGE) - before
        if appeared:
            window_pid = sorted(appeared)[0]
            break
        time.sleep(0.3)

    note = ("базу выбираете вы, в окне 1С" if window_pid else
            "окно не найдено среди процессов; возможно, 1С ещё "
            "запускается или окно уже было открыто")
    # This function was already observing -- diffing the process list is
    # what the loop above does. It just had nowhere to say so in a form
    # anything else could read.
    outcome = Outcome(
        action="onec_launch_selector",
        goal="открыть окно выбора базы 1С",
        expected={"window": True},
        observed={"window": True} if window_pid else {},
        evidence={"starter_pid": shim.pid, "window_pid": window_pid},
        note="" if window_pid else note)
    return {"launched": bool(window_pid), "what": "окно выбора базы 1С",
            "executable": starter,
            "starter_pid": shim.pid,
            "pid": window_pid,
            "bases_offered": [b["name"] for b in bases()],
            "outcome": outcome.as_dict(),
            "verified": outcome.verified,
            "note": note}


def launch(base: str = "", user: str = "", password: str = "",
           designer: bool = False, thin: bool = False,
           goal: str = "", settle: float = 4.0) -> Dict[str, Any]:
    """Open one base. Scenario 2.

    With no `base` this falls through to the selector, because launching
    "some base" is not a thing a person means.

    A password given here is passed as /P, and the returned `warning` says
    what that costs: Windows lets any process running as this user read
    another process's command line without administrator rights, which was
    verified here rather than assumed. Omit the password and 1C asks for
    it in its own dialog -- the safer half of a real trade-off, not a
    lecture.
    """
    if not base:
        return launch_selector()

    known = {b["name"]: b for b in bases()}
    if base not in known:
        raise AppUnavailable(
            f"базы {base!r} нет в списке 1С. Есть: "
            f"{', '.join(known) or '(список пуст)'}")

    executable = _require("thin" if thin else "client")
    parts = [_quoted(executable), "DESIGNER" if designer else "ENTERPRISE",
             "/IBName" + _quoted(base)]
    if user:
        parts.append("/N" + _quoted(user))
    warning = ""
    if password:
        parts.append("/P" + _quoted(password))
        warning = ("пароль передан в командной строке процесса 1С; пока она "
                   "открыта, его может прочитать любая программа, запущенная "
                   "от вашего имени, без прав администратора")

    mode = "конфигуратор" if designer else "предприятие"
    process = subprocess.Popen(_command_line(parts))

    # Everything above this line is the request. Everything below is what
    # the machine says happened, and they are kept apart on purpose: the
    # previous version of this function returned `base` and `mode` as
    # results when they were arguments, so "запустила базу X" was said
    # with equal confidence whether X opened or 1С exited on the spot.
    seen = _observe(process, base, settle)
    outcome = Outcome(
        action="onec_launch",
        goal=goal or f"открыть {base} в режиме {mode}",
        expected={"running": True, "base": base, "mode": mode},
        observed=seen["observed"],
        evidence=seen["evidence"],
        note=seen["note"])

    return {
        # Now means what it says: a process that is alive. It used to mean
        # that Popen returned.
        "launched": bool(seen["observed"].get("running")),
        "base": base,
        "kind": known[base]["kind"],
        "location": known[base]["location"],
        "mode": mode,
        "user": user or "(спросит 1С)",
        # Returned for diagnosis with the password stripped: this dict ends
        # up in a trace, and a trace is a file on disk.
        "command": " ".join(part for part in parts
                            if not part.startswith("/P")),
        "pid": process.pid,
        "outcome": outcome.as_dict(),
        "verified": outcome.verified,
        "observed": outcome.summary(),
        "warning": warning,
    }


def create_base(path: str = "", name: str = "", open_designer: bool = True,
                add_to_list: bool = True) -> Dict[str, Any]:
    """Create a file infobase and open the Configurator on it. Scenario 3.

    Refuses without a path, and that refusal is the question the person
    expects to be asked. Choosing a directory for them would put a
    database somewhere nobody picked, to be found later by accident.

    The directory is left for 1C to create. An empty one it made itself
    and an empty one we made look identical on disk, but only the first
    is a state 1C put itself in, and there is no reason to take that over.
    """
    if not path:
        raise AppUnavailable(
            "не указано, где разместить базу. Нужен путь к папке, например "
            r"C:\1С\Базы\Новая база")

    target = Path(path)
    if target.exists() and any(target.iterdir()):
        raise AppUnavailable(
            f"каталог {target} не пуст; 1С создаёт базу в пустой папке. "
            f"Укажите другой путь или очистите этот")

    executable = _require("client")
    base_name = name or target.name
    # 1C writes its result here rather than to stderr, so without /Out a
    # failure arrives as a bare exit code 1 with nothing to act on -- which
    # is exactly how the first version of this function failed.
    report = target.parent / f".mana-1c-{os.getpid()}.txt"

    parts = [_quoted(executable), "CREATEINFOBASE", _connection(target),
             "/Out", _quoted(report)]
    if add_to_list:
        # Without this the base exists and does not appear in the launcher,
        # so the person cannot find it the ordinary way.
        parts.append("/AddInList" + _quoted(base_name))

    created = subprocess.run(_command_line(parts), capture_output=True)
    message = ""
    try:
        if report.exists():
            message = " ".join(report.read_text(encoding="utf-8-sig",
                                                errors="replace").split())
            report.unlink()
    except OSError:
        pass

    if created.returncode != 0 or not (target / "1Cv8.1CD").exists():
        raise AppUnavailable(
            f"1С не создала базу (код {created.returncode}). "
            f"{message or 'сообщения нет'}")

    result = {"created": True, "path": str(target), "name": base_name,
              "added_to_list": add_to_list, "designer_opened": False,
              "message": message}
    if open_designer:
        designer = subprocess.Popen(_command_line(
            [_quoted(executable), "DESIGNER", "/F" + _quoted(target)]))
        result["designer_opened"] = True
        result["pid"] = designer.pid
    return result
