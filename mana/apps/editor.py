"""
mana.apps.editor — put a file in front of the user in Notepad++.

The honest scope of this module
-------------------------------
Notepad++ has no automation API. It has no COM object, no scripting
interface without the NppExec or PythonScript plugins, and nothing that
reports back what is on screen. What it does have is a command line.

So this module does exactly one thing: it opens a file, optionally at a
line and column. It cannot read what the user typed, cannot save on their
behalf, and cannot tell whether they looked at it. Editing a file happens
by writing the file -- MANA already does that -- and this is how the
result gets shown to a person.

Pretending otherwise was the alternative, and it is worth naming why it
was rejected: driving the editor by synthesising keystrokes into its
window "works" in a demo and silently types into whatever window happened
to have focus when it does not. A tool whose failure mode is entering text
into an unknown application is not a tool.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict

from . import find_executable, require

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

EXECUTABLE = "notepad++.exe"


def open_file(path: str, line: int = 0, column: int = 0,
              new_instance: bool = False) -> Dict[str, Any]:
    """Open a file in Notepad++, optionally at a position.

    Returns as soon as the editor has been launched, and says so: the
    process is not waited on, because waiting would block MANA until the
    user closed their editor. `shown` means "handed to the editor", never
    "the user has read it" -- nothing here can know the second.
    """
    require("editor")
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"нет файла {target}")

    executable = find_executable(EXECUTABLE)
    command = [executable]
    if new_instance:
        # -multiInst opens a separate process; without it the file goes
        # into the window the user already has open, which is what they
        # usually want and why it is not the default.
        command += ["-multiInst", "-nosession"]
    if line > 0:
        command.append(f"-n{int(line)}")
    if column > 0:
        command.append(f"-c{int(column)}")
    command.append(str(target.resolve()))

    process = subprocess.Popen(command)
    return {
        "shown": True,
        "path": str(target.resolve()),
        "line": int(line) or None,
        "editor": executable,
        "pid": process.pid,
        "note": "файл передан редактору; прочитал ли его человек — отсюда не видно",
    }
