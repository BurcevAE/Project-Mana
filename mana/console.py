"""
mana.console — make what this program prints readable.

Every message MANA prints is in Russian, and a frozen build was writing
them as cp1251 while the console was set to 65001. `--peer show` came out
as a wall of question marks, and the user could not tell whether the
command had worked -- let alone read the public key it had just printed,
which was the entire point of running it.

Measured rather than assumed: the console was ALREADY UTF-8 and the
output was still wrong, so this is not something `chcp 65001` fixes.
Python picks the locale encoding for a redirected stream, and a frozen
build on a Russian Windows gets cp1251 whichever way the console is set.

Reconfigured at startup rather than through PYTHONUTF8, because that
variable has to exist before the interpreter starts and by the time any
of this runs it already has.

One implementation, called from every entry point. Three copies of a
two-line fix is how one of them gets missed and the bug comes back on
whichever route nobody tested.
"""
from __future__ import annotations

import sys

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"


def speak_utf8() -> bool:
    """Put stdout and stderr into UTF-8. True if anything was changed.

    `errors="replace"` so a character with no representation costs one
    glyph rather than raising and taking the whole message with it -- a
    diagnostic that cannot print is worse than one printed imperfectly.
    """
    changed = False
    for stream in (sys.stdout, sys.stderr):
        try:
            if getattr(stream, "encoding", "").lower().replace("-", "") == "utf8":
                continue
            stream.reconfigure(encoding="utf-8", errors="replace")
            changed = True
        except Exception:
            # A windowed build can have no streams at all, and a stream
            # that cannot be reconfigured is not a reason to refuse to
            # start.
            continue
    return changed
