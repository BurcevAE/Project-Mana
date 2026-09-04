"""
mana.core — the immutable boundary.

Everything in this package answers one of two questions: *what is true?*
and *what counts as an improvement?* Nothing here may be modified by
MANA's own evolution machinery. The rule is enforced by directory, not by
a list of file names, so adding a file here protects it automatically --
`code_evolution._NEVER_PATCHABLE` had to be edited by hand every time,
which is exactly the kind of boundary that erodes.

What this means in practice
---------------------------
MANA must not be able to improve itself by editing the thing that declares
it improved. The phase-0 audit found that invariant holding by accident:
`BenchmarkSuite` lives in `pipeline.py`, which nothing protected, and only
the whitelist happening to contain a single pure function kept the agent
away from its own test set.

So the boundary moves from convention to structure:

  * task oracles and ground truth live here;
  * the hidden holdout lives here and is **never returned as data** --
    `hidden_score()` gives a number, and there is no function that hands
    out the tasks. An agent cannot overfit to a set it cannot read;
  * acceptance, regression and rollback rules live here;
  * the evaluation-mode flag lives here rather than on the agent, so
    "am I being measured right now?" is not a property the measured thing
    owns.

Honest limit
------------
Python has no real privacy. A determined patch could import and rebind
anything in this package, and `is_immutable_path()` is a check the
patching machinery performs on itself. What this boundary buys is that
crossing it requires an explicit, visible act rather than a mutation that
looks like ordinary tuning -- and every route MANA actually has to change
code (`code_evolution.apply_patch`) refuses at the boundary. That is a
real guarantee about the mechanisms that exist, not a sandbox against an
adversary.
"""
from __future__ import annotations

from pathlib import Path

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"

#: The protected directory itself.
CORE_ROOT = Path(__file__).resolve().parent


def is_immutable_path(path: str | Path) -> bool:
    """True when `path` lies inside the immutable core.

    Resolved before comparing: a relative path, a symlink or a `..`
    traversal must not be able to point at core and read as outside it.
    """
    try:
        candidate = Path(path).resolve()
    except (OSError, ValueError):
        return False
    if candidate == CORE_ROOT:
        return True
    try:
        candidate.relative_to(CORE_ROOT)
        return True
    except ValueError:
        return False
