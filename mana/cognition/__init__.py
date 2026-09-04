"""
mana.cognition — the part of MANA that is allowed to change.

The counterpart to `mana.core`. Core answers "what is true?" and "what
counts as an improvement?"; nothing here may answer either. This package
holds what MANA thinks *with*: operators, representations, programs, the
genome that describes the space of all three, and later the machinery that
searches that space.

The split is the point of the whole rebuild. A system that can improve
itself by editing its own success criterion has not improved; it has moved
the goalposts, and no measurement taken afterwards means anything. So the
criterion lives on the other side of a boundary this package cannot reach:
`code_evolution` refuses any target under `mana/core/`, the hidden holdout
is never returned as data, and the acceptance gate is the only function
permitted to conclude that something worked.

What "evolvable" means here
---------------------------
Not "the agent rewrites these files at runtime". It means the *content* of
these structures is data that mutation operates on, and every mutation is
a proposal that must survive `core.gates.judge` before it becomes the
current genome. Proposing is free; being adopted is not.
"""
from __future__ import annotations

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"
