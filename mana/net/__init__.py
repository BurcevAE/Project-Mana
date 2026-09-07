"""
mana.net — the link between two installations.

The decision this package is built around
------------------------------------------
`mana_desktop.server` is **never** exposed to a network. It answers
/api/ask, which runs code in the sandbox; /api/onec/confirm, which writes
to a live trade database; and /api/key, which stores API credentials. On
an open Wi-Fi port that is remote code execution with access to somebody's
accounting records, and no amount of authentication in front of it would
make it a reasonable thing to publish.

So the peer link is a separate endpoint that does exactly one thing:
accept and serve exchange bundles. It has no session, no agent, no
sandbox and no route to any of them. The surface is small enough to read
in one sitting, which is the property that matters for something
listening on a network.

Why signatures are enough, and encryption is not required here
---------------------------------------------------------------
Bundles carry no private data by construction -- `check_shareable` refuses
free text, and `Attempt.shared()` strips everything that names a
configuration or a question. So confidentiality is not what this traffic
needs. What it needs is **authenticity** (this really came from that
installation) and **integrity** (nobody altered it), and an Ed25519
signature gives both.

That is a claim about this traffic and not a general one. The moment
anything private travels this way, it stops being true.

A consideration written down rather than hidden
------------------------------------------------
This package lives under `mana/`, which the agent can rewrite -- self
patching is the thing this project exists to do. A network listener that
the agent can modify is a real consideration. The mitigations are the ones
already here: every patch is recorded in the changelog, `--self-check`
verifies the installation, and the listener is off unless started
explicitly. It is worth knowing rather than discovering.
"""
from __future__ import annotations

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: The port the peer link listens on when no other is given. Fixed rather
#: than OS-assigned: a peer has to be reachable at an address somebody
#: wrote down, and an address that changes every restart is not one.
DEFAULT_PORT = 8787

#: How far apart two clocks may be before a signed request is refused.
#: Replay protection: a captured request stops working once it ages out.
#: Five minutes is loose enough for machines that never synchronised and
#: tight enough that a recording is not a key.
CLOCK_SKEW_SECONDS = 300.0
