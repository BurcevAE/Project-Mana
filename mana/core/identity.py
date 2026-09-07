"""
mana.core.identity — which installation this is.

Exists for one reason: a hidden set that is identical on every machine is
not hidden once machines start talking to each other.

The problem it fixes, which is already present
-----------------------------------------------
`HOLDOUT_V0` and `HOLDOUT_V1` are defined by a fixed seed, so every copy
of MANA generates *the same* hidden tasks. While there is one copy that is
harmless. The moment two instances exchange anything -- a proposal, a
result, a genome -- it becomes a leak channel by construction: one
instance publishing something fitted to the hidden set invalidates the
acceptance gates of every other instance at once, and silently, because
nothing in a score says which set produced it.

Salting the seed per installation makes each instance's hidden set its
own. A leak stops being contagious, and agreement between two instances
stops being an artefact of them having drawn the same tasks -- which is
what turns it into replication rather than coincidence.

Why the id is random and not the hostname
------------------------------------------
A fingerprint derived from a machine name or a MAC address would travel
with every shared record and identify the business it came from. This one
is random, generated once, and the only thing derived from it that ever
leaves is an 8-character digest that says "a different instance" and
nothing else.

The id must persist
--------------------
If it changes, the hidden set changes with it, and scores recorded before
stop being comparable to scores recorded after. That is why it lives in a
file beside the agent's state rather than being derived at each start --
and why `Holdout.identity` carries the fingerprint, so a comparison across
a lost id fails loudly instead of quietly comparing two different sets.
"""
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Optional

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Overrides the stored id. Set in tests, and by anyone who deliberately
#: wants two installations to share a hidden set (comparing them directly
#: is the one case where that is what you want).
INSTANCE_ENV = "MANA_INSTANCE_ID"

#: Name of the file holding it, beside the rest of the agent's state.
FILENAME = "instance_id"

_CACHED: Optional[str] = None


def _store() -> Path:
    from .. import paths
    return Path(paths.data_root()) / FILENAME


def instance_id() -> str:
    """This installation's stable random id.

    Read once and cached: it is consulted on every hidden evaluation, and
    a value that could change between two calls in one run would give two
    different hidden sets inside a single experiment.
    """
    global _CACHED
    override = os.environ.get(INSTANCE_ENV, "").strip()
    if override:
        return override
    if _CACHED is not None:
        return _CACHED

    path = _store()
    try:
        if path.is_file():
            stored = path.read_text(encoding="utf-8").strip()
            if stored:
                _CACHED = stored
                return stored
    except OSError:
        pass

    fresh = uuid.uuid4().hex
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(fresh, encoding="utf-8")
    except OSError:
        # Unwritable state directory: the run still works, but the id
        # lasts only for this process. `Holdout.identity` will differ from
        # the stored one, which is exactly the loud failure wanted here --
        # scores from this run must not be compared with the others.
        pass
    _CACHED = fresh
    return fresh


def reset_cache() -> None:
    """Forget the cached id. For tests, and after the state dir moves."""
    global _CACHED
    _CACHED = None


def fingerprint(value: str = "") -> str:
    """Short digest of an instance id: shareable, and says nothing else.

    Eight hex characters. Enough to tell instances apart and to count how
    many of them reproduced a result; not enough, and not the right shape,
    to say anything about whose machine it is.
    """
    source = value or instance_id()
    return hashlib.blake2b(source.encode("utf-8"), digest_size=4).hexdigest()


def salted_seed(seed: int, value: str = "") -> int:
    """A seed of this installation's own, derived from the shared one.

    Deterministic in (seed, instance): the same installation regenerates
    exactly the same hidden set every run, which is what makes scores
    comparable over time. Different installations get different sets,
    which is what makes agreement mean something.
    """
    source = f"{int(seed)}:{value or instance_id()}".encode("utf-8")
    digest = hashlib.blake2b(source, digest_size=8).digest()
    # Kept inside the range a Python/NumPy seed is happy with.
    return int.from_bytes(digest, "big") % (2 ** 31 - 1)
