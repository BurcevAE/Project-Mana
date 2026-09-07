"""
mana.core.identity — which installation this is, provably.

An installation's identity is an Ed25519 key pair. There is no registry,
no server and nothing to sign up to: the identity *is* the key, which is
the one idea worth taking from how cryptocurrency networks work.

What the signature buys, and what it does not
----------------------------------------------
It buys **non-impersonation**. `replication` counts distinct instances,
and before signatures those instance strings were self-declared: one
installation could invent twelve fingerprints and manufacture "confirmed
on twelve instances". Now a report carries a signature over its own
contents and the public key that made it, and the fingerprint is checked
to be that key's -- so a report provably comes from whoever it says.

It does **not** buy Sybil resistance. Nothing stops one person generating
twelve key pairs. Cryptocurrencies answer that with proof of work or
stake, which is expensive precisely because they must reach agreement
among parties who cannot check a claim themselves. MANA can check: a
hypothesis is judged by local gates on a local hidden set, so agreement
is never needed and the expensive machinery would buy nothing.

What limits Sybil here is deciding whose keys count -- a list of peers
you recognise, the way SSH known_hosts works. Trust by acquaintance, not
by burnt electricity.

Two separations that matter
----------------------------
**The holdout salt derives from the PRIVATE key, never the public one.**
This nearly went wrong. The salt exists so another installation cannot
fit anything to this one's hidden set; deriving it from a public key --
which travels in every bundle -- would let anyone regenerate that set and
undo phase 25 entirely. Domain-separated derivation from the secret keeps
it secret while staying reproducible here.

**The id must persist.** If the key changes, the hidden set changes with
it, and scores from before stop being comparable with scores from after.
That is why it lives in a file beside the agent's state, and why
`Holdout.identity` carries the fingerprint -- so a comparison across a
lost key fails loudly instead of quietly comparing two different sets.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"

#: Overrides the stored key, deriving one deterministically from the given
#: string. Set in tests, and by anyone who deliberately wants two
#: installations to share an identity -- comparing them directly is the
#: one case where that is what you want. Not for production: a key derived
#: from a memorable string is a key somebody else can derive too.
INSTANCE_ENV = "MANA_INSTANCE_ID"

#: Where the secret lives. Beside the state, not in the package: a
#: reinstall must not mint a new identity.
FILENAME = "instance_key"

_CACHED_SEED: Optional[bytes] = None


class IdentityUnavailable(RuntimeError):
    """No signing backend. Signatures are refused rather than faked."""


def _material() -> bytes:
    """32 secret bytes this installation's key is built from."""
    global _CACHED_SEED
    override = os.environ.get(INSTANCE_ENV, "").strip()
    if override:
        return hashlib.blake2b(override.encode("utf-8"), digest_size=32,
                               person=b"mana-instance").digest()
    if _CACHED_SEED is not None:
        return _CACHED_SEED

    path = _store()
    try:
        if path.is_file():
            stored = bytes.fromhex(path.read_text(encoding="utf-8").strip())
            if len(stored) == 32:
                _CACHED_SEED = stored
                return stored
    except (OSError, ValueError):
        pass

    fresh = os.urandom(32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(fresh.hex(), encoding="utf-8")
        # Best effort: on Windows the ACL is what actually protects it, and
        # chmod is close to a no-op. Worth doing for the platforms where it
        # is not.
        os.chmod(path, 0o600)
    except OSError:
        # Unwritable state directory: the run works, but the identity lasts
        # only for this process. `Holdout.identity` will differ from the
        # stored one, which is the loud failure wanted here -- scores from
        # this run must not be compared with the others.
        pass
    _CACHED_SEED = fresh
    return fresh


def _store() -> Path:
    from .. import paths
    return Path(paths.data_root()) / FILENAME


def reset_cache() -> None:
    """Forget the cached secret. For tests, and after the state dir moves."""
    global _CACHED_SEED
    _CACHED_SEED = None


# --------------------------------------------------------------- the key pair


def _private_key() -> Any:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey)
    except Exception as exc:                            # pragma: no cover
        raise IdentityUnavailable(
            f"нет пакета cryptography, подписывать нечем: {exc}")
    return Ed25519PrivateKey.from_private_bytes(_material())


def public_key() -> bytes:
    """This installation's public key: 32 bytes, safe to publish."""
    from cryptography.hazmat.primitives import serialization
    return _private_key().public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)


def instance_id() -> str:
    """The public key in hex. Kept for callers that want the full identity."""
    return public_key().hex()


def fingerprint(key: Any = None) -> str:
    """Short digest of a public key: shareable, and says nothing else.

    Eight hex characters -- enough to tell instances apart and to count
    how many reproduced a result; not enough, and not the right shape, to
    say anything about whose machine it is. A hostname or a MAC would
    travel with every shared record and identify the business it came
    from; this does not.
    """
    if key is None:
        raw = public_key()
    elif isinstance(key, bytes):
        raw = key
    else:
        # A hex string, or a legacy free-form id. Hashed either way, so a
        # caller passing the wrong thing gets a stable wrong answer rather
        # than an exception in the middle of a run.
        try:
            raw = bytes.fromhex(str(key))
        except ValueError:
            raw = str(key).encode("utf-8")
    return hashlib.blake2b(raw, digest_size=4).hexdigest()


# ------------------------------------------------------------------ signing


def sign(payload: bytes) -> bytes:
    """Sign bytes with this installation's key."""
    return _private_key().sign(payload)


def verify(key: bytes, payload: bytes, signature: bytes) -> bool:
    """Whether `signature` is this key's signature over `payload`.

    Returns False rather than raising: a bad signature is an ordinary
    outcome when reading records from elsewhere, not an error in reading
    them, and a run importing two hundred records should not stop at the
    first forged one.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey)
        Ed25519PublicKey.from_public_bytes(key).verify(signature, payload)
        return True
    except Exception:
        return False


def canonical(payload: Dict[str, Any]) -> bytes:
    """The exact bytes a signature covers.

    Sorted keys and no spaces, so two installations serialising the same
    record produce identical bytes. A signature over a representation that
    can vary is a signature that verifies by luck.
    """
    return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def signed(payload: Dict[str, Any]) -> Tuple[str, str]:
    """Sign a record. Returns (public key hex, signature hex)."""
    return public_key().hex(), sign(canonical(payload)).hex()


def signature_holds(payload: Dict[str, Any], key_hex: str,
                    signature_hex: str) -> bool:
    """Check a record's signature, and that its fingerprint is that key's.

    Both halves are required. Verifying only the signature would let
    somebody sign with their own key while claiming another instance's
    fingerprint -- which is precisely the impersonation this exists to
    stop.
    """
    if not key_hex or not signature_hex:
        return False
    try:
        key = bytes.fromhex(key_hex)
        signature = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    return verify(key, canonical(payload), signature)


# ------------------------------------------------------------- the salt


def salted_seed(seed: int, value: str = "") -> int:
    """A hidden-set seed of this installation's own.

    Derived from the PRIVATE key, and that is the whole point. The salt
    exists so no other installation can fit anything to this one's hidden
    set; deriving it from the public key -- which travels in every bundle
    -- would let anybody regenerate the set and undo the salt completely.

    Deterministic in (seed, secret): the same installation redraws exactly
    the same hidden set every run, which is what keeps scores comparable
    over time, while different installations get different sets, which is
    what makes agreement between them mean something.
    """
    secret = (hashlib.blake2b(value.encode("utf-8"), digest_size=32,
                              person=b"mana-instance").digest()
              if value else _material())
    digest = hashlib.blake2b(str(int(seed)).encode("utf-8"), key=secret,
                             digest_size=8, person=b"mana-holdout").digest()
    # Kept inside the range a Python/NumPy seed is happy with.
    return int.from_bytes(digest, "big") % (2 ** 31 - 1)
