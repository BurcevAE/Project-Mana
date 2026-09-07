"""
mana.net.peers — the installations this one recognises.

Shaped like `known_hosts`, and for the same reason. Signatures prove a
message came from whoever holds a key; they say nothing about whether
that key should be listened to. One person can generate a thousand key
pairs, so Sybil resistance cannot come from cryptography here -- it comes
from a human deciding whose keys count.

Cryptocurrencies answer the same question with proof of work, which is
expensive precisely because they need agreement among parties who cannot
check a claim for themselves. MANA checks locally, so it never needs
agreement, and a list somebody curated is both cheaper and stronger.

What a peer entry is not
-------------------------
It is not permission to be believed. An added peer may send hypotheses,
which are things to try, and reports, which say what happened elsewhere.
Neither becomes true here: the local gates still rule on everything. So
adding a peer you later regret costs you some disk and no correctness.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

FILENAME = "peers.json"


class UnknownPeer(PermissionError):
    """A signed request from a key nobody added. Refused, not trusted."""


@dataclass
class Peer:
    """One installation this one will talk to."""
    fingerprint: str
    public_key: str                 # hex, 32 bytes
    address: str = ""               # host:port, empty for inbound-only peers
    label: str = ""                 # what a person calls this machine
    added: float = field(default_factory=time.time)
    last_seen: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PeerBook:
    """The peer list on disk. Small, human-readable, hand-editable.

    Deliberately a plain JSON file: somebody removing a peer at three in
    the morning should be able to do it with a text editor, and a format
    that requires the application to be running is a format that fails
    exactly when it is needed.
    """

    def __init__(self, path: Any = None) -> None:
        if path is None:
            from ..paths import data_root
            path = Path(data_root()) / FILENAME
        self.path = Path(path)

    # ---------- storage ----------

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.is_file():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data.get("peers", {}) if isinstance(data, dict) else {}

    def _save(self, peers: Dict[str, Dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"peers": peers}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        temporary.replace(self.path)

    # ---------- the list ----------

    def add(self, public_key: str, address: str = "", label: str = "") -> Peer:
        """Recognise an installation by its public key.

        The fingerprint is derived here rather than accepted from the
        caller. Taking a claimed fingerprint would let somebody file their
        key under another instance's name, which is the impersonation the
        whole scheme exists to stop -- one level up from where the
        signature check catches it.
        """
        from ..core.identity import fingerprint as digest

        key = str(public_key).strip().lower()
        try:
            raw = bytes.fromhex(key)
        except ValueError as exc:
            raise ValueError(f"ключ не шестнадцатеричный: {exc}") from exc
        if len(raw) != 32:
            raise ValueError(
                f"ключ Ed25519 — 32 байта, здесь {len(raw)}; "
                f"похоже, скопирована не та строка")

        peers = self._load()
        mark = digest(raw)
        existing = peers.get(mark, {})
        peer = Peer(fingerprint=mark, public_key=key, address=address,
                    label=label or existing.get("label", ""),
                    added=existing.get("added", time.time()),
                    last_seen=existing.get("last_seen", 0.0))
        peers[mark] = peer.as_dict()
        self._save(peers)
        return peer

    def remove(self, fingerprint: str) -> bool:
        peers = self._load()
        removed = peers.pop(fingerprint, None)
        if removed is not None:
            self._save(peers)
        return removed is not None

    def all(self) -> List[Peer]:
        return [Peer(**record) for record in self._load().values()]

    def by_fingerprint(self, fingerprint: str) -> Optional[Peer]:
        record = self._load().get(fingerprint)
        return Peer(**record) if record else None

    def by_key(self, public_key: str) -> Optional[Peer]:
        """Look one up by the key that signed a request.

        Keyed on the key, not on what the request claims to be: a caller
        may say anything about who they are, and only the key is evidence.
        """
        from ..core.identity import fingerprint as digest
        try:
            mark = digest(bytes.fromhex(str(public_key).strip().lower()))
        except ValueError:
            return None
        peer = self.by_fingerprint(mark)
        # A key that hashes to a known fingerprint but is not the stored
        # key would be a hash collision, and treating that as a match
        # would turn an 8-character digest into the credential.
        if peer is not None and peer.public_key != str(public_key).strip().lower():
            return None
        return peer

    def require(self, public_key: str) -> Peer:
        peer = self.by_key(public_key)
        if peer is None:
            raise UnknownPeer(
                "этот ключ не в списке известных; добавьте его командой "
                "--peer add <ключ>, если это ваша вторая установка")
        return peer

    def seen(self, fingerprint: str) -> None:
        peers = self._load()
        if fingerprint in peers:
            peers[fingerprint]["last_seen"] = time.time()
            self._save(peers)

    def describe(self) -> str:
        peers = self.all()
        if not peers:
            return "известных узлов нет"
        lines = []
        for peer in sorted(peers, key=lambda p: p.label or p.fingerprint):
            when = (time.strftime("%Y-%m-%d %H:%M", time.localtime(peer.last_seen))
                    if peer.last_seen else "не отвечал")
            lines.append(f"  {peer.fingerprint}  {peer.address or '(входящий)':22s} "
                         f"{peer.label or '':16s} {when}")
        return "\n".join(lines)
