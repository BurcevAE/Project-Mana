"""
mana.core.transaction — every accepted change is a transaction, and an
interrupted one is findable afterwards.

The requirement is short: after a crash the system must know whether a
self-modification was left half-done. The reason it is not optional: MANA
writes to its own source files, its genome and its memory, and a process
killed between "patch applied" and "genome updated" leaves a system whose
recorded state and actual behaviour disagree. That disagreement is
invisible -- everything keeps running -- and it corrupts every measurement
taken afterwards, because the baseline is no longer what the journal says
it is.

Shape
-----
    OPENED -> SNAPSHOT -> MEASURED -> DECIDED -> COMMITTED
                                             -> ROLLED_BACK

Each step is appended to a JSONL journal and flushed before the step it
records is performed, not after. Recording after would put the crash
window exactly where it does the most damage: the change happens, the
record does not.

`unfinished()` returns transactions with no terminal state. That is the
whole crash-recovery contract -- the caller decides what to do about them,
because "roll back automatically" is wrong for a transaction that had
already committed its file write and merely died before logging it.

The journal lives in the user's data directory, not in the package: it is
evidence about the code, and evidence stored inside the thing it describes
disappears with it on reinstall.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"

OPENED = "OPENED"
SNAPSHOT = "SNAPSHOT"
MEASURED = "MEASURED"
DECIDED = "DECIDED"
COMMITTED = "COMMITTED"
ROLLED_BACK = "ROLLED_BACK"
FAILED = "FAILED"

#: A transaction in any of these is finished; anything else was interrupted.
TERMINAL = (COMMITTED, ROLLED_BACK, FAILED)

_lock = threading.RLock()


def journal_path() -> Path:
    from ..paths import data_root
    root = data_root() / "experiments"
    root.mkdir(parents=True, exist_ok=True)
    return root / "transactions.jsonl"


@dataclass
class Transaction:
    txn_id: str
    claim_id: str
    kind: str
    description: str
    opened_at: float
    states: List[str] = field(default_factory=list)
    last_state: str = OPENED
    last_at: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return self.last_state in TERMINAL

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _append(record: Dict[str, Any]) -> None:
    """Write one journal line and get it onto the disk before returning.

    flush + fsync rather than relying on buffering: the whole point of the
    journal is to survive a process that does not exit cleanly, and a
    record still sitting in a userspace buffer survives nothing.
    """
    path = journal_path()
    with _lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def open_transaction(claim_id: str, kind: str, description: str = "",
                     **data: Any) -> str:
    txn_id = uuid.uuid4().hex[:16]
    _append({"txn_id": txn_id, "claim_id": claim_id, "kind": kind,
             "description": description, "state": OPENED, "at": time.time(),
             "data": data})
    return txn_id


def record(txn_id: str, state: str, **data: Any) -> None:
    """Log a step. Called BEFORE performing it.

    The ordering is the point. A record written afterwards leaves a window
    in which the change exists and the journal denies it -- which is the
    one failure recovery cannot detect, because there is nothing to find.
    A record written first can at worst report an intention that did not
    complete, and that is exactly what `unfinished()` is for.
    """
    _append({"txn_id": txn_id, "state": state, "at": time.time(), "data": data})


def close(txn_id: str, outcome: str, **data: Any) -> None:
    if outcome not in TERMINAL:
        raise ValueError(f"{outcome!r} is not a terminal state; use record() instead")
    _append({"txn_id": txn_id, "state": outcome, "at": time.time(), "data": data})


def read_journal() -> List[Transaction]:
    """Replay the journal into transactions, newest state last.

    Tolerant of a truncated final line: a process killed mid-write leaves
    exactly that, and refusing to read the journal because of it would
    make recovery impossible in precisely the case recovery exists for.
    """
    path = journal_path()
    if not path.exists():
        return []
    transactions: Dict[str, Transaction] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                      # truncated tail: skip, do not fail
            txn_id = rec.get("txn_id")
            if not txn_id:
                continue
            txn = transactions.get(txn_id)
            if txn is None:
                txn = Transaction(txn_id=txn_id, claim_id=rec.get("claim_id", ""),
                                  kind=rec.get("kind", ""),
                                  description=rec.get("description", ""),
                                  opened_at=rec.get("at", 0.0))
                transactions[txn_id] = txn
            state = rec.get("state", "")
            if state:
                txn.states.append(state)
                txn.last_state = state
                txn.last_at = rec.get("at", 0.0)
            if rec.get("data"):
                txn.data.update(rec["data"])
    return list(transactions.values())


def unfinished() -> List[Transaction]:
    """Transactions that never reached a terminal state.

    The crash-recovery contract in one function. Deliberately does NOT act
    on them: a transaction that died after writing a patch but before
    logging COMMITTED must not be rolled back blindly, and only the
    subsystem that owns the change knows which case it is.
    """
    return [t for t in read_journal() if not t.finished]


def history(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    txns = sorted(read_journal(), key=lambda t: t.opened_at, reverse=True)
    return [t.as_dict() for t in (txns[:limit] if limit else txns)]


class TransactionScope:
    """Context manager that cannot leave a transaction silently open.

        with TransactionScope("claim-1", "operator", "COUNTERFACTUAL") as txn:
            txn.step(SNAPSHOT, path=...)
            ...
            txn.commit(verdict=...)

    Exiting without commit() or rollback() records FAILED with the
    exception, because an unlabelled abandonment is the state that makes a
    journal untrustworthy.
    """

    def __init__(self, claim_id: str, kind: str, description: str = "", **data: Any):
        self.txn_id = open_transaction(claim_id, kind, description, **data)
        self._closed = False

    def step(self, state: str, **data: Any) -> None:
        record(self.txn_id, state, **data)

    def commit(self, **data: Any) -> None:
        close(self.txn_id, COMMITTED, **data)
        self._closed = True

    def rollback(self, **data: Any) -> None:
        close(self.txn_id, ROLLED_BACK, **data)
        self._closed = True

    def __enter__(self) -> "TransactionScope":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self._closed:
            close(self.txn_id, FAILED,
                  error=f"{exc_type.__name__}: {exc}" if exc_type else "scope exited without a decision")
        return False        # never swallow the exception
