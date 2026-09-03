"""
mana.core.evaluation — "am I being measured right now?" is not a property
the measured thing owns.

The audit found `_benchmark_holdout` as a mutable attribute on ManaAgent,
read by confidence.py and routing.py to switch learning off during a
holdout run. The intent was right. The placement meant the agent knew when
it was under evaluation and behaved differently, on a flag it also owned.

Here the mode is an object created by core and handed *in*. The agent
reads it and cannot construct a legitimate one for itself: `EvaluationMode`
carries a token issued by `open_evaluation()`, and code outside core that
fabricates a context gets one that reports `authorized=False` -- visible in
every audit record rather than silently equivalent.

Honest limit, same as the rest of the boundary: Python has no privacy, and
a determined patch can forge anything. What this buys is that forging it
is an explicit act that shows up in the record, instead of a one-line
assignment that looks like ordinary state.
"""
from __future__ import annotations

import secrets
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: What the agent is doing right now, from the evaluator's point of view.
NORMAL = "normal"          # ordinary work; learn from everything
DEV = "dev"                # measured on visible tasks; learning allowed
HELD_OUT = "held_out"      # measured on hidden/transfer; learning forbidden

_lock = threading.RLock()
_issued: set = set()


@dataclass(frozen=True)
class EvaluationMode:
    """The mode, plus proof that core issued it."""
    mode: str = NORMAL
    label: str = ""
    token: str = ""
    started: float = field(default_factory=time.time)

    @property
    def authorized(self) -> bool:
        """False for a context nobody in core issued. Not an exception --
        an unauthorized context still works, it is just marked, because a
        gate that crashes on a forged flag tells you less than a record
        that shows one was used."""
        with _lock:
            return bool(self.token) and self.token in _issued

    @property
    def learning_enabled(self) -> bool:
        """Whether outcomes from this run may update anything persistent.

        The single rule that matters: nothing learns from held-out tasks.
        Routing statistics, brain reputation, the learned router and the
        graph memory all consult this.
        """
        return self.mode != HELD_OUT

    @property
    def is_measured(self) -> bool:
        return self.mode != NORMAL

    def as_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "label": self.label, "authorized": self.authorized,
                "learning_enabled": self.learning_enabled, "started": self.started}


#: What an agent gets when nobody is measuring it. Authorized by
#: construction: "not being evaluated" is the default state of the world,
#: not a claim that needs proof.
NORMAL_MODE = EvaluationMode(mode=NORMAL, label="", token="")


@contextmanager
def open_evaluation(mode: str, label: str = "") -> Iterator[EvaluationMode]:
    """Issue an evaluation context for the duration of a measurement.

    A context manager rather than a setter, so the mode cannot outlive the
    run that needed it -- the failure the old boolean invited was leaving
    it set and silently disabling learning for the rest of the session.
    """
    if mode not in (NORMAL, DEV, HELD_OUT):
        raise ValueError(f"unknown evaluation mode: {mode!r}")
    token = secrets.token_hex(8)
    with _lock:
        _issued.add(token)
    ctx = EvaluationMode(mode=mode, label=label, token=token)
    try:
        yield ctx
    finally:
        with _lock:
            _issued.discard(token)


def normal() -> EvaluationMode:
    return NORMAL_MODE
