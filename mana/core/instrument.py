"""
mana.core.instrument — did the thing we were varying actually run?

A live meta-evolution run compared two search policies and got an
identical 1.398 from both. The natural reading is "this parameter does
not matter". The true reading was that at the budget used, the cycle
spent every step measuring untouched slices, where gap ranking is not
applied at all -- the weight under test was never read. An experiment on
a parameter the episodes do not consult shows no effect, and that is not
evidence about the parameter.

Nothing in the system could tell those two readings apart, and no amount
of statistics can: they produce the same numbers. The only thing that
separates them is a counter at the point of use.

Why this lives in the immutable core
------------------------------------
Same reason as `cost`: it is not an acceptance rule and no gate reads it,
but it decides whether a measurement is admissible evidence at all. A
layer that could switch off its own activation counters could report
experiments that never touched what they claimed to test.

What it detects, and what it cannot
-----------------------------------
It catches the accident: code that never ran. It does not catch
deception: a caller that calls `record_read` without using the value
would satisfy it. That is the honest limit, and it is acceptable here
because the counter's job is to stop MANA fooling itself, not to defend
against a lying caller -- and in the real path the read happens inside
`gaps._build`, which the meta layer does not control.

Deliberately dumb
-----------------
A counter and a lock. No sampling, no decay, no inference about what the
reads meant. `record_read` is called where a value is actually used in a
computation -- not where the module is imported, which would prove
nothing.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"

_lock = threading.Lock()
_counts: Dict[str, int] = {}


def record_read(name: str) -> None:
    """One use of a named parameter, at the point it was used.

    Never raises: instrumentation that can break the thing it measures is
    worse than no instrumentation.
    """
    try:
        with _lock:
            _counts[name] = _counts.get(name, 0) + 1
    except Exception:                                  # pragma: no cover
        pass


def reads(name: Optional[str] = None):
    """How many times a parameter was used, or the whole table."""
    with _lock:
        if name is None:
            return dict(_counts)
        return _counts.get(name, 0)


def reset() -> None:
    with _lock:
        _counts.clear()


@contextmanager
def watching() -> Iterator[Dict[str, int]]:
    """Count uses that happen inside this block.

        with instrument.watching() as used:
            run_one_episode()
        if not used.get("gap.cost"):
            # the episode never consulted the parameter under test

    The dict is filled on exit, so read it after the block. A snapshot
    diff rather than a reset, because resetting would clobber counting
    that another thread is in the middle of.
    """
    before = reads()
    delta: Dict[str, int] = {}
    try:
        yield delta
    finally:
        after = reads()
        for key, value in after.items():
            change = value - before.get(key, 0)
            if change:
                delta[key] = change


def exercised(delta: Dict[str, int], name: str) -> bool:
    """Was this parameter actually consulted during the watched block?"""
    return delta.get(name, 0) > 0
