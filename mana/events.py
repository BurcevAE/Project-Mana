"""
mana.events — one output channel for everything MANA says about itself.

Why this module exists
----------------------
MANA reported its work with 110 `print()` calls spread across cli.py,
agent_parts/core.py, voice.py, evolution.py, benchmarking.py, knowledge.py
and code_evolution.py. That works in a terminal and fails in two ways
outside one:

  * **A windowed application has no stdout.** Built with PyInstaller's
    `--windowed`, stdout is not a console -- on Windows it can be an
    invalid handle, and writing to it raises. The agent would die inside
    a status banner.

  * **The console it does have cannot encode what MANA writes.** Already
    observed in a real run: `_load_state()` printing a warning with an
    emoji raised `UnicodeEncodeError: 'charmap' codec can't encode` under
    cp1251. An unrelated failure (a stale pickle) turned into a crash
    because of how the *message about it* was printed.

So library code emits events; a sink decides what to do with them. The CLI
installs a console sink; the desktop app installs one that pushes to the
window. Neither the agent nor its subsystems know which.

What is deliberately NOT changed
--------------------------------
`cli.py` keeps its own `print()` calls. That module *is* the console
presentation layer -- routing its output through a bus only to print it
again would be indirection with no reader. The library modules under
`mana/` are the ones that must not assume a terminal.

Contract
--------
`emit()` never raises. An output channel that can bring down the agent is
worse than one that occasionally drops a line, and every call site here is
reporting on work that already succeeded or failed on its own.
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Event kinds. Keep this list short: a sink should be able to style every
#: one of them without a lookup table full of one-offs.
BANNER = "banner"      # startup identity block
STATUS = "status"      # steady-state fact about the agent
PROGRESS = "progress"  # a step inside long work (evolution cycle, benchmark)
WARNING = "warning"    # something degraded but the run continues
ERROR = "error"        # something failed
RESULT = "result"      # a finished artifact the user asked for


@dataclass
class Event:
    kind: str
    text: str
    data: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


Sink = Callable[[Event], None]


class EventBus:
    """Fan-out with no back-pressure and no failure propagation.

    A slow or broken sink must not stall or crash the agent, so a sink that
    raises is dropped from the bus rather than retried -- a sink failing
    repeatedly during evolution would otherwise emit an error per cycle
    forever.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sinks: List[Sink] = []
        self._history: List[Event] = []
        self._history_limit = 500

    def subscribe(self, sink: Sink, replay: bool = False) -> Sink:
        """Add a sink. `replay=True` hands it the events already emitted --
        the desktop window opens after the agent has started, and the
        startup banner is worth seeing."""
        with self._lock:
            self._sinks.append(sink)
            backlog = list(self._history) if replay else []
        for event in backlog:
            try:
                sink(event)
            except Exception:
                break
        return sink

    def unsubscribe(self, sink: Sink) -> None:
        with self._lock:
            if sink in self._sinks:
                self._sinks.remove(sink)

    def emit(self, kind: str, text: str, **data: Any) -> Event:
        event = Event(kind=kind, text=str(text), data=data)
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_limit:
                del self._history[:-self._history_limit]
            sinks = list(self._sinks)
        for sink in sinks:
            try:
                sink(event)
            except Exception:
                self.unsubscribe(sink)
        return event

    def history(self, limit: Optional[int] = None) -> List[Event]:
        with self._lock:
            events = list(self._history)
        return events[-limit:] if limit else events

    def clear(self) -> None:
        with self._lock:
            self._history.clear()


#: The process-wide bus. A single agent per process is the only shape MANA
#: supports (SQLite state, a state pickle, one evolution thread), so a
#: module-level bus matches reality and keeps call sites to `emit(...)`.
BUS = EventBus()


def emit(kind: str, text: str, **data: Any) -> Event:
    return BUS.emit(kind, text, **data)


def subscribe(sink: Sink, replay: bool = False) -> Sink:
    return BUS.subscribe(sink, replay=replay)


def unsubscribe(sink: Sink) -> None:
    BUS.unsubscribe(sink)


def write_console(text: str) -> None:
    """Print without ever raising, whatever the console can encode.

    Three layers, because all three failure modes are real on Windows:
    cp1251 cannot encode emoji or box-drawing characters; a `--windowed`
    build has no valid stdout handle at all; and a closed pipe raises on
    flush rather than on write.
    """
    stream = sys.stdout
    if stream is None:
        return
    try:
        stream.write(text + "\n")
        stream.flush()
        return
    except UnicodeEncodeError:
        pass
    except Exception:
        return
    try:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        stream.write(text.encode(encoding, errors="replace").decode(encoding) + "\n")
        stream.flush()
    except Exception:
        return


def console_sink(event: Event) -> None:
    """Default sink for terminal runs: warnings and errors get a marker,
    everything else prints as-is so existing CLI output is unchanged."""
    if event.kind == WARNING:
        write_console(f"[!] {event.text}")
    elif event.kind == ERROR:
        write_console(f"[x] {event.text}")
    else:
        write_console(event.text)


def install_console_sink() -> Sink:
    """Called by cli.main() so command-line runs look exactly as before."""
    return subscribe(console_sink)
