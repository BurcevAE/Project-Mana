"""
mana_desktop — the Windows application shell around the MANA agent.

Deliberately a separate package from `mana`: the agent must stay usable
(and testable) without a GUI, and nothing here may become a dependency of
it. The arrow points one way -- mana_desktop imports mana, never the
reverse.

  session.py  the agent on a worker thread, one question at a time
  server.py   a loopback HTTP surface + SSE event stream for the window
  window.py   pywebview shell, single-instance guard
  web/        the interface itself
"""
from __future__ import annotations

__version__ = "2.0"
