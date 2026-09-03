"""
mana.version — the single source of truth for version numbers.

Why this module exists: version strings used to be hardcoded in at least
five places and had already drifted into THREE different values at once
("5.4.0" in __init__, "5.4" in CoreMixin and MemoryManager, "5.3.1" in the
CLI description and the interactive banner). Anyone reading the code, a
log, or a benchmark report could not tell which number was authoritative.
Nothing here is computed -- these are declarations -- but they are
declared exactly once.

Two levels, deliberately:

  * PRODUCT_VERSION -- the release as a whole, what you name the download.
  * per-module __version__ -- each subsystem versions independently, so a
    change to, say, the verifier is traceable without inflating the number
    of everything else. Every module under mana/ declares its own
    `__version__`; component_versions() collects them.

Bump rules (conventions, not enforced by code):
  * patch (1.0 -> 1.0.1): bugfix, no behaviour change for correct callers
  * minor (1.0 -> 1.1): new behaviour, backwards compatible
  * major (1.0 -> 2.0): callers must change

Components started at 1.0 as the agreed baseline for release 5.7.8 --
this is a starting line, not a claim that each subsystem is mature.
"""
from __future__ import annotations

import importlib
from typing import Dict

__version__ = "1.0"

#: The release as a whole.
PRODUCT_VERSION = "5.12.2"

#: Modules that carry their own version. Kept as an explicit list rather
#: than discovered by scanning, so a module added without a version number
#: shows up as a test failure instead of silently disappearing from
#: reports.
VERSIONED_MODULES = (
    "config", "optional_deps", "hardware", "knowledge", "web", "llm",
    "brains", "decompose", "paths", "events",
    "pipeline", "experience", "verifier", "memory", "graph_memory", "intent",
    "tools", "code_evolution", "voice", "cli", "version", "episode_affinity",
    "agent_parts.core", "agent_parts.context", "agent_parts.routing",
    "agent_parts.confidence", "agent_parts.execution",
    "agent_parts.benchmarking", "agent_parts.evolution",
    "agent_parts.knowledge_ops",
)


def component_versions() -> Dict[str, str]:
    """Collect every module's declared __version__.

    Imports lazily and reports a missing/unimportable module rather than
    raising -- version reporting must never be the thing that breaks a
    run.
    """
    out: Dict[str, str] = {}
    for name in VERSIONED_MODULES:
        try:
            module = importlib.import_module(f"mana.{name}")
            out[name] = getattr(module, "__version__", "unversioned")
        except Exception as exc:
            out[name] = f"unavailable ({type(exc).__name__})"
    return out


def version_report() -> Dict[str, object]:
    return {"product": PRODUCT_VERSION, "components": component_versions()}


def format_version_report() -> str:
    lines = [f"MANA {PRODUCT_VERSION}", "components:"]
    for name, ver in sorted(component_versions().items()):
        lines.append(f"  {name:34} {ver}")
    return "\n".join(lines)
