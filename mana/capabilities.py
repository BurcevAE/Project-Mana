"""
mana.capabilities — what the agent can use right now, in one place.

Why this exists
----------------
The agent had no answer to "what can you do?". The pieces lived apart: the
tool registry knows which tools report themselves available, the brain
pool knows which models are keyed, healthy and within quota, the
application tools sit in the same registry under their own names, and
`cognition/self_model.py` keeps measured capability intervals that nothing
on the answer path reads. A planner choosing between a tool and a brain,
and a detector saying which capability failed, both need the same list
first.

Read-only on purpose. Nothing here changes what the agent does. It reports
what is there, carries the measured numbers where something measured
them, and says `None` where nothing did -- a brain nobody has called has
an unknown success rate, not a zero one.

What is not here yet
---------------------
Capabilities acquired through `mana/acquire.py` (verified external
oracles). `acquire.status_all()` runs the verification checks on every
call, which is right for `--capabilities` and wrong for something the
answer path may consult on every turn.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

TOOL = "tool"
BRAIN = "brain"


@dataclass
class Capability:
    """One thing the agent can use, and whether it can use it now."""
    capability_id: str
    kind: str
    name: str
    available: bool
    description: str = ""
    requires: Tuple[str, ...] = ()
    measured: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row["requires"] = list(self.requires)
        return row


def from_tools(registry: Any) -> List[Capability]:
    """Every registered tool, with the availability the tool reports.

    Tools keep no call statistics of their own, so nothing is measured
    here; the journal records calls per episode, not per tool.
    """
    out: List[Capability] = []
    for row in registry.list_tools():
        needs = tuple(name for name, flag in (("network", row["requires_network"]),
                                              ("exec", row["requires_exec"]),
                                              ("llm", row["requires_llm"])) if flag)
        out.append(Capability(
            capability_id=f"{TOOL}:{row['name']}", kind=TOOL, name=row["name"],
            available=bool(row["available"]), description=row["description"],
            requires=needs, measured={"cost_hint": row["cost_hint"]}))
    return out


def from_brains(pool: Any) -> List[Capability]:
    """Every brain in the catalog: configured, ready, and how it has done."""
    out: List[Capability] = []
    if pool is None:
        return out
    for brain_id, spec in sorted(pool.brains.items()):
        health = pool.health.get(brain_id)
        calls = int(getattr(health, "calls", 0) or 0)
        measured: Dict[str, Any] = {
            "configured": bool(pool.usable(spec)),
            "calls": calls,
            "success_rate": (round(float(health.success_rate()), 3)
                             if health is not None and calls else None),
            "latency": (round(float(health.ewma_latency), 3)
                        if health is not None and calls else None),
            "tier": getattr(spec, "tier", ""),
            "substrate": getattr(spec, "substrate", ""),
            "strengths": list(getattr(spec, "strengths", ()) or ()),
        }
        needs = () if getattr(spec, "local", False) else ("network",)
        out.append(Capability(
            capability_id=f"{BRAIN}:{brain_id}", kind=BRAIN, name=brain_id,
            available=bool(pool.ready(brain_id)),
            description=f"{getattr(spec, 'provider', '')}/{getattr(spec, 'model', '')}",
            requires=needs, measured=measured))
    return out


def snapshot(registry: Any, pool: Optional[Any] = None) -> List[Capability]:
    """Tools and brains, side by side."""
    return from_tools(registry) + from_brains(pool)


def describe(rows: List[Capability]) -> str:
    lines = []
    for kind, title in ((TOOL, "инструменты"), (BRAIN, "мозги")):
        chosen = [row for row in rows if row.kind == kind]
        if not chosen:
            continue
        ready = sum(1 for row in chosen if row.available)
        lines.append(f"{title}: готово {ready} из {len(chosen)}")
        for row in chosen:
            mark = "+" if row.available else "-"
            lines.append(f"  {mark} {row.name}: {row.description}")
    return "\n".join(lines)
