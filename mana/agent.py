"""
mana.agent — ManaAgent: composition root.

ManaAgent itself now contains no logic — every method lives in one of the
mixins under mana/agent_parts/, grouped by concern (see that package's
docstring). This file's only job is to wire them together into one class,
exactly preserving the original single-class runtime behavior: all mixins
share the same `self`, so cross-concern calls (e.g. evolution code calling
into benchmarking, or execution code calling into routing) work unchanged.
"""
from __future__ import annotations

from .agent_parts.core import CoreMixin
from .agent_parts.context import ContextMixin
from .agent_parts.routing import RoutingMixin
from .agent_parts.confidence import ConfidenceMixin
from .agent_parts.execution import ExecutionMixin
from .agent_parts.benchmarking import BenchmarkingMixin
from .agent_parts.evolution import EvolutionMixin
from .agent_parts.knowledge_ops import KnowledgeOpsMixin


class ManaAgent(
    CoreMixin,
    ContextMixin,
    RoutingMixin,
    ConfidenceMixin,
    ExecutionMixin,
    BenchmarkingMixin,
    EvolutionMixin,
    KnowledgeOpsMixin,
):
    """Persistent, self-improving MANA agent.

    See mana/agent_parts/*.py for what each mixin owns. None of the mixins
    have overlapping method names (verified at split time), so inheritance
    order here only affects Python's MRO for attribute lookup, not behavior.
    """
    pass
