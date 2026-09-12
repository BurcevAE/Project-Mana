"""
mana.agent_parts.planning — PlanningMixin: one decision about how a turn is
served, made before any of it runs.

What this replaces
-------------------
Three selectors decided a turn without knowing about one another: the
imperative matcher in `apps/intent.py` (launch a program, or not), the
router in `routing.py` (local, web or mixed), and the brain strategy and
graph in `execution.py` (one brain, consensus or decomposition; which
nodes run). Each was right on its own ground. Nothing put the decision in
one place, so there was no single answer to "why was this turn served the
way it was", and nothing a gap detector could compare the outcome against.

The planner calls the same functions, in the same order, with the same
inputs -- a test holds it to the decisions the answer path then carries
out. What it adds is that the decision exists as one object, is taken
once, and travels on the result.

What it does not decide
------------------------
The adaptive loop may still vary the brain strategy step by step, because
each step runs a different architecture of the same spec. The plan records
the strategy for the spec in force; what each step actually used is in the
trace, where it always was.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

from ..pipeline import PipelineSpec

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

CLARIFY = "clarify"
REMEMBER = "remember"
APP_ACTION = "app_action"
ANSWER = "answer"


@dataclass
class Plan:
    """How one turn will be served, and why."""
    kind: str
    capability: str
    reason: str
    route: str = ""
    path: str = ""
    graph: Tuple[str, ...] = ()
    required_nodes: Tuple[str, ...] = ()
    brain_strategy: str = ""
    #: The ambiguity or the matched intent. Carried to the executor, never
    #: serialised: it is an object, and the record keeps what it said.
    detail: Any = None

    def as_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row.pop("detail", None)
        row["graph"] = list(self.graph)
        row["required_nodes"] = list(self.required_nodes)
        return row


class PlanningMixin:
    def plan_task(self, task: str, spec: Optional[PipelineSpec] = None) -> Plan:
        """Decide how this turn is served. Reads state; changes nothing."""
        if self.config.clarify_ambiguous_followups:
            ambiguous = self._ambiguous_followup(task)
            if ambiguous:
                return Plan(kind=CLARIFY, capability="",
                            reason=str(getattr(ambiguous, "reason", "")),
                            detail=ambiguous)
        if self._is_memory_write_request(task):
            return Plan(kind=REMEMBER, capability="memory:explicit",
                        reason="просьба запомнить")
        found = self._match_app_intent(task)
        if found is not None:
            return Plan(kind=APP_ACTION, capability=f"tool:{found.tool}",
                        reason=f"повелительная просьба: {found.action}",
                        detail=found)
        return self._plan_answer(task, spec)

    def _plan_answer(self, task: str, spec: Optional[PipelineSpec] = None) -> Plan:
        """The answer path: route, execution path, graph and brain strategy."""
        spec = PipelineSpec(**asdict(spec or self.pipeline)).normalize(self.config)
        route = self._effective_route(task, spec)
        adaptive = getattr(spec, "route_mode", "auto") == "auto"
        graph: Tuple[str, ...] = ()
        required: Tuple[str, ...] = ()
        if adaptive:
            graph = tuple(self._graph_for_task(task, spec, route))
            required = tuple(sorted(self._required_nodes_for_task(task, spec, graph, route)))
        uses_brain = bool(spec.use_llm) and self._tool_available("llm_generate")
        return Plan(kind=ANSWER,
                    capability="brain:pool" if uses_brain else "fallback:local",
                    reason="ответ" if uses_brain else "ответ без модели: ни одного мозга",
                    route=route, path="adaptive" if adaptive else "routed",
                    graph=graph, required_nodes=required,
                    brain_strategy=self._brain_strategy(task, spec) if uses_brain else "")

    def _match_app_intent(self, task: str) -> Any:
        """The instruction this message gives to a program, or None.

        Any failure is None, exactly as `_perform_app_intent` treated it:
        a matcher that raises must never stop the agent from answering.
        """
        try:
            from ..apps import intent as app_intent
            return app_intent.match(task)
        except Exception as exc:
            self._vlog(f"app intent match failed: {exc}")
            return None
