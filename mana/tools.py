"""
mana.tools — the Tool Engine: every capability the agent can invoke (LLM
generation, web search, code/answer verification, memory read/write,
graph-memory traversal) goes through one uniform protocol and registry,
with enough metadata (cost, required capability) to support a future
planner choosing between them.

Design boundary, deliberately drawn and documented here rather than left
implicit: a capability becomes a Tool when the agent *conditionally
chooses* to invoke it based on the task or PipelineSpec (web search is
optional, code execution is optional, which LLM provider is optional,
whether to consult the flat knowledge base or the graph memory is
optional). A capability that is unconditional bookkeeping the agent
always performs regardless of task -- writing to the event log,
persistent_memory.remember_*, graph_memory.record_turn during solve_task,
MemoryManager.context_for's internal orchestration of its own tables --
stays as direct method calls. The registry mediates cross-subsystem
capability choices at the orchestration layer (agent_parts/*), not every
internal call within a single subsystem (memory.py/verifier.py/llm.py
each still call their own helper methods directly). Blurring that line
would turn the registry into decoration around calls that were never
actually optional, which is the opposite of what it's for.

Every primitive the agent's real execution/context path uses now goes
through here -- this replaced direct self.llm/self.web/self.verifier/
self.memory calls throughout agent_parts/execution.py, context.py,
confidence.py, core.py and knowledge_ops.py in one coordinated pass (see
CHANGELOG note in each file), not incrementally one node at a time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolResult:
    ok: bool
    output: Any = None
    error: str = ""
    latency: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


class BaseTool:
    """Subclass and implement `run()`, or use FunctionTool for the common
    case of wrapping an existing method. `name` must be unique in a
    registry. The metadata fields below are what a future planner would
    weigh a tool by; they are illustrative/qualitative hints set by
    whoever registers the tool, not measured/calibrated costs -- treat
    them as "roughly how expensive/risky is calling this", not as
    guarantees."""
    name: str = "unnamed_tool"
    description: str = ""
    requires_network: bool = False
    requires_exec: bool = False
    requires_llm: bool = False
    cost_hint: float = 1.0

    def run(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError

    def is_available(self) -> bool:
        """Whether this tool can usefully be called right now (e.g. the
        underlying LLM/web backend is actually enabled). Default: always
        available; FunctionTool lets a factory override this per-tool."""
        return True

    def __call__(self, **kwargs: Any) -> ToolResult:
        started = time.perf_counter()
        try:
            result = self.run(**kwargs)
        except Exception as exc:  # a broken tool must never crash the caller
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}",
                               latency=time.perf_counter() - started)
        if result.latency == 0.0:
            result.latency = time.perf_counter() - started
        return result


class FunctionTool(BaseTool):
    """Wrap a plain callable as a Tool without writing a subclass -- the
    common case for adapting an existing method (WebSearcher.search, ...)."""

    def __init__(self, name: str, description: str, fn: Callable[..., ToolResult],
                 requires_network: bool = False, requires_exec: bool = False,
                 requires_llm: bool = False, cost_hint: float = 1.0,
                 available_fn: Optional[Callable[[], bool]] = None):
        self.name = name
        self.description = description
        self._fn = fn
        self.requires_network = requires_network
        self.requires_exec = requires_exec
        self.requires_llm = requires_llm
        self.cost_hint = float(cost_hint)
        self._available_fn = available_fn

    def run(self, **kwargs: Any) -> ToolResult:
        return self._fn(**kwargs)

    def is_available(self) -> bool:
        if self._available_fn is None:
            return True
        try:
            return bool(self._available_fn())
        except Exception:
            return False


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool, replace: bool = False) -> None:
        if not replace and tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' is already registered "
                              f"(pass replace=True to override deliberately)")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        """The single dispatch point every capability invocation in the
        live execution/context path now goes through. Calling an unknown
        tool never raises -- it comes back as a normal failed ToolResult,
        same as any other tool failure, so callers only need one error
        -handling path."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"no such tool: '{name}'")
        return tool(**kwargs)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [{"name": t.name, "description": t.description,
                 "requires_network": t.requires_network, "requires_exec": t.requires_exec,
                 "requires_llm": t.requires_llm, "cost_hint": t.cost_hint,
                 "available": t.is_available()} for t in self._tools.values()]

    def describe(self) -> str:
        """Human/LLM-readable tool directory, one line per tool."""
        if not self._tools:
            return "(no tools registered)"
        return "\n".join(f"- {t.name} (cost~{t.cost_hint}, available={t.is_available()}): {t.description}"
                          for t in self._tools.values())


# ---------------------------------------------------------------------------
# Adapters for MANA's existing capabilities
# ---------------------------------------------------------------------------

def make_web_search_tool(web_searcher: Any) -> BaseTool:
    def _run(**kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, error="query is required")
        max_results = kwargs.get("max_results")
        results, meta = web_searcher.search(query, max_results=max_results)
        return ToolResult(ok=bool(results) or meta.get("ok", False), output=results, meta=meta)
    return FunctionTool(
        "web_search",
        "Search the web for current information (news, prices, current versions/facts). "
        "Args: query (str, required), max_results (int, optional).",
        _run, requires_network=True, cost_hint=3.0,
        available_fn=lambda: web_searcher.enabled,
    )


def make_verify_arithmetic_tool(verifier: Any) -> BaseTool:
    def _run(**kwargs: Any) -> ToolResult:
        expr = str(kwargs.get("expression", "")).strip()
        if not expr:
            return ToolResult(ok=False, error="expression is required")
        result = verifier.verify_expression(expr)
        return ToolResult(ok=bool(result.get("ok")), output=result)
    return FunctionTool(
        "verify_arithmetic",
        "Evaluate a pure arithmetic expression safely (no LLM, no network). "
        "Args: expression (str, required, e.g. '17*23').",
        _run, cost_hint=0.1,
    )


def make_verify_answer_tool(verifier: Any) -> BaseTool:
    def _run(**kwargs: Any) -> ToolResult:
        task = str(kwargs.get("task", ""))
        answer = str(kwargs.get("answer", ""))
        category = str(kwargs.get("category", "general"))
        result = verifier.verify(task, answer, category)
        return ToolResult(ok=bool(result.get("verified")), output=result)
    return FunctionTool(
        "verify_answer",
        "Check whether a claimed answer matches a ground-truth value MANA can "
        "extract from the task itself (currently: arithmetic claims only). "
        "Args: task (str, required), answer (str, required), category (str, required).",
        _run, cost_hint=0.2,
    )


def make_code_exec_tool(verifier: Any) -> BaseTool:
    def _run(**kwargs: Any) -> ToolResult:
        code = str(kwargs.get("code", "")).strip()
        if not code:
            return ToolResult(ok=False, error="code is required")
        tests = kwargs.get("tests")
        result = (verifier.verify_code_with_tests(code, str(tests))
                  if tests else verifier.verify_code(code))
        return ToolResult(ok=bool(result.get("ok")), output=result,
                           error="" if result.get("ok") else str(result.get("stderr") or result.get("reason") or result.get("error") or ""))
    return FunctionTool(
        "run_code",
        "Run a small Python snippet in the local sandbox (subject to "
        "config.local_exec_enabled and the sandbox's import/timeout policy). "
        "Args: code (str, required), tests (str, optional -- assert/print checks run alongside code).",
        _run, requires_exec=True, cost_hint=2.0,
        available_fn=lambda: bool(verifier.config.local_exec_enabled),
    )


def make_knowledge_search_tool(knowledge_base: Any) -> BaseTool:
    def _run(**kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, error="query is required")
        top_k = int(kwargs.get("top_k", 5) or 5)
        min_confidence = float(kwargs.get("min_confidence", 0.0) or 0.0)
        entries = knowledge_base.search(query, top_k=top_k, min_confidence=min_confidence)
        return ToolResult(ok=bool(entries), output=[e.to_dict() for e in entries])
    return FunctionTool(
        "search_knowledge_base",
        "Search MANA's own acquired/learned knowledge base (v3.4-compatible "
        "flat store, populated by --learn / acquire_knowledge). "
        "Args: query (str, required), top_k (int, optional), min_confidence (float, optional).",
        _run, cost_hint=0.3,
    )


def make_write_memory_tool(knowledge_base: Any) -> BaseTool:
    def _run(**kwargs: Any) -> ToolResult:
        content = str(kwargs.get("content", "")).strip()
        if not content:
            return ToolResult(ok=False, error="content is required")
        knowledge_base.add(
            content,
            source=str(kwargs.get("source", "agent")),
            confidence=float(kwargs.get("confidence", 0.5)),
            metadata=kwargs.get("metadata") or {},
            status=str(kwargs.get("status", "unverified")),
        )
        return ToolResult(ok=True, output={"stored": True})
    return FunctionTool(
        "write_memory",
        "Store a distilled answer into MANA's flat knowledge base. "
        "Args: content (str, required), source (str), confidence (float), "
        "status (str), metadata (dict).",
        _run, cost_hint=0.1,
    )


def make_memory_search_tool(memory_manager: Any) -> BaseTool:
    def _run(**kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, error="query is required")
        session_id = str(kwargs.get("session_id", "") or "")
        limit = kwargs.get("limit")
        items = memory_manager.semantic_search(query, limit=limit, session_id=session_id)
        return ToolResult(ok=bool(items), output=items)
    return FunctionTool(
        "search_conversation_memory",
        "Semantic search over MANA's persistent multi-tier conversation "
        "memory (events/facts/episodes). Args: query (str, required), "
        "session_id (str, optional), limit (int, optional).",
        _run, cost_hint=0.3,
    )


def make_graph_memory_search_tool(graph_memory_store: Any) -> BaseTool:
    def _run(**kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, error="query is required")
        session_id = str(kwargs.get("session_id", "") or "")
        context, trace = graph_memory_store.graph_context(
            session_id, query,
            depth=int(kwargs.get("depth", 2)),
            limit=int(kwargs.get("limit", 8)),
            seed_limit=int(kwargs.get("seed_limit", 3)),
            char_budget=int(kwargs.get("char_budget", 2000)),
        )
        return ToolResult(ok=bool(context), output=context, meta={"trace": trace})
    return FunctionTool(
        "search_graph_memory",
        "Retrieve context by traversing MANA's graph memory (distilled turns "
        "linked by FOLLOWS/MENTIONS/PART_OF edges), not just top-k similarity. "
        "Args: query (str, required), session_id (str, required), depth/limit/"
        "seed_limit/char_budget (int, optional).",
        _run, cost_hint=0.6,
    )


def make_llm_generate_tool(llm_client: Any) -> BaseTool:
    def _run(**kwargs: Any) -> ToolResult:
        prompt = str(kwargs.get("prompt", ""))
        if not prompt:
            return ToolResult(ok=False, error="prompt is required")
        difficulty = kwargs.get("difficulty")
        text, meta = llm_client.ask_detailed(
            prompt,
            system=str(kwargs.get("system", "") or ""),
            temperature=float(kwargs.get("temperature", 0.2)),
            provider=str(kwargs.get("provider", "auto") or "auto"),
            context_tag=str(kwargs.get("context_tag", "") or ""),
            kind=str(kwargs.get("kind", "general") or "general"),
            difficulty=None if difficulty is None else float(difficulty),
            task=str(kwargs.get("task", "") or ""),
            policy=str(kwargs.get("policy", "") or ""),
        )
        return ToolResult(ok=bool(text), output=text,
                           error="" if text else (meta.error or "no response"),
                           latency=meta.latency, meta=asdict(meta))
    return FunctionTool(
        "llm_generate",
        "Generate text from MANA's brain pool. The pool -- not this caller "
        "-- decides which model answers, from task kind/difficulty plus "
        "measured health, and fails over if that brain is down or out of "
        "free-tier quota. Every LLM call in the agent (answers, critic, "
        "repair, synthesis, verification-bundle generation, graph-memory "
        "distillation, self-improvement patches) routes through this one "
        "tool. Args: prompt (str, required); system/provider/context_tag/"
        "kind/policy/task (str, optional); temperature/difficulty (float, "
        "optional). provider='auto' or a brain id (see list_brains); "
        "legacy provider names still resolve.",
        _run, requires_llm=True, requires_network=True, cost_hint=5.0,
        available_fn=lambda: llm_client.enabled,
    )


def make_llm_consensus_tool(llm_client: Any) -> BaseTool:
    """Ask several brains the same thing and report their agreement.

    Separate from llm_generate on purpose: this one costs N calls against N
    free-tier quotas, so a caller must opt into it explicitly rather than
    getting it as a hidden default. Its distinctive output is `agreement`
    -- with one brain MANA can only report what a model said; with two
    independent ones it can report whether they said the same thing.
    """
    def _run(**kwargs: Any) -> ToolResult:
        prompt = str(kwargs.get("prompt", ""))
        if not prompt:
            return ToolResult(ok=False, error="prompt is required")
        difficulty = kwargs.get("difficulty")
        res = llm_client.ask_consensus(
            prompt,
            n=int(kwargs.get("n", 2) or 2),
            system=str(kwargs.get("system", "") or ""),
            temperature=float(kwargs.get("temperature", 0.2)),
            kind=str(kwargs.get("kind", "general") or "general"),
            difficulty=None if difficulty is None else float(difficulty),
            task=str(kwargs.get("task", "") or ""),
            policy=str(kwargs.get("policy", "") or ""),
            context_tag=str(kwargs.get("context_tag", "") or ""),
        )
        return ToolResult(
            ok=bool(res.get("ok")), output=res.get("text"),
            error=str(res.get("error") or ""), latency=float(res.get("latency", 0.0)),
            meta={"agreement": float(res.get("agreement", 0.0)),
                  "disagreement": bool(res.get("disagreement")),
                  "brains": list(res.get("brains") or []),
                  "single": bool(res.get("single")),
                  "brain": str(res.get("brain") or "")},
        )
    return FunctionTool(
        "llm_consensus",
        "Ask N different brains the same prompt in parallel and return the "
        "medoid answer plus an agreement score in meta. Use when being "
        "wrong is expensive: agreement is weak corroboration, disagreement "
        "is a strong signal to verify or to say the answer is uncertain. "
        "Costs N calls. Args: prompt (str, required), n (int, default 2), "
        "system/kind/policy/task/context_tag (str), temperature/difficulty "
        "(float).",
        _run, requires_llm=True, requires_network=True, cost_hint=12.0,
        available_fn=lambda: llm_client.enabled and len(llm_client.pool.available()) >= 2,
    )


def make_decompose_tool(agent: Any) -> BaseTool:
    """Split a task, solve the parts on different brains, synthesize.

    Registered as a tool rather than buried in ExecutionMixin so that the
    same visibility rules as every other capability apply: it shows up in
    --list-tools, it reports availability honestly (it needs at least two
    ready brains to be worth anything), and its trace says which brain
    answered which part.
    """
    def _run(**kwargs: Any) -> ToolResult:
        from .decompose import solve
        task = str(kwargs.get("task", "") or kwargs.get("prompt", ""))
        if not task:
            return ToolResult(ok=False, error="task is required")
        cfg = agent.config
        use_planner = bool(kwargs.get("llm_planner", True)) and agent._tool_available("llm_generate")
        res = solve(
            task, agent.llm.pool,
            ask_planner=(agent._llm_call if use_planner else None),
            temperature=float(kwargs.get("temperature", 0.2)),
            context_tag=str(kwargs.get("context_tag", "") or "DECOMPOSE"),
            max_subtasks=int(kwargs.get("max_subtasks", cfg.decompose_max_subtasks)),
            max_parallel=int(cfg.decompose_max_parallel),
            policy=str(kwargs.get("policy", "") or ""),
        )
        return ToolResult(ok=bool(res.get("ok")), output=res.get("answer"),
                           error=str(res.get("error") or ""), latency=float(res.get("latency", 0.0)),
                           meta={k: v for k, v in res.items() if k not in {"answer", "ok"}})
    return FunctionTool(
        "decompose_task",
        "Break a complex task into subtasks, route each subtask to the "
        "brain best suited to it (running independent ones in parallel on "
        "different providers), then synthesize one answer. Meta carries the "
        "plan, per-subtask results and which brain produced each. Args: "
        "task (str, required), max_subtasks (int), temperature (float), "
        "llm_planner (bool, default true -- false uses the offline "
        "heuristic planner), policy/context_tag (str).",
        _run, requires_llm=True, requires_network=True, cost_hint=20.0,
        available_fn=lambda: (agent.config.decompose_enabled
                              and agent.llm.enabled
                              and len(agent.llm.pool.available()) >= 1),
    )

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.1"


def build_default_registry(agent: Any) -> ToolRegistry:
    """Wrap `agent`'s already-constructed subsystems (llm/web/verifier/
    memory/persistent_memory/graph_memory) as Tools and return a populated
    registry. Called once from CoreMixin.__init__ after all of those
    subsystems exist."""
    registry = ToolRegistry()
    registry.register(make_llm_generate_tool(agent.llm))
    registry.register(make_llm_consensus_tool(agent.llm))
    registry.register(make_decompose_tool(agent))
    registry.register(make_web_search_tool(agent.web))
    registry.register(make_verify_arithmetic_tool(agent.verifier))
    registry.register(make_verify_answer_tool(agent.verifier))
    registry.register(make_code_exec_tool(agent.verifier))
    registry.register(make_knowledge_search_tool(agent.memory))
    registry.register(make_write_memory_tool(agent.memory))
    registry.register(make_memory_search_tool(agent.persistent_memory))
    registry.register(make_graph_memory_search_tool(agent.graph_memory))
    return registry
