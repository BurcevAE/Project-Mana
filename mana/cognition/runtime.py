"""
mana.cognition.runtime — running a program of thought.

An interpreter, not a second agent
----------------------------------
Each operator is bound to something MANA can already do: GENERATE to
`_llm_call`, VERIFY to the `verify_answer` tool, RETRIEVE to memory and
web search, DECOMPOSE to `decompose.solve`. The runtime's whole job is to
walk the chain, keep the table of intermediate values, count what is
spent, and stop when the budget says so.

That restraint is deliberate. If the runtime re-implemented answering, the
project would have two execution paths whose behaviour drifts, and every
measured difference between "the old path" and "a cognitive program" would
be contaminated by the difference between two implementations rather than
between two strategies. Binding to the existing methods keeps the
comparison honest: the same machinery, in a different order.

Failure is a value, not an exception
------------------------------------
A step that fails records the failure and the chain continues to the next
step, which may still be able to work with what is on the table. A chain
that raises tells the experiment layer nothing about *where* it broke, and
"which step failed" is the first question about any bad answer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .ir import (ABSTRACTION, ANSWER, CONTEXT, CRITIQUE, DRAFT, EVIDENCE, PLAN,
                 PREDICTION, TASK, CognitiveOperator)
from .programs import (Budget, CognitiveProgram, ProgramResult, ProgramState,
                       StepRecord)

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: An operator implementation: given the table, produce new values.
#: Returning {} means "ran, produced nothing" -- different from raising,
#: which means "could not run".
OperatorImpl = Callable[[ProgramState], Dict[str, Any]]


@dataclass
class StepOutcome:
    """What an implementation reports back beyond its values."""
    values: Dict[str, Any]
    calls: int = 0
    brain: str = ""
    detail: Optional[Dict[str, Any]] = None


class Runtime:
    """Walks a program. Knows nothing about brains, tools or memory --
    only about the implementations it was handed."""

    def __init__(self, operators: Dict[str, CognitiveOperator],
                 implementations: Dict[str, OperatorImpl]) -> None:
        self.operators = operators
        self.implementations = implementations

    def missing_implementations(self, program: CognitiveProgram) -> List[str]:
        """Operators the program needs and this runtime cannot run.

        Checked before starting rather than discovered at step four: a
        program that dies halfway has already spent budget, and its
        partial trace would be scored as a capability failure of the
        strategy rather than of the binding.
        """
        expanded = program.expand(self.operators)
        return sorted({op for op in expanded if op not in self.implementations})

    def run(self, program: CognitiveProgram, task: str,
            budget: Optional[Budget] = None) -> ProgramResult:
        budget = budget or Budget()
        state = ProgramState(task=task)
        steps = program.expand(self.operators)
        brains: List[str] = []

        missing = self.missing_implementations(program)
        if missing:
            return ProgramResult(
                program=program, answer="", ok=False, stopped_early=True,
                stop_reason=f"no implementation for {missing}",
                calls_used=0, elapsed=0.0, trace=[])

        stop_reason = ""
        for index, op_id in enumerate(steps):
            spent = budget.exhausted(state.calls_used, state.elapsed)
            if spent:
                stop_reason = spent
                break

            operator = self.operators.get(op_id)
            if operator is not None and not operator.accepts(state.available):
                # Not fatal: a chain may reach a step whose input an
                # earlier failure never produced. Recording and continuing
                # keeps the trace honest about which step was skipped and
                # why, instead of collapsing the run.
                state.trace.append(StepRecord(
                    index=index, op_id=op_id, ok=False,
                    error=f"needs {list(operator.inputs)}, has {state.available}"))
                continue

            started = time.perf_counter()
            try:
                outcome = self.implementations[op_id](state)
            except Exception as exc:
                state.trace.append(StepRecord(
                    index=index, op_id=op_id, ok=False,
                    latency=time.perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}"))
                continue

            if not isinstance(outcome, StepOutcome):
                outcome = StepOutcome(values=dict(outcome or {}))
            for type_name, value in (outcome.values or {}).items():
                state.put(type_name, value)
            state.calls_used += max(0, int(outcome.calls))
            if outcome.brain:
                brains.append(outcome.brain)
            state.trace.append(StepRecord(
                index=index, op_id=op_id, ok=True,
                produced=tuple(outcome.values or {}),
                calls=int(outcome.calls), latency=time.perf_counter() - started,
                brain=outcome.brain, detail=outcome.detail or {}))

        answer = state.answer()
        return ProgramResult(
            program=program, answer=answer, ok=bool(answer),
            stopped_early=bool(stop_reason), stop_reason=stop_reason,
            calls_used=state.calls_used, elapsed=state.elapsed,
            trace=state.trace, brains_used=tuple(dict.fromkeys(brains)))


# ---------------------------------------------------------------------------
# binding operators to the agent MANA already is
# ---------------------------------------------------------------------------

def bind_agent(agent: Any) -> Dict[str, OperatorImpl]:
    """Map the primitive operators onto existing agent capabilities.

    This is the integration point the audit identified: every one of these
    goes through a choke point that already exists (`_llm_call`,
    `tools.call`, `decompose.solve`), so a cognitive program uses exactly
    the machinery the old path uses -- brain selection, failover, the
    independent critic, the trust levels -- and the only difference
    between them is the order of the steps.
    """

    def observe(state: ProgramState) -> StepOutcome:
        # Free by construction: reading the task is not a call. It exists
        # as an operator so that a program can be a complete description
        # of the work rather than assuming a hidden first step.
        return StepOutcome(values={}, calls=0)

    def retrieve(state: ProgramState) -> StepOutcome:
        pieces: List[str] = []
        try:
            found = agent.tools.call("search_conversation_memory", query=state.task, limit=3)
            if found.ok and found.output:
                pieces.append(str(found.output)[:1200])
        except Exception:
            pass
        if agent._tool_available("web_search"):
            try:
                web = agent.tools.call("web_search", query=state.task, max_results=3)
                if web.ok and web.output:
                    pieces.append(str(web.output)[:1200])
            except Exception:
                pass
        return StepOutcome(values={CONTEXT: "\n\n".join(pieces)} if pieces else {}, calls=0)

    def generate(state: ProgramState) -> StepOutcome:
        context = state.get(CONTEXT) or ""
        prompt = state.task if not context else f"{context}\n\nЗадача: {state.task}"
        text, meta = agent._llm_call(prompt, temperature=0.0, task=state.task,
                                     context_tag="COGPROG GENERATE")
        return StepOutcome(values={DRAFT: text} if text else {}, calls=1,
                           brain=meta.brain or meta.provider,
                           detail={"attempts": list(meta.attempts)})

    def critique(state: ProgramState) -> StepOutcome:
        draft = state.get(DRAFT)
        if not draft:
            return StepOutcome(values={}, calls=0)
        author = ""
        for record in reversed(state.trace):
            if record.op_id == "GENERATE" and record.brain:
                author = record.brain
                break
        text, meta = agent._llm_call(
            f"Проверь ответ на задачу. Первая строка: SCORE: число от 0 до 1.\n"
            f"Задача: {state.task}\nОтвет: {draft}",
            temperature=0.0, kind="reasoning", avoid=(author,) if author else (),
            context_tag="COGPROG CRITIQUE")
        return StepOutcome(values={CRITIQUE: text} if text else {}, calls=1,
                           brain=meta.brain or meta.provider,
                           detail={"independent": bool(meta.independent)})

    def repair(state: ProgramState) -> StepOutcome:
        draft, note = state.get(DRAFT), state.get(CRITIQUE)
        if not draft or not note:
            return StepOutcome(values={}, calls=0)
        text, meta = agent._llm_call(
            f"Исправь ответ только при наличии ошибок.\nЗадача: {state.task}\n"
            f"Черновик: {draft}\nКритик: {note}\nВерни только исправленный ответ.",
            temperature=0.0, context_tag="COGPROG REPAIR")
        return StepOutcome(values={DRAFT: text} if text else {}, calls=1,
                           brain=meta.brain or meta.provider)

    def verify(state: ProgramState) -> StepOutcome:
        draft = state.get(DRAFT)
        if not draft:
            return StepOutcome(values={}, calls=0)
        result = agent._verify_answer(state.task, str(draft), "general")
        return StepOutcome(values={EVIDENCE: result}, calls=0,
                           detail={"kind": result.get("kind"),
                                   "verified": bool(result.get("verified"))})

    def decompose_step(state: ProgramState) -> StepOutcome:
        from ..decompose import plan_heuristic, plan_with_llm
        planner = agent._llm_call if agent._tool_available("llm_generate") else None
        plan = plan_with_llm(state.task, planner) if planner else plan_heuristic(state.task)
        return StepOutcome(values={PLAN: [p.text for p in plan]}, calls=1 if planner else 0)

    def synthesize(state: ProgramState) -> StepOutcome:
        draft, plan = state.get(DRAFT), state.get(PLAN)
        if not draft:
            return StepOutcome(values={}, calls=0)
        if not plan:
            # Nothing to assemble: the draft IS the answer, and spending a
            # call to restate it would cost budget and risk losing content.
            return StepOutcome(values={ANSWER: draft}, calls=0)
        text, meta = agent._llm_call(
            f"Собери единый ответ из частей, ничего не добавляя.\n"
            f"Задача: {state.task}\nЧасти: {draft}",
            temperature=0.0, kind="synthesis", context_tag="COGPROG SYNTHESIZE")
        return StepOutcome(values={ANSWER: text or draft}, calls=1,
                           brain=meta.brain or meta.provider)

    def abstract(state: ProgramState) -> StepOutcome:
        text, meta = agent._llm_call(
            f"Переформулируй задачу в более общем виде, одним предложением.\n{state.task}",
            temperature=0.0, kind="reasoning", context_tag="COGPROG ABSTRACT")
        return StepOutcome(values={ABSTRACTION: text} if text else {}, calls=1,
                           brain=meta.brain or meta.provider)

    def predict(state: ProgramState) -> StepOutcome:
        draft = state.get(DRAFT)
        if not draft:
            return StepOutcome(values={}, calls=0)
        text, meta = agent._llm_call(
            f"Если этот ответ верен, что должно из него следовать? Одним предложением.\n"
            f"Задача: {state.task}\nОтвет: {draft}",
            temperature=0.0, kind="reasoning", context_tag="COGPROG PREDICT")
        return StepOutcome(values={PREDICTION: text} if text else {}, calls=1,
                           brain=meta.brain or meta.provider)

    def counterexample(state: ProgramState) -> StepOutcome:
        draft = state.get(DRAFT)
        if not draft:
            return StepOutcome(values={}, calls=0)
        text, meta = agent._llm_call(
            f"Найди случай, в котором этот ответ неверен. Если такого нет — ответь «нет».\n"
            f"Задача: {state.task}\nОтвет: {draft}",
            temperature=0.0, kind="reasoning", context_tag="COGPROG COUNTEREXAMPLE")
        return StepOutcome(values={CRITIQUE: text} if text else {}, calls=1,
                           brain=meta.brain or meta.provider)

    def compare(state: ProgramState) -> StepOutcome:
        drafts = state.history.get(DRAFT) or []
        if len(drafts) < 2:
            return StepOutcome(values={}, calls=0)
        text, meta = agent._llm_call(
            f"Какой из двух ответов лучше отвечает на задачу и почему? Кратко.\n"
            f"Задача: {state.task}\nА: {drafts[-2]}\nБ: {drafts[-1]}",
            temperature=0.0, kind="reasoning", context_tag="COGPROG COMPARE")
        return StepOutcome(values={CRITIQUE: text} if text else {}, calls=1,
                           brain=meta.brain or meta.provider)

    def answer(state: ProgramState) -> StepOutcome:
        draft = state.get(DRAFT)
        return StepOutcome(values={ANSWER: draft} if draft else {}, calls=0)

    return {
        "OBSERVE": observe, "RETRIEVE": retrieve, "GENERATE": generate,
        "CRITIQUE": critique, "REPAIR": repair, "VERIFY": verify,
        "DECOMPOSE": decompose_step, "SYNTHESIZE": synthesize,
        "ABSTRACT": abstract, "PREDICT": predict,
        "COUNTEREXAMPLE": counterexample, "COMPARE": compare, "ANSWER": answer,
    }


def capabilities_of(agent: Any):
    """Read the current capabilities from a live agent."""
    from .compiler import Capabilities
    try:
        brains = len(agent.llm.pool.available())
    except Exception:
        brains = 0
    return Capabilities(
        brains=brains,
        has_sandbox=bool(agent._tool_available("run_code")),
        has_web=bool(agent._tool_available("web_search")),
        has_memory=True,
    )
