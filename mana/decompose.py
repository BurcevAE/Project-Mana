"""
mana.decompose — split a hard task across several brains, then synthesize.

Why this is a separate module from brains.py
--------------------------------------------
`brains.py` answers "which model should say this?". This module answers
"is this actually one question?". They compose, but they fail
independently: a broken planner must degrade to "one subtask == the whole
task", which is exactly the single-brain behaviour MANA had before, and a
broken pool must not make the planner unusable in tests. Keeping them
apart is what makes that degradation expressible as one early return
instead of a special case threaded through the router.

The shape of the thing
----------------------
    plan(task)                 -> [Subtask, ...]        (LLM or heuristic)
    execute(plan, pool)        -> [SubtaskResult, ...]  (parallel per layer)
    synthesize(task, results)  -> final answer          (one strong brain)

Each subtask carries its own `kind` and `difficulty`, so the pool routes
each one separately: arithmetic goes to a cheap fast brain, the reasoning
step goes to a large one, and independent subtasks run *concurrently* on
different providers. That is the load-distribution win -- three free tiers
working at once rather than one queue.

Honesty constraints, carried over from the rest of MANA
-------------------------------------------------------
  * The synthesis prompt forbids adding facts that no subtask produced.
    Decomposition multiplies generation, and generation is where
    fabrication comes from; nothing here upgrades a model's claim to a
    fact. The result is still `verification_kind: none` unless MANA's own
    verifier checks it.
  * A failed subtask is reported as failed and carried into synthesis as a
    gap, not silently dropped. An answer synthesized from 2 of 4 parts
    while claiming to answer all 4 is worse than an incomplete answer that
    says which part is missing.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"

MAX_SUBTASKS = 6


@dataclass
class Subtask:
    sid: str
    text: str
    kind: str = "general"
    difficulty: float = 0.4
    depends_on: Tuple[str, ...] = ()

    def normalize(self) -> "Subtask":
        from .brains import KINDS
        self.sid = str(self.sid or "s1")[:16]
        self.text = str(self.text or "").strip()
        if self.kind not in KINDS:
            self.kind = "general"
        self.difficulty = max(0.0, min(1.0, float(self.difficulty)))
        self.depends_on = tuple(str(d)[:16] for d in (self.depends_on or ()) if str(d) != self.sid)
        return self


@dataclass
class SubtaskResult:
    sid: str
    text: str
    ok: bool
    answer: str = ""
    brain: str = ""
    error: str = ""
    latency: float = 0.0


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

_ENUM_RE = re.compile(r"(?:^|\n)\s*(?:\d+[\.\)]|[-*•])\s+(.+)")
_KIND_MARKERS = (
    ("math", ("посчитай", "вычисли", "сколько будет", "calculate", "сумм", "процент", "раздели", "умнож")),
    ("programming", ("код", "функци", "python", "скрипт", "класс", "sql", "1с", "баг", "code", "debug")),
    ("current", ("сегодня", "сейчас", "последние новости", "актуальн", "курс", "погода", "свежие")),
    ("reasoning", ("почему", "объясни", "сравни", "обоснуй", "докажи", "why", "explain", "compare")),
)


def guess_kind(text: str) -> str:
    t = (text or "").lower()
    for kind, markers in _KIND_MARKERS:
        if any(m in t for m in markers):
            return kind
    return "general"


def plan_heuristic(task: str, max_subtasks: int = MAX_SUBTASKS) -> List[Subtask]:
    """Split without an LLM. Conservative by design.

    It only splits on structure a human actually wrote -- an enumerated
    list, or several complete questions. It deliberately does NOT split on
    "и"/commas: "объясни разницу между списком и кортежем" is one question
    with an "и" in it, and splitting it produces two subtasks that each
    answer half of nothing. When in doubt this returns a single subtask,
    which makes the whole feature a no-op rather than a regression.
    """
    text = (task or "").strip()
    if not text:
        return []
    parts = [m.group(1).strip() for m in _ENUM_RE.finditer(text)]
    if len(parts) < 2:
        questions = [q.strip() for q in re.split(r"(?<=\?)\s+", text) if q.strip().endswith("?")]
        parts = questions if len(questions) >= 2 else []
    parts = [p for p in parts if len(p) >= 8][:max_subtasks]
    if len(parts) < 2:
        return [Subtask("s1", text, guess_kind(text), 0.5).normalize()]
    return [Subtask(f"s{i+1}", p, guess_kind(p), 0.4).normalize() for i, p in enumerate(parts)]


_PLAN_PROMPT = """Разбей задачу на независимые подзадачи. Правила:
- От 1 до {max_n} подзадач. Если задача цельная -- верни ровно одну.
- Каждая подзадача формулируется как самостоятельный вопрос, понятный без остальных.
- kind: одно из math, programming, reasoning, current, general.
- difficulty: число от 0 до 1.
- depends_on: список id подзадач, чей ответ нужен ДО этой (обычно пустой).
Верни ТОЛЬКО JSON-массив, без пояснений и без markdown-ограждения:
[{{"sid":"s1","text":"...","kind":"...","difficulty":0.4,"depends_on":[]}}]

Задача: {task}"""


def _extract_json_array(text: str) -> Optional[List[Any]]:
    """Pull the first JSON array out of an LLM response.

    Models wrap JSON in prose and in ```json fences no matter how the
    prompt asks them not to, so parsing has to survive that. A failure
    here returns None and the caller falls back to the heuristic planner --
    a bad plan must never become a bad answer.
    """
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.I | re.M).strip()
    start = cleaned.find("[")
    while start != -1:
        depth, in_str, escape = 0, False, False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(cleaned[start:i + 1])
                    except Exception:
                        break
                    return parsed if isinstance(parsed, list) else None
        start = cleaned.find("[", start + 1)
    return None


def plan_with_llm(task: str, ask: Callable[..., Tuple[Optional[str], Any]],
                  max_subtasks: int = MAX_SUBTASKS) -> List[Subtask]:
    """Ask a planner brain for a decomposition; fall back to the heuristic.

    `ask` matches ManaAgent._llm_call's signature, so the planner call goes
    through the same tool-registry choke point as every other LLM call and
    is itself routed to whichever brain is best at planning.
    """
    try:
        text, _meta = ask(_PLAN_PROMPT.format(task=task, max_n=max_subtasks),
                          temperature=0.0, context_tag="DECOMPOSE-PLAN")
    except Exception:
        return plan_heuristic(task, max_subtasks)
    items = _extract_json_array(text or "")
    if not items:
        return plan_heuristic(task, max_subtasks)
    subs: List[Subtask] = []
    for i, item in enumerate(items[:max_subtasks]):
        if not isinstance(item, dict):
            continue
        body = str(item.get("text", "")).strip()
        if len(body) < 4:
            continue
        subs.append(Subtask(
            sid=str(item.get("sid") or f"s{i+1}"),
            text=body,
            kind=str(item.get("kind") or guess_kind(body)),
            difficulty=float(item.get("difficulty", 0.4) or 0.4),
            depends_on=tuple(item.get("depends_on") or ()),
        ).normalize())
    if not subs:
        return plan_heuristic(task, max_subtasks)
    return prune_dependencies(subs)


def prune_dependencies(subs: Sequence[Subtask]) -> List[Subtask]:
    """Drop edges to unknown ids and break cycles.

    A planner LLM will happily emit `depends_on: ["s3"]` on s3 itself, or
    reference an id it never defined. Either one deadlocks `layers()`, so
    the graph is repaired here rather than trusted: unknown ids are
    dropped, and any edge that would close a cycle is dropped in declared
    order (earlier subtasks may not depend on later ones).
    """
    known = {s.sid for s in subs}
    position = {s.sid: i for i, s in enumerate(subs)}
    out: List[Subtask] = []
    for s in subs:
        deps = tuple(d for d in s.depends_on
                     if d in known and position.get(d, 0) < position[s.sid])
        s.depends_on = deps
        out.append(s)
    return out


def layers(subs: Sequence[Subtask]) -> List[List[Subtask]]:
    """Group subtasks into dependency layers. Everything inside one layer
    is independent and therefore safe to run in parallel on different
    brains -- which is where the wall-clock win comes from."""
    remaining = {s.sid: s for s in subs}
    done: set = set()
    out: List[List[Subtask]] = []
    while remaining:
        ready = [s for s in remaining.values() if all(d in done for d in s.depends_on)]
        if not ready:                      # unreachable after prune_dependencies, but
            ready = list(remaining.values())   # a stuck plan must still execute
        out.append(ready)
        for s in ready:
            done.add(s.sid)
            remaining.pop(s.sid, None)
    return out


# ---------------------------------------------------------------------------
# execution + synthesis
# ---------------------------------------------------------------------------

_SUBTASK_PROMPT = """{context}Ответь строго на этот вопрос, кратко и по существу.
Если данных недостаточно -- так и скажи, не додумывай.

Вопрос: {question}"""

_SYNTH_PROMPT = """Собери единый ответ на исходную задачу из готовых частей.
Правила:
- Используй ТОЛЬКО факты из частей ниже. Не добавляй ничего от себя.
- Если часть не получена -- прямо укажи, что этот пункт остался без ответа.
- Без служебных пометок, без перечисления моделей, просто ответ.

Исходная задача: {task}

{parts}"""


def execute(subs: Sequence[Subtask], pool: Any, *, system: str = "", temperature: float = 0.2,
            context_tag: str = "", max_parallel: int = 4,
            policy: str = "") -> List[SubtaskResult]:
    """Run the plan: layer by layer, parallel within a layer.

    Answers from a subtask's dependencies are prepended to its prompt, so a
    chain (`s2 depends_on s1`) actually carries information forward instead
    of being a decorative field.
    """
    answers: Dict[str, str] = {}
    results: List[SubtaskResult] = []
    for layer in layers(subs):
        # Assign distinct brains BEFORE launching the layer. Letting each
        # thread call select() on its own does not spread the load: they
        # all rank the pool at the same instant, before any of them has
        # incremented `inflight`, so every subtask picks the same
        # top-ranked brain and the "parallel" layer serializes behind one
        # provider. Assigning up front -- each subtask still choosing by
        # its own kind/difficulty, just excluding brains already taken --
        # is what actually turns N free tiers into N workers. Failover is
        # unaffected: `brain=` is a preference, and pool.ask still walks
        # the ranking if that brain fails.
        preferred: Dict[str, str] = {}
        if len(layer) > 1:
            taken: List[str] = []
            for sub in layer:
                pick = pool.select(kind=sub.kind, difficulty=sub.difficulty, task=sub.text,
                                   policy=policy, exclude=taken, limit=1)
                if not pick:
                    break          # fewer brains than subtasks: the rest share
                preferred[sub.sid] = pick[0]
                taken.append(pick[0])

        def run(sub: Subtask) -> SubtaskResult:
            context = ""
            if sub.depends_on:
                known = [f"- {answers[d]}" for d in sub.depends_on if answers.get(d)]
                if known:
                    context = "Уже установлено:\n" + "\n".join(known) + "\n\n"
            res = pool.ask(_SUBTASK_PROMPT.format(context=context, question=sub.text),
                           system=system, temperature=temperature, kind=sub.kind,
                           difficulty=sub.difficulty, task=sub.text, policy=policy,
                           brain=preferred.get(sub.sid, "auto"),
                           context_tag=f"{context_tag} SUB-{sub.sid}")
            return SubtaskResult(sid=sub.sid, text=sub.text, ok=bool(res.get("ok")),
                                 answer=str(res.get("text") or ""), brain=str(res.get("brain") or ""),
                                 error=str(res.get("error") or ""),
                                 latency=float(res.get("latency_total", res.get("latency", 0.0))))

        if len(layer) == 1:
            layer_results = [run(layer[0])]
        else:
            layer_results = []
            with ThreadPoolExecutor(max_workers=min(len(layer), max(1, max_parallel))) as ex:
                futures = {ex.submit(run, s): s for s in layer}
                for fut in as_completed(futures):
                    sub = futures[fut]
                    try:
                        layer_results.append(fut.result())
                    except Exception as exc:
                        layer_results.append(SubtaskResult(sid=sub.sid, text=sub.text, ok=False,
                                                           error=f"{type(exc).__name__}: {exc}"))
        order = {s.sid: i for i, s in enumerate(layer)}
        layer_results.sort(key=lambda r: order.get(r.sid, 99))
        for r in layer_results:
            if r.ok and r.answer:
                answers[r.sid] = r.answer
        results.extend(layer_results)
    return results


def format_parts(results: Sequence[SubtaskResult]) -> str:
    lines = []
    for r in results:
        if r.ok and r.answer:
            lines.append(f"[{r.sid}] Вопрос: {r.text}\nОтвет: {r.answer}")
        else:
            lines.append(f"[{r.sid}] Вопрос: {r.text}\nОтвет НЕ ПОЛУЧЕН ({r.error or 'нет ответа'})")
    return "\n\n".join(lines)


def synthesize(task: str, results: Sequence[SubtaskResult], pool: Any, *,
               temperature: float = 0.1, context_tag: str = "",
               policy: str = "") -> Dict[str, Any]:
    """Merge the parts with a brain that is good at synthesis.

    A single successful subtask needs no synthesis pass -- returning its
    answer directly saves a whole LLM call and cannot lose information,
    which matters when every brain in the pool is a rate-limited free tier.
    """
    usable = [r for r in results if r.ok and r.answer]
    if not usable:
        return {"ok": False, "answer": "", "brain": "", "error": "no subtask produced an answer",
                "latency": sum(r.latency for r in results)}
    if len(results) == 1:
        return {"ok": True, "answer": usable[0].answer, "brain": usable[0].brain, "error": "",
                "latency": usable[0].latency, "skipped_synthesis": True}
    res = pool.ask(_SYNTH_PROMPT.format(task=task, parts=format_parts(results)),
                   temperature=temperature, kind="synthesis", difficulty=0.5,
                   task=task, policy=policy, context_tag=f"{context_tag} SYNTH")
    if not res.get("ok"):
        # Synthesis failed -- concatenating the parts is a worse answer than
        # a synthesized one, but it is a real answer built only from what
        # the subtasks actually returned, so nothing is fabricated.
        joined = "\n\n".join(f"{r.text.rstrip('?')}: {r.answer}" for r in usable)
        return {"ok": True, "answer": joined, "brain": "", "error": str(res.get("error") or ""),
                "latency": float(res.get("latency", 0.0)), "degraded": True}
    return {"ok": True, "answer": str(res.get("text") or ""), "brain": str(res.get("brain") or ""),
            "error": "", "latency": float(res.get("latency_total", res.get("latency", 0.0)))}


def solve(task: str, pool: Any, *, ask_planner: Optional[Callable[..., Tuple[Optional[str], Any]]] = None,
          system: str = "", temperature: float = 0.2, context_tag: str = "",
          max_subtasks: int = MAX_SUBTASKS, max_parallel: int = 4,
          policy: str = "") -> Dict[str, Any]:
    """Full decompose -> route -> execute -> synthesize cycle.

    Returns a trace naming which brain answered which part. That is not
    decoration: without it a wrong answer produced by five models is
    undebuggable, and MANA's whole reporting discipline is built on being
    able to say where an answer came from.
    """
    plan = plan_with_llm(task, ask_planner, max_subtasks) if ask_planner else plan_heuristic(task, max_subtasks)
    if not plan:
        return {"ok": False, "answer": "", "error": "empty task", "plan": [], "results": []}
    results = execute(plan, pool, system=system, temperature=temperature,
                      context_tag=context_tag, max_parallel=max_parallel, policy=policy)
    final = synthesize(task, results, pool, context_tag=context_tag, policy=policy)
    return {
        "ok": bool(final.get("ok")),
        "answer": final.get("answer", ""),
        "error": final.get("error", ""),
        "subtasks": len(plan),
        "plan": [asdict(s) for s in plan],
        "results": [asdict(r) for r in results],
        "brains_used": sorted({r.brain for r in results if r.brain} |
                              ({final["brain"]} if final.get("brain") else set())),
        "synthesis_brain": final.get("brain", ""),
        "degraded": bool(final.get("degraded")),
        "failed_subtasks": [r.sid for r in results if not r.ok],
        "latency": sum(r.latency for r in results) + float(final.get("latency", 0.0)),
    }
