"""
mana_desktop.session — the agent as a long-lived session, not a REPL.

Why this exists
---------------
MANA's only interactive entry point was `CoreMixin.interactive()`: a
`while True: input()` loop with twenty `if low == "/status"` branches and
`print()` for every answer. Everything a window needs was in there --
asking, the slash commands, memory writes -- but locked inside a loop that
owns the thread and talks to a terminal.

This class turns that into something a window can drive:

  * **the agent lives on a worker thread**, so a 30-second answer does not
    freeze the UI;
  * **one question at a time**, because the agent is not re-entrant: it
    holds a single SQLite connection, one pipeline and one session id;
  * **cancellation**, which the REPL never had. `solve_task` cannot be
    interrupted mid-call, so cancelling abandons the *result* rather than
    the work -- stated plainly here because the honest version ("we stop
    waiting") is different from what a Stop button usually implies;
  * **events**, not return values. The window subscribes once and receives
    everything the agent says about itself, including work started
    elsewhere -- a background evolution cycle reports through the same
    channel as an answer.

Nothing here reimplements agent behaviour. Every operation is a call into
the existing API (`solve_task`, `brains_status`, `evolution_status`,
`code_history`, `rollback_code`), which is why the CLI and the window
cannot drift apart.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from mana import events


class AgentSession:
    """Owns one ManaAgent and serialises access to it."""

    def __init__(self, config: Any) -> None:
        self._config = config
        self._agent: Any = None
        self._agent_error: Optional[str] = None
        self._lock = threading.RLock()
        self._busy = threading.Event()
        self._cancel = threading.Event()
        self._current_id = 0
        self._ready = threading.Event()
        #: Cost of the pool at the last answer, so each reply can report
        #: what IT cost rather than the running total. Before phase 14
        #: there was nothing to report but a call count, in which a 120B
        #: remote model and a local 7B were the same number.
        self._cost_mark = None
        self._cycle_lock = threading.RLock()
        self._cycle: Dict[str, Any] = {"running": False, "report": None, "note": ""}
        self._cycle_stop = threading.Event()
        threading.Thread(target=self._build_agent, name="MANA-Startup", daemon=True).start()

    # ---------- lifecycle ----------

    def _build_agent(self) -> None:
        """Construct the agent off the UI thread.

        Startup is not instant -- it opens SQLite, rebuilds the FTS index,
        may load an embedding model and probes the brain pool. Doing it
        before the window exists would show the user a blank screen with no
        explanation of what is taking so long.
        """
        try:
            from mana import ManaAgent
            events.emit(events.STATUS, "Запуск агента...")
            self._agent = ManaAgent(self._config)
        except Exception as exc:
            self._agent_error = f"{type(exc).__name__}: {exc}"
            events.emit(events.ERROR, f"Не удалось запустить агента: {self._agent_error}",
                        traceback=traceback.format_exc())
        finally:
            self._ready.set()

    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        return self._ready.wait(timeout)

    @property
    def agent(self) -> Any:
        return self._agent

    def state(self) -> Dict[str, Any]:
        return {
            "ready": self._ready.is_set() and self._agent is not None,
            "error": self._agent_error,
            "busy": self._busy.is_set(),
        }

    def close(self) -> None:
        agent = self._agent
        if agent is None:
            return
        for closer in (getattr(agent, "persistent_memory", None), getattr(agent, "experience", None)):
            try:
                closer.close()
            except Exception:
                pass

    # ---------- asking ----------

    def ask(self, text: str) -> Dict[str, Any]:
        """Answer one question. Blocking; call it from a worker thread.

        Returns a plain dict rather than raising, because every caller here
        is an HTTP handler whose job is to report the failure to the
        window, not to propagate it.
        """
        if not self._ready.is_set():
            return {"ok": False, "error": "агент ещё запускается"}
        if self._agent is None:
            return {"ok": False, "error": self._agent_error or "агент не запустился"}
        if self._busy.is_set():
            return {"ok": False, "error": "уже отвечаю на предыдущий вопрос"}

        with self._lock:
            self._busy.set()
            self._cancel.clear()
            self._current_id += 1
            request_id = self._current_id

        started = time.perf_counter()
        events.emit(events.PROGRESS, "Думаю...", request_id=request_id, phase="start")
        try:
            result = self._agent.solve_task(text)
        except Exception as exc:
            events.emit(events.ERROR, f"Ошибка при ответе: {type(exc).__name__}: {exc}",
                        request_id=request_id, traceback=traceback.format_exc())
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "request_id": request_id}
        finally:
            self._busy.clear()

        if self._cancel.is_set():
            # Honest wording matters: the call could not be interrupted, so
            # the work was done and paid for. What was cancelled is the
            # answer being shown, not the computation.
            events.emit(events.STATUS, "Ответ отменён (запрос уже был выполнен)",
                        request_id=request_id)
            return {"ok": False, "cancelled": True, "request_id": request_id}

        trace = result.get("trace", {}) or {}
        return {
            "ok": True,
            "request_id": request_id,
            "answer": result.get("answer", ""),
            "elapsed": time.perf_counter() - started,
            "provenance": self._provenance(result, trace),
            "cost": self._cost_since_mark(),
            "trace": trace,
        }

    def _pool(self) -> Any:
        agent = self._agent
        client = getattr(agent, "llm", None) if agent is not None else None
        return getattr(client, "pool", None)

    def _cost_since_mark(self) -> Dict[str, Any]:
        """What THIS answer cost, not what the session has spent.

        A running total tells a user nothing about the question they just
        asked, and the substrate mix is the interesting part: an answer
        that came from an algorithmic brain cost no tokens at all, and
        that is invisible in a call count.
        """
        pool = self._pool()
        if pool is None:
            return {}
        try:
            total = pool.total_cost()
        except Exception:                              # pragma: no cover
            return {}
        previous, self._cost_mark = self._cost_mark, total
        if previous is None:
            return total.as_dict()
        from mana.core.cost import CostVector
        delta = CostVector(
            calls=total.calls - previous.calls,
            wall_seconds=max(0.0, total.wall_seconds - previous.wall_seconds),
            tokens_in=total.tokens_in - previous.tokens_in,
            tokens_out=total.tokens_out - previous.tokens_out,
            unmeasured_token_calls=(total.unmeasured_token_calls
                                    - previous.unmeasured_token_calls),
            by_substrate={k: v - previous.by_substrate.get(k, 0)
                          for k, v in total.by_substrate.items()
                          if v - previous.by_substrate.get(k, 0) > 0})
        return delta.as_dict()

    def cancel(self) -> Dict[str, Any]:
        """Stop waiting for the current answer.

        Deliberately not called `stop`: solve_task runs a graph of LLM,
        web and verification calls with no interruption point, so nothing
        is actually aborted. Naming it honestly here keeps the UI from
        promising something the engine cannot do.
        """
        if not self._busy.is_set():
            return {"ok": False, "error": "нечего отменять"}
        self._cancel.set()
        return {"ok": True, "note": "запрос продолжает выполняться, но результат будет отброшен"}

    @staticmethod
    def _provenance(result: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
        """The line under each answer: where it came from and how checked.

        This is the whole reason the window is worth building over the
        REPL. MANA already tracks which brain answered, whether the critic
        was independent, whether brains agreed and whether anything was
        verified -- and the terminal showed it as a JSON blob nobody read.
        """
        consensus = trace.get("consensus") or {}
        decompose = trace.get("decompose") or {}
        return {
            "brain": trace.get("brain") or "",
            "attempts": list(trace.get("brain_attempts") or []),
            "strategy": trace.get("brain_strategy") or "single",
            "route": trace.get("route") or "",
            "verification": trace.get("verification_kind") or "none",
            "verified": bool(trace.get("verification_used")),
            "critic_brain": trace.get("critic_brain") or "",
            "critic_independent": bool(trace.get("critic_independent")),
            "agreement": consensus.get("agreement"),
            "disagreement": bool(consensus.get("disagreement")),
            "subtasks": decompose.get("subtasks"),
            "brains_used": list(decompose.get("brains_used") or []),
            "fallback": bool(result.get("fallback")),
            "confidence": result.get("confidence"),
        }

    # ---------- the cognitive layer ----------

    def cycle_status(self) -> Dict[str, Any]:
        with self._cycle_lock:
            return dict(self._cycle)

    def start_cycle(self, budget: int = 200, steps: int = 6,
                    genome_path: str = "") -> Dict[str, Any]:
        """Run a research cycle in the background and stream its steps.

        Started from the window because that is where someone watching it
        actually is. Everything built in phases 14-21 was reachable only
        by running a script and reading stdout, which is a fine way to
        develop and a poor way to watch.
        """
        if self._agent is None:
            return {"ok": False, "error": "агент не запустился"}
        with self._cycle_lock:
            if self._cycle["running"]:
                return {"ok": False, "error": "цикл уже идёт"}
            self._cycle = {"running": True, "report": None, "note": "",
                           "budget": int(budget), "steps": int(steps)}
        self._cycle_stop.clear()
        threading.Thread(target=self._run_cycle, name="MANA-Cycle", daemon=True,
                         args=(int(budget), int(steps), genome_path)).start()
        return {"ok": True}

    def stop_cycle(self) -> Dict[str, Any]:
        """Ask the cycle to stop after the step it is on.

        Named for what it does: a step is a batch of brain calls with no
        interruption point, so the current one finishes and is paid for.
        Promising otherwise would be the same lie `cancel` refuses to
        tell.
        """
        if not self.cycle_status().get("running"):
            return {"ok": False, "error": "цикл не запущен"}
        self._cycle_stop.set()
        return {"ok": True, "note": "остановится после текущего шага"}

    def _run_cycle(self, budget: int, steps: int, genome_path: str) -> None:
        from mana.cognition.research import ResearchCycle
        from mana.cognition.self_model import SelfModel
        from mana.core import oracle
        from mana.core.cost import efficiency

        pool = self._pool()
        texts: Dict[str, str] = {}
        model = SelfModel()

        def runner(task):
            if self._cycle_stop.is_set():
                raise RuntimeError("остановлено пользователем")
            texts[task.task_id] = task.prompt
            reply = pool.ask(task.prompt + "\n\nОтветь только итоговым значением, "
                             "без пояснений.", kind=task.domain,
                             difficulty=task.difficulty, context_tag="ui-cycle")
            graded = oracle.grade(task, reply.get("text") or "")
            return graded.correct, reply.get("brain", ""), 1

        before = pool.total_cost() if pool is not None else None
        try:
            cycle = ResearchCycle(model, task_texts=texts, budget_calls=budget,
                                  max_steps=steps,
                                  genome_path=genome_path or None)
            events.emit(events.STATUS, f"Когнитивный цикл: бюджет {budget}, шагов {steps}")
            report = cycle.run(runner)
            for i, step in enumerate(report.get("history", []), 1):
                events.emit(events.PROGRESS,
                            f"{i}. [{step['activity']}] {step['description']}"
                            f" → {step['outcome']}")
            spent = (pool.total_cost() if pool is not None else None)
            cost = {}
            if before is not None and spent is not None:
                from mana.core.cost import CostVector
                cost = CostVector(
                    calls=spent.calls - before.calls,
                    wall_seconds=max(0.0, spent.wall_seconds - before.wall_seconds),
                    tokens_in=spent.tokens_in - before.tokens_in,
                    tokens_out=spent.tokens_out - before.tokens_out,
                    unmeasured_token_calls=(spent.unmeasured_token_calls
                                            - before.unmeasured_token_calls),
                    by_substrate={k: v - before.by_substrate.get(k, 0)
                                  for k, v in spent.by_substrate.items()
                                  if v - before.by_substrate.get(k, 0) > 0}).as_dict()
            profile = []
            for key, cap in sorted(model.capabilities().items()):
                if cap.band == "all":
                    continue
                lo, hi = cap.confidence_interval
                profile.append({"slice": key, "score": round(cap.score, 3),
                                "low": round(lo, 3), "high": round(hi, 3),
                                "observations": cap.observations})
            report["cost"] = cost
            report["profile"] = profile
            # Переставить отметку, иначе стоимость цикла попадёт в счёт
            # следующего ответа: пользователь спросит одну задачу и увидит
            # под ней сотню вызовов, потраченных не на неё.
            self._cost_mark = spent
            with self._cycle_lock:
                self._cycle = {"running": False, "report": report, "note": ""}
            events.emit(events.STATUS,
                        f"Цикл завершён: {report['stop_reason']}")
        except Exception as exc:
            with self._cycle_lock:
                self._cycle = {"running": False, "report": None,
                               "note": f"{type(exc).__name__}: {exc}"}
            events.emit(events.ERROR, f"Цикл прерван: {type(exc).__name__}: {exc}")

    def genome(self, genome_path: str = "") -> Dict[str, Any]:
        """What has been adopted, and whether it came from disk.

        Reads rather than builds: a window that constructed a fresh
        genome to display would show a baseline and call it the system's
        state, which is exactly the confusion phase 20 was about.
        """
        from mana.cognition import genome as genome_mod
        path = genome_path or "mana_state/genome.json"
        loaded, note = genome_mod.load_report(path)
        if loaded is None:
            return {"path": str(path), "note": note, "present": False}
        return {
            "path": str(path), "note": note, "present": True,
            "genome_id": loaded.genome_id, "parent_id": loaded.parent_id,
            "mutation": loaded.mutation, "size": loaded.size(),
            "brains": [{"brain_id": g.brain_id, "substrate": g.substrate,
                        "applicability": list(g.applicability),
                        "notes": g.notes}
                       for g in loaded.brains.values()],
            "templates": sorted(loaded.program_templates),
        }

    # ---------- inspection, straight through to the existing API ----------

    def brains(self) -> Dict[str, Any]:
        if self._agent is None:
            return {"brains": [], "available": [], "configured": [], "policy": ""}
        return self._agent.brains_status()

    def evolution(self) -> Dict[str, Any]:
        if self._agent is None:
            return {"running": False}
        return self._agent.evolution_status()

    def start_evolution(self, cycles: int = 1) -> Dict[str, Any]:
        if self._agent is None:
            return {"ok": False, "error": "агент не запущен"}
        try:
            return {"ok": True, "started": self._agent.start_evolution_background(int(cycles))}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def stop_evolution(self) -> Dict[str, Any]:
        if self._agent is None:
            return {"ok": False, "error": "агент не запущен"}
        try:
            return {"ok": True, "result": self._agent.stop_evolution()}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def code_history(self) -> List[Dict[str, Any]]:
        if self._agent is None:
            return []
        return self._agent.code_history()

    def memory_search(self, query: str) -> Dict[str, Any]:
        if self._agent is None or not query.strip():
            return {"context": "", "trace": {}}
        return self._agent.graph_memory_search(query)
