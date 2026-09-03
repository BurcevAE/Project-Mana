"""
mana.agent_parts.core — CoreMixin: lifecycle (init/state/cache/reports persistence), logging helpers, top-level solve_task()/interactive() entry points.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import pickle
import random
import re
import statistics
import sys
import threading
import time
import subprocess
import tempfile
import shutil
import platform
import ast
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import Config, RandomManager
from ..knowledge import KnowledgeBase
from ..web import WebSearcher
from ..llm import LLMClient, LLMCallMeta
from ..pipeline import PipelineSpec, PipelineFactory, BenchmarkTask, BenchmarkSuite
from ..experience import ExperienceDB
from ..verifier import LocalVerifier
from ..memory import MemoryManager
from ..version import PRODUCT_VERSION, format_version_report
from ..hardware import detect_hardware, apply_hardware_profile
from ..tools import build_default_registry
from .. import events
from ..core import evaluation
from ..graph_memory import GraphMemoryStore, extract_entities
from ..intent import is_ambiguous_followup, format_clarifying_question
from ..optional_deps import fitz, HAS_FITZ, HAS_SKLEARN, LogisticRegression, HAS_TORCH, DEVICE, HAS_WEB, WEB_BACKEND, torch

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.4"


class CoreMixin:
    # Product version, declared once in mana/version.py. Never hardcode it
    # here again -- this file, memory.py and cli.py had drifted to three
    # different values before that module existed.
    VERSION = PRODUCT_VERSION

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.config.ensure_dirs()

        # Adapt Config to the host machine *before* anything below reads
        # values it touches (evolution_workers, timeouts, population size,
        # use_embeddings). See mana/hardware.py for exactly what changes
        # and why; this is the "works when copied to a different machine"
        # requirement -- best-effort and fully inspectable, never silent:
        # hardware_status is kept on the instance and printed at startup.
        self.hardware_profile = None
        self.hardware_adaptation: Dict[str, Any] = {}
        if getattr(self.config, "hardware_auto_adapt", True):
            try:
                self.hardware_profile = detect_hardware()
                self.hardware_adaptation = apply_hardware_profile(self.config, self.hardware_profile)
            except Exception as exc:
                self._vlog(f"hardware detection failed, keeping defaults: {exc}")

        self.rm = RandomManager(self.config.seed)
        self.pipeline = PipelineSpec().normalize(self.config)
        self.memory = KnowledgeBase(self.config)
        self.persistent_memory = MemoryManager(self.config, embedder=self.memory.embedder)
        self.graph_memory = GraphMemoryStore(self.persistent_memory)
        self.session_id = self.config.memory_session_id
        self.llm = LLMClient(self.config, self._vlog)
        # Tell the genome which brains exist on THIS machine. Done here,
        # before _load_state() restores a pipeline and before any evolution
        # runs, so mutations propose providers the user actually has a key
        # for instead of "learning" that unconfigured ones always fail.
        self.config.brain_ids = tuple(["auto"] + sorted(self.llm.pool.configured()))
        self.web = WebSearcher(self.config, self._vlog)
        self.experience = ExperienceDB(self.config.experience_db_path)

        self.cycle = 0
        self.best_pipeline_fitness = 0.0
        self.best_metrics: Dict[str, Any] = {}
        self.stagnation = 0
        self.frozen_params: Dict[str, int] = {}
        self.param_stability: Dict[str, Dict[str, Any]] = {}
        self.rollback_snapshot: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, Any]] = []
        self.reason_log: List[Dict[str, Any]] = []
        self.fitness_cache: Dict[str, Dict[str, Any]] = {}
        self._evolution_running = False
        self._evolution_stop = threading.Event()
        self._evolution_started_at = None
        self._evolution_cycle_started_at = None
        self._evolution_total_elapsed = 0.0
        self._evolution_last_cycle_elapsed = 0.0
        self._evolution_target = None
        self._report_lock = threading.RLock()
        self.evolution_reports: List[Dict[str, Any]] = []
        self.routing_stats: Dict[str, Dict[str, Any]] = {}
        self.confidence_stats: Dict[str, Dict[str, Any]] = {}
        self.stop_policy_stats: Dict[str, Dict[str, Any]] = {}
        self.verifier = LocalVerifier(self.config, self._vlog)
        self.tools = build_default_registry(self)
        self.learned_route_examples: List[Dict[str, Any]] = []
        self.learned_router_model = None
        self.learned_router_trained_n = 0
        self.learned_router_classes: List[str] = []
        self.mutation_failure_history: List[Dict[str, Any]] = []
        self._benchmark_learning = False
        # The evaluation mode is NOT a boolean the agent owns any more.
        # It is a context issued by mana.core.evaluation and handed in;
        # `_benchmark_holdout` survives as a read-only view so the three
        # existing readers need no change. See enter_evaluation().
        self._eval_mode = evaluation.normal()

        self._load_state()
        self._load_cache()
        self._load_reports()

        self._emit_banner()

    def _emit_banner(self) -> None:
        """Startup identity, as one event instead of 15 prints.

        It moved off `print()` because a windowed build has no stdout to
        print to, and because the console it does have could not always
        encode this text -- the same cp1251 failure that turned a stale
        state file into a crash in `_load_state`. The event carries the
        same facts as structured data alongside the text, so the desktop
        app can render a status panel instead of re-parsing lines.
        """
        from ..paths import status as paths_status
        brains = self.llm.pool.available()
        lines = [
            "=" * 62,
            f"MANA v{self.VERSION}  (подсистемы: --version)",
            f"device={DEVICE if HAS_TORCH else 'cpu/no-torch'}",
        ]
        if self.hardware_profile is not None:
            hp = self.hardware_profile
            gpu = f", gpu={hp.gpu_name}" if hp.has_cuda else ""
            lines.append(f"hardware=tier:{hp.tier} cpu:{hp.cpu_count} ram:{hp.total_ram_gb}GB{gpu}")
            if self.hardware_adaptation:
                changed = ", ".join(f"{k}={v['before']}->{v['after']}" for k, v in self.hardware_adaptation.items())
                lines.append(f"hardware_adapted: {changed}")
        lines += [
            f"cycle={self.cycle}",
            f"llm={'on' if self._tool_available('llm_generate') else 'off'} "
            f"({len(brains)} мозгов готово: {', '.join(brains) or 'нет'})",
            f"web={'on' if self._tool_available('web_search') else 'off'} ({WEB_BACKEND or 'none'})",
            f"memory={len(self.memory.entries)}",
            f"memory_db={self.config.memory_db_path} session={self.session_id} "
            f"events={self._memory_event_count(self.session_id)}",
            f"tools={', '.join(t['name'] for t in self.tools.list_tools())}",
            f"pipeline={self.pipeline.pretty()}",
            "=" * 62,
        ]
        events.emit(events.BANNER, "\n".join(lines),
                    version=self.VERSION, cycle=self.cycle, session=self.session_id,
                    brains=brains, paths=paths_status())

    def _memory_event_count(self, session_id: str) -> int:
        try:
            with self.persistent_memory.lock:
                row = self.persistent_memory.con.execute("SELECT COUNT(*) AS n FROM events WHERE session_id=?", (session_id,)).fetchone()
                return int(row["n"]) if row else 0
        except Exception:
            return 0

    def _vlog(self, message: str) -> None:
        """Verbose trace line.

        Emitted rather than printed: this is the busiest output path in the
        agent (every brain call, every evolution step logs through it), so
        it is also the one most likely to hit a console that cannot take
        it. The verbosity gate stays here so a subscriber never has to
        filter, and so nothing is built when logging is off.
        """
        if self.config.verbose_logging:
            events.emit(events.PROGRESS, f"[MANA {time.strftime('%H:%M:%S')}] {message}")


    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        if h: return f"{h} ч {m:02d} мин {s:04.1f} с"
        if m: return f"{m} мин {s:04.1f} с"
        return f"{s:.1f} с"


    # ---------- persistence ----------
    def _load_state(self) -> None:
        p = Path(self.config.state_file)
        if not p.exists(): return
        try:
            with p.open("rb") as fh: d = pickle.load(fh)
            self.cycle = int(d.get("cycle", 0))
            self.best_pipeline_fitness = float(d.get("best_pipeline_fitness", 0.0))
            self.best_metrics = dict(d.get("best_metrics", {}))
            self.stagnation = int(d.get("stagnation", 0))
            self.frozen_params = dict(d.get("frozen_params", {}))
            self.param_stability = dict(d.get("param_stability", {}))
            self.rollback_snapshot = d.get("rollback_snapshot")
            self.history = list(d.get("history", []))[-500:]
            self.reason_log = list(d.get("reason_log", []))[-1000:]
            self.routing_stats = dict(d.get("routing_stats", {}))
            self.confidence_stats = dict(d.get("confidence_stats", {}))
            self.stop_policy_stats = dict(d.get("stop_policy_stats", {}))
            self.learned_route_examples = list(d.get("learned_route_examples", []))[-self.config.learned_router_history_limit:]
            self.learned_router_trained_n = int(d.get("learned_router_trained_n", 0))
            self.mutation_failure_history = list(d.get("mutation_failure_history", []))[-300:]
            self._fit_learned_router(force=True)
            p_data = d.get("pipeline")
            if p_data: self.pipeline = PipelineSpec(**p_data).normalize(self.config)
            rng = d.get("rng_state")
            if rng: self.rm.load_state(rng)
        except Exception as exc:
            # Was a print() with an emoji -- which is how a merely stale
            # state file became a hard crash under a cp1251 console
            # (UnicodeEncodeError inside the handler for another error).
            events.emit(events.WARNING, f"Не удалось восстановить state: {exc}", error=str(exc))

    def _save_state(self) -> None:
        p = Path(self.config.state_file); tmp = p.with_suffix(p.suffix + ".tmp")
        payload = {
            "version": self.VERSION,
            "cycle": self.cycle,
            "pipeline": asdict(self.pipeline),
            "best_pipeline_fitness": self.best_pipeline_fitness,
            "best_metrics": self.best_metrics,
            "stagnation": self.stagnation,
            "frozen_params": self.frozen_params,
            "param_stability": self.param_stability,
            "rollback_snapshot": self.rollback_snapshot,
            "history": self.history[-500:],
            "reason_log": self.reason_log[-1000:],
            "routing_stats": self.routing_stats,
            "confidence_stats": self.confidence_stats,
            "stop_policy_stats": self.stop_policy_stats,
            "learned_route_examples": self.learned_route_examples[-self.config.learned_router_history_limit:],
            "learned_router_trained_n": self.learned_router_trained_n,
            "mutation_failure_history": self.mutation_failure_history[-300:],
            "rng_state": self.rm.save_state(),
            "timestamp": time.time(),
        }
        with tmp.open("wb") as fh: pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, p)

    def _load_cache(self) -> None:
        p = Path(self.config.cache_file)
        if not p.exists(): return
        try:
            with p.open("rb") as fh: self.fitness_cache = dict(pickle.load(fh).get("fitness_cache", {}))
        except Exception: self.fitness_cache = {}

    def _save_cache(self) -> None:
        p = Path(self.config.cache_file); tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("wb") as fh: pickle.dump({"fitness_cache": self.fitness_cache}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, p)

    def _load_reports(self) -> None:
        p = Path(self.config.evolution_report_file)
        if not p.exists(): return
        try:
            with p.open("r", encoding="utf-8") as fh: self.evolution_reports = list(json.load(fh))[-200:]
        except Exception: self.evolution_reports = []

    def _save_reports(self) -> None:
        p = Path(self.config.evolution_report_file); tmp = p.with_suffix(p.suffix + ".tmp")
        with self._report_lock:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(self.evolution_reports[-200:], fh, ensure_ascii=False, indent=2)
        os.replace(tmp, p)


    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, bytes):
            return {"type":"bytes", "size":len(value)}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): CoreMixin._json_safe(v) for k,v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [CoreMixin._json_safe(v) for v in value]
        return str(value)

    def hardware_status(self) -> Dict[str, Any]:
        from dataclasses import asdict as _asdict
        return {
            "profile": _asdict(self.hardware_profile) if self.hardware_profile else None,
            "adapted": self.hardware_adaptation,
            "auto_adapt_enabled": bool(getattr(self.config, "hardware_auto_adapt", True)),
        }

    def tools_status(self) -> Dict[str, Any]:
        return {"tools": self.tools.list_tools()}

    def _tool_available(self, name: str) -> bool:
        """Availability check routed through the registry (tool.is_available())
        instead of reading self.llm.enabled / self.web.enabled / config.
        local_exec_enabled directly -- the same check a future planner would
        run before choosing a tool, used here for the agent's own internal
        decisions too, so there's one source of truth for 'can this tool
        usefully be called right now', not two."""
        tool = self.tools.get(name)
        return bool(tool and tool.is_available())

    def _llm_call(self, prompt: str, *, system: str = "", temperature: float = 0.2,
                  provider: str = "auto", context_tag: str = "", kind: str = "general",
                  difficulty: Optional[float] = None, task: str = "",
                  policy: str = "", avoid: Sequence[str] = ()) -> Tuple[Optional[str], LLMCallMeta]:
        """The single choke point for every LLM invocation in the agent.
        Routes through self.tools ('llm_generate') instead of
        self.llm.ask_detailed directly -- the registry is the real dispatch
        path now, not just an inspectable side list. Preserves the exact
        (text, meta) contract every existing call site already expects, so
        migrating a call site to this is a one-line change, not a rewrite
        of the logic built around it.

        v5.10: the optional kind/difficulty/task/policy arguments are the
        only thing a call site needs to add to get brain selection tuned to
        what it is doing (a critic pass is not the same job as a synthesis
        pass). Omitting them is fully supported and is what every
        un-migrated call site does -- the pool then infers difficulty from
        the prompt itself."""
        result = self.tools.call("llm_generate", prompt=prompt, system=system, temperature=temperature,
                                  provider=provider, context_tag=context_tag, kind=kind,
                                  difficulty=difficulty, task=task, policy=policy,
                                  avoid=tuple(avoid))
        meta = LLMCallMeta(**result.meta) if result.meta else LLMCallMeta(ok=False, error=result.error)
        return result.output, meta

    @property
    def _benchmark_holdout(self) -> bool:
        """Read-only view of the evaluation mode.

        Was a plain mutable attribute the agent set on itself, which meant
        the thing being measured owned the flag saying whether it was
        being measured. Kept under the old name because three call sites
        (confidence.py, routing.py, benchmarking.py) read it and their
        logic is unchanged -- what changed is that nothing can assign to
        it.
        """
        return not self._eval_mode.learning_enabled

    @property
    def evaluation_mode(self):
        return self._eval_mode

    def enter_evaluation(self, mode) -> None:
        """Accept an evaluation context issued by mana.core.evaluation.

        Rejects anything the core did not issue. Not for security -- Python
        offers none -- but so that an unauthorized context is a visible
        error at the point of use rather than a silent equivalence.
        """
        if mode.is_measured and not mode.authorized:
            raise PermissionError(
                "evaluation mode was not issued by mana.core.evaluation; "
                "use open_evaluation() rather than constructing one")
        self._eval_mode = mode

    def leave_evaluation(self) -> None:
        self._eval_mode = evaluation.normal()

    def brains_status(self) -> Dict[str, Any]:
        """Everything about the pool: which brains exist, which are ready,
        which are cooling down or out of free-tier quota, and their measured
        latency/quality. API keys are never included (BrainSpec.public_dict
        drops them) so this is safe to print, log and put in a report."""
        return self.llm.pool.status()

    def ask_consensus(self, task: str, n: int = 2, **kwargs: Any) -> Dict[str, Any]:
        """Ask N brains the same question and report their agreement.
        Exposed on the agent (not only as a tool) because the interactive
        REPL and the CLI both need it as a first-class operation."""
        result = self.tools.call("llm_consensus", prompt=task, n=int(n), task=task, **kwargs)
        return {"answer": result.output, "ok": result.ok, "error": result.error,
                "latency": result.latency, **result.meta}

    def solve_decomposed(self, task: str, **kwargs: Any) -> Dict[str, Any]:
        """Split a task across brains and synthesize. See mana.decompose."""
        result = self.tools.call("decompose_task", task=task, **kwargs)
        return {"answer": result.output, "ok": result.ok, "error": result.error,
                "latency": result.latency, **result.meta}

    def _verify_answer(self, task: str, answer: str, category: str) -> Dict[str, Any]:
        """Choke point for the 'does this claimed answer match an
        extractable ground truth' capability (agent_parts/verifier.verify),
        routed through self.tools ('verify_answer')."""
        result = self.tools.call("verify_answer", task=task, answer=answer, category=category)
        return result.output if result.output is not None else {"kind": "none", "verified": False}

    def _llm_ask_plain(self, prompt: str) -> str:
        """Thin str->str adapter for callers (graph_memory distillation)
        that just want text back, not the full (text, meta) tuple."""
        text, _meta = self._llm_call(prompt, temperature=0.0, context_tag="GRAPH-MEMORY-DISTILL")
        return text or ""

    def graph_memory_status(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        return self.graph_memory.stats(session_id or self.session_id)

    def graph_memory_search(self, query: str, depth: int = 2, limit: int = 8) -> Dict[str, Any]:
        result = self.tools.call("search_graph_memory", query=query, session_id=self.session_id,
                                  depth=depth, limit=limit)
        return {"context": result.output or "", "trace": result.meta.get("trace", {})}

    def _ambiguous_followup(self, task: str):
        """Would answering require guessing which earlier topic is meant?

        Asking is itself a refusal to answer, so this must stay rare: it
        fires only when the question names no subject AND at least two
        distinct topics are in play. Any failure degrades to "not
        ambiguous", i.e. answer as before -- an unnecessary question is
        worse than an occasional wrong guess.
        """
        try:
            # UNITS BUG (found live): recent() returns EVENTS -- user and
            # assistant interleaved -- so asking for 6 showed only 3 user
            # turns. In the observed session the AI-news and football
            # topics had already scrolled out, and the clarifying question
            # offered "про Воронеже или Gismeteo?" for a question about
            # news. Ask for enough events to actually see N user turns.
            recent = self.persistent_memory.recent(
                self.session_id, self.config.clarify_history_turns * 2 + 2)
            # Only the USER's earlier subjects count as candidates. An
            # assistant answer names many incidental entities -- sources,
            # places, products -- and counting those inflated the topic
            # count so badly that a single-subject conversation triggered
            # "про ИИ или РИА?", asking about a news agency the user never
            # raised. Ambiguity is about which of THEIR subjects is meant.
            topics = [extract_entities(str(r.get("content", "")))
                      for r in recent if r.get("kind") == "USER_MESSAGE"]
            return is_ambiguous_followup(task, topics,
                                          min_candidates=self.config.clarify_min_topics)
        except Exception as exc:
            self._vlog(f"ambiguity check unavailable: {exc}")
            from ..intent import AmbiguousReference
            return AmbiguousReference(False, reason=f"check failed: {exc}")

    def solve_task(self, task: str) -> Dict[str, Any]:
        if self.config.clarify_ambiguous_followups:
            ambiguous = self._ambiguous_followup(task)
            if ambiguous:
                question = format_clarifying_question(ambiguous.candidates)
                result = {"task": task, "answer": question, "latency": 0.0,
                          "trace": {"clarification_requested": True,
                                    "candidates": ambiguous.candidates},
                          "pipeline": asdict(self.pipeline), "critic_score": 0.0,
                          "critic_trace": {}, "llm_ok": False, "passes_used": 0,
                          "timeout_count": 0, "fallback": False, "llm_latency": 0.0,
                          "verification_trust": "UNVERIFIED",
                          "clarification": {"asked": True, "candidates": ambiguous.candidates,
                                            "reason": ambiguous.reason}}
                self.history.append({"type": "clarification", "task": task, "cycle": self.cycle,
                                      "latency": 0.0, "reliability": 1.0, "quality": None,
                                      "timestamp": time.time(), **result})
                self._save_state()
                return result

        if self._is_memory_write_request(task):
            text=self._extract_memory_text(task)
            saved=self.persistent_memory.store_explicit_memory(self.session_id,text,scope="session")
            answer = f"Запомнила: {text}" if saved.get("stored") else "Не удалось сохранить эту запись в память."
            result={"task":task,"answer":answer,"latency":0.0,"trace":{"persistent_memory":1,"memory_action":"STORE"},"pipeline":asdict(self.pipeline),"critic_score":0.0,"critic_trace":{},"llm_ok":False,"passes_used":0,"timeout_count":0,"fallback":False,"llm_latency":0.0,"memory_action":"STORE","memory_store":saved}
            self.history.append({"type":"memory_action","task":task,"cycle":self.cycle,"latency":0.0,"reliability":1.0,"quality":None,"timestamp":time.time(),**result})
            self._save_state()
            return result
        result = self.answer(task, self.pipeline, save_memory=True, context_tag="USER")
        self.history.append({
            "type": "user_task", "task": task, "cycle": self.cycle,
            "latency": result["latency"], "reliability": 1.0 if result.get("llm_ok") else 0.0,
            "quality": None, "timestamp": time.time(), **result,
        })
        self._save_state()
        # Graph memory is written only for genuine user-facing exchanges
        # (this method), never from benchmark/screening/evolution calls to
        # answer() -- those pass save_memory=False and go through
        # _answer_core/_adaptive_answer_v41 directly, not solve_task, so
        # synthetic benchmark queries never pollute the long-term graph.
        try:
            answer_text = str(result.get("answer") or "")
            if answer_text:
                self.graph_memory.record_turn(self.session_id, task, answer_text,
                                               llm_ask=self._llm_ask_plain if self._tool_available("llm_generate") else None)
                self.graph_memory.maybe_rollup_episode(
                    self.session_id, every_n_turns=self.config.graph_memory_episode_every_n_turns,
                    llm_ask=self._llm_ask_plain if self._tool_available("llm_generate") else None)
        except Exception as exc:
            self._vlog(f"graph memory write failed: {exc}")
        return result

    def interactive(self) -> None:
        print(f"\nMANA {self.VERSION} interactive")
        print("Команды: /status /improve /benchmark /routing /routing-holdout /adaptive /adaptive-holdout /report /memory /memory-search /remember /knowledge-status /learn /experience /verify /exec-on /clear-cache /exit")
        while True:
            try: task = input("\nВы: ").strip()
            except (EOFError, KeyboardInterrupt): print(); break
            if not task: continue
            low = task.lower()
            if low in {"/exit", "exit", "quit", "выход"}: break
            if self._is_cli_command(task):
                print("Это командная строка MANA, а не пользовательская задача. Выполните её в PowerShell или используйте соответствующую /команду."); continue
            if self._is_memory_write_request(task):
                memory_text=self._extract_memory_text(task)
                saved=self.persistent_memory.store_explicit_memory(self.session_id, memory_text, scope="session")
                print("Память обновлена: " + (memory_text or "пусто") if saved.get("stored") else "Не удалось сохранить: " + str(saved.get("reason"))); continue
            if low == "/status":
                print(json.dumps(self.evolution_status(), ensure_ascii=False, indent=2)); continue
            if low == "/report":
                print(json.dumps(self.latest_evolution_report() or {"report": None}, ensure_ascii=False, indent=2)); continue
            if low == "/memory":
                print(f"memory entries={len(self.memory.entries)} persistent={self._memory_event_count(self.session_id)} semantic={self.knowledge_status().get('memory_items',0)}"); continue
            if low.startswith("/memory-search"):
                q=task[len("/memory-search"):].strip(); print(json.dumps(self._json_safe(self.persistent_memory.safe_search_global(q,self.config.memory_semantic_top_k)),ensure_ascii=False,indent=2)); continue
            if low.startswith("/remember"):
                txt=task[len("/remember"):].strip(); saved=self.persistent_memory.store_explicit_memory(self.session_id,txt,scope="session")
                print(json.dumps(saved,ensure_ascii=False,indent=2)); continue
            if low == "/knowledge-status":
                print(json.dumps(self.knowledge_status(),ensure_ascii=False,indent=2)); continue
            if low.startswith("/learn"):
                src=task[len("/learn"):].strip() or input("Источник (файл/каталог/тема): ").strip(); print(json.dumps(self.acquire_knowledge(src,domain="1c" if "1с" in src.lower() else "manual"),ensure_ascii=False,indent=2)); continue
            if low == "/experience":
                print(f"experience records={self.experience.count()}"); continue
            if low == "/verify":
                expr = input("Выражение: ").strip(); print(json.dumps(self.tools.call("verify_arithmetic", expression=expr).output, ensure_ascii=False, indent=2)); continue
            if low == "/exec-on":
                self.config.local_exec_enabled = True; print("Локальный sandbox включён. Права не повышаются."); continue
            if low == "/clear-cache":
                self.fitness_cache = {}; self._save_cache(); print("Кэш fitness очищен"); continue
            if low == "/improve":
                e = self.self_improve(); print(f"cycle={e['cycle']} verdict={e['report']['verdict']} accepted={e['improved']}"); continue
            if low == "/benchmark":
                self.benchmark(); continue
            if low == "/routing":
                self.routing_benchmark(); continue
            if low == "/routing-holdout":
                print(json.dumps(self.routing_holdout(), ensure_ascii=False, indent=2)); continue
            if low == "/adaptive":
                print(json.dumps(self.adaptive_benchmark(False), ensure_ascii=False, indent=2)); continue
            if low == "/adaptive-holdout":
                print(json.dumps(self.adaptive_benchmark(True), ensure_ascii=False, indent=2)); continue
            r = self.solve_task(task)
            # Show the verification trust level inline: without it the person
            # has no way to tell whether an answer was independently checked,
            # merely passed the model's own tests, or not checked at all --
            # which is exactly the distinction P0 #2 introduced.
            trust = r.get("verification_trust") or "UNVERIFIED"
            trust_label = {"INDEPENDENTLY_VERIFIED": "проверено независимо",
                           "MODEL_TESTED": "прошло тесты, написанные самой моделью (не доказательство)",
                           "UNVERIFIED": "не проверено"}.get(trust, trust)
            # Audit #16 made the difference between "the web worked" and "we
            # tried the web and answered without it" real in the data --
            # surface it here too, otherwise the person still cannot tell
            # whether an answer about current events actually rests on
            # anything fetched, or is the model reciting from memory.
            rex = r.get("route_execution") or {}
            web_label = ""
            if rex.get("web_required"):
                web_label = {
                    "success": f"веб: {r.get('trace', {}).get('web', 0)} результатов",
                    "degraded_no_evidence": "ВЕБ НЕ ДАЛ ДАННЫХ — ответ без источников"
                                            + (f" ({rex.get('web_failure_reason')})"
                                               if rex.get("web_failure_reason") not in (None, "", "unknown") else ""),
                    "tool_not_attempted": "веб не вызывался",
                    "tool_unavailable": "веб отключён",
                }.get(rex.get("status", ""), f"веб: {rex.get('status')}")
            elif rex.get("status") == "unexpected_web_use":
                web_label = "локальный маршрут неожиданно обратился к вебу"
            parts = [f"latency={r['latency']:.2f}s", trust_label]
            if web_label:
                parts.append(web_label)
            parts.append("p50/p95 benchmark see /report")
            print(f"\nMANA: {r['answer']}\n[{' | '.join(parts)}]")
