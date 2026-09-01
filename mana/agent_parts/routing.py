"""
mana.agent_parts.routing — RoutingMixin: local/web/mixed route classification, learned router, and route outcome bookkeeping.
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
from ..llm import LLMClient
from ..pipeline import PipelineSpec, PipelineFactory, BenchmarkTask, BenchmarkSuite
from ..experience import ExperienceDB
from ..verifier import LocalVerifier
from ..memory import MemoryManager
from ..optional_deps import fitz, HAS_FITZ, HAS_SKLEARN, LogisticRegression, HAS_TORCH, DEVICE, HAS_WEB, WEB_BACKEND, torch

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.1"


#: The words that mark a question as needing CURRENT information.
#: Previously this list was written out FOUR separate times in this file
#: (_task_category, _should_use_web, _route_signature, classify_route) and
#: they had already drifted apart. Measured consequence, from a live
#: session: "какая завтра погода в Воронеже?" was classified `general` and
#: never reached the web, while "какая сегодня погода" did -- because
#: "сегодня" was in the lists and "завтра" was in none of them. Likewise
#: "как сыграли ЦСКА и Локомотив?" had no matching word at all, so a
#: sports result question was answered from the model's memory with no
#: search. Same failure mode as the two programming classifiers that
#: disagreed about "функция"; one list is the fix.
CURRENT_INFO_TERMS = (
    # time references -- "завтра"/"вчера" were missing entirely
    "сегодня", "сейчас", "завтра", "вчера", "на этой неделе", "текущ",
    "на данный момент", "кто сейчас", "в 2026", "2026",
    # freshness
    "актуаль", "последн", "новост", "свеж", "новейш", "современн",
    # markets
    "цена", "стоимость", "курс", "котиров",
    # results and events -- sports, elections, scores had no coverage
    "сыгра", "матч", "счёт", "счет", "результат", "победил", "выиграл",
    "проиграл", "чемпионат", "турнир", "погод", "прогноз",
    # english
    "latest", "today", "tomorrow", "current", "recent", "news", "weather",
)

#: Matched as PREFIXES AT A WORD BOUNDARY, not as bare substrings.
#: Plain `in` matching was wrong and measurably so: "курс" fired inside
#: "реКУРСия", "дисКУРС" and "конКУРС", and "счёт" inside "обСЧЁТ", so
#: "объясни, что такое рекурсия" was routed to the web as a
#: currency-rate question. A leading \b keeps "курс"/"курса"/"курсы"
#: while rejecting the words that merely contain those letters.
_CURRENT_INFO_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term in CURRENT_INFO_TERMS) + r")",
    re.IGNORECASE)


def mentions_current_info(text: str) -> bool:
    """Does this text ask for information that changes over time?"""
    return bool(_CURRENT_INFO_RE.search(text or ""))


class RoutingMixin:

    # ---------- pipeline ----------
    def _task_category(self, task: str) -> str:
        t = task.lower()
        if any(x in t for x in ["python", "код", "программ", "sql", "1с", "git", "тест", "функци", "алгоритм"]): return "programming"
        if any(x in t for x in ["сколько", "вычисл", "математ", "процент", "арифмет", "умнож", "раздели"]): return "math"
        if any(x in t for x in ["почему", "сравни", "логик", "объясни"]): return "reasoning"
        if mentions_current_info(t): return "current"
        if any(x in t for x in ["больше", "меньше", "0."]): return "logic"
        return "general"

    def _should_use_web(self, task: str, spec: PipelineSpec) -> bool:
        if spec.web_mode == "never" or not spec.use_web: return False
        if spec.web_mode == "always": return True
        t = task.lower()
        return self._task_category(task) == "current" or mentions_current_info(t)


    # ---------- learned routing (v4.6) ----------
    def _routing_feature_vector(self, task: str) -> List[float]:
        t = (task or "").lower()
        cat = self._task_category(task)
        f = self._task_features(task)
        tokens = len(re.findall(r"\w+", t))
        return [
            float(cat == "math"), float(cat == "programming"), float(cat == "reasoning"),
            float(cat == "current"), float(cat == "general"),
            float(f.get("current", False)), float(f.get("research", False)),
            float(f.get("compare", False)), float(f.get("verifiable", False)),
            float(f.get("programming", False)), float(f.get("reasoning", False)),
            min(1.0, tokens / 80.0), min(1.0, len(t) / 500.0),
            float("цена" in t or "стоимость" in t),
            float("код" in t or "python" in t or "1с" in t),
            float("источник" in t or "ссылк" in t),
        ]

    def _fit_learned_router(self, force: bool = False) -> None:
        if not self.config.learned_router_enabled or not HAS_SKLEARN or LogisticRegression is None:
            self.learned_router_model = None
            return
        n = len(self.learned_route_examples)
        if n < self.config.learned_router_min_samples:
            return
        if not force and self.learned_router_trained_n and n - self.learned_router_trained_n < self.config.learned_router_retrain_every:
            return
        X=[]; y=[]
        for ex in self.learned_route_examples[-self.config.learned_router_history_limit:]:
            try:
                route=str(ex.get("route","local")); X.append(list(ex["x"])); y.append(route)
            except Exception:
                continue
        if len(set(y)) < 2:
            return
        try:
            model=LogisticRegression(max_iter=400, multi_class="auto", class_weight="balanced", random_state=self.config.seed)
            model.fit(np.asarray(X,dtype=float), np.asarray(y,dtype=object))
            self.learned_router_model=model
            self.learned_router_classes=[str(x) for x in model.classes_]
            self.learned_router_trained_n=n
        except Exception as exc:
            self._vlog(f"LEARNED ROUTER TRAIN ERROR: {type(exc).__name__}: {exc}")

    def _learned_route_probabilities(self, task: str) -> Dict[str, float]:
        model=self.learned_router_model
        if model is None:
            return {}
        try:
            probs=model.predict_proba(np.asarray([self._routing_feature_vector(task)],dtype=float))[0]
            return {str(c): float(p) for c,p in zip(model.classes_, probs)}
        except Exception:
            return {}

    def _record_learned_route_example(self, task: str, expected_route: str, score: float) -> None:
        if self._benchmark_holdout or expected_route not in {"local","web","mixed"}:
            return
        self.learned_route_examples.append({"x":self._routing_feature_vector(task), "route":expected_route,
                                            "quality":float(score), "timestamp":time.time()})
        if len(self.learned_route_examples) > self.config.learned_router_history_limit:
            self.learned_route_examples=self.learned_route_examples[-self.config.learned_router_history_limit:]
        self._fit_learned_router(force=False)


    # ---------- adaptive routing (v3.4.11) ----------
    def _route_signature(self, task: str) -> str:
        t = (task or "").lower()
        flags = []
        for name, words in {
            "current": list(CURRENT_INFO_TERMS),
            "research": ["найди", "поищи", "источники", "ссылк", "исследуй", "проверь в интернете", "по данным", "рынок", "inference", "llm"],
            "compare": ["сравни", "сопоставь", "что выгоднее", "какой лучше", "против"],
            "calculation": ["сколько", "вычисл", "процент", "арифмет", "умнож", "раздели"],
            "programming": ["python", "код", "программ", "sql", "1с", "git", "тест"],
        }.items():
            if any(w in t for w in words): flags.append(name)
        return f"{self._task_category(task)}|" + "+".join(flags or ["general"])

    def _route_stat(self, signature: str, route: str) -> Dict[str, Any]:
        key = f"{signature}|{route}"
        return self.routing_stats.setdefault(key, {"n": 0, "quality_sum": 0.0, "exec_sum": 0.0, "web_ok_sum": 0.0, "latency_sum": 0.0, "last": 0.0})

    @staticmethod
    def evaluate_route_execution(route: str, web_attempted: bool, web_ok: bool,
                                  web_enabled: bool = True,
                                  web_reason: str = "") -> Dict[str, Any]:
        """Decide whether a route actually did what the route promised.

        Audit issue #16: the old rule counted `web_attempted` as success --
        so a search that was refused, throttled or returned nothing still
        trained the router as if the web arm had worked. The learner then
        preferred a route on the strength of attempts, not results.

        The distinction that matters is between three outcomes, not two:
          * success  -- the route did its job (web actually returned rows)
          * degraded -- the tool was tried and failed; we answered anyway,
                        from the model alone. Useful to know, but it is NOT
                        evidence the route was a good choice.
          * failure  -- the route could not even be attempted.

        `degraded` is deliberately reported as execution_success=False. An
        answer produced without the evidence the route existed to fetch is
        not a successful execution of that route, however fluent it reads.
        """
        web_required = route in {"web", "mixed"}
        if not web_required:
            success = not web_attempted          # a local route should not have hit the network
            status = "success" if success else "unexpected_web_use"
            return {"execution_success": success, "status": status, "degraded": False,
                    "web_required": False, "web_attempted": web_attempted, "web_ok": web_ok}
        if not web_enabled:
            return {"execution_success": False, "status": "tool_unavailable", "degraded": False,
                    "web_required": True, "web_attempted": web_attempted, "web_ok": web_ok}
        if not web_attempted:
            return {"execution_success": False, "status": "tool_not_attempted", "degraded": False,
                    "web_required": True, "web_attempted": False, "web_ok": web_ok}
        if web_ok:
            return {"execution_success": True, "status": "success", "degraded": False,
                    "web_required": True, "web_attempted": True, "web_ok": True}
        # Attempted but produced nothing usable -- the case that used to be
        # scored as success.
        return {"execution_success": False, "status": "degraded_no_evidence", "degraded": True,
                "web_required": True, "web_attempted": True, "web_ok": False,
                "web_failure_reason": web_reason or "unknown"}

    def _record_route_outcome(self, task: str, route: str, quality: float, execution_success: bool,
                              web_ok: bool, latency: float) -> None:
        if route not in {"local", "web", "mixed"}:
            return
        sig = self._route_signature(task)
        st = self._route_stat(sig, route)
        st["n"] += 1
        st["quality_sum"] += float(max(0.0, min(1.0, quality)))
        st["exec_sum"] += float(bool(execution_success))
        st["web_ok_sum"] += float(bool(web_ok)) if route in {"web", "mixed"} else 1.0
        st["latency_sum"] += float(max(0.0, latency))
        st["last"] = time.time()
        # Keep the routing memory bounded.
        if len(self.routing_stats) > 500:
            old = sorted(self.routing_stats.items(), key=lambda kv: kv[1].get("last", 0.0))[:100]
            for k, _ in old:
                self.routing_stats.pop(k, None)

    def _adaptive_route_score(self, task: str, route: str) -> Optional[float]:
        if not self.config.routing_adaptive_enabled:
            return None
        st = self.routing_stats.get(f"{self._route_signature(task)}|{route}")
        if not st or int(st.get("n", 0)) < self.config.routing_min_observations:
            return None
        n = max(1, int(st["n"]))
        q = st["quality_sum"] / n
        e = st["exec_sum"] / n
        w = st["web_ok_sum"] / n
        lat = st["latency_sum"] / n
        latency_score = 1.0 / (1.0 + lat / 5.0)
        return (self.config.routing_quality_weight * q +
                self.config.routing_reliability_weight * (0.5 * e + 0.5 * w) +
                self.config.routing_latency_weight * latency_score)

    def classify_route(self, task: str) -> str:
        """Conservative deterministic router. Adaptive history may override only after enough evidence."""
        t = (task or "").lower().strip()
        current = mentions_current_info(t)
        research = any(x in t for x in [
            "найди", "поищи", "источники", "ссылк", "исследуй", "проверь в интернете",
            "сравни цены", "сравни актуаль", "по данным", "свежие данные", "рынок"])
        compare = any(x in t for x in ["сравни", "сопоставь", "что выгоднее", "какой лучше", "против", "trade-off", "компромисс"])
        technical_compare = compare and any(x in t for x in ["python", "java", "llm", "нейросет", "inference", "модел", "подход", "характеристик", "технолог"])
        base = "mixed" if (compare and (current or research or technical_compare)) else ("web" if current or research else "local")
        if not self.config.routing_adaptive_enabled:
            return base
        candidates = [base]
        # Only consider alternatives after enough observations; never make a blind jump.
        for r in ("local", "web", "mixed"):
            if r == base: continue
            if self._adaptive_route_score(task, r) is not None:
                candidates.append(r)
        scored = [(r, self._adaptive_route_score(task, r)) for r in candidates]
        scored = [(r, v) for r, v in scored if v is not None]
        if not scored:
            return base
        best_route, best_score = max(scored, key=lambda x: x[1])
        base_score = self._adaptive_route_score(task, base)
        # Historical utility can override only with a measured margin.
        if best_route != base and base_score is not None and best_score >= base_score + 0.08:
            base = best_route
        # v4.6 learned classifier: generalizes benchmark labels instead of relying only on keywords.
        probs = self._learned_route_probabilities(task) if self.config.learned_router_enabled else {}
        if probs:
            learned_route, learned_prob = max(probs.items(), key=lambda kv: kv[1])
            base_prob = probs.get(base, 0.0)
            if learned_route in {"local","web","mixed"} and learned_prob >= 0.55 and learned_prob >= base_prob + self.config.learned_router_margin:
                blend = float(self.config.learned_router_blend)
                if learned_prob >= base_prob + 0.20:
                    return learned_route
                if blend >= 0.20 and learned_prob >= base_prob + self.config.learned_router_margin:
                    return learned_route
        return base

    def _effective_route(self, task: str, spec: PipelineSpec) -> str:
        mode = getattr(spec, "route_mode", "auto")
        if mode != "auto":
            return mode
        return self.classify_route(task)

    def _answer_routed(self, task: str, spec: PipelineSpec, save_memory: bool, context_tag: str) -> Dict[str, Any]:
        route = self._effective_route(task, spec)
        routed = PipelineSpec(**asdict(spec)).normalize(self.config)
        routed.route_mode = route
        if route == "local":
            routed.use_web = False
        elif route == "web":
            routed.use_web = bool(self.config.enable_web and HAS_WEB)
            routed.web_mode = "always"
        elif route == "mixed":
            routed.use_memory = True
            routed.use_web = bool(self.config.enable_web and HAS_WEB)
            routed.web_mode = "always"
        result = self._answer_core(task, routed, save_memory, context_tag)
        web_attempted = bool(result.get("trace", {}).get("web_attempted", False))
        web_ok = bool(result.get("trace", {}).get("web_ok", False))
        route_exec = self.evaluate_route_execution(
            route, web_attempted, web_ok,
            web_enabled=bool(self.config.enable_web),
            web_reason=str(result.get("trace", {}).get("web_error") or ""))
        execution_success = route_exec["execution_success"]
        result["route"] = route
        result["route_expected"] = None
        result["route_execution"] = dict(route_exec, route=route)
        # Runtime telemetry is intentionally weak supervision. Benchmark/holdout supplies exact quality.
        proxy_quality = 0.0 if result.get("fallback") else (float(result.get("critic_score", 0.0)) if result.get("critic_score", 0.0) > 0 else (1.0 if result.get("llm_ok") else 0.0))
        self._record_route_outcome(task, route, proxy_quality, execution_success, web_ok, float(result.get("latency", 0.0)))
        if save_memory:
            self._save_state()
        return result
