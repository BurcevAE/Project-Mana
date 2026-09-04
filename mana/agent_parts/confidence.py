"""
mana.agent_parts.confidence — ConfidenceMixin: architecture selection, confidence estimation/calibration and stop-policy learning for the adaptive-compute loop.
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
__version__ = "2.0"


class ConfidenceMixin:

    # ---------- MANA 4.0 adaptive compute ----------
    def _task_features(self, task: str) -> Dict[str, Any]:
        t = (task or "").lower()
        return {
            "verifiable": bool(re.search(r"\b(?:сколько|посчитай|вычисли|calculate)\b", t) or re.search(r"\d+\s*[\*\+\-/]\s*\d+", t)),
            "current": any(x in t for x in ["сегодня", "сейчас", "актуаль", "последн", "новост", "цена", "курс", "свеж", "текущ", "latest", "today", "current"]),
            "research": any(x in t for x in ["найди", "поищи", "исследуй", "источники", "ссылк", "проверь в интернете", "по данным"]),
            "compare": any(x in t for x in ["сравни", "сопоставь", "что выгоднее", "какой лучше", "против"]),
            "reasoning": self._task_category(task) in {"reasoning", "planning"},
            "programming": self._task_category(task) == "programming",
        }

    def _select_architecture(self, task: str, spec: PipelineSpec, route: str) -> str:
        if spec.architecture != "adaptive":
            return spec.architecture
        f = self._task_features(task)
        if spec.compute_budget >= 4 and (f["research"] or (f["current"] and f["compare"])):
            return "deep"
        if f["current"] or f["research"]:
            return "research" if route in {"web", "mixed"} else "verify"
        if f["reasoning"] or f["programming"]:
            return "verify"
        return "minimal"

    def _architecture_spec(self, base: PipelineSpec, architecture: str, route: str) -> PipelineSpec:
        s = PipelineSpec(**asdict(base)).normalize(self.config)
        s.route_mode = route
        # Architecture is a graph policy; route determines required tools.
        if route == "local":
            s.use_web = False
        elif route == "web":
            s.use_web = bool(self.config.enable_web and HAS_WEB)
            s.web_mode = "always"
        elif route == "mixed":
            s.use_memory = True
            s.use_web = bool(self.config.enable_web and HAS_WEB)
            s.web_mode = "always"
        if architecture == "minimal":
            s.use_critic = False
            s.second_pass_mode = "never"
            if route == "local":
                s.use_memory = False
        elif architecture == "verify":
            s.use_critic = bool(self._tool_available("llm_generate"))
            s.second_pass_mode = "always" if s.compute_budget >= 3 else "auto"
            s.critic_prompt_strategy = "balanced"
        elif architecture == "research":
            s.use_memory = True
            s.use_critic = bool(self._tool_available("llm_generate"))
            s.second_pass_mode = "auto"
            s.prompt_strategy = "researcher"
            s.critic_prompt_strategy = "factcheck"
        elif architecture == "deep":
            s.use_memory = True
            s.use_critic = bool(self._tool_available("llm_generate"))
            s.second_pass_mode = "always"
            s.llm_passes = max(2, s.llm_passes)
            s.prompt_strategy = "analytical"
            s.critic_prompt_strategy = "strict"
        return s.normalize(self.config)

    def _estimate_confidence(self, task: str, result: Dict[str, Any], route: str) -> float:
        """Operational confidence, not a claim of truth. It is deliberately conservative."""
        if result.get("fallback"):
            return 0.15
        c = 0.45 if result.get("llm_ok") else 0.20
        critic = float(result.get("critic_score", 0.0) or 0.0)
        if critic > 0:
            c = 0.35 + 0.60 * critic
        else:
            ans = str(result.get("answer", ""))
            c += min(0.18, len(ans) / 2500.0)
        trace = result.get("trace", {}) or {}
        if route in {"web", "mixed"}:
            if trace.get("web_ok"):
                c += 0.08
            else:
                c -= 0.20
        if int(result.get("timeout_count", 0)) > 0:
            c -= 0.12
        return max(0.0, min(0.99, c))

    def _confidence_bucket(self, task: str) -> str:
        return self._task_category(task)

    def _learn_confidence_calibration(self, task: str, predicted: float, observed_quality: float) -> None:
        if getattr(self, "_benchmark_holdout", False):
            return
        bucket = self._confidence_bucket(task)
        st = self.confidence_stats.setdefault(bucket, {"n": 0, "pred_sum": 0.0, "quality_sum": 0.0, "abs_error_sum": 0.0, "bias": 0.0})
        lr = float(self.config.confidence_learning_rate)
        err = float(observed_quality) - float(predicted)
        st["n"] += 1
        st["pred_sum"] += float(predicted)
        st["quality_sum"] += float(observed_quality)
        st["abs_error_sum"] += abs(err)
        st["bias"] = max(-0.25, min(0.25, float(st.get("bias", 0.0)) * (1.0-lr) + err*lr))

    def _calibration_adjustment(self, task: str) -> float:
        st = self.confidence_stats.get(self._confidence_bucket(task))
        if not st or int(st.get("n", 0)) < self.config.confidence_history_min_observations:
            return 0.0
        return float(st.get("bias", 0.0))

    def _learn_stop_outcome(self, task: str, steps: int, success: bool, confidence: float) -> None:
        if getattr(self, "_benchmark_holdout", False):
            return
        bucket = self._confidence_bucket(task)
        st = self.stop_policy_stats.setdefault(bucket, {"n": 0, "success_sum": 0.0, "steps_sum": 0.0, "last_confidence": 0.0})
        st["n"] += 1
        st["success_sum"] += float(bool(success))
        st["steps_sum"] += float(steps)
        st["last_confidence"] = float(confidence)

    def _learned_stop_adjustment(self, task: str) -> float:
        st = self.stop_policy_stats.get(self._confidence_bucket(task))
        if not st or int(st.get("n", 0)) < self.config.stop_min_observations:
            return 0.0
        n=max(1,int(st["n"]))
        success=float(st["success_sum"])/n
        steps=float(st["steps_sum"])/n
        adj = 0.08*(0.5-success) + 0.03*max(0.0, steps-1.0)
        return max(-0.08,min(0.10,adj))

    def _risk_threshold(self, task: str, spec: PipelineSpec) -> float:
        f = self._task_features(task)
        risk = 0.0
        if f["research"]: risk += .12
        if f["current"]: risk += .10
        if f["compare"]: risk += .08
        if f["reasoning"] or f["programming"]: risk += .05
        base = float(spec.confidence_threshold)
        learned = self._learned_stop_adjustment(task)
        if spec.stop_policy == "fixed":
            return max(.35, min(.95, base))
        if spec.stop_policy == "risk_aware":
            return max(base, min(.95, base + risk + .05 + learned))
        return max(.35, min(.94, base + risk + learned))

    def _evaluate_confidence_v41(self, task: str, result: Dict[str, Any], route: str,
                                 spec: PipelineSpec, previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        trace=result.get("trace",{}) or {}; ans=str(result.get("answer","")).strip(); f=self._task_features(task)
        signals: Dict[str,float]={}
        signals["llm"]=1.0 if result.get("llm_ok") else 0.0
        signals["execution"]=0.0 if result.get("fallback") else (0.45 if result.get("timeout_count",0) else 1.0)
        if route in {"web","mixed"}:
            web_n=int(trace.get("web",0) or 0); signals["evidence"]=1.0 if trace.get("web_ok") and web_n>0 else 0.0
            if f.get("research") and web_n>=3 and trace.get("web_ok"): signals["evidence"]=min(1.0,signals["evidence"]+0.08)
        else:
            mem_n=int(trace.get("memory",0) or 0); signals["evidence"]=min(1.0,0.35+0.12*min(4,mem_n)) if mem_n else 0.35
        critic=float(result.get("critic_score",0.0) or 0.0); signals["critic"]=critic if critic>0 else (0.30 if (f["current"] or f["research"] or f["compare"]) else 0.42)
        if previous and previous.get("answer"):
            old_words=set(re.findall(r"\w+",str(previous.get("answer","")).lower())); cur_words=set(re.findall(r"\w+",ans.lower())); signals["stability"]=len(old_words&cur_words)/max(1,len(old_words|cur_words)) if old_words and cur_words else 0.0
        else: signals["stability"]=min(0.80,0.35+min(0.30,len(ans)/1800.0))
        hist=self._counterfactual_estimates(task,route)["alternatives"].get(route,{})
        signals["history"]=float(hist.get("quality",0.45)) if hist.get("observed") else 0.45
        signals["answer_density"]=min(1.0,len(ans)/900.0) if ans else 0.0
        signals["tool_health"]=float(trace.get("tool_health",1.0) or 1.0)
        # P0 #2: the verification signal is scaled by TRUST LEVEL. Passing
        # LLM-authored tests must not contribute the same 1.0 as an oracle
        # we computed ourselves -- otherwise a model that writes a lenient
        # test for its own buggy code buys full confidence with it.
        _v = result.get("verification") or {}
        _trust = self.verification_trust_level(_v)
        if _trust == self.TRUST_INDEPENDENTLY_VERIFIED:
            signals["verification"] = 1.0
        elif _trust == self.TRUST_MODEL_TESTED:
            signals["verification"] = 0.55
        else:
            signals["verification"] = 0.15 if _v.get("kind") not in {None, "none"} else 0.0
        weights={"llm":.10,"execution":.12,"evidence":.16,"critic":.14,"stability":.08,"history":.10,"answer_density":.04,"tool_health":.04,"verification":.22}
        raw=sum(weights[k]*signals[k] for k in weights)
        if route in {"web","mixed"} and not trace.get("web_ok"): raw-=0.18
        if result.get("fallback"): raw=min(raw,0.20)
        threshold=self._risk_threshold(task,spec); adj=self._calibration_adjustment(task); # Value-of-computation verifier is stronger evidence than answer length.
        confidence=max(0.0,min(.99,raw*float(spec.confidence_calibration)+adj))
        v=(result.get("verification") or {})
        # P0 #2 (second half): this floor used to key off the same boolean
        # `verified`, so passing self-authored tests forced confidence to
        # ~0.92 regardless of the weighted signal above -- silently undoing
        # any grading done there. The floor is now per trust level, so an
        # independently-checked answer still gets its strong floor while
        # self-certification cannot buy one.
        if v.get("verified"):
            if _trust == self.TRUST_INDEPENDENTLY_VERIFIED:
                confidence=max(confidence, min(.99, 0.82 + 0.10*float(v.get("ok",1.0))))
            else:
                # MODEL_TESTED: a real signal, but it must not floor
                # confidence above the point where computation would stop.
                confidence=max(confidence, min(threshold, 0.60))
        elif v.get("kind") == "arithmetic" and not v.get("verified",False):
            confidence=min(confidence, 0.35)
        return {"confidence":confidence,"threshold":threshold,"signals":signals,"raw":raw,"calibration_adjustment":adj}

    def _counterfactual_estimates(self, task: str, chosen: str) -> Dict[str, Any]:
        """Historical alternatives with uncertainty-aware estimates; no extra calls."""
        estimates = {}
        sig = self._route_signature(task)
        for route in ("local", "web", "mixed"):
            st = self.routing_stats.get(f"{sig}|{route}")
            n = int(st.get("n", 0)) if st else 0
            if st and n > 0:
                q = st.get("quality_sum", 0.0) / n
                e = st.get("exec_sum", 0.0) / n
                w = st.get("web_ok_sum", 0.0) / n
                lat = st.get("latency_sum", 0.0) / n
                uncertainty = 1.0 / (n ** 0.5)
                utility = (self.config.adaptive_quality_weight * q +
                           0.18 * e + 0.10 * w +
                           0.10 * (1.0 / (1.0 + lat / 5.0)) -
                           self.config.adaptive_uncertainty_penalty * uncertainty)
                estimates[route] = {"observed": n >= self.config.counterfactual_min_observations,
                                    "observations": n, "quality": q, "execution": e,
                                    "web_success": w, "latency": lat, "uncertainty": uncertainty,
                                    "utility": utility}
            else:
                estimates[route] = {"observed": False, "observations": 0, "uncertainty": 1.0}
        return {"chosen": chosen, "alternatives": estimates}

    def _value_of_computation(self, task: str, current: Optional[Dict[str, Any]], action: str,
                              route: str, spec: PipelineSpec) -> Dict[str, Any]:
        """Estimate marginal value per unit cost of the next computation step."""
        if action == "EXECUTE":
            gain = self.config.voc_execute_gain if self._verify_answer(task, str((current or {}).get("answer", "")), self._task_category(task)).get("kind") != "none" else 0.02
            cost = 0.12
        elif action == "CRITIC":
            gain = self.config.voc_critic_gain
            cost = 0.45
        elif action == "WEB":
            gain = self.config.voc_web_gain if route in {"web","mixed"} else 0.04
            cost = 1.20
        elif action == "MEMORY":
            gain = 0.05
            cost = 0.18
        elif action == "LLM":
            gain = self.config.voc_default_gain
            cost = 0.80
        elif action == "REPAIR":
            gain = 0.08
            cost = 0.55
        elif action == "SYNTHESIS":
            gain = 0.04
            cost = 0.45
        else:
            gain = 0.02; cost = 0.50
        confidence=float((current or {}).get("_eval",{}).get("confidence",0.0) or 0.0)
        # Less confident answers benefit more from additional computation.
        gain *= (1.0 + max(0.0, min(1.0, 0.8-confidence)))
        # Historical quality helps suppress useless repeated work.
        cf=self._counterfactual_estimates(task, route)["alternatives"].get(route,{})
        if cf.get("observed"):
            gain *= (0.75 + 0.50*float(cf.get("quality",0.5)))
        value=gain/max(0.01,cost*self.config.voc_cost_scale)
        return {"action":action,"expected_gain":gain,"expected_cost":cost,"value":value}

    def _next_graph_action(self, graph: Tuple[str, ...], completed: set, confidence: float,
                           threshold: float) -> Optional[str]:
        # MEMORY/WEB are tool prerequisites and are materialized by the next
        # LLM stage; they do not consume a cognitive step by themselves.
        for node in graph:
            if node in completed: continue
            if node in {"MEMORY", "WEB"}:
                return node
            if node == "EVALUATE": return node
            if node == "EXECUTE": return node
            # Do not spend critic/repair/synthesis work while confidence is already high.
            if node in {"CRITIC", "REPAIR", "SYNTHESIS"} and confidence >= threshold:
                completed.add(node)
                continue
            return node
        return None

    def _verification_policy(self, task: str, answer: str, spec: PipelineSpec) -> str:
        if not self.config.verification_policy_enabled or spec.verification_policy == "never":
            return "none"
        arithmetic = self._verify_answer(task, answer, self._task_category(task))
        if arithmetic.get("kind") == "arithmetic":
            return "arithmetic"
        if spec.verification_policy == "arithmetic_only":
            return "none"
        if spec.verification_policy in {"adaptive","always","code_only"} and self._is_code_verifiable_task(task, answer):
            if self._tool_available("run_code"):
                return "code_tests"
        return "none"
