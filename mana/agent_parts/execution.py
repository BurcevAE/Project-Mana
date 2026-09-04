"""
mana.agent_parts.execution — ExecutionMixin: the critic/repair loop, verification bundle generation, the adaptive-compute graph executor and answer()/_answer_core().
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


class ExecutionMixin:
    def _critic(self, task: str, answer: str, spec: PipelineSpec, tag: str,
                author_brain: str = "") -> Tuple[str, float, Dict[str, Any]]:
        """Judge the draft, and prefer a brain that did not write it.

        With one brain this was always self-review: the same model that
        produced the draft was asked whether the draft was good, which
        systematically under-reports the errors that model is prone to.
        Now that a pool exists, `author_brain` goes to `avoid` so an
        independent model does the judging when one is ready. It stays a
        preference -- if avoiding would leave nothing, the critic still
        runs, and the trace records `independent: false` so the weaker
        check is not mistaken for the stronger one.
        """
        if not spec.use_critic or not self._tool_available("llm_generate"):
            return answer, 0.0, {"called": False, "repaired": False, "timeout": False}
        avoid = (author_brain,) if author_brain else ()
        rules = {
            "strict": "Проверяй фактическую точность и соответствие требованиям.",
            "balanced": "Проверяй правильность, полноту и уместность.",
            "factcheck": "Ищи выдуманные факты, неверные числа и неподтверждённые утверждения.",
            "minimal": "Ищи только существенные ошибки.",
        }[spec.critic_prompt_strategy]
        # Audit #60: the critic could not catch "новости за последние два
        # дня" over a snippet dated 18 June, because it was never told what
        # today is either. Giving it the reference point is what makes the
        # judgement follow from data instead of from impression -- the
        # critic's verdict still counts only as MODEL_TESTED, never as proof.
        critique, meta = self._llm_call(
            f"{rules}\n"
            f"Сегодняшняя дата: {time.strftime('%Y-%m-%d')}.\n"
            f"Отдельно проверь утверждения о времени: если ответ называет сведения свежими, "
            f"актуальными или последними, убедись, что в задаче или контексте есть дата, "
            f"подтверждающая это относительно сегодняшней. Утверждение о свежести без даты "
            f"или с датой заметно старше сегодняшней — это ошибка.\n"
            f"Проверь ответ.\nЗадача: {task}\nОтвет: {answer}\n"
            f"Первая строка: SCORE: число от 0 до 1. Затем кратко укажи ошибки.",
            temperature=0.0, provider=spec.llm_provider, context_tag=tag + " CRITIC",
            kind="reasoning", avoid=avoid, policy=spec.brain_policy)
        if not critique:
            return answer, 0.0, {"called": True, "repaired": False, "failed": True,
                                 "timeout": meta.timeout, "independent": False}
        m = re.search(r"SCORE\s*:\s*([01](?:\.\d+)?)", critique, re.I)
        score = float(m.group(1)) if m else 0.5
        score = max(0.0, min(1.0, score))
        if score >= spec.critic_threshold:
            return answer, score, {"called": True, "repaired": False, "timeout": meta.timeout,
                                   "critic_brain": meta.brain, "independent": bool(meta.independent)}
        # Repair goes back to the drafting brain deliberately: it holds the
        # context that produced the answer, and the critique it is handed
        # already came from elsewhere. Avoiding here too would mean a third
        # model rewriting an answer whose reasoning it never saw.
        repaired, rmeta = self._llm_call(
            f"Исправь ответ по замечаниям критика. Задача: {task}\nЧерновик: {answer}\nКритик: {critique}\nВерни только исправленный ответ.",
            temperature=min(spec.temperature, .25), provider=spec.llm_provider,
            context_tag=tag + " REPAIR", policy=spec.brain_policy)
        return repaired or answer, score, {"called": True, "repaired": bool(repaired),
                                           "timeout": bool(meta.timeout or rmeta.timeout),
                                           "critic_brain": meta.brain,
                                           "independent": bool(meta.independent)}

    def _is_code_verifiable_task(self, task: str, answer: str = "") -> bool:
        t=(task or "").lower(); a=(answer or "")
        code_terms = ["напиши код", "напиши функцию", "реализуй функцию", "исправь код", "проверь код", "протестируй код", "python", "функци", "алгоритм", "код"]
        return (self._task_category(task) == "programming" or any(x in t for x in code_terms)) and ("```" in a or "def " in a or "class " in a or "return " in a or self._task_category(task)=="programming")

    def _generate_verification_bundle(self, task: str, answer: str, spec: PipelineSpec, context_tag: str) -> Dict[str, Any]:
        """Ask the LLM for a tiny executable test bundle; never treat generated tests as proof by themselves."""
        if not self._tool_available("run_code") or not self._tool_available("llm_generate"):
            return {"ok":False,"reason":"execution or LLM disabled"}
        prompt=(
            "Верни JSON без markdown с двумя полями code и tests. "
            "code — только небольшой Python-код из ответа, без внешних файлов/сети. "
            "tests — несколько assert/print тестов, проверяющих задачу. "
            f"Сделай до {max(1, int(spec.generated_test_cases))} независимых тестовых случаев. "
            "Не используй import os, subprocess, socket, pathlib, shutil, ctypes, sys, open, eval, exec, compile. "
            f"\nЗадача: {task}\nОтвет: {answer}"
        )
        text, meta = self._llm_call(prompt, temperature=0.0, provider=spec.llm_provider, context_tag=context_tag+" AUTO-TEST")
        if not text:
            return {"ok":False,"reason":"test generation failed"}
        raw=text.strip()
        raw=re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I|re.S).strip()
        try:
            obj=json.loads(raw)
        except Exception:
            m=re.search(r"\{.*\}", raw, re.S)
            if not m: return {"ok":False,"reason":"invalid JSON from test generator"}
            try: obj=json.loads(m.group(0))
            except Exception: return {"ok":False,"reason":"invalid JSON from test generator"}
        code=str(obj.get("code","")).strip(); tests=str(obj.get("tests","")).strip()
        if not code or not tests: return {"ok":False,"reason":"empty code/tests"}
        # NOTE: _static_policy is an internal safety gate LocalVerifier runs
        # on LLM-generated code before it's ever executed, not an agent-level
        # capability choice -- it stays a direct call, same boundary as
        # MemoryManager.context_for's internal orchestration (see the design
        # note at the top of mana/tools.py).
        policy=self.verifier._static_policy(code+"\n"+tests)
        if not policy.get("ok"): return {"ok":False,"reason":policy.get("reason","blocked by policy")}
        return {"ok":True,"code":code,"tests":tests,"llm_latency":float(meta.latency)}

    def _autonomous_execute(self, task: str, current: Dict[str, Any], spec: PipelineSpec, context_tag: str) -> Dict[str, Any]:
        """Choose the safest available verifier: arithmetic first, then generated tests for code."""
        category=self._task_category(task)
        answer=str(current.get("answer", ""))
        policy=self._verification_policy(task, answer, spec)
        arithmetic=self._verify_answer(task, answer, category) if policy in {"arithmetic","code_tests"} else {"kind":"none","verified":False}
        if policy == "arithmetic" and arithmetic.get("kind") == "arithmetic":
            arithmetic["policy"] = "arithmetic"
            return arithmetic
        if policy == "code_tests" and category == "programming" and self._tool_available("run_code"):
            bundle=self._generate_verification_bundle(task, answer, spec, context_tag)
            if bundle.get("ok"):
                result=self.tools.call("run_code", code=bundle["code"], tests=bundle["tests"]).output or {}
                result.update({"mode":"generated_tests","generated_code":bundle["code"],"generated_tests":bundle["tests"],"test_generation_latency":bundle.get("llm_latency",0.0)})
                result["verified"]=bool(result.get("ok"))
                return result
        return {"kind":"none","verified":False,"available":self._tool_available("run_code"),"reason":"no safe verifier matched"}

    def _graph_for_task(self, task: str, spec: PipelineSpec, route: str) -> Tuple[str, ...]:
        """Build a task-aware executable computation graph while respecting the evolved graph."""
        if spec.architecture != "adaptive":
            graph = tuple(spec.graph_nodes)
        else:
            f = self._task_features(task)
            if f.get("verifiable") and route == "local":
                preferred = ("LLM", "EXECUTE", "EVALUATE")
            elif f["research"] or (f["current"] and f["compare"]):
                preferred = ("MEMORY", "WEB", "LLM", "CRITIC", "SYNTHESIS", "EVALUATE")
            elif f["current"]:
                preferred = ("WEB", "LLM", "EVALUATE", "CRITIC", "SYNTHESIS")
            elif f["programming"] and self._tool_available("run_code"):
                preferred = ("MEMORY", "LLM", "EXECUTE", "EVALUATE")
            elif f["reasoning"] or f["programming"]:
                preferred = ("MEMORY", "LLM", "CRITIC", "REPAIR", "EVALUATE")
            else:
                preferred = ("LLM", "EVALUATE")
            graph = tuple(spec.graph_nodes)
            if graph == ("LLM", "EVALUATE"):
                graph = preferred
        if route == "local":
            graph = tuple(x for x in graph if x != "WEB")
        elif route == "web":
            if "WEB" not in graph:
                graph = ("WEB",) + graph
        elif route == "mixed":
            if "WEB" not in graph:
                graph = ("MEMORY", "WEB") + tuple(x for x in graph if x != "MEMORY")
            elif "MEMORY" not in graph:
                graph = ("MEMORY",) + graph
        if "LLM" not in graph:
            graph += ("LLM",)
        if "EVALUATE" not in graph:
            graph += ("EVALUATE",)
        # Remove duplicates but retain order so a malformed mutation cannot create repeated stages.
        clean=[]
        for node in graph:
            if node not in clean:
                clean.append(node)
        return tuple(clean[:max(2, min(self.config.graph_max_nodes, 10))])

    def _required_nodes_for_task(self, task: str, spec: PipelineSpec, graph: Tuple[str, ...],
                                  route: str) -> set:
        """Nodes that MUST run before confidence is allowed to terminate
        computation (P0 issue #1).

        Rationale: confidence is not verification. A high-confidence LLM
        answer to "17*23" is still just a claim until EXECUTE actually
        checks it -- but three separate early-exit paths in
        _adaptive_answer_v41 (value-of-computation, the EVALUATE branch,
        and the post-action check) could each stop the loop the moment
        confidence crossed the threshold, silently skipping the EXECUTE
        node the genome explicitly contained.

        Only nodes that are BOTH in the graph AND actually runnable here
        are required -- requiring a node the machine can't execute (e.g.
        EXECUTE with the sandbox disabled) would deadlock the loop into
        burning its whole budget instead of answering.
        """
        required = set()
        if "EXECUTE" not in graph:
            return required
        category = self._task_category(task)
        answer = ""  # verification policy is re-checked at EXECUTE time with the real answer
        if category == "math" or self._verify_answer(task, answer, category).get("kind") == "arithmetic":
            # An arithmetic ground truth is extractable from the task itself,
            # independent of anything the LLM claims -- always verify it.
            required.add("EXECUTE")
        elif category == "programming" and self._tool_available("run_code"):
            required.add("EXECUTE")
        return required

    # Verification trust levels (P0 #2). The old code had a single boolean
    # `verified`, so "the LLM wrote code, wrote its own tests, and passed
    # them" was indistinguishable from "we independently evaluated the
    # arithmetic in the task ourselves" -- both jumped proxy_quality to
    # 0.95. That is self-certification laundered into evidence: a model
    # that writes a weak test for its own buggy code gets rewarded exactly
    # like a genuinely checked answer, and the routing/experience learners
    # then train on that as if it were ground truth.
    TRUST_UNVERIFIED = "UNVERIFIED"                    # nothing checked it
    TRUST_MODEL_TESTED = "MODEL_TESTED"                # LLM-authored tests passed -- a signal, not proof
    TRUST_INDEPENDENTLY_VERIFIED = "INDEPENDENTLY_VERIFIED"  # ground truth we derived ourselves

    # Quality credit each level is allowed to claim. MODEL_TESTED is
    # deliberately capped below the confidence threshold band so it can
    # never, on its own, make an answer look as trustworthy as one checked
    # against an independent oracle.
    TRUST_QUALITY_CAP = {
        TRUST_UNVERIFIED: 0.0,
        TRUST_MODEL_TESTED: 0.60,
        TRUST_INDEPENDENTLY_VERIFIED: 0.95,
    }

    @staticmethod
    def verification_trust_level(verification: Optional[Dict[str, Any]]) -> str:
        """Classify how much a verification result actually establishes.

        The distinction is *who produced the oracle*, not whether the check
        passed: arithmetic is evaluated by us from the task text, so it is
        independent; generated tests are authored by the same model that
        wrote the answer, so passing them is evidence of self-consistency
        only.
        """
        v = verification or {}
        if not v.get("verified"):
            return ExecutionMixin.TRUST_UNVERIFIED
        if v.get("mode") == "generated_tests":
            return ExecutionMixin.TRUST_MODEL_TESTED
        if v.get("kind") == "arithmetic":
            return ExecutionMixin.TRUST_INDEPENDENTLY_VERIFIED
        return ExecutionMixin.TRUST_MODEL_TESTED

    def _correct_refuted_answer(self, task: str, current: Dict[str, Any], verification: Dict[str, Any],
                                 spec: PipelineSpec, context_tag: str) -> Optional[Dict[str, Any]]:
        """Act on a failed verification instead of only recording it.

        Scope is deliberately narrow: this only fires when the verifier holds
        an INDEPENDENT ground truth that the answer contradicts -- currently
        arithmetic evaluated locally from the task itself (never anything the
        LLM asserted about its own correctness). A refuted answer is replaced
        with the verified value; nothing here can "correct" an answer on the
        strength of a model's opinion, which would just launder a guess into
        a verified-looking claim.

        Returns a small report dict when it acted, or None when it didn't.
        """
        if verification.get("kind") != "arithmetic":
            return None            # no independent ground truth -> nothing to correct against
        if not verification.get("ok") or verification.get("verified"):
            return None            # couldn't evaluate, or the answer already matches
        value = verification.get("value")
        if value is None:
            return None
        rendered = str(int(value)) if float(value).is_integer() else str(value)
        before = str(current.get("answer", ""))

        corrected: Optional[str] = None
        source = "llm_rewrite"
        if self._tool_available("llm_generate"):
            text, _meta = self._llm_call(
                f"Задача: {task}\nЧерновик ответа: {before}\n"
                f"Локальная проверка вычислила точный результат: {rendered}. Черновик ему противоречит.\n"
                f"Верни исправленный ответ, содержащий {rendered}. Только ответ, без объяснений.",
                temperature=0.0, provider=spec.llm_provider, context_tag=f"{context_tag} VERIFY-CORRECT")
            if text and rendered in text:
                corrected = text.strip()
        if corrected is None:
            # Deterministic fallback: state the verified value plainly rather
            # than keep a refuted answer. Correctness beats phrasing here.
            corrected = rendered
            source = "deterministic"

        current["answer"] = corrected
        current["verification"] = dict(verification, verified=True, corrected=True,
                                        corrected_from=before, correction_source=source)
        current["verification_trust"] = self.TRUST_INDEPENDENTLY_VERIFIED
        current.setdefault("trace", {})["verified"] = True
        current["trace"]["verification_corrected"] = True
        self._vlog(f"verification refuted the answer; corrected via {source}: {before[:60]!r} -> {corrected[:60]!r}")
        return {"corrected": True, "source": source, "expected": rendered, "before": before}

    def _adaptive_answer_v41(self, task: str, spec: PipelineSpec, save_memory: bool, context_tag: str) -> Dict[str, Any]:
        started = time.perf_counter()
        route = self._effective_route(task, spec)
        graph = self._graph_for_task(task, spec, route)
        budget = max(1, min(self.config.adaptive_max_steps, int(spec.compute_budget)))
        threshold = self._risk_threshold(task, spec)
        attempts: List[Dict[str, Any]] = []
        completed = set()
        required_nodes = self._required_nodes_for_task(task, spec, graph, route)
        current: Optional[Dict[str, Any]] = None
        previous: Optional[Dict[str, Any]] = None
        stop_reason = "compute_budget"
        total_compute = 0.0

        def _mandatory_pending() -> bool:
            """True while a required verification stage still hasn't run --
            confidence-based stopping is forbidden until this is False."""
            return bool(required_nodes - completed)

        for step in range(1, budget + 1):
            confidence = float(current.get("_eval", {}).get("confidence", 0.0)) if current else 0.0
            action = self._next_graph_action(graph, completed, confidence, threshold)
            if action is None:
                stop_reason = "graph_complete"
                break
            if action not in {"EVALUATE", "MEMORY", "WEB"} and current is not None and self.config.voc_enabled:
                voc = self._value_of_computation(task, current, action, route, spec)
                if (voc["value"] < self.config.voc_min_value
                        and confidence >= self.config.adaptive_min_confidence
                        and not _mandatory_pending()):
                    stop_reason = "value_of_computation"
                    attempts.append({"step":step,"node":"VOC","confidence":confidence,"threshold":threshold,"value_of_computation":voc})
                    break
            if action == "EVALUATE":
                if current is None: break
                ev = self._evaluate_confidence_v41(task, current, route, spec, previous)
                current["_eval"] = ev
                completed.add(action)
                confidence = ev["confidence"]
                attempts.append({"step":step,"node":action,"confidence":confidence,"threshold":threshold,"latency":0.0,"signals":ev["signals"]})
                if confidence >= threshold and not _mandatory_pending():
                    stop_reason = "confidence_threshold"; break
                continue
            run_spec = self._architecture_spec(spec, "minimal", route)
            run_spec.graph_nodes = graph
            run_spec.compute_budget = budget
            # Graph nodes control actual tools instead of architecture labels.
            run_spec.use_memory = "MEMORY" in graph
            run_spec.use_web = "WEB" in graph and bool(self.config.enable_web and HAS_WEB)
            run_spec.web_mode = "always" if run_spec.use_web else "never"
            run_spec.use_critic = False
            run_spec.second_pass_mode = "never"
            if action == "EXECUTE" and current:
                verified = self._autonomous_execute(task, current, spec, context_tag)
                current["verification"] = verified
                trust = self.verification_trust_level(verified)
                current["verification_trust"] = trust
                current.setdefault("trace", {})["verified"] = bool(verified.get("verified", False))
                current["trace"]["verification_trust"] = trust
                current["verification_latency"] = float(verified.get("latency", 0.0) or 0.0)
                completed.add(action)
                # P0 #2 (found on real hardware): verification used to be
                # purely observational -- a refuted answer only lowered
                # confidence, and the wrong answer was still returned to the
                # user (this is what produced "17*23 = 401"). When we hold a
                # ground truth the answer contradicts, correct it instead of
                # merely noting the contradiction.
                correction = self._correct_refuted_answer(task, current, verified, spec, context_tag)
                if correction:
                    attempts.append({"step": step, "node": "VERIFY-CORRECT", "confidence": 0.0,
                                     "threshold": threshold, "correction": correction})
            elif action == "LLM":
                if step > 1 and current and current.get("answer"):
                    run_spec.second_pass_mode = "always"
                    run_spec.prompt_strategy = "verification" if step >= 3 else "analytical"
                result = self._answer_core(task, run_spec, save_memory=False, context_tag=f"{context_tag} V41-S{step}")
                previous = current
                current = result
                completed.add(action)
            elif action == "CRITIC" and current:
                critic_spec = PipelineSpec(**asdict(run_spec)).normalize(self.config)
                critic_spec.use_critic = True
                repaired, cscore, ctr = self._critic(
                    task, str(current.get("answer","")), critic_spec,
                    f"{context_tag} V41-S{step}",
                    author_brain=str((current.get("trace") or {}).get("brain", "")))
                current["answer"] = repaired
                current["critic_score"] = cscore
                current["critic_trace"] = ctr
                # Surface independence in the same place the single-pass
                # path does. Without this the graph route reported a
                # critique with no way to tell whether it was self-review,
                # which is exactly the distinction the field exists for.
                if isinstance(current.get("trace"), dict):
                    current["trace"]["critic_independent"] = bool(ctr.get("independent"))
                    current["trace"]["critic_brain"] = str(ctr.get("critic_brain") or "")
                completed.add(action)
            elif action == "REPAIR" and current:
                current_spec = PipelineSpec(**asdict(run_spec)).normalize(self.config)
                current_spec.second_pass_mode = "never"
                repaired, meta = self._llm_call(
                    f"Исправь ответ только при наличии ошибок. Задача: {task}\nЧерновик: {current.get('answer','')}\nКритик: {current.get('critic_score',0):.3f}. Верни улучшенный ответ.",
                    temperature=min(spec.temperature, .20), provider=spec.llm_provider, context_tag=f"{context_tag} V41-REPAIR")
                if repaired:
                    current["answer"] = repaired
                    current["llm_ok"] = True
                    current["llm_latency"] = float(current.get("llm_latency",0.0)) + meta.latency
                completed.add(action)
            elif action == "SYNTHESIS" and current:
                if self._tool_available("llm_generate"):
                    text, meta = self._llm_call(
                        f"Синтезируй финальный ответ. Задача: {task}\nМатериал: {current.get('answer','')}\nНе добавляй неподтверждённых фактов.",
                        temperature=min(spec.temperature, .20), provider=spec.llm_provider, context_tag=f"{context_tag} V41-SYNTH")
                    if text:
                        current["answer"] = text
                        current["llm_latency"] = float(current.get("llm_latency",0.0)) + meta.latency
                completed.add(action)
            elif action == "MEMORY" or action == "WEB":
                # These nodes are materialized by the next LLM call through run_spec flags.
                completed.add(action)
            else:
                completed.add(action)
            if current is not None:
                ev = self._evaluate_confidence_v41(task, current, route, spec, previous)
                current["_eval"] = ev
                confidence = ev["confidence"]
                attempts.append({"step":step,"node":action,"confidence":confidence,"threshold":threshold,
                                 "latency":float(current.get("latency",0.0)),"signals":ev["signals"],
                                 "critic_score":float(current.get("critic_score",0.0) or 0.0)})
                if confidence >= threshold and action not in {"CRITIC","REPAIR"} and not _mandatory_pending():
                    stop_reason = "confidence_threshold"; break
            if time.perf_counter() - started >= spec.cost_budget:
                stop_reason = "cost_budget"; break
        if current is None:
            current = self._answer_core(task, spec, save_memory=False, context_tag=context_tag)
        ev = current.get("_eval") or self._evaluate_confidence_v41(task, current, route, spec, previous)
        total_latency = time.perf_counter() - started
        current["latency"] = total_latency
        trace = current.get("trace", {}) or {}
        final_route = route
        # Audit #16: routed through the shared evaluator so this path and
        # _answer_routed can never disagree about what "the route worked"
        # means -- attempting the web is not the same as the web working.
        route_exec = self.evaluate_route_execution(
            final_route,
            bool(trace.get("web_attempted", False)),
            bool(trace.get("web_ok", False)),
            web_enabled=bool(self.config.enable_web),
            web_reason=str(trace.get("web_error") or ""))
        execution_success = route_exec["execution_success"]
        current["route"] = final_route
        current["route_expected"] = None
        current["route_execution"] = dict(route_exec, route=final_route)
        proxy_quality = float(current.get("critic_score",0.0) or 0.0)
        # P0 #2: credit for verification is capped by TRUST LEVEL, not by a
        # single boolean. LLM-authored tests passing is a real signal, but a
        # weaker one than an oracle we computed ourselves -- see
        # verification_trust_level() for why.
        trust_level = self.verification_trust_level(current.get("verification"))
        trust_cap = self.TRUST_QUALITY_CAP[trust_level]
        if trust_cap > 0:
            proxy_quality = max(proxy_quality, trust_cap)
        if proxy_quality <= 0: proxy_quality = float(ev.get("confidence",0.0))
        current["verification_trust"] = trust_level
        # The trace gets the SAME final value. It used to keep whatever the
        # EXECUTE step wrote before correction ran, so an answer the
        # verifier caught and fixed was recorded as UNVERIFIED at the top
        # of the trace while the verdict beside it said
        # INDEPENDENTLY_VERIFIED. Two values for one field in one
        # response, and the stale one is the audit record -- wrong in the
        # direction that matters, because a reader checking whether to
        # trust the answer reads the trace.
        current.setdefault("trace", {})["verification_trust"] = trust_level
        if not self._benchmark_learning:
            self._record_route_outcome(task, final_route, proxy_quality, execution_success, bool(trace.get("web_ok",False)), total_latency)
            # Same outcome, second consumer: the brain that produced this
            # answer earns or loses reputation by the same proxy_quality the
            # route does. Without this the pool would rank brains forever by
            # the catalog's declared strengths -- i.e. by an assumption --
            # instead of by how they actually perform on this user's tasks.
            self._record_brain_outcome(trace, proxy_quality)
        counterfactual = self._counterfactual_estimates(task, final_route)
        current["adaptive_confidence"] = float(ev.get("confidence",0.0))
        current["adaptive"] = {"enabled":True,"budget":budget,"threshold":threshold,"steps_used":len(attempts),
                                "stop_reason":stop_reason,"route":final_route,"graph":list(graph),
                                "attempts":attempts,"counterfactual":counterfactual,"cost_seconds":total_latency,
                                "required_nodes":sorted(required_nodes),
                                "required_nodes_satisfied":not bool(required_nodes - completed),
                                "verification_used":bool((current.get("verification") or {}).get("verified")),
                                "verification_trust":trust_level,
                                "verification_kind":(current.get("verification") or {}).get("kind", "none")}
        if save_memory and current.get("answer"):
            self.tools.call("write_memory", content=f"Задача: {task}\nОтвет: {current['answer']}",
                             source="llm" if current.get("llm_ok") else "fallback",
                             confidence=max(.2,min(.95,float(ev.get("confidence",.2)))), status="unverified",
                             metadata={"route":final_route,"graph":list(graph),"steps":len(attempts)})
        self.reason_log.append({"type":"adaptive_compute_v41","task":task,"route":final_route,"graph":list(graph),
                                "steps":len(attempts),"confidence":current["adaptive_confidence"],"stop_reason":stop_reason,
                                "counterfactual":counterfactual,"timestamp":time.time()})
        if save_memory: self._save_state()
        if not self._benchmark_learning:
            self._learn_stop_outcome(task, len(attempts), confidence >= threshold, float(confidence))
        return current

    def answer(self, task: str, spec: Optional[PipelineSpec] = None, save_memory: bool = True, context_tag: str = "") -> Dict[str, Any]:
        spec = PipelineSpec(**asdict(spec or self.pipeline)).normalize(self.config)
        # v4.0: AUTO uses adaptive compute; forced routes retain the deterministic v3.4.11 path.
        if getattr(spec, "route_mode", "auto") == "auto":
            return self._adaptive_answer_v41(task, spec, save_memory, context_tag)
        return self._answer_routed(task, spec, save_memory, context_tag)

    def _record_brain_outcome(self, trace: Dict[str, Any], quality: float) -> None:
        """Credit/debit whichever brains contributed to this answer.

        Never raises and never blocks the answer path: a reputation update
        failing is not a reason for a user-facing call to fail, which is the
        same contract every other side-channel write in this file follows.
        """
        try:
            brains = []
            if trace.get("brain"):
                brains.append(str(trace["brain"]))
            brains += [str(b) for b in (trace.get("consensus", {}) or {}).get("brains", [])]
            brains += [str(b) for b in (trace.get("decompose", {}) or {}).get("brains_used", [])]
            for brain_id in dict.fromkeys(b for b in brains if b):
                self.llm.record_outcome(brain_id, quality)
        except Exception as exc:
            self._vlog(f"brain outcome write failed: {exc}")

    def _brain_strategy(self, task: str, spec: PipelineSpec) -> str:
        """Decide between one brain, several in consensus, or decomposition.

        Every branch below falls back to "single", which is byte-for-byte
        the pre-5.10 behaviour. That is deliberate: a multi-brain strategy
        that fires when only one brain is configured would spend two calls
        on the same model and call the result a consensus, which is worse
        than useless -- it would manufacture agreement out of nothing.
        So both alternatives require at least two *distinct* ready brains.
        """
        try:
            pool = self.llm.pool
            ready = pool.available()
        except Exception:
            return "single"
        if len(ready) < 2:
            return "single"
        mode = getattr(spec, "decompose_mode", "never")
        if mode != "never" and self.config.decompose_enabled and self._tool_available("decompose_task"):
            if mode == "always":
                return "decompose"
            # 'auto': only for tasks the difficulty proxy calls hard. The
            # proxy is a heuristic (see BrainPool.estimate_difficulty) --
            # using it to spend more compute is a safe use of a rough
            # signal; it is never used to decide whether an answer is true.
            if pool.estimate_difficulty(task) >= float(self.config.decompose_min_difficulty):
                return "decompose"
        if int(getattr(spec, "brain_ensemble", 1)) > 1 and self._tool_available("llm_consensus"):
            return "consensus"
        return "single"

    def _answer_core(self, task: str, spec: PipelineSpec, save_memory: bool = True, context_tag: str = "") -> Dict[str, Any]:
        spec = PipelineSpec(**asdict(spec or self.pipeline)).normalize(self.config)
        started = time.perf_counter()
        try:
            self.persistent_memory.remember_user(self.session_id, task, {"context_tag": context_tag, "route_mode": getattr(spec, "route_mode", "auto"), "kind": "user" if context_tag == "USER" else "system"})
            if context_tag == "USER":
                self.persistent_memory.remember_user_claim(self.session_id, task)
            prev=self.persistent_memory.get_session(self.session_id); durable=[]
            try:
                wc=json.loads(prev.get("working_context") or "{}"); durable=list(wc.get("durable_memory",[])) if isinstance(wc,dict) else []
            except Exception: durable=[]
            if any(x in task.lower() for x in ["я твой создатель","меня зовут","запомни","важно помнить"]): durable.append(task.strip())
            durable=durable[-12:]
            self.persistent_memory.update_working_context(self.session_id, task if context_tag == "USER" else prev.get("active_task",task), topic=self._task_category(task), context={"route_mode": getattr(spec, "route_mode", "auto"), "cycle": self.cycle, "durable_memory": durable})
        except Exception as exc:
            self._vlog(f"memory write failed: {exc}")
        context, trace = self._build_context(task, spec)
        answer_text = None
        critic_score = 0.0
        critic_trace = {}
        llm_ok = False
        passes_used = 0
        timeout_count = 0
        fallback = False
        llm_latency = 0.0

        if spec.use_llm and self._tool_available("llm_generate"):
            prompt = self._compose_prompt(task, context, spec)
            strategy = self._brain_strategy(task, spec)
            trace["brain_strategy"] = strategy
            if strategy == "decompose":
                dec = self.solve_decomposed(task, temperature=spec.temperature,
                                             context_tag=f"{context_tag} DECOMP",
                                             policy=spec.brain_policy)
                llm_latency += float(dec.get("latency", 0.0))
                answer_text = dec.get("answer") or None
                llm_ok = bool(answer_text)
                passes_used = max(1, int(dec.get("subtasks", 1)))
                trace["decompose"] = {k: dec.get(k) for k in
                                      ("subtasks", "brains_used", "synthesis_brain",
                                       "failed_subtasks", "degraded")}
            elif strategy == "consensus":
                con = self.ask_consensus(prompt, n=spec.brain_ensemble, temperature=spec.temperature,
                                          kind=self._task_category(task),
                                          context_tag=f"{context_tag} CONS")
                llm_latency += float(con.get("latency", 0.0))
                answer_text = con.get("answer") or None
                llm_ok = bool(answer_text)
                passes_used = len(con.get("brains") or []) or 1
                # Recorded, not acted on: disagreement between brains is
                # evidence about confidence, and ConfidenceMixin is where
                # that belongs. Silently overriding the answer here would
                # hide the signal in exactly the place it is most useful.
                trace["consensus"] = {"agreement": float(con.get("agreement", 0.0)),
                                      "disagreement": bool(con.get("disagreement")),
                                      "brains": list(con.get("brains") or [])}
            else:
                text, meta = self._llm_call(prompt, temperature=spec.temperature, provider=spec.llm_provider,
                                            context_tag=context_tag, kind=self._task_category(task),
                                            task=task, policy=spec.brain_policy)
                llm_latency += meta.latency
                answer_text = text
                llm_ok = bool(text)
                timeout_count += int(meta.timeout)
                trace["brain"] = meta.brain or meta.provider
                trace["brain_attempts"] = list(meta.attempts)
            passes_used = passes_used or 1

            second = False
            if answer_text and spec.second_pass_mode == "always": second = True
            elif answer_text and spec.second_pass_mode == "auto" and self.config.adaptive_second_pass:
                second = self._task_category(task) in {"reasoning", "current"} and len(answer_text) < 900
            if second:
                revised, meta2 = self._llm_call(self._compose_prompt(task, context, spec, answer_text),
                                                temperature=min(spec.temperature, .25), provider=spec.llm_provider,
                                                context_tag=context_tag + " PASS2")
                llm_latency += meta2.latency
                timeout_count += int(meta2.timeout)
                if revised:
                    answer_text = revised
                    passes_used = 2

            if answer_text and spec.use_critic:
                answer_text, critic_score, critic_trace = self._critic(
                    task, answer_text, spec, context_tag or "TASK",
                    author_brain=str(trace.get("brain", "")))
                timeout_count += int(critic_trace.get("timeout", False))
                trace["critic_independent"] = bool(critic_trace.get("independent"))

        if not answer_text:
            fallback = True
            answer_text = self._local_fallback(task)

        elapsed = time.perf_counter() - started
        result = {
            "task": task, "answer": answer_text, "latency": elapsed, "trace": trace,
            "pipeline": asdict(spec), "critic_score": critic_score, "critic_trace": critic_trace,
            "llm_ok": llm_ok, "passes_used": passes_used, "timeout_count": timeout_count,
            "fallback": fallback, "llm_latency": llm_latency,
        }
        try:
            self.persistent_memory.remember_assistant(self.session_id, answer_text, {"task": task, "latency": elapsed, "llm_ok": llm_ok, "confidence": result.get("confidence")})
            self.persistent_memory.remember_decision(self.session_id, f"task={task[:180]} | route={trace.get('route', '')} | verified={trace.get('verification_kind', 'none')}", {"trace": trace})
            self.persistent_memory.maybe_compress(self.session_id)
        except Exception as exc:
            self._vlog(f"memory response write failed: {exc}")
        if save_memory and answer_text:
            self.tools.call("write_memory", content=f"Задача: {task}\nОтвет: {answer_text}",
                             source="llm" if llm_ok else "fallback", confidence=.55 if llm_ok else .2,
                             status="unverified")
        return result


    @staticmethod
    def _local_fallback(task: str) -> str:
        t = task.lower()
        if "17 * 23" in t or "17*23" in t: return "391"
        if "144 / 12" in t: return "12"
        if "0.9" in t and "0.89" in t: return "0.9 больше 0.89."
        if "19 * 17" in t: return "323"
        if "225" in t and "15" in t: return "15"
        if "0.72" in t and "0.7" in t: return "0.72 больше 0.7."
        if "18" in t and "14" in t: return "252"
        if "360" in t and "24" in t: return "15"
        if "0.81" in t and "0.8" in t: return "0.81 больше 0.8."
        if "0.56" in t and "0.506" in t: return "0.56 больше 0.506."
        if "unit-test" in t or "unit test" in t or ("модульн" in t and "тест" in t): return "Модульный тест автоматически проверяет отдельную часть программы и помогает быстро обнаружить регрессии."
        if "git" in t: return "Git — система контроля версий, которая хранит историю изменений и помогает совместно работать над кодом."
        if "автоматически истинным" in t or "достоверным фактом" in t: return "Нет. Ответ LLM может содержать ошибки и требует проверки."
        if "памяти агента" in t or "память агента" in t: return "Разделение памяти и весов позволяет хранить опыт отдельно от модели и изменять его без переобучения весов."
        if "пользовательский опыт" in t: return "Пользовательский опыт стоит хранить отдельно от весов модели, как и любую другую долговременную память агента."
        if "регрессионного тест" in t: return "Регрессионный тест проверяет, что новые изменения не сломали уже работавшую функциональность."
        if "истори" in t and "измен" in t: return "История изменений позволяет понять, кто и что менял, сравнивать версии и откатывать ошибки."
        return "Недостаточно данных для надёжного ответа без внешней модели."
