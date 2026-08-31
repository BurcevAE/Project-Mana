"""
mana.agent_parts.benchmarking — BenchmarkingMixin: metric aggregation and the routing/adaptive/control benchmark runners (measurement only, no selection logic).
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
__version__ = "1.0"


class BenchmarkingMixin:
    def adaptive_benchmark(self, holdout: bool = False, spec: Optional[PipelineSpec] = None, repetitions: Optional[int] = None) -> Dict[str, Any]:
        """v4.2 benchmark with repeated measurements and strict train/holdout isolation."""
        repetitions = int(repetitions or (self.config.adaptive_holdout_repetitions if holdout else self.config.adaptive_benchmark_repetitions))
        if holdout:
            tasks = [
                ("hold_math_1","Посчитай 37 * 16. Нужен только результат.","local",["592"],"math"),
                ("hold_math_2","Раздели 528 на 24.","local",["22"],"math"),
                ("hold_prog_1","Объясни назначение регрессионных тестов.","local",["тест"],"programming"),
                ("hold_prog_2","Зачем хранить историю изменений исходников?","local",["измен"],"programming"),
                ("hold_reason_1","Почему опыт агента полезно хранить отдельно от весов модели?","local",["памят"],"reasoning"),
                ("hold_current_1","Какая сейчас актуальная версия Python?","web",[],"current"),
                ("hold_news_1","Какие свежие события сейчас важны для локальных LLM?","web",[],"news"),
                ("hold_price_1","Какая сейчас цена золота и на какую дату относится значение?","web",[],"price"),
                ("hold_compare_1","Сравни актуальные Python и Java для нового проекта.","mixed",[],"compare"),
                ("hold_research_1","Исследуй свежие подходы к локальным LLM и сопоставь их.","mixed",[],"research"),
                ("hold_compare_2","Сравни два современных подхода к inference локальных моделей.","mixed",[],"compare"),
                ("hold_hard_1","Разбери компромисс качество, стоимость и задержку LLM routing.","local",[],"hard_reasoning"),
            ]
        else:
            tasks = [
                ("train_math_1","Сколько будет 18 * 14?","local",["252"],"math"),
                ("train_math_2","Вычисли 27 * 13.","local",["351"],"math"),
                ("train_prog_1","Зачем нужны unit-тесты?","local",["тест"],"programming"),
                ("train_prog_2","Что делает Git?","local",["верси"],"programming"),
                ("train_reason_1","Почему память агента лучше отделять от весов?","local",["памят"],"reasoning"),
                ("train_reason_2","Объясни пользу проверки ответа перед публикацией.","local",["пров"],"reasoning"),
                ("train_current_1","Какая сейчас последняя стабильная версия Python?","web",[],"current"),
                ("train_news_1","Какие сейчас новости о локальных LLM?","web",[],"news"),
                ("train_price_1","Какая сейчас цена золота?","web",[],"price"),
                ("train_compare_1","Сравни актуальные Python и Java.","mixed",[],"compare"),
                ("train_compare_2","Сравни современные локальные LLM подходы.","mixed",[],"compare"),
                ("train_research_1","Найди свежие данные о локальных LLM и сравни два подхода.","mixed",[],"research"),
                ("train_hard_1","Проанализируй trade-off качества, стоимости и latency LLM routing.","local",[],"hard_reasoning"),
                ("train_hard_2","Предложи стратегию проверки противоречивых источников.","mixed",[],"hard_reasoning"),
            ]
        base = PipelineSpec(**asdict(spec or self.pipeline)).normalize(self.config)
        rows=[]
        self._benchmark_learning = True
        self._benchmark_holdout = bool(holdout)
        try:
            for rep in range(max(1, repetitions)):
                for tid,q,expected,must,cat in tasks:
                    r=self.answer(q, spec=base, save_memory=False, context_tag=("HOLD-" if holdout else "ADAPT-")+tid+f" R{rep+1}")
                    ans=str(r.get("answer","")); low=ans.lower()
                    score = 1.0 if (not must or all(x.lower() in low for x in must)) else .25
                    if not ans.strip(): score=0.0
                    adaptive=r.get("adaptive",{}) or {}; trace=r.get("trace",{}) or {}
                    route=r.get("route")
                    execution=bool(r.get("route_execution",{}).get("execution_success"))
                    verification = r.get("verification") or {}
                    row={"id":tid,"rep":rep+1,"category":cat,"expected_route":expected,"chosen_route":route,
                         "route_correct":route==expected,"execution_success":execution,
                         "score":score,"confidence":r.get("adaptive_confidence",0.0),"threshold":adaptive.get("threshold"),
                         "steps":adaptive.get("steps_used",1),"stop_reason":adaptive.get("stop_reason"),"graph":adaptive.get("graph",[]),
                         "latency":r.get("latency",0.0),"web_attempted":bool(trace.get("web_attempted",False)),"web_ok":bool(trace.get("web_ok",False)),
                         "critic_score":float(r.get("critic_score",0.0) or 0.0),
                         "verification_used":bool(verification),"verification_success":bool(verification.get("verified",False)),
                         "verification_kind":verification.get("kind","none")}
                    rows.append(row)
                    if not holdout and route in {"local","web","mixed"}:
                        self._record_route_outcome(q, route, score, execution, bool(trace.get("web_ok",False)), float(r.get("latency",0.0)))
                        self._record_learned_route_example(q, expected, score)
                        self._learn_confidence_calibration(q, float(row["confidence"]), score)
                        self._learn_stop_outcome(q, int(row["steps"]), score >= 0.75, float(row["confidence"]))
        finally:
            self._benchmark_learning = False
            self._benchmark_holdout = False
        n=max(1,len(rows)); webrows=[x for x in rows if x["web_attempted"]]
        bycat={}
        for cat in sorted(set(x["category"] for x in rows)):
            rr=[x for x in rows if x["category"]==cat]
            bycat[cat]={"tasks":len(rr),"quality":float(np.mean([x["score"] for x in rr])),
                        "route_accuracy":float(np.mean([x["route_correct"] for x in rr])),"execution_accuracy":float(np.mean([x["execution_success"] for x in rr])),
                        "avg_steps":float(np.mean([x["steps"] for x in rr])),"p50_latency":self._quantile([x["latency"] for x in rr],.5),
                        "confidence":float(np.mean([x["confidence"] for x in rr]))}
        return {"version":self.VERSION,"holdout":holdout,"tasks":len(rows),"route_accuracy":float(np.mean([x["route_correct"] for x in rows])),
                "execution_accuracy":float(np.mean([x["execution_success"] for x in rows])),"quality":float(np.mean([x["score"] for x in rows])),
                "avg_steps":float(np.mean([x["steps"] for x in rows])),"p50_latency":self._quantile([x["latency"] for x in rows],.5),
                "p95_latency":self._quantile([x["latency"] for x in rows],.95),"web_attempt_rate":float(np.mean([x["web_attempted"] for x in rows])),
                "web_success_rate":float(np.mean([x["web_ok"] for x in webrows])) if webrows else 1.0,
                "repetitions": max(1,repetitions),
                "confidence_calibration": {k:{"n":v.get("n",0),"bias":v.get("bias",0.0),"mae":(v.get("abs_error_sum",0.0)/max(1,v.get("n",0)))} for k,v in self.confidence_stats.items()},
                "learned_router": {"enabled":bool(self.config.learned_router_enabled),"trained":self.learned_router_model is not None,
                                   "samples":len(self.learned_route_examples),"trained_n":self.learned_router_trained_n,
                                   "classes":list(self.learned_router_classes)},
                "verification": {"available":bool(self.config.local_exec_enabled),
                                  "verified_rows":int(sum(1 for x in rows if x.get("verification_success"))),
                                  "attempted_rows":int(sum(1 for x in rows if x.get("verification_used")))},
                "by_category":bycat,"rows":rows,"timestamp":time.time()}


    # ---------- metrics ----------
    @staticmethod
    def _quantile(values: List[float], q: float) -> float:
        if not values: return 0.0
        return float(np.quantile(np.asarray(values, dtype=float), q))

    def _metrics_from_rows(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        scores = [float(x["score"]) for x in rows]
        lat = [float(x["latency"]) for x in rows]
        timeouts = sum(int(x.get("timeouts", 0)) for x in rows)
        fallbacks = sum(int(x.get("fallback", False)) for x in rows)
        llm_ok = sum(int(x.get("llm_ok", False)) for x in rows)
        n = max(1, len(rows))
        return {
            "tasks": len(rows),
            "quality": float(np.mean(scores)) if scores else 0.0,
            "pass_rate": float(sum(s >= .75 for s in scores) / n) if scores else 0.0,
            "reliability": float(llm_ok / n) if scores else 1.0,
            "avg_latency": float(np.mean(lat)) if lat else 0.0,
            "median_latency": float(np.median(lat)) if lat else 0.0,
            "p50_latency": self._quantile(lat, .50),
            "p95_latency": self._quantile(lat, .95),
            "timeouts": timeouts,
            "timeout_rate": timeouts / n,
            "fallbacks": fallbacks,
            "fallback_rate": fallbacks / n,
            "task_scores": scores,
            "avg_steps": float(np.mean([float(x.get("adaptive_steps", 1)) for x in rows])) if rows else 1.0,
        }


    # ---------- benchmark runs ----------
    def _benchmark_tasks(self, spec: PipelineSpec, tasks: List[BenchmarkTask], label: str,
                         repetitions: int = 1, tag_prefix: str = "") -> Dict[str, Any]:
        rows = []
        started = time.perf_counter()
        total = len(tasks) * max(1, repetitions)
        done = 0
        for rep in range(max(1, repetitions)):
            for task in tasks:
                done += 1
                self._vlog(f"{label} TASK {done}/{total} START | id={task.task_id} | rep={rep+1}")
                t0 = time.perf_counter()
                r = self.answer(task.query, spec=spec, save_memory=False,
                                context_tag=f"{tag_prefix}{label} {task.task_id} r{rep+1}")
                score = BenchmarkSuite.score(task, r["answer"])
                row = {
                    "id": task.task_id, "category": task.category, "rep": rep + 1,
                    "score": score, "latency": float(time.perf_counter() - t0),
                    "answer": r["answer"], "llm_ok": bool(r["llm_ok"]),
                    "timeouts": int(r.get("timeout_count", 0)), "fallback": bool(r.get("fallback", False)),
                    "web": int(r["trace"].get("web", 0)), "web_attempted": bool(r["trace"].get("web_attempted", False)),
                    "adaptive_steps": int((r.get("adaptive", {}) or {}).get("steps_used", 1)),
                    "adaptive_confidence": float(r.get("adaptive_confidence", 0.0) or 0.0),
                }
                rows.append(row)
                self._vlog(f"{label} TASK {done}/{total} END | id={task.task_id} | score={score:.3f} | time={row['latency']:.2f}s | timeout={row['timeouts']} | fallback={row['fallback']}")
        elapsed = time.perf_counter() - started
        metrics = self._metrics_from_rows(rows)
        metrics["wall_time"] = elapsed
        metrics["rows"] = rows
        self._vlog(f"{label} END | quality={metrics['quality']:.3f} | p50={metrics['p50_latency']:.2f}s | p95={metrics['p95_latency']:.2f}s | timeout_rate={metrics['timeout_rate']:.1%} | fallback_rate={metrics['fallback_rate']:.1%} | wall={elapsed:.1f}s")
        return metrics

    def _routing_tasks(self) -> List[Dict[str, Any]]:
        return [
            {"id":"route_local_math","query":"Сколько будет 17 * 23? Ответь кратко.","category":"math","expected":"local","must":["391"]},
            {"id":"route_local_programming","query":"Объясни кратко, зачем нужен unit-test.","category":"programming","expected":"local","must":["тест"]},
            {"id":"route_local_reasoning","query":"Почему полезно отделять память агента от весов модели?", "category":"reasoning","expected":"local","must":["памят"]},
            {"id":"route_web_current","query":"Какая сейчас последняя версия Python? Ответь кратко.","category":"current","expected":"web","must":[]},
            {"id":"route_web_news","query":"Какие последние новости о разработке искусственного интеллекта? Кратко.","category":"current","expected":"web","must":[]},
            {"id":"route_web_price","query":"Какая сейчас цена золота? Укажи, что данные актуальны на момент поиска.","category":"current","expected":"web","must":[]},
            {"id":"route_mixed_compare","query":"Сравни актуальные версии Python и Java и объясни, какую выбрать для нового проекта.","category":"current","expected":"mixed","must":[]},
            {"id":"route_mixed_research","query":"Найди свежие данные о двух подходах к локальным LLM и сравни их преимущества.","category":"current","expected":"mixed","must":[]},
            {"id":"route_mixed_price","query":"Сравни текущие цены двух вариантов и рассчитай, какой выгоднее.","category":"current","expected":"mixed","must":[]},
        ]

    def routing_benchmark(self, spec: Optional[PipelineSpec] = None) -> Dict[str, Any]:
        base = PipelineSpec(**asdict(spec or self.pipeline)).normalize(self.config)
        tasks = self._routing_tasks()
        print("\\n🧭 ROUTING BENCHMARK v3.4.11")
        rows = []
        for task in tasks:
            forced = PipelineSpec(**asdict(base)).normalize(self.config)
            forced.route_mode = "auto"
            t0 = time.perf_counter()
            r = self.answer(task["query"], forced, save_memory=False, context_tag=f"ROUTING {task['id']}")
            elapsed = time.perf_counter() - t0
            chosen = r.get("route") or self.classify_route(task["query"])
            expected = task["expected"]
            route_correct = chosen == expected
            web_attempted = bool(r.get("trace", {}).get("web_attempted", False))
            web_ok = bool(r.get("trace", {}).get("web_ok", False))
            execution_correct = bool(r.get("route_execution", {}).get("execution_success", False))
            base_score = BenchmarkSuite.score(BenchmarkTask(task["id"], task["query"], task["must"], category=task["category"]), r["answer"])
            answer_quality = base_score
            # Current WEB/MIXED benchmark tasks require a successful web execution;
            # an offline fallback must never score as a correct current-information answer.
            if expected in {"web", "mixed"}:
                answer_quality = base_score if web_ok and r.get("answer") else 0.0
            score = answer_quality
            rows.append({"id":task["id"],"category":task["category"],"expected_route":expected,"chosen_route":chosen,
                         "route_correct":route_correct,"execution_correct":execution_correct,"web_attempted":web_attempted,
                         "web_ok":web_ok,"score":score,"answer_quality":answer_quality,"latency":elapsed,"fallback":bool(r.get("fallback",False))})
            self._record_route_outcome(task["query"], chosen, answer_quality, execution_correct, web_ok, elapsed)
            self._vlog(f"ROUTING {task['id']} | expected={expected} chosen={chosen} route_ok={route_correct} exec_ok={execution_correct} web={web_attempted}/{web_ok} score={score:.3f} time={elapsed:.2f}s")
        by_arm = {}
        for arm in ["LOCAL","WEB","MIXED"]:
            forced_rows=[]
            for task in tasks:
                forced=PipelineSpec(**asdict(base)).normalize(self.config); forced.route_mode=arm.lower()
                t0=time.perf_counter(); rr=self.answer(task["query"], forced, save_memory=False, context_tag=f"ROUTING ARM {arm} {task['id']}")
                forced_web_attempted=bool(rr.get("trace",{}).get("web_attempted",False)); forced_web_ok=bool(rr.get("trace",{}).get("web_ok",False))
                base_score=BenchmarkSuite.score(BenchmarkTask(task["id"],task["query"],task["must"],category=task["category"]),rr["answer"])
                forced_score=base_score if task["must"] else (1.0 if forced_web_ok and rr.get("answer") else 0.0)
                forced_rows.append({"id":task["id"],"category":task["category"],"score":forced_score,"latency":time.perf_counter()-t0,"web_attempted":forced_web_attempted,"web_ok":forced_web_ok})
            by_arm[arm]={"arm":arm,"tasks":len(forced_rows),"quality":float(np.mean([x["score"] for x in forced_rows])),"p50_latency":self._quantile([x["latency"] for x in forced_rows],.5),"p95_latency":self._quantile([x["latency"] for x in forced_rows],.95),"web_attempt_rate":float(np.mean([x["web_attempted"] for x in forced_rows])),"web_success_rate":float(np.mean([x["web_ok"] for x in forced_rows if x["web_attempted"]])) if any(x["web_attempted"] for x in forced_rows) else 1.0,"rows":forced_rows}
        n=max(1,len(rows)); quality=float(np.mean([x["score"] for x in rows])); route_acc=float(np.mean([x["route_correct"] for x in rows])); exec_acc=float(np.mean([x["execution_correct"] for x in rows])); web_rate=float(np.mean([x["web_attempted"] for x in rows])); web_success=float(np.mean([x["web_ok"] for x in rows if x["web_attempted"]])) if any(x["web_attempted"] for x in rows) else 1.0
        cats={}
        for cat in sorted(set(x["category"] for x in rows)):
            cr=[x for x in rows if x["category"]==cat]; cats[cat]={"tasks":len(cr),"route_accuracy":float(np.mean([x["route_correct"] for x in cr])),"execution_accuracy":float(np.mean([x["execution_correct"] for x in cr])),"quality":float(np.mean([x["score"] for x in cr])),"web_attempt_rate":float(np.mean([x["web_attempted"] for x in cr]))}
        result={"version":self.VERSION,"arms":by_arm,"auto":{"arm":"AUTO","tasks":len(rows),"quality":quality,"p50_latency":self._quantile([x["latency"] for x in rows],.5),"p95_latency":self._quantile([x["latency"] for x in rows],.95),"route_accuracy":route_acc,"execution_accuracy":exec_acc,"web_attempt_rate":web_rate,"web_success_rate":web_success,"rows":rows,"by_category":cats},"timestamp":time.time()}
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return result

    def routing_holdout(self, spec: Optional[PipelineSpec] = None) -> Dict[str, Any]:
        base=PipelineSpec(**asdict(spec or self.pipeline)).normalize(self.config)
        base.route_mode = "auto"
        tasks=[
            ("hold_route_local_1","Вычисли 19 * 17. Только результат.","local",["323"]),
            ("hold_route_local_2","Для чего разработчику Git? Кратко.","local",["верси"]),
            ("hold_route_local_3","Что больше: 0.72 или 0.7?","local",["0.72"]),
            ("hold_route_web_1","Какая самая свежая стабильная версия Python сейчас?","web",[]),
            ("hold_route_web_2","Какие последние новости в области локальных нейросетей?","web",[]),
            ("hold_route_web_3","Какой сегодня курс евро к доллару?","web",[]),
            ("hold_route_mixed_1","Сравни актуальные характеристики двух последних версий Python и объясни различия.","mixed",[]),
            ("hold_route_mixed_2","Найди свежие данные о двух локальных LLM и оцени компромисс скорость/качество.","mixed",[]),
            ("hold_route_mixed_3","Сравни актуальные цены двух вариантов и посчитай экономию.","mixed",[]),
        ]
        rows=[]
        for tid,q,expected,must in tasks:
            t0=time.perf_counter(); r=self.answer(q,base,save_memory=False,context_tag=f"ROUTING HOLDOUT {tid}")
            chosen=r.get("route"); trace=r.get("trace",{}); attempted=bool(trace.get("web_attempted",False)); web_ok=bool(trace.get("web_ok",False))
            execution_correct=bool(r.get("route_execution",{}).get("execution_success",False))
            quality=BenchmarkSuite.score(BenchmarkTask(tid,q,must,category="current" if expected!="local" else "general"),r.get("answer", "")) if must else (1.0 if r.get("answer") and (web_ok if expected in {"web","mixed"} else True) else 0.0)
            elapsed=time.perf_counter()-t0
            route_correct=chosen==expected
            rows.append({"id":tid,"category":"current" if expected!="local" else "local","expected_route":expected,"chosen_route":chosen,"route_correct":route_correct,"execution_correct":execution_correct,"answer_quality":quality,"web_attempted":attempted,"web_ok":web_ok,"latency":elapsed,"fallback":bool(r.get("fallback",False))})
            if not getattr(self, "_benchmark_holdout", False):
                self._record_route_outcome(q, chosen, quality, execution_correct, web_ok, elapsed)
        web_rows=[r for r in rows if r["web_attempted"]]
        by_cat={}
        for cat in sorted(set(r["category"] for r in rows)):
            cr=[r for r in rows if r["category"]==cat]
            by_cat[cat]={"tasks":len(cr),"route_accuracy":float(np.mean([r["route_correct"] for r in cr])),"execution_accuracy":float(np.mean([r["execution_correct"] for r in cr])),"quality":float(np.mean([r["answer_quality"] for r in cr])),"web_attempt_rate":float(np.mean([r["web_attempted"] for r in cr]))}
        return {"arm":"AUTO","tasks":len(rows),"route_accuracy":float(np.mean([r["route_correct"] for r in rows])),"execution_accuracy":float(np.mean([r["execution_correct"] for r in rows])),"quality":float(np.mean([r["answer_quality"] for r in rows])),"web_attempt_rate":float(np.mean([r["web_attempted"] for r in rows])),"web_success_rate":float(np.mean([r["web_ok"] for r in web_rows])) if web_rows else 1.0,"p50_latency":self._quantile([r["latency"] for r in rows],.5),"p95_latency":self._quantile([r["latency"] for r in rows],.95),"by_category":by_cat,"rows":rows}

    def run_control_benchmark(self, spec: Optional[PipelineSpec] = None) -> Dict[str, Any]:
        spec = PipelineSpec(**asdict(spec or self.pipeline)).normalize(self.config)
        self._vlog("CONTROL BEFORE BENCHMARK START")
        train = self._benchmark_tasks(spec, BenchmarkSuite.train_tasks(), "BASELINE-TRAIN", self.config.benchmark_repetitions, "BASE ")
        gen = self._benchmark_tasks(spec, BenchmarkSuite.generalization_tasks(), "BASELINE-GEN", self.config.benchmark_repetitions, "BASE ")
        hold = self._benchmark_tasks(spec, BenchmarkSuite.holdout_tasks(), "BASELINE-HOLDOUT", self.config.holdout_repetitions, "BASE ")
        return {"train": train, "generalization": gen, "holdout": hold, "pipeline": asdict(spec), "timestamp": time.time()}


    # ---------- evolution fitness ----------
    def _fitness_key(self, spec: PipelineSpec) -> str:
        tasks_key = ",".join(t.task_id for t in BenchmarkSuite.train_tasks() + BenchmarkSuite.generalization_tasks())
        return hashlib.sha256(f"{spec.key()}|{tasks_key}|{self.config.seed}|v473".encode()).hexdigest()

    def _evaluate_task_set(self, spec: PipelineSpec, tasks: List[BenchmarkTask], label: str, max_tasks: Optional[int] = None) -> Tuple[float, Dict[str, Any]]:
        selected = tasks[:max_tasks] if max_tasks else tasks
        m = self._benchmark_tasks(spec, selected, label, 1, "EVO ")
        # Fitness for evolution uses training/generalization plus modest speed/reliability terms.
        quality = m["quality"]
        pass_rate = m["pass_rate"]
        reliability = m["reliability"]
        latency_score = max(0.0, min(1.0, 1.0 - m["p50_latency"] / 20.0))
        avg_steps = float(m.get("avg_steps", 1.0))
        # v4.2: quality remains dominant, but evolution pays a measurable
        # penalty for unnecessary compute and latency.
        raw = 0.70 * quality + 0.15 * pass_rate + 0.10 * reliability + 0.05 * latency_score
        cost_penalty = self.config.adaptive_cost_penalty * max(0.0, avg_steps - 1.0)
        latency_penalty = self.config.adaptive_latency_penalty * min(2.0, max(0.0, m["p50_latency"] / 10.0))
        failure_penalty = self.config.adaptive_failure_penalty * float(m.get("fallback_rate", 0.0))
        raw = max(0.0, raw - cost_penalty - latency_penalty - failure_penalty)
        m["avg_steps"] = avg_steps
        m["cost_penalty"] = cost_penalty
        m["latency_penalty"] = latency_penalty
        m["failure_penalty"] = failure_penalty
        return raw * 100.0, m

    def benchmark(self) -> Dict[str, Any]:
        print("\n🧪 BENCHMARK current pipeline")
        r = self.run_control_benchmark(self.pipeline)
        print(f"TRAIN: quality={r['train']['quality']:.3f} p50={r['train']['p50_latency']:.2f}s p95={r['train']['p95_latency']:.2f}s timeouts={r['train']['timeouts']}")
        print(f"GEN:   quality={r['generalization']['quality']:.3f} p50={r['generalization']['p50_latency']:.2f}s p95={r['generalization']['p95_latency']:.2f}s")
        print(f"HOLD:  quality={r['holdout']['quality']:.3f} p50={r['holdout']['p50_latency']:.2f}s p95={r['holdout']['p95_latency']:.2f}s")
        return r
