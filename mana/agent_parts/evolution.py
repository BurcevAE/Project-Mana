"""
mana.agent_parts.evolution — EvolutionMixin: pipeline fitness evaluation, the causal one-mutation-at-a-time GA loop (evolve_pipeline) and the self_improve() cycle driver.
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
from .. import events
from ..knowledge import KnowledgeBase
from ..web import WebSearcher
from ..llm import LLMClient
from ..pipeline import PipelineSpec, PipelineFactory, BenchmarkTask, BenchmarkSuite
from ..experience import ExperienceDB
from ..verifier import LocalVerifier
from ..memory import MemoryManager
from .. import code_evolution as _code_evolution
from ..optional_deps import fitz, HAS_FITZ, HAS_SKLEARN, LogisticRegression, HAS_TORCH, DEVICE, HAS_WEB, WEB_BACKEND, torch

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"

def _out(*parts: Any, **_kw: Any) -> None:
    """print()-shaped adapter onto the event bus.

    Kept print-shaped on purpose: these call sites are status lines whose
    wording and formatting are already right, and rewriting each one into
    a bespoke emit() would have been a large diff with no behavioural
    gain. What changes is where the text goes -- a windowed build has no
    stdout, and a cp1251 console could not encode the markers these lines
    use. Severity is taken from the marker the line already carries.
    """
    text = " ".join(str(p) for p in parts)
    stripped = text.lstrip("\n ")
    if stripped.startswith(("⚠", "❌")):
        events.emit(events.WARNING, text)
    else:
        events.emit(events.STATUS, text)



class EvolutionMixin:
    def list_code_targets(self) -> List[Dict[str, Any]]:
        return _code_evolution.list_targets()

    def code_history(self, target_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return _code_evolution.history(target_id)

    def rollback_code(self, target_id: str, backup_path: Optional[str] = None) -> Dict[str, Any]:
        return _code_evolution.rollback(target_id, backup_path)

    def self_improve_code(self, target_id: str, instruction: str,
                           candidate_source: Optional[str] = None) -> Dict[str, Any]:
        """Gated self-improvement of one whitelisted source-code target (see
        mana/code_evolution.py for the full design/safety rationale).

        Pass `candidate_source` directly to test/replay a specific patch
        (bypassing the LLM) -- this is what makes the pipeline reproducible
        and testable without a live model. Otherwise the agent's own LLM is
        asked to propose one. Either way, acceptance is decided purely by
        the sandbox test results, never by anything the LLM claims.
        """
        if target_id not in _code_evolution.WHITELIST:
            return {"ok": False, "reason": f"unknown target_id: {target_id!r}",
                    "available": [t["target_id"] for t in _code_evolution.list_targets()]}
        if not self._tool_available("run_code"):
            return {"ok": False, "reason": "local execution disabled; start with --enable-local-exec "
                                            "(self-improve-code runs candidates in the same sandbox as run_code)"}
        if candidate_source is None:
            if not self._tool_available("llm_generate"):
                return {"ok": False, "reason": "no candidate_source given and LLM is disabled"}
            # Route the patch proposal to a brain that declares strength in
            # programming, at the top tier: this is the highest-stakes
            # generation MANA performs -- its output is written into MANA's
            # own source -- so it should not land on whichever brain the
            # prompt-length difficulty heuristic happened to pick. The
            # acceptance gate is unchanged; a better proposer only means
            # fewer candidates rejected for being nonsense.
            def _ask_code_brain(prompt, **kwargs):
                kwargs.setdefault("kind", "programming")
                kwargs.setdefault("difficulty", 0.9)
                kwargs.setdefault("policy", "strongest")
                return self._llm_call(prompt, **kwargs)

            proposal = _code_evolution.propose_patch_llm(target_id, _ask_code_brain, instruction)
            if not proposal.get("ok"):
                return proposal
            candidate_source = proposal["candidate_source"]
        evaluation = _code_evolution.evaluate_candidate(target_id, candidate_source, self.verifier)
        decision = _code_evolution.decide(evaluation)
        report: Dict[str, Any] = {"target_id": target_id, "instruction": instruction,
                                   "candidate_source": candidate_source,
                                   "evaluation": evaluation, "decision": decision, "applied": False}
        if decision.get("accepted"):
            apply_result = _code_evolution.apply_patch(target_id, candidate_source, evaluation, decision, instruction)
            report["applied"] = apply_result.get("applied", False)
            report["apply_result"] = apply_result
            self._vlog(f"self_improve_code({target_id}): ACCEPTED and applied -- {decision.get('reason')}")
        else:
            self._vlog(f"self_improve_code({target_id}): rejected -- {decision.get('reason')}")
        return report

    def evaluate_pipeline(self, spec: PipelineSpec, screen: bool = False, screen_tasks: int = 4,
                          context: str = "") -> Tuple[float, Dict[str, Any]]:
        spec = PipelineSpec(**asdict(spec)).normalize(self.config)
        key = self._fitness_key(spec) + ("|screen" if screen else "|full")
        cached = self.fitness_cache.get(key)
        if cached and isinstance(cached, dict) and "fitness" in cached:
            self._vlog(f"CACHE HIT | {'SCREEN' if screen else 'FULL'} | fitness={float(cached['fitness']):.3f}")
            return float(cached["fitness"]), dict(cached["metrics"])
        if screen:
            fit_train, train = self._evaluate_task_set(spec, BenchmarkSuite.train_tasks(), context + "SCREEN-TRAIN", screen_tasks)
            fit_gen = 0.0
            gen = {"quality": 0.0, "pass_rate": 0.0, "reliability": 1.0, "p50_latency": train["p50_latency"], "timeouts": 0, "fallbacks": 0}
            fitness = fit_train
        else:
            fit_train, train = self._evaluate_task_set(spec, BenchmarkSuite.train_tasks(), context + "FULL-TRAIN")
            fit_gen, gen = self._evaluate_task_set(spec, BenchmarkSuite.generalization_tasks(), context + "FULL-GEN")
            fitness = self.config.benchmark_weight * fit_train + self.config.generalization_weight * fit_gen + self.config.latency_weight * max(0.0, min(100.0, (1.0 - train["p50_latency"] / 20.0) * 100.0))
        total_rows = int(train.get("tasks", 0) + gen.get("tasks", 0))
        total_timeouts = int(train.get("timeouts", 0) + gen.get("timeouts", 0))
        total_fallbacks = int(train.get("fallbacks", 0) + gen.get("fallbacks", 0))
        metrics = {
            "fitness": float(fitness),
            "train": train,
            "generalization": gen,
            "evaluated": total_rows,
            "timeouts": total_timeouts,
            "fallbacks": total_fallbacks,
            "timeout_rate": total_timeouts / max(1, total_rows),
            "fallback_rate": total_fallbacks / max(1, total_rows),
            "pipeline_key": spec.key(),
        }
        self.fitness_cache[key] = {"fitness": float(fitness), "metrics": metrics}
        return float(fitness), metrics


    # ---------- population ----------
    def _exploration_level(self) -> float:
        if self.stagnation <= 0: return self.config.exploration_levels[0]
        if self.stagnation < self.config.exploration_stagnation_threshold: return self.config.exploration_levels[1]
        return self.config.exploration_levels[2]

    def _initial_population(self) -> List[PipelineSpec]:
        total = max(1, self.experience.count())
        candidates = [self.pipeline] + self.experience.best("programming", 2) + self.experience.best("reasoning", 2)
        scored = []
        seen = set()
        for p in candidates:
            p = PipelineSpec(**asdict(p)).normalize(self.config)
            if p.key() in seen: continue
            seen.add(p.key())
            scored.append((self.experience.ucb(p.key(), total, self.config.ucb_exploration), p))
        scored.sort(key=lambda x: x[0], reverse=True)
        population = [p for _, p in scored[:self.config.elite_count]]
        rate = min(.95, self.config.mutation_rate + self._exploration_level())
        while len(population) < self.config.strategy_population:
            if self.stagnation >= self.config.exploration_stagnation_threshold and len(population) % 2 == 0:
                p = PipelineFactory.random(self.rm, self.config)
            else:
                base = self.rm.choice(population or [self.pipeline])
                p = PipelineFactory.mutate(base, self.rm, self.config, rate, self.frozen_params.keys())
            failed_keys = {str(x.get("candidate_key")) for x in self.mutation_failure_history[-80:]}
            failed_sigs = self._recent_failed_signatures(80)
            if p.key() in failed_keys and self.rm.random() < 0.90:
                continue
            if self._mutation_signature(self._pipeline_changes(self.pipeline, p).get("changed", {})) in failed_sigs and self.rm.random() < 0.70:
                continue
            if p.key() not in {x.key() for x in population}:
                population.append(p)
        return population


    # ---------- champion comparison ----------
    def _precheck_diagnostics(self, candidate_fit: float, candidate_metrics: Dict[str, Any]) -> Dict[str, Any]:
        base = self.best_metrics or {}
        cand = candidate_metrics or {}
        def q(container, key):
            try: return float(container.get(key, 0.0) or 0.0)
            except Exception: return 0.0
        btr, ctr = base.get("train", {}), cand.get("train", {})
        bgen, cgen = base.get("generalization", {}), cand.get("generalization", {})
        deltas = {
            "fitness": float(candidate_fit - float(self.best_pipeline_fitness)),
            "train_quality": q(ctr,"quality") - q(btr,"quality"),
            "generalization_quality": q(cgen,"quality") - q(bgen,"quality"),
            "train_p50_latency": q(ctr,"p50_latency") - q(btr,"p50_latency"),
            "train_p95_latency": q(ctr,"p95_latency") - q(btr,"p95_latency"),
            "train_timeout_rate": q(ctr,"timeout_rate") - q(btr,"timeout_rate"),
            "train_fallback_rate": q(ctr,"fallback_rate") - q(btr,"fallback_rate"),
        }
        failed=[]
        if deltas["fitness"] < self.config.min_improvement: failed.append("fitness_margin")
        if deltas["train_quality"] < -self.config.acceptance_quality_tolerance: failed.append("train_quality_regression")
        if deltas["generalization_quality"] < -self.config.acceptance_generalization_tolerance: failed.append("generalization_regression")
        if deltas["train_timeout_rate"] > self.config.reliability_regression_tolerance: failed.append("train_timeout_regression")
        if deltas["train_fallback_rate"] > self.config.reliability_regression_tolerance: failed.append("train_fallback_regression")
        dominant = failed[0] if failed else ("promising" if deltas["fitness"] >= self.config.min_improvement else "insufficient_fitness_gain")
        return {"deltas": deltas, "failed_gates": failed, "dominant_reason": dominant, "candidate_fitness": float(candidate_fit), "champion_fitness": float(self.best_pipeline_fitness)}

    def _mutation_signature(self, changed: Dict[str, Any]) -> str:
        return "|".join(sorted(str(k) for k in changed))

    def _recent_failed_signatures(self, limit: int = 80) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for item in self.mutation_failure_history[-limit:]:
            sig = str(item.get("mutation_signature", ""))
            if sig:
                out[sig] = out.get(sig, 0) + 1
        return out

    def _statistical_precheck(self, cand_scores: np.ndarray, base_scores: np.ndarray) -> Dict[str, Any]:
        n = min(len(cand_scores), len(base_scores))
        if n <= 0:
            return {"n": 0, "mean_delta": 0.0, "std_delta": 0.0, "se": 1.0, "z": 0.0, "borderline": False}
        d = cand_scores[:n] - base_scores[:n]
        mean = float(np.mean(d))
        std = float(np.std(d, ddof=1)) if n > 1 else 0.0
        se = std / max(1.0, float(n) ** 0.5)
        z = mean / max(se, 1e-6)
        return {"n": int(n), "mean_delta": mean, "std_delta": std, "se": se, "z": z,
                "borderline": bool(mean < 0 and z > -float(self.config.statistical_precheck_z))}

    def _causal_probe(self, champion: PipelineSpec, candidate: PipelineSpec, changed: Dict[str, Any]) -> Dict[str, Any]:
        """Cheap one-change-at-a-time probe. It is diagnostic, never an acceptance gate."""
        if not self.config.causal_probe_enabled or not changed:
            return {"enabled": False, "probes": []}
        fields = list(changed.keys())[:max(1, int(self.config.causal_probe_max_fields))]
        probes = []
        for field in fields:
            data = asdict(champion)
            data[field] = asdict(candidate).get(field)
            probe_spec = PipelineSpec(**data).normalize(self.config)
            try:
                fit, metrics = self.evaluate_pipeline(
                    probe_spec, screen=True, screen_tasks=max(2, int(self.config.causal_probe_screen_tasks)),
                    context=f"C{self.cycle+1} CAUSAL {field} "
                )
                probes.append({"field": field, "screen_fitness": float(fit),
                               "delta_vs_champion": float(fit - self.best_pipeline_fitness),
                               "pipeline_key": probe_spec.key()})
            except Exception as exc:
                probes.append({"field": field, "error": str(exc)})
        return {"enabled": True, "probes": probes}

    def _compare_to_champion(self, candidate_fit: float, candidate_metrics: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Controlled precheck: use paired statistics and permit borderline candidates to reach full evaluation."""
        champion_fit = float(self.best_pipeline_fitness)
        delta = candidate_fit - champion_fit
        if champion_fit <= 0 or not self.best_metrics:
            return candidate_fit > 0, {"reason": "no_previous_champion", "delta": delta, "significant": candidate_fit > 0}
        cand_train = candidate_metrics.get("train", {})
        base_train = self.best_metrics.get("train", {})
        cand_scores = np.asarray(cand_train.get("task_scores", []), dtype=float)
        base_scores = np.asarray(base_train.get("task_scores", []), dtype=float)
        stats = self._statistical_precheck(cand_scores, base_scores)
        diag = self._precheck_diagnostics(candidate_fit, candidate_metrics)
        significant = delta >= self.config.min_improvement and (stats["mean_delta"] >= -self.config.acceptance_quality_tolerance or stats["borderline"])
        if delta < 0 and stats["borderline"]:
            # Borderline negative results deserve verification instead of immediate rejection.
            significant = True
        return significant, {
            "reason": "candidate_promising" if significant else "candidate_rejected_precheck",
            "delta": delta, "paired_mean": stats["mean_delta"], "significant": significant,
            "paired_statistics": stats, **diag
        }


    @staticmethod
    def _relative_change(before: float, after: float) -> float:
        before = float(before or 0.0)
        after = float(after or 0.0)
        if abs(before) <= 1e-12:
            return 0.0 if abs(after) <= 1e-12 else float("inf")
        return (after - before) / abs(before)

    def _strict_acceptance(self, before: Dict[str, Any], candidate_after: Dict[str, Any],
                           candidate_fit: float, champion_fit: float) -> Tuple[bool, Dict[str, Any]]:
        """Final conservative gate. No candidate may be accepted merely because its fitness rose.

        Requirements:
          * candidate fitness must clear the configured evolutionary margin;
          * train/generalization/holdout quality may not materially regress;
          * reliability may not materially regress;
          * a speed-only win must be visible on both train and holdout and cannot trade away quality;
          * meaningful quality improvement must survive holdout.
        """
        tb, cb = before["train"], candidate_after["train"]
        gb, ga = before["generalization"], candidate_after["generalization"]
        hb, ha = before["holdout"], candidate_after["holdout"]

        fit_delta = float(candidate_fit - champion_fit)
        train_q_delta = float(cb.get("quality", 0.0) - tb.get("quality", 0.0))
        gen_q_delta = float(ga.get("quality", 0.0) - gb.get("quality", 0.0))
        hold_q_delta = float(ha.get("quality", 0.0) - hb.get("quality", 0.0))

        train_p50_rel = self._relative_change(tb.get("p50_latency", 0.0), cb.get("p50_latency", 0.0))
        train_p95_rel = self._relative_change(tb.get("p95_latency", 0.0), cb.get("p95_latency", 0.0))
        hold_p50_rel = self._relative_change(hb.get("p50_latency", 0.0), ha.get("p50_latency", 0.0))
        hold_p95_rel = self._relative_change(hb.get("p95_latency", 0.0), ha.get("p95_latency", 0.0))

        reliability_regressions = {
            "train_timeout": float(cb.get("timeout_rate", 0.0)) - float(tb.get("timeout_rate", 0.0)),
            "train_fallback": float(cb.get("fallback_rate", 0.0)) - float(tb.get("fallback_rate", 0.0)),
            "generalization_timeout": float(ga.get("timeout_rate", 0.0)) - float(gb.get("timeout_rate", 0.0)),
            "holdout_timeout": float(ha.get("timeout_rate", 0.0)) - float(hb.get("timeout_rate", 0.0)),
            "holdout_fallback": float(ha.get("fallback_rate", 0.0)) - float(hb.get("fallback_rate", 0.0)),
        }
        reliability_regression = self.config.acceptance_require_reliability_non_regression and any(
            v > self.config.reliability_regression_tolerance for v in reliability_regressions.values()
        )

        quality_floor_ok = (
            train_q_delta >= -self.config.acceptance_quality_tolerance and
            gen_q_delta >= -self.config.acceptance_generalization_tolerance and
            (not self.config.acceptance_require_holdout_non_regression or hold_q_delta >= -self.config.acceptance_holdout_tolerance)
        )

        meaningful_quality_gain = (
            train_q_delta >= self.config.acceptance_min_quality_gain and
            gen_q_delta >= -self.config.acceptance_generalization_tolerance and
            (not self.config.acceptance_require_holdout_non_regression or hold_q_delta >= self.config.acceptance_holdout_tolerance)
        )

        speed_gain_train = train_p50_rel <= -self.config.acceptance_min_speed_gain
        speed_gain_hold = hold_p50_rel <= -self.config.acceptance_min_speed_gain
        speed_gain_both = speed_gain_train and speed_gain_hold
        speed_regression = (
            train_p50_rel > self.config.acceptance_speed_regression_tolerance or
            hold_p50_rel > self.config.acceptance_speed_regression_tolerance or
            train_p95_rel > self.config.acceptance_p95_regression_tolerance or
            hold_p95_rel > self.config.acceptance_p95_regression_tolerance
        )
        speed_win = speed_gain_both and not speed_regression

        fit_gate = champion_fit <= 0 or fit_delta >= self.config.min_improvement
        accepted = bool(fit_gate and quality_floor_ok and not reliability_regression and not speed_regression and (meaningful_quality_gain or speed_win))

        failed = []
        if not fit_gate: failed.append("fitness_gain_too_small")
        if train_q_delta < -self.config.acceptance_quality_tolerance: failed.append("train_quality_regression")
        if gen_q_delta < -self.config.acceptance_generalization_tolerance: failed.append("generalization_regression")
        if self.config.acceptance_require_holdout_non_regression and hold_q_delta < -self.config.acceptance_holdout_tolerance:
            failed.append("holdout_regression")
        if reliability_regression: failed.append("reliability_regression")
        if speed_regression: failed.append("latency_regression")
        if not meaningful_quality_gain and not speed_win: failed.append("no_verified_improvement")

        if accepted:
            reason = "verified_quality_improvement" if meaningful_quality_gain else "verified_speed_improvement"
        else:
            reason = failed[0] if failed else "verified_improvement_missing"

        return accepted, {
            "accepted": accepted, "reason": reason, "fit_delta": fit_delta,
            "train_quality_delta": train_q_delta, "generalization_quality_delta": gen_q_delta,
            "holdout_quality_delta": hold_q_delta,
            "train_p50_relative": train_p50_rel, "holdout_p50_relative": hold_p50_rel,
            "train_p95_relative": train_p95_rel, "holdout_p95_relative": hold_p95_rel,
            "meaningful_quality_gain": meaningful_quality_gain,
            "speed_gain_both": speed_gain_both, "speed_win": speed_win,
            "quality_floor_ok": quality_floor_ok, "reliability_regression": reliability_regression,
            "reliability_regressions": reliability_regressions, "speed_regression": speed_regression,
            "failed_gates": failed,
        }


    # ---------- evolution ----------
    def evolve_pipeline(self, budget: Optional[int] = None) -> Tuple[PipelineSpec, float, Dict[str, Any], Dict[str, Any]]:
        budget = budget or self.config.max_pipeline_evaluations_per_cycle
        # Strict causal search: the only valid incumbent is the current champion.
        # Historical/experience candidates may seed *which field* to explore, but
        # they must never become the returned candidate themselves.
        champion = PipelineSpec(**asdict(self.pipeline)).normalize(self.config)
        population = [champion]
        global_best = PipelineSpec(**asdict(champion)).normalize(self.config)
        global_best_fit = float(self.best_pipeline_fitness) if self.best_pipeline_fitness > 0 else -1.0
        global_best_metrics: Dict[str, Any] = dict(self.best_metrics or {})
        evaluated = 0
        generation_stats = []
        workers = max(1, min(8, int(self.config.evolution_workers)))
        started = time.perf_counter()
        self._vlog(f"EVOLUTION START | workers={workers} | population={len(population)} | generations={self.config.strategy_generations} | budget={budget} | screen_tasks={self.config.screen_tasks_per_candidate} | finalists={self.config.finalists_per_generation}")

        for gen in range(self.config.strategy_generations):
            if evaluated >= budget: break
            self._vlog(f"GENERATION {gen+1}/{self.config.strategy_generations} SCREEN START | candidates={len(population)}")
            screen_results: List[Tuple[float, PipelineSpec, Dict[str, Any]]] = []

            def screen_one(idx: int, spec: PipelineSpec):
                t0 = time.perf_counter()
                self._vlog(f"candidate={idx} SCREEN START | key={spec.key()[:10]}")
                try:
                    # Screen cache is allowed because the result is only a ranking hint.
                    fit, metrics = self.evaluate_pipeline(spec, screen=True, screen_tasks=self.config.screen_tasks_per_candidate, context=f"C{self.cycle+1} G{gen+1} CAND{idx} ")
                    return idx, fit, spec, metrics, time.perf_counter() - t0, None
                except Exception as exc:
                    return idx, 0.0, spec, {}, time.perf_counter() - t0, exc

            batch = list(enumerate(population, 1))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="MANA-EvoScreen") as ex:
                fs = [ex.submit(screen_one, i, p) for i, p in batch]
                for f in as_completed(fs):
                    i, fit, spec, metrics, elapsed, err = f.result()
                    evaluated += 1
                    if err:
                        self._vlog(f"candidate={i} SCREEN FAIL | {err}")
                    else:
                        self._vlog(f"candidate={i} SCREEN END | fitness={fit:.3f} | time={elapsed:.1f}s")
                        screen_results.append((fit, spec, metrics))
                    if evaluated >= budget: break

            screen_results.sort(key=lambda x: x[0], reverse=True)
            finalists = [x[1] for x in screen_results[:self.config.finalists_per_generation]]
            self._vlog(f"GENERATION {gen+1}/{self.config.strategy_generations} SCREEN END | screened={len(screen_results)} | finalists={len(finalists)} | screen_best={(screen_results[0][0] if screen_results else 0):.3f}")
            if not finalists: break

            full_results: List[Tuple[float, PipelineSpec, Dict[str, Any]]] = []
            self._vlog(f"GENERATION {gen+1}/{self.config.strategy_generations} FULL START | finalists={len(finalists)}")

            def full_one(idx: int, spec: PipelineSpec):
                t0 = time.perf_counter()
                self._vlog(f"candidate={idx} FULL START | key={spec.key()[:10]}")
                old = getattr(self.config, "_active_llm_timeout", None)
                self.config._active_llm_timeout = self.config.evolution_llm_timeout
                try:
                    fit, metrics = self.evaluate_pipeline(spec, screen=False, context=f"C{self.cycle+1} G{gen+1} CAND{idx} ")
                    return idx, fit, spec, metrics, time.perf_counter() - t0, None
                except Exception as exc:
                    return idx, 0.0, spec, {}, time.perf_counter() - t0, exc
                finally:
                    if old is None:
                        try: delattr(self.config, "_active_llm_timeout")
                        except AttributeError: pass
                    else:
                        self.config._active_llm_timeout = old

            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="MANA-EvoFull") as ex:
                fs = [ex.submit(full_one, i, p) for i, p in enumerate(finalists, 1)]
                for f in as_completed(fs):
                    i, fit, spec, metrics, elapsed, err = f.result()
                    evaluated += 1
                    if err:
                        self._vlog(f"candidate={i} FULL FAIL | {err} | time={elapsed:.1f}s")
                    else:
                        timeouts = metrics.get("timeouts", 0)
                        fallbacks = metrics.get("fallbacks", 0)
                        self._vlog(f"candidate={i} FULL END | fitness={fit:.3f} | time={elapsed:.1f}s | timeouts={timeouts} | fallbacks={fallbacks}")
                        full_results.append((fit, spec, metrics))

            if full_results:
                # Only a candidate that is exactly one local mutation away from
                # the champion may become the returned global best. This prevents
                # experience-seeded/elite pipelines from bypassing causal locality.
                causal_candidates = []
                for item in full_results:
                    _fit, _spec, _metrics = item
                    _changed = self._pipeline_changes(champion, _spec).get("changed", {})
                    if len(_changed) <= 1:
                        causal_candidates.append(item)
                gen_best = max(causal_candidates or full_results, key=lambda x: x[0])
                if gen_best[0] > global_best_fit and len(self._pipeline_changes(champion, gen_best[1]).get("changed", {})) <= 1:
                    global_best_fit, global_best, global_best_metrics = gen_best
                avg_fit = float(np.mean([x[0] for x in full_results]))
            else:
                gen_best = (global_best_fit, global_best, global_best_metrics)
                avg_fit = 0.0

            generation_stats.append({
                "generation": gen + 1,
                "screened": len(screen_results),
                "finalists": len(finalists),
                "full_evaluated": len(full_results),
                "screen_best": screen_results[0][0] if screen_results else 0.0,
                "best": gen_best[0],
                "avg": avg_fit,
                "elapsed": time.perf_counter() - started,
            })
            self._vlog(f"GENERATION {gen+1}/{self.config.strategy_generations} END | best={gen_best[0]:.3f} | avg={avg_fit:.3f} | elapsed={self._fmt_duration(time.perf_counter()-started)}")

            # Next generation: elites + offspring.
            ranked = sorted(full_results, key=lambda x: x[0], reverse=True)
            new = [PipelineSpec(**asdict(x[1])).normalize(self.config) for x in ranked[:self.config.elite_count]]
            while len(new) < self.config.strategy_population and evaluated < budget:
                # Strict causal mode: every offspring starts from the CURRENT CHAMPION.
                # Ordinary crossover is disabled because it can import unrelated field changes
                # from parents and destroy one-variable-at-a-time causality.
                child = PipelineSpec(**asdict(self.pipeline)).normalize(self.config)
                child = PipelineFactory.mutate(
                    child, self.rm, self.config,
                    min(.95, self.config.mutation_rate + self._exploration_level()),
                    self.frozen_params.keys(), max_changes=1
                )
                if child.key() not in {x.key() for x in new}:
                    new.append(child)
            population = new or population

        causal_delta = self._pipeline_changes(champion, global_best)
        self._vlog(
            f"EVOLUTION END | evaluated={evaluated} | best={global_best_fit:.3f} | "
            f"causal_changes={causal_delta.get('changed_count', 0)} | "
            f"total={self._fmt_duration(time.perf_counter()-started)}"
        )
        evo_meta = {
            "evaluations": evaluated,
            "generations": generation_stats,
            "workers": workers,
            "causal_changes": causal_delta,
            "strict_causal": True,
        }
        return global_best, max(0.0, global_best_fit), evo_meta, global_best_metrics


    # ---------- before/after and holdout ----------
    @staticmethod
    def _effect(before: Dict[str, Any], after: Dict[str, Any], key: str, inverse: bool = False) -> Dict[str, float]:
        b = float(before.get(key, 0.0) or 0.0)
        a = float(after.get(key, 0.0) or 0.0)
        delta = a - b
        pct = (delta / abs(b) * 100.0) if abs(b) > 1e-12 else (100.0 if delta > 0 else 0.0)
        return {"before": b, "after": a, "delta": delta, "percent": pct, "better": (delta < 0 if inverse else delta > 0)}

    def _compare_benchmarks(self, before: Dict[str, Any], after: Dict[str, Any], hold_before: Dict[str, Any], hold_after: Dict[str, Any]) -> Dict[str, Any]:
        train_b, train_a = before["train"], after["train"]
        gen_b, gen_a = before["generalization"], after["generalization"]
        hold_b, hold_a = hold_before, hold_after
        effect = {
            "train_quality": self._effect(train_b, train_a, "quality"),
            "train_p50_latency": self._effect(train_b, train_a, "p50_latency", inverse=True),
            "train_p95_latency": self._effect(train_b, train_a, "p95_latency", inverse=True),
            "train_timeout_rate": self._effect(train_b, train_a, "timeout_rate", inverse=True),
            "train_fallback_rate": self._effect(train_b, train_a, "fallback_rate", inverse=True),
            "generalization_quality": self._effect(gen_b, gen_a, "quality"),
            "holdout_quality": self._effect(hold_b, hold_a, "quality"),
            "holdout_p50_latency": self._effect(hold_b, hold_a, "p50_latency", inverse=True),
            "holdout_p95_latency": self._effect(hold_b, hold_a, "p95_latency", inverse=True),
            "holdout_timeout_rate": self._effect(hold_b, hold_a, "timeout_rate", inverse=True),
            "holdout_fallback_rate": self._effect(hold_b, hold_a, "fallback_rate", inverse=True),
        }
        quality_improved = (
            effect["train_quality"]["delta"] >= 0.01 and
            effect["holdout_quality"]["delta"] >= 0.01
        )
        # _effect percent for inverse latency is negative for an improvement, e.g. -20%
        train_p50_better = effect["train_p50_latency"]["percent"] <= -self.config.speed_improvement_threshold * 100.0
        hold_p50_better = effect["holdout_p50_latency"]["percent"] <= -self.config.speed_improvement_threshold * 100.0
        train_p95_better = effect["train_p95_latency"]["percent"] <= -self.config.p95_speed_improvement_threshold * 100.0
        hold_p95_better = effect["holdout_p95_latency"]["percent"] <= -self.config.p95_speed_improvement_threshold * 100.0
        # Speed may only count as a real improvement when reliability did not regress materially.
        reliability_regression = (
            (effect["train_timeout_rate"]["after"] - effect["train_timeout_rate"]["before"] > self.config.reliability_regression_tolerance) or
            (effect["train_fallback_rate"]["after"] - effect["train_fallback_rate"]["before"] > self.config.reliability_regression_tolerance) or
            (effect["holdout_timeout_rate"]["after"] - effect["holdout_timeout_rate"]["before"] > self.config.reliability_regression_tolerance)
        )
        speed_improved = (train_p50_better or hold_p50_better or train_p95_better or hold_p95_better) and not reliability_regression
        reliability_improved = (
            effect["train_timeout_rate"]["after"] <= effect["train_timeout_rate"]["before"] and
            effect["train_fallback_rate"]["after"] <= effect["train_fallback_rate"]["before"] and
            effect["holdout_timeout_rate"]["after"] <= effect["holdout_timeout_rate"]["before"] and
            effect["holdout_fallback_rate"]["after"] <= effect["holdout_fallback_rate"]["before"]
        )
        return {
            "effect": effect,
            "quality_improved": quality_improved,
            "speed_improved": speed_improved,
            "reliability_improved": reliability_improved,
            "reliability_regression": reliability_regression,
            "speed_thresholds": {
                "p50": self.config.speed_improvement_threshold,
                "p95": self.config.p95_speed_improvement_threshold,
            },
        }


    @staticmethod
    def _pipeline_changes(before: PipelineSpec, after: PipelineSpec) -> Dict[str, Any]:
        b = asdict(before); a = asdict(after)
        changed = {}
        for k in sorted(set(b) | set(a)):
            if b.get(k) != a.get(k):
                changed[k] = {"before": b.get(k), "after": a.get(k)}
        return {"changed": changed, "changed_count": len(changed)}

    def _cycle_report(self, cycle_number: int, before: Dict[str, Any], after: Dict[str, Any],
                      hold_before: Dict[str, Any], hold_after: Dict[str, Any],
                      candidate_fit: float, accepted: bool, reason: str, evo_stats: Dict[str, Any],
                      cycle_elapsed: float) -> Dict[str, Any]:
        comparison = self._compare_benchmarks(before, after, hold_before, hold_after)
        verdict = "NO_CHANGE"
        if not accepted:
            if reason in {"train_quality_regression", "generalization_regression", "holdout_regression"}:
                verdict = "REGRESSION_REJECTED"
            elif reason == "latency_regression":
                verdict = "SLOWER_REJECTED"
            elif reason == "reliability_regression":
                verdict = "RELIABILITY_REGRESSION_REJECTED"
            elif reason == "no_verified_improvement" or reason == "candidate_rejected_precheck":
                verdict = "NO_VERIFIED_IMPROVEMENT"
        elif comparison["quality_improved"] and comparison["speed_improved"]:
            verdict = "QUALITY_AND_SPEED_IMPROVED"
        elif comparison["quality_improved"]:
            verdict = "QUALITY_IMPROVED"
        elif comparison["speed_improved"]:
            verdict = "SPEED_IMPROVED"
        elif comparison["reliability_improved"]:
            verdict = "RELIABILITY_IMPROVED"

        report = {
            "version": self.VERSION,
            "cycle": cycle_number,
            "timestamp": time.time(),
            "cycle_elapsed": cycle_elapsed,
            "accepted": accepted,
            "reason": reason,
            "candidate_fitness": candidate_fit,
            "champion_fitness_after": self.best_pipeline_fitness,
            "baseline": before,
            "after": after,
            "holdout_before": hold_before,
            "holdout_after": hold_after,
            "comparison": comparison,
            "verdict": verdict,
            "evolution": evo_stats,
            "pipeline_after": asdict(self.pipeline),
            "acceptance_policy": {
                "quality_tolerance": self.config.acceptance_quality_tolerance,
                "generalization_tolerance": self.config.acceptance_generalization_tolerance,
                "holdout_tolerance": self.config.acceptance_holdout_tolerance,
                "min_speed_gain": self.config.acceptance_min_speed_gain,
                "speed_regression_tolerance": self.config.acceptance_speed_regression_tolerance,
                "p95_regression_tolerance": self.config.acceptance_p95_regression_tolerance,
            },
        }
        report["adaptive_compute"] = {
            "budget": int(getattr(self.pipeline, "compute_budget", 2)),
            "confidence_threshold": float(getattr(self.pipeline, "confidence_threshold", 0.62)),
            "verification_mode": getattr(self.pipeline, "verification_mode", "adaptive"),
            "architecture": getattr(self.pipeline, "architecture", "adaptive"),
            "graph_nodes": list(getattr(self.pipeline, "graph_nodes", ("LLM", "EVALUATE"))),
            "stop_policy": getattr(self.pipeline, "stop_policy", "adaptive"),
            "cost_budget": float(getattr(self.pipeline, "cost_budget", 12.0)),
            "confidence_calibration": float(getattr(self.pipeline, "confidence_calibration", 1.0)),
        }
        with self._report_lock:
            self.evolution_reports.append(report)
            self.evolution_reports = self.evolution_reports[-200:]
        self._save_reports()
        return report

    def self_improve(self) -> Dict[str, Any]:
        cycle_no = self.cycle + 1
        cycle_started = time.perf_counter()
        if self._evolution_started_at is None:
            self._evolution_started_at = time.perf_counter()
        self._evolution_cycle_started_at = cycle_started
        self._vlog(f"CYCLE {cycle_no} START")

        # 1. Genuine baseline: uncached control run. HOLDOUT is locked and never enters evolution.
        self._vlog("BASELINE START | train + generalization + LOCKED HOLDOUT")
        baseline_full = self.run_control_benchmark(self.pipeline)
        baseline_routing = self.routing_holdout(self.pipeline) if self.config.routing_gate_enabled else None
        champion_before_fit = self.best_pipeline_fitness
        champion_before_pipeline = PipelineSpec(**asdict(self.pipeline)).normalize(self.config)

        if self.best_pipeline_fitness <= 0 or "train" not in self.best_metrics:
            self.best_pipeline_fitness = 100.0 * (
                0.65 * baseline_full["train"]["quality"] +
                0.25 * baseline_full["generalization"]["quality"] +
                0.10 * max(0.0, 1.0 - baseline_full["train"]["p50_latency"] / 20.0)
            )
            self.best_metrics = {
                "train": baseline_full["train"],
                "generalization": baseline_full["generalization"],
                "holdout": baseline_full["holdout"],
                "fitness": self.best_pipeline_fitness,
                "migrated_from_legacy": True,
            }
            self._save_state()

        self.rollback_snapshot = {
            "pipeline": asdict(self.pipeline),
            "best_fitness": self.best_pipeline_fitness,
            "best_metrics": self.best_metrics,
        }

        # 2. Evolution still NEVER sees holdout tasks.
        candidate, fit, evo_stats, candidate_metrics = self.evolve_pipeline()
        precheck_ok, precheck = self._compare_to_champion(fit, candidate_metrics)
        candidate_pipeline = PipelineSpec(**asdict(candidate)).normalize(self.config)
        changed = self._pipeline_changes(champion_before_pipeline, candidate_pipeline)

        # 3. Only promising candidates get a real candidate benchmark. This benchmark includes the locked holdout.
        if precheck_ok:
            self._vlog("CANDIDATE AFTER START | actual candidate + LOCKED HOLDOUT")
            candidate_after = self.run_control_benchmark(candidate_pipeline)
            candidate_after_for_compare = candidate_after
            accepted, final_gate = self._strict_acceptance(
                baseline_full, candidate_after, fit, champion_before_fit
            )
            candidate_routing = self.routing_holdout(candidate_pipeline) if self.config.routing_gate_enabled else None
            routing_gate = {"enabled": bool(self.config.routing_gate_enabled), "baseline": baseline_routing, "candidate": candidate_routing, "failed_gates": []}
            if self.config.routing_gate_enabled and candidate_routing is not None:
                if candidate_routing["route_accuracy"] < self.config.routing_holdout_min_accuracy:
                    routing_gate["failed_gates"].append("routing_holdout_accuracy")
                if candidate_routing["execution_accuracy"] < self.config.routing_holdout_min_execution_accuracy:
                    routing_gate["failed_gates"].append("routing_holdout_execution_accuracy")
                if candidate_routing.get("quality", 0.0) < self.config.routing_holdout_min_quality:
                    routing_gate["failed_gates"].append("routing_holdout_quality")
                if candidate_routing.get("web_success_rate", 1.0) < self.config.routing_holdout_min_web_success:
                    routing_gate["failed_gates"].append("routing_holdout_web_success")
                if routing_gate["failed_gates"]:
                    accepted = False
                    final_gate.setdefault("failed_gates", []).extend(routing_gate["failed_gates"])
                    final_gate["reason"] = routing_gate["failed_gates"][0]
                elif baseline_routing is not None:
                    if candidate_routing["route_accuracy"] + 1e-9 < baseline_routing["route_accuracy"] - 0.001:
                        accepted = False; final_gate.setdefault("failed_gates", []).append("routing_accuracy_regression"); final_gate["reason"] = "routing_accuracy_regression"
                    elif candidate_routing["execution_accuracy"] + 1e-9 < baseline_routing["execution_accuracy"] - 0.001:
                        accepted = False; final_gate.setdefault("failed_gates", []).append("routing_execution_regression"); final_gate["reason"] = "routing_execution_regression"
            final_gate["routing_gate"] = routing_gate
            # v4.0 adaptive-compute gate: protect the new decision policy from regression.
            adaptive_gate = {"enabled": True, "baseline": None, "candidate": None, "failed_gates": []}
            try:
                adaptive_gate["baseline"] = self.adaptive_benchmark(holdout=True, spec=champion_before_pipeline)
                adaptive_gate["candidate"] = self.adaptive_benchmark(holdout=True, spec=candidate_pipeline)
                ac = adaptive_gate["candidate"]
                if ac.get("route_accuracy", 0.0) < 0.78: adaptive_gate["failed_gates"].append("adaptive_holdout_route_accuracy")
                if ac.get("execution_accuracy", 0.0) < 0.78: adaptive_gate["failed_gates"].append("adaptive_holdout_execution_accuracy")
                if ac.get("quality", 0.0) < 0.45: adaptive_gate["failed_gates"].append("adaptive_holdout_quality")
                if adaptive_gate["failed_gates"]:
                    accepted = False
                    final_gate.setdefault("failed_gates", []).extend(adaptive_gate["failed_gates"])
                    final_gate["reason"] = adaptive_gate["failed_gates"][0]
            except Exception as exc:
                adaptive_gate["error"] = str(exc)
                # Do not reject an otherwise valid candidate solely because optional adaptive telemetry failed.
            final_gate["adaptive_gate"] = adaptive_gate
        else:
            candidate_after = None
            candidate_after_for_compare = baseline_full
            candidate_routing = None
            routing_gate = {"enabled": bool(self.config.routing_gate_enabled), "baseline": baseline_routing, "candidate": None, "failed_gates": ["candidate_rejected_precheck"] if self.config.routing_gate_enabled else []}
            accepted = False
            precheck_failures = list(precheck.get("failed_gates", [])) or ["candidate_rejected_precheck"]
            final_gate = {
                "accepted": False, "reason": "candidate_rejected_precheck",
                "fit_delta": fit - champion_before_fit,
                "failed_gates": precheck_failures,
                "dominant_reason": precheck.get("dominant_reason", "candidate_rejected_precheck"),
            }
            causal = self._causal_probe(champion_before_pipeline, candidate_pipeline, changed.get("changed", {}))
            self.mutation_failure_history.append({
                "cycle": cycle_no, "candidate_key": candidate_pipeline.key(),
                "reason": precheck.get("dominant_reason", "candidate_rejected_precheck"),
                "failed_gates": precheck_failures, "fit_delta": fit - champion_before_fit,
                "pipeline_changes": changed, "mutation_signature": self._mutation_signature(changed.get("changed", {})),
                "causal_probe": causal, "timestamp": time.time()
            })
            self.mutation_failure_history = self.mutation_failure_history[-300:]
            self._vlog(f"CANDIDATE SKIP | precheck failed: delta={precheck.get('delta', 0.0):+.3f}")

        # 4. Apply only a VERIFIED change. Rejected candidates never touch the champion state.
        if accepted:
            self.pipeline = candidate_pipeline
            self.best_pipeline_fitness = fit
            self.best_metrics = candidate_metrics
            self.stagnation = 0
            after_full = candidate_after_for_compare
        else:
            self.pipeline = champion_before_pipeline
            self.best_pipeline_fitness = champion_before_fit
            self.best_metrics = self.rollback_snapshot["best_metrics"] if self.rollback_snapshot else self.best_metrics
            self.stagnation += 1
            after_full = baseline_full

        # 5. Build a report from actual before/after data. There is no post-hoc 'accept because holdout looked okay'.
        report = self._cycle_report(
            cycle_no,
            baseline_full,
            after_full,
            baseline_full["holdout"],
            after_full["holdout"],
            fit,
            accepted,
            final_gate.get("reason", "unknown"),
            evo_stats,
            time.perf_counter() - cycle_started,
        )
        report["routing_gate"] = routing_gate
        report["adaptive_gate"] = final_gate.get("adaptive_gate", {})
        report["routing_stats"] = {"entries": len(self.routing_stats), "observations": int(sum(v.get("n", 0) for v in self.routing_stats.values())), "adaptive_enabled": bool(self.config.routing_adaptive_enabled)}
        report["precheck"] = precheck
        report["final_acceptance_gate"] = final_gate
        report["candidate_pipeline"] = asdict(candidate_pipeline)
        report["candidate_after"] = candidate_after
        report["pipeline_changes"] = changed
        if not accepted and final_gate.get("reason") != "candidate_rejected_precheck":
            self.mutation_failure_history.append({
                "cycle": cycle_no, "candidate_key": candidate_pipeline.key(),
                "reason": final_gate.get("reason", "unknown"),
                "failed_gates": final_gate.get("failed_gates", []),
                "fit_delta": fit - champion_before_fit, "pipeline_changes": changed,
                "mutation_signature": self._mutation_signature(changed.get("changed", {})),
                "timestamp": time.time()
            })
            self.mutation_failure_history = self.mutation_failure_history[-300:]
        report["decision"] = {
            "accepted": accepted,
            "reason": final_gate.get("reason", "unknown"),
            "failed_gates": final_gate.get("failed_gates", []),
            "dominant_reason": final_gate.get("dominant_reason", ""),
        }
        report["mutation_learning"] = {"history_entries": len(self.mutation_failure_history), "recent": self.mutation_failure_history[-10:]}
        self._save_reports()

        self.cycle = cycle_no
        elapsed = time.perf_counter() - cycle_started
        self._evolution_last_cycle_elapsed = elapsed
        if self._evolution_started_at is not None:
            self._evolution_total_elapsed = time.perf_counter() - self._evolution_started_at

        event = {
            "type": "self_improve",
            "cycle": cycle_no,
            "improved": bool(accepted),
            "fitness": fit,
            "best_fitness": self.best_pipeline_fitness,
            "reason": report["reason"],
            "elapsed": elapsed,
            "report": report,
            "timestamp": time.time(),
            "pipeline": asdict(self.pipeline),
        }
        self.history.append(event)
        self._save_cache(); self._save_state()
        self._vlog(
            f"CYCLE {cycle_no} END | accepted={accepted} | reason={report['reason']} | "
            f"cycle_time={self._fmt_duration(elapsed)} | total_time={self._fmt_duration(self._evolution_total_elapsed)}"
        )
        return event


    # ---------- public execution ----------
    def evolution_status(self) -> Dict[str, Any]:
        now = time.perf_counter()
        total = self._evolution_total_elapsed
        cur = self._evolution_last_cycle_elapsed
        if self._evolution_running and self._evolution_started_at is not None:
            total = now - self._evolution_started_at
        if self._evolution_running and self._evolution_cycle_started_at is not None:
            cur = now - self._evolution_cycle_started_at
        return {
            "running": bool(self._evolution_running), "cycle": self.cycle, "target_cycle": self._evolution_target,
            "best_fitness": self.best_pipeline_fitness, "current_cycle_elapsed": cur, "total_elapsed": total,
            "last_verdict": self.evolution_reports[-1].get("verdict") if self.evolution_reports else None,
            "adaptive": {
                "graph_nodes": list(getattr(self.pipeline, "graph_nodes", ("LLM", "EVALUATE"))),
                "stop_policy": getattr(self.pipeline, "stop_policy", "adaptive"),
                "compute_budget": int(getattr(self.pipeline, "compute_budget", 3)),
                "cost_budget": float(getattr(self.pipeline, "cost_budget", 12.0)),
            },
        }

    def run_cycles(self, n: int) -> None:
        n = max(1, int(n)); target = self.cycle + n
        _out(f"\n🚀 Самоулучшение: {n} циклов (до cycle={target})")
        while self.cycle < target and not self._evolution_stop.is_set():
            event = self.self_improve()
            comp = event["report"]["comparison"]
            _out(
                f"[{time.strftime('%H:%M:%S')}] cycle={event['cycle']} | "
                f"fitness={event['fitness']:.3f} | best={event['best_fitness']:.3f} | "
                f"accepted={'YES' if event['improved'] else 'NO'} | verdict={event['report']['verdict']} | "
                f"cycle_time={self._fmt_duration(event['elapsed'])}"
            )
            e = comp["effect"]
            _out(
                f"  quality: {e['train_quality']['before']:.3f} -> {e['train_quality']['after']:.3f} "
                f"({e['train_quality']['delta']:+.3f}) | "
                f"holdout: {e['holdout_quality']['before']:.3f} -> {e['holdout_quality']['after']:.3f} "
                f"({e['holdout_quality']['delta']:+.3f})"
            )
            _out(
                f"  p50: {e['train_p50_latency']['before']:.2f}s -> {e['train_p50_latency']['after']:.2f}s | "
                f"p95: {e['train_p95_latency']['before']:.2f}s -> {e['train_p95_latency']['after']:.2f}s | "
                f"timeouts: {e['train_timeout_rate']['before']:.1%} -> {e['train_timeout_rate']['after']:.1%}"
            )
            _out(
                f"  decision: {event['report'].get('reason', 'unknown')} | "
                f"failed_gates={event['report'].get('decision', {}).get('failed_gates', [])}"
            )
        _out("✅ Самоулучшение завершено")

    def start_evolution_background(self, cycles: int = 3) -> bool:
        if self._evolution_running:
            return False
        self._evolution_stop.clear()
        count = max(1, int(cycles))
        self._evolution_target = self.cycle + count
        def worker():
            self._evolution_running = True
            self._evolution_started_at = time.perf_counter()
            try:
                _out(f"\n🚀 Фоновое самоулучшение: {count} циклов (до cycle={self._evolution_target})")
                while self.cycle < self._evolution_target and not self._evolution_stop.is_set():
                    event = self.self_improve()
                    r = event["report"]
                    e = r["comparison"]["effect"]
                    _out(
                        f"[{time.strftime('%H:%M:%S')}] cycle={event['cycle']} | "
                        f"verdict={r['verdict']} | accepted={'YES' if r['accepted'] else 'NO'} | "
                        f"quality={e['train_quality']['before']:.3f}->{e['train_quality']['after']:.3f} | "
                        f"holdout={e['holdout_quality']['before']:.3f}->{e['holdout_quality']['after']:.3f} | "
                        f"p50={e['train_p50_latency']['before']:.2f}->{e['train_p50_latency']['after']:.2f}s | "
                        f"cycle_time={self._fmt_duration(event['elapsed'])}"
                    )
            except Exception as exc:
                _out(f"❌ Ошибка фонового самоулучшения: {exc}")
            finally:
                self._evolution_total_elapsed = time.perf_counter() - self._evolution_started_at
                self._evolution_running = False
                self._evolution_target = None
        threading.Thread(target=worker, daemon=True, name="MANA-Evolution").start()
        return True

    def stop_evolution(self) -> None:
        self._evolution_stop.set()

    def latest_evolution_report(self) -> Optional[Dict[str, Any]]:
        return self.evolution_reports[-1] if self.evolution_reports else None
