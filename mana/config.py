"""
mana.config — runtime Config dataclass and the seeded RandomManager.
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

from .optional_deps import torch, HAS_TORCH

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.2"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    seed: int = 42

    # Hardware adaptation (mana.hardware) -- applied once at ManaAgent
    # startup, before evolution_workers/strategy_population/etc. below are
    # read anywhere else, so the rest of this dataclass's defaults are
    # this module's *portable baseline*, and hardware.py narrows them for
    # the actual machine. Set to False to keep exactly the defaults below
    # (or whatever CLI flags override) regardless of the host machine.
    hardware_auto_adapt: bool = True

    # LLM
    enable_llm: bool = True
    llm_backend: str = "ollama"
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "qwen2.5:0.5b"
    llm_timeout: int = 45
    evolution_llm_timeout: int = 20
    llm_max_tokens: int = 700

    # Optional external providers
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")
    gemini_model: str = os.environ.get("MANA_GEMINI_MODEL", "gemini-2.5-flash-lite")
    gemini_url: str = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    openrouter_api_key: str = os.environ.get("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.environ.get("MANA_OPENROUTER_MODEL", "openrouter/free")
    openrouter_url: str = "https://openrouter.ai/api/v1/chat/completions"
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    openai_model: str = os.environ.get("MANA_OPENAI_MODEL", "gpt-5.6-luna")
    openai_url: str = os.environ.get("MANA_OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses")
    openai_timeout: int = 60
    openai_max_output_tokens: int = 700
    openai_web_enabled: bool = True

    # Web
    enable_web: bool = True
    max_web_results: int = 5
    web_region: str = "ru-ru"
    web_failure_limit: int = 3
    web_cooldown_seconds: float = 60.0
    web_max_retries: int = 2
    web_retry_delay_seconds: float = 0.35
    web_serialized_requests: bool = True
    web_network_error_penalty: float = 0.08

    # Memory
    knowledge_db_path: str = "mana_v3_4_knowledge.pkl"
    max_knowledge_entries: int = 3000
    embedding_model: str = "all-MiniLM-L6-v2"
    use_embeddings: bool = True
    tfidf_max_features: int = 300

    # v5.0 persistent memory architecture
    memory_root: str = "mana_memory"
    memory_db_path: str = "mana_memory/mana_memory.sqlite3"
    memory_session_id: str = "default"
    # Minimum similarity for a stored memory to be injected as context.
    # Without a floor, any query retrieves any entry (see KnowledgeBase.search)
    # -- a greeting once pulled back a stored answer about AI news.
    #
    # Two separate floors because the scoring paths are NOT on the same
    # scale: embedding cosine between unrelated sentences sits around
    # 0.2-0.4, while TF-IDF cosine and word overlap sit near 0.0. A single
    # threshold silently filtered nothing on machines that had
    # sentence-transformers installed.
    #
    # The embedding figure is a reasoned default, not a calibrated one --
    # see scripts/calibrate_memory_relevance.py to measure it on your own
    # data and model before trusting it.
    # Ask instead of guessing when a follow-up names no subject and several
    # topics are in play (mana/intent.py). Asking is a refusal to answer, so
    # the trigger is deliberately narrow and the false-ask rate is measured.
    clarify_ambiguous_followups: bool = True
    clarify_min_topics: int = 2
    clarify_history_turns: int = 6

    memory_min_relevance_embedding: float = 0.45
    memory_min_relevance_lexical: float = 0.20
    memory_recent_messages: int = 12
    memory_retrieval_limit: int = 8
    memory_max_summary_chars: int = 5000
    memory_auto_compress_after: int = 40
    memory_store_assistant_responses: bool = True
    memory_store_user_messages: bool = True
    memory_wal: bool = True
    memory_fts_enabled: bool = True
    memory_vector_enabled: bool = True
    memory_fact_confidence_default: float = 0.55
    memory_context_budget_chars: int = 11000
    memory_recent_weight: float = 1.00
    memory_semantic_weight: float = 0.78
    memory_fact_weight: float = 0.95
    memory_cross_session: bool = True
    memory_semantic_top_k: int = 8
    memory_chunk_chars: int = 1800
    memory_chunk_overlap: int = 220
    memory_ingest_max_chunks: int = 20000
    memory_auto_extract_facts: bool = True
    memory_auto_extract_concepts: bool = True
    knowledge_root: str = "mana_memory/knowledge"
    knowledge_acquisition_max_sources: int = 12

    # v5.5 graph memory (mana.graph_memory) -- read side, used by _build_context
    graph_memory_context_enabled: bool = True
    graph_memory_depth: int = 2          # hops to walk from semantic seed nodes
    graph_memory_seed_limit: int = 3     # direct semantic seeds (kept independent of result limit -- see graph_memory.py)
    graph_memory_limit: int = 5          # nodes surfaced into the prompt
    graph_memory_char_budget: int = 1200 # this source's share of max_context_chars
    graph_memory_episode_every_n_turns: int = 12

    # Evolution
    strategy_population: int = 8
    strategy_generations: int = 3
    max_pipeline_evaluations_per_cycle: int = 24
    evolution_workers: int = 2
    screen_tasks_per_candidate: int = 4
    finalists_per_generation: int = 4
    min_improvement: float = 0.50
    benchmark_weight: float = 0.65
    generalization_weight: float = 0.25
    latency_weight: float = 0.10
    mutation_rate: float = 0.30
    max_mutations_per_offspring: int = 1
    exploration_max_mutations: int = 1
    causal_probe_enabled: bool = True
    causal_probe_max_fields: int = 2
    causal_probe_screen_tasks: int = 4
    statistical_precheck_z: float = 1.0
    elite_count: int = 3
    stagnation_limit: int = 3
    exploration_stagnation_threshold: int = 2
    exploration_levels: Tuple[float, float, float] = (0.0, 0.20, 0.45)
    ucb_exploration: float = 0.35
    freeze_after_successes: int = 2
    persist_rejected_experiments: bool = True
    rollback_on_regression: bool = True
    rollback_tolerance: float = 0.015
    holdout_min_delta: float = 0.005
    speed_improvement_threshold: float = 0.05
    p95_speed_improvement_threshold: float = 0.10
    reliability_regression_tolerance: float = 0.02

    # Strict acceptance / measured before-after gates
    acceptance_quality_tolerance: float = 0.01
    acceptance_generalization_tolerance: float = 0.01
    acceptance_holdout_tolerance: float = 0.01
    acceptance_speed_regression_tolerance: float = 0.10
    acceptance_p95_regression_tolerance: float = 0.10
    acceptance_min_speed_gain: float = 0.05
    acceptance_min_quality_gain: float = 0.01
    acceptance_require_holdout_non_regression: bool = True
    acceptance_require_reliability_non_regression: bool = True

    # Routing regression gate
    routing_gate_enabled: bool = True
    routing_min_accuracy: float = 0.80
    routing_min_execution_accuracy: float = 0.80
    routing_holdout_min_accuracy: float = 0.78
    routing_holdout_min_execution_accuracy: float = 0.78
    routing_allow_web_failures: bool = True
    routing_adaptive_enabled: bool = True
    routing_min_observations: int = 3
    routing_exploration_rate: float = 0.08
    routing_quality_weight: float = 0.65
    routing_reliability_weight: float = 0.25
    routing_latency_weight: float = 0.10
    routing_quality_floor: float = 0.45
    routing_holdout_min_quality: float = 0.45
    routing_holdout_min_web_success: float = 0.40

    # v4.2 adaptive cognition / cost-aware evolution
    adaptive_min_confidence: float = 0.42
    adaptive_confidence_margin: float = 0.05
    adaptive_hardness_weight: float = 0.12
    adaptive_critic_trigger: float = 0.68
    adaptive_max_steps: int = 6
    adaptive_cost_penalty: float = 0.035
    adaptive_latency_penalty: float = 0.020
    adaptive_failure_penalty: float = 0.15
    counterfactual_min_observations: int = 3
    confidence_history_min_observations: int = 4
    # v4.6 learned routing / verification policy
    learned_router_enabled: bool = True
    learned_router_min_samples: int = 12
    learned_router_margin: float = 0.12
    learned_router_blend: float = 0.35
    learned_router_history_limit: int = 1200
    learned_router_retrain_every: int = 4
    verification_policy_enabled: bool = True
    generated_test_cases: int = 5
    verification_quality_floor: float = 0.50
    confidence_learning_rate: float = 0.18
    stop_policy_learning_rate: float = 0.15
    stop_min_observations: int = 5
    adaptive_benchmark_repetitions: int = 2
    adaptive_holdout_repetitions: int = 2
    adaptive_route_margin: float = 0.10
    adaptive_quality_weight: float = 0.72
    adaptive_route_penalty: float = 0.10
    adaptive_uncertainty_penalty: float = 0.08
    graph_mutation_rate: float = 0.35
    graph_max_nodes: int = 10
    architecture_gate_min_quality: float = 0.60

    # v4.6 Value-of-Computation / local verification
    voc_enabled: bool = True
    voc_min_value: float = 0.018
    voc_cost_scale: float = 1.0
    voc_default_gain: float = 0.10
    voc_web_gain: float = 0.14
    voc_critic_gain: float = 0.10
    voc_execute_gain: float = 0.28
    voc_max_steps: int = 6
    local_exec_enabled: bool = False
    local_exec_timeout: float = 3.0
    local_exec_max_output: int = 12000
    local_exec_max_code_chars: int = 12000
    local_exec_memory_mb: int = 256
    local_exec_cpu_seconds: int = 2
    local_exec_max_processes: int = 1
    local_exec_workdir: str = "mana_exec_sandbox"
    local_exec_allow_filesystem: bool = False
    local_exec_allow_network: bool = False
    local_exec_allowed_modules: Tuple[str, ...] = ("math", "statistics", "decimal", "fractions", "json", "re")
    confidence_quality_weight: float = 0.55
    confidence_execution_weight: float = 0.15
    confidence_critic_weight: float = 0.20
    confidence_verification_weight: float = 0.25
    confidence_history_weight: float = 0.10
    # Evaluation
    benchmark_timeout_per_task: float = 60.0
    benchmark_repetitions: int = 1
    holdout_repetitions: int = 1
    real_task_window: int = 100
    generated_task_seed: int = 3401

    # Pipeline
    prompt_strategies: Tuple[str, ...] = ("direct", "structured", "analytical", "verification", "researcher")
    critic_prompt_strategies: Tuple[str, ...] = ("strict", "balanced", "factcheck", "minimal")
    adaptive_second_pass: bool = True

    # Voice
    enable_voice: bool = False
    voice_whisper_model: str = os.environ.get("MANA_WHISPER_MODEL", "small")
    voice_sample_rate: int = 16000
    voice_seconds: float = 30.0
    voice_silence_sec: float = 1.8
    voice_pre_roll_sec: float = 0.5
    voice_language: str = "ru"
    voice_tts_backend: str = os.environ.get("MANA_VOICE_TTS_BACKEND", "auto")
    voice_silero_speaker: str = os.environ.get("MANA_SILERO_SPEAKER", "xenia")
    voice_silero_sample_rate: int = 48000
    voice_output_device: Optional[int] = None
    voice_command_fuzzy_threshold: float = 0.72
    voice_compute_preferences: Tuple[str, ...] = ("float16", "int8_float16", "float32", "int8")
    voice_wakeword_required: bool = True

    # Persistence
    state_file: str = "mana_v3_4_state.pkl"
    history_file: str = "mana_v3_4_history.json"
    cache_file: str = "mana_v3_4_cache.pkl"
    experience_db_path: str = "mana_v3_4_experience.sqlite3"
    evolution_report_file: str = "mana_v4_6_evolution_reports.json"

    # Logging
    verbose_logging: bool = False

    def ensure_dirs(self) -> None:
        for p in [self.knowledge_db_path, self.state_file, self.history_file,
                  self.cache_file, self.experience_db_path, self.evolution_report_file, self.memory_db_path]:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(self.memory_root).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Random manager
# ---------------------------------------------------------------------------

class RandomManager:
    def __init__(self, seed: int):
        self.seed = seed
        self.evolution_rng = random.Random(seed)
        random.seed(seed)
        np.random.seed(seed)
        if HAS_TORCH:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def save_state(self) -> Dict[str, Any]:
        data = {
            "seed": self.seed,
            "evolution_rng_state": self.evolution_rng.getstate(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
        }
        if HAS_TORCH:
            data["torch_rng_state"] = torch.random.get_rng_state()
            if torch.cuda.is_available():
                data["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return data

    def load_state(self, data: Dict[str, Any]) -> None:
        if not data:
            return
        self.seed = int(data.get("seed", self.seed))
        if "evolution_rng_state" in data:
            self.evolution_rng.setstate(data["evolution_rng_state"])
        if "python_random_state" in data:
            random.setstate(data["python_random_state"])
        if "numpy_random_state" in data:
            np.random.set_state(data["numpy_random_state"])
        if HAS_TORCH:
            if "torch_rng_state" in data:
                torch.random.set_rng_state(data["torch_rng_state"])
            if torch.cuda.is_available() and "cuda_rng_state_all" in data:
                torch.cuda.set_rng_state_all(data["cuda_rng_state_all"])

    def random(self) -> float:
        return self.evolution_rng.random()

    def choice(self, seq: Sequence[Any]) -> Any:
        return self.evolution_rng.choice(seq)

    def randint(self, a: int, b: int) -> int:
        return self.evolution_rng.randint(a, b)

    def uniform(self, a: float, b: float) -> float:
        return self.evolution_rng.uniform(a, b)
