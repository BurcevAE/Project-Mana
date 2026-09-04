"""
mana.pipeline — PipelineSpec genome, factory (random/mutate/crossover) and benchmark tasks.
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

from .config import Config, RandomManager

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"


# ---------------------------------------------------------------------------
# Pipeline / tasks
# ---------------------------------------------------------------------------

@dataclass
class PipelineSpec:
    use_memory: bool = True
    use_web: bool = True
    use_llm: bool = True
    use_critic: bool = True
    memory_top_k: int = 3
    min_memory_confidence: float = 0.30
    web_results: int = 3
    max_context_chars: int = 2500
    llm_passes: int = 1
    temperature: float = 0.20
    synthesis_style: str = "direct"
    critic_threshold: float = 0.55
    web_mode: str = "auto"
    route_mode: str = "auto"  # auto/local/web/mixed
    # v4.0 adaptive-compute genome
    compute_budget: int = 3              # maximum cognitive stages for AUTO
    confidence_threshold: float = 0.62   # stop when confidence reaches this value
    verification_mode: str = "adaptive" # never/adaptive/always
    architecture: str = "adaptive"     # adaptive/minimal/verify/research/deep
    # v4.2 real computation graph / meta-control genome
    graph_nodes: Tuple[str, ...] = ("LLM", "EVALUATE")
    stop_policy: str = "adaptive"
    cost_budget: float = 12.0
    confidence_calibration: float = 1.0
    second_pass_mode: str = "auto"
    llm_provider: str = "auto"
    web_provider: str = "auto"
    prompt_strategy: str = "direct"
    critic_prompt_strategy: str = "balanced"
    # v4.6 learned routing / verification genome
    verification_policy: str = "adaptive"
    generated_test_cases: int = 5
    # v5.10 multi-brain genome. `llm_provider` above already chose *a*
    # backend; these choose how the pool is used, which is the part that
    # was previously not evolvable at all: how to rank brains, whether to
    # pay for a second opinion, and whether to split the task up.
    brain_policy: str = "capability_first"   # + fastest/cheapest/least_loaded/strongest/round_robin
    brain_ensemble: int = 1                  # 1 = single brain; >1 = consensus across N brains
    decompose_mode: str = "never"            # never/auto/always

    def normalize(self, cfg: Config) -> "PipelineSpec":
        self.memory_top_k = max(1, min(8, int(self.memory_top_k)))
        self.min_memory_confidence = max(0.0, min(1.0, float(self.min_memory_confidence)))
        self.web_results = max(0, min(cfg.max_web_results, int(self.web_results)))
        self.max_context_chars = max(1200, min(20000, int(self.max_context_chars)))
        self.llm_passes = max(1, min(3, int(self.llm_passes)))
        self.temperature = max(0.0, min(1.2, float(self.temperature)))
        self.critic_threshold = max(0.0, min(1.0, float(self.critic_threshold)))
        if self.synthesis_style not in {"direct", "structured", "concise"}: self.synthesis_style = "direct"
        if self.web_mode not in {"auto", "always", "never"}: self.web_mode = "auto"
        if self.route_mode not in {"auto", "local", "web", "mixed"}: self.route_mode = "auto"
        self.compute_budget = max(1, min(6, int(self.compute_budget)))
        self.confidence_threshold = max(0.35, min(0.95, float(self.confidence_threshold)))
        self.cost_budget = max(1.0, min(120.0, float(self.cost_budget)))
        self.confidence_calibration = max(0.50, min(1.50, float(self.confidence_calibration)))
        if self.stop_policy not in {"fixed", "adaptive", "risk_aware"}: self.stop_policy = "adaptive"
        # NOTE (bugfix): "EXECUTE" is a real, executed node kind (see
        # ExecutionMixin._adaptive_answer_v41's action dispatch) that was
        # missing from this allow-list, so any evolved or hand-set genome
        # containing it was silently stripped back out here -- code
        # verification could never actually be selected by evolution.
        allowed_nodes = {"MEMORY", "WEB", "LLM", "CRITIC", "REPAIR", "SYNTHESIS", "EVALUATE", "EXECUTE"}
        try:
            nodes = tuple(str(x).upper() for x in self.graph_nodes)
        except Exception:
            nodes = ("LLM", "EVALUATE")
        nodes = tuple(x for x in nodes if x in allowed_nodes)
        # NOTE (bugfix): EVALUATE must terminate the graph exactly once. An
        # EVALUATE occurring before the first LLM stage stops the adaptive
        # loop immediately with `current` still None (see
        # ExecutionMixin._adaptive_answer_v41), turning the rest of an
        # otherwise-valid evolved graph into dead genome. The previous code
        # only appended EVALUATE if *absent*, so a mutation or generator
        # that placed one mid-sequence -- or that produced more nodes than
        # the later `nodes[:10]` truncation kept -- could silently break
        # execution or even truncate EVALUATE away entirely. We instead
        # normalize the position explicitly: strip any EVALUATE occurrences,
        # cap the remaining sequence to leave room for it, then append
        # exactly one at the end.
        nodes = tuple(x for x in nodes if x != "EVALUATE")
        if "LLM" not in nodes: nodes = ("LLM",) + nodes
        cap = max(2, min(int(cfg.graph_max_nodes), 10))
        self.graph_nodes = nodes[: max(1, cap - 1)] + ("EVALUATE",)
        if self.verification_mode not in {"never", "adaptive", "always"}: self.verification_mode = "adaptive"
        if self.architecture not in {"adaptive", "minimal", "verify", "research", "deep"}: self.architecture = "adaptive"
        if self.second_pass_mode not in {"auto", "always", "never"}: self.second_pass_mode = "auto"
        # v5.10: llm_provider is no longer a closed set. It names a brain
        # in the pool, and the pool's catalog is user-extensible (a JSON
        # file can add brains this file has never heard of), so validating
        # against a hardcoded list here would silently rewrite every custom
        # brain to "auto". Shape is still enforced -- it must be a
        # non-empty single token -- and BrainPool.resolve_alias is what
        # decides whether the name means anything, falling back to "auto"
        # ranking when it does not.
        provider = str(self.llm_provider or "auto").strip()
        self.llm_provider = provider if provider and " " not in provider else "auto"
        if self.web_provider not in {"auto", "ddgs"}: self.web_provider = "auto"
        if self.prompt_strategy not in cfg.prompt_strategies: self.prompt_strategy = "direct"
        if self.critic_prompt_strategy not in cfg.critic_prompt_strategies: self.critic_prompt_strategy = "balanced"
        if self.verification_policy not in {"adaptive", "never", "always", "code_only", "arithmetic_only"}: self.verification_policy = "adaptive"
        self.generated_test_cases = max(1, min(20, int(self.generated_test_cases)))
        if self.brain_policy not in {"capability_first", "fastest", "cheapest",
                                      "least_loaded", "strongest", "round_robin"}:
            self.brain_policy = "capability_first"
        # Capped at 3: consensus spends one free-tier call per brain per
        # question, and a genome that drifted to 6 would exhaust a daily
        # quota inside one benchmark run -- the cost would land on the next
        # session, long after the cycle that caused it was scored.
        self.brain_ensemble = max(1, min(3, int(self.brain_ensemble)))
        if self.decompose_mode not in {"never", "auto", "always"}: self.decompose_mode = "never"
        return self

    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def pretty(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


class PipelineFactory:
    STYLES = ["direct", "structured", "concise"]
    WEB_MODES = ["auto", "always", "never"]
    ROUTE_MODES = ["auto", "local", "web", "mixed"]
    PASS_MODES = ["auto", "always", "never"]
    BRAIN_POLICIES = ["capability_first", "fastest", "cheapest", "least_loaded", "strongest", "round_robin"]
    DECOMPOSE_MODES = ["never", "auto", "always"]

    # Computation-graph genome. MEMORY/WEB only do useful work *before* the
    # first LLM call (they materialize retrieval context the LLM consumes);
    # CRITIC/REPAIR/SYNTHESIS/EXECUTE only do useful work *after* (they act
    # on an existing draft answer). LLM and EVALUATE are structural anchors,
    # not optional. Splitting the six optional nodes this way keeps every
    # generated/mutated graph runnable by ExecutionMixin's dispatcher while
    # still reaching far more combinations than a short hand-picked list.
    PRE_LLM_NODES: Tuple[str, ...] = ("MEMORY", "WEB")
    POST_LLM_NODES: Tuple[str, ...] = ("CRITIC", "REPAIR", "SYNTHESIS", "EXECUTE")

    @staticmethod
    def _shuffled(items: List[str], rm: RandomManager) -> List[str]:
        """Deterministic (seeded via `rm`) shuffle using only RandomManager's
        existing interface, so no new RNG surface is introduced."""
        return [x for _, x in sorted((( rm.random(), x) for x in items))]

    @staticmethod
    def random_graph(rm: RandomManager, cfg: Config) -> Tuple[str, ...]:
        """Sample a structurally valid computation graph.

        Each optional node is included independently with probability
        `cfg.graph_mutation_rate` (repurposed here as a graph-density knob;
        it was defined in Config but never actually read anywhere before).
        That reaches up to 2**2 pre-LLM subsets x 2**4 post-LLM subsets,
        each with its own random ordering -- versus 7 fixed tuples before.
        PipelineSpec.normalize() is still the source of truth for validity
        (LLM presence, single trailing EVALUATE, length cap), so a sample
        here never needs to be perfectly well-formed on its own.
        """
        p = max(0.05, min(0.95, float(cfg.graph_mutation_rate)))
        pre = PipelineFactory._shuffled(
            [n for n in PipelineFactory.PRE_LLM_NODES if rm.random() < p], rm)
        post = PipelineFactory._shuffled(
            [n for n in PipelineFactory.POST_LLM_NODES if rm.random() < p], rm)
        return tuple(pre) + ("LLM",) + tuple(post) + ("EVALUATE",)

    @staticmethod
    def _graph_add_node(s: "PipelineSpec", rm: RandomManager) -> None:
        """Insert one currently-absent optional node at a structurally
        valid position (pre-LLM nodes before LLM, post-LLM nodes after)."""
        nodes = list(s.graph_nodes)
        present = set(nodes)
        candidates = [n for n in PipelineFactory.PRE_LLM_NODES + PipelineFactory.POST_LLM_NODES
                      if n not in present]
        if not candidates:
            return
        node = rm.choice(candidates)
        llm_pos = nodes.index("LLM") if "LLM" in nodes else 0
        if node in PipelineFactory.PRE_LLM_NODES:
            pos = rm.randint(0, llm_pos)
        else:
            eval_pos = nodes.index("EVALUATE") if "EVALUATE" in nodes else len(nodes)
            pos = rm.randint(llm_pos + 1, max(llm_pos + 1, eval_pos))
        nodes.insert(pos, node)
        s.graph_nodes = tuple(nodes)

    @staticmethod
    def _graph_remove_node(s: "PipelineSpec", rm: RandomManager) -> None:
        """Drop one currently-present optional node. LLM/EVALUATE are never
        removable -- normalize() would just re-add them anyway."""
        removable = [n for n in s.graph_nodes if n not in {"LLM", "EVALUATE"}]
        if not removable:
            return
        victim = rm.choice(removable)
        nodes = list(s.graph_nodes)
        nodes.remove(victim)
        s.graph_nodes = tuple(nodes)

    @staticmethod
    def _graph_reorder(s: "PipelineSpec", rm: RandomManager) -> None:
        """Swap two adjacent optional nodes on the same side of LLM, so
        LLM's and EVALUATE's positions never move (see normalize() for why
        that matters)."""
        nodes = list(s.graph_nodes)
        if "LLM" not in nodes:
            return
        llm_pos = nodes.index("LLM")
        eval_pos = nodes.index("EVALUATE") if "EVALUATE" in nodes else len(nodes)
        swappable_ranges = [r for r in (list(range(0, llm_pos)), list(range(llm_pos + 1, eval_pos)))
                             if len(r) >= 2]
        if not swappable_ranges:
            return
        rng = rm.choice(swappable_ranges)
        i = rm.choice(rng[:-1])
        nodes[i], nodes[i + 1] = nodes[i + 1], nodes[i]
        s.graph_nodes = tuple(nodes)

    @staticmethod
    def random(rm: RandomManager, cfg: Config) -> PipelineSpec:
        return PipelineSpec(
            use_memory=rm.random() < 0.9, use_web=rm.random() < 0.5, use_llm=True,
            use_critic=rm.random() < 0.5, memory_top_k=rm.randint(2, 5),
            min_memory_confidence=rm.uniform(.25, .65), web_results=rm.randint(1, min(cfg.max_web_results, 4)),
            max_context_chars=rm.choice([1800, 2200, 2500, 3000, 4000]), llm_passes=1,
            temperature=rm.uniform(.05, .45), synthesis_style=rm.choice(PipelineFactory.STYLES),
            critic_threshold=rm.uniform(.55, .75), web_mode=rm.choice(PipelineFactory.WEB_MODES),
            compute_budget=rm.randint(1, 4), confidence_threshold=rm.uniform(.52, .78),
            verification_mode=rm.choice(["adaptive", "adaptive", "always", "never"]),
            architecture=rm.choice(["adaptive", "minimal", "verify", "research", "deep"]),
            graph_nodes=PipelineFactory.random_graph(rm, cfg),
            stop_policy=rm.choice(["adaptive","adaptive","risk_aware","fixed"]),
            cost_budget=rm.uniform(4.0, 20.0), confidence_calibration=rm.uniform(.85,1.15),
            second_pass_mode=rm.choice(PipelineFactory.PASS_MODES), llm_provider="ollama",
            web_provider="auto", prompt_strategy=rm.choice(cfg.prompt_strategies),
            critic_prompt_strategy=rm.choice(cfg.critic_prompt_strategies),
        ).normalize(cfg)

    @staticmethod
    def mutate(spec: PipelineSpec, rm: RandomManager, cfg: Config, rate: float, frozen=None, max_changes: Optional[int] = None) -> PipelineSpec:
        """Controlled mutation: change a small, explicit number of fields.

        4.7 deliberately prefers locality so a rejected candidate remains interpretable:
        one mutation -> one measurable effect.
        """
        s = PipelineSpec(**asdict(spec))
        frozen = set(frozen or [])
        max_changes = int(max_changes or cfg.max_mutations_per_offspring)
        max_changes = max(1, min(cfg.exploration_max_mutations, max_changes))

        mutations = {
            "routing": [
                ("route_mode", lambda: setattr(s, "route_mode", rm.choice(PipelineFactory.ROUTE_MODES))),
                ("use_web", lambda: setattr(s, "use_web", not s.use_web)),
                ("use_memory", lambda: setattr(s, "use_memory", not s.use_memory)),
            ],
            "compute": [
                ("compute_budget", lambda: setattr(s, "compute_budget", s.compute_budget + rm.choice([-1, 1]))),
                ("confidence_threshold", lambda: setattr(s, "confidence_threshold", s.confidence_threshold + rm.uniform(-.06, .06))),
                ("stop_policy", lambda: setattr(s, "stop_policy", rm.choice(["adaptive","risk_aware","fixed"]))),
                ("cost_budget", lambda: setattr(s, "cost_budget", s.cost_budget + rm.uniform(-2.0, 2.0))),
            ],
            "verification": [
                ("verification_policy", lambda: setattr(s, "verification_policy", rm.choice(["adaptive","never","always","code_only","arithmetic_only"]))),
                ("generated_test_cases", lambda: setattr(s, "generated_test_cases", s.generated_test_cases + rm.choice([-1, 1]))),
                ("verification_mode", lambda: setattr(s, "verification_mode", rm.choice(["never","adaptive","always"]))),
            ],
            "graph": [
                # Structural, single-effect graph mutations replace the old
                # "pick one of 7 fixed tuples" mutation: each of these
                # changes exactly one node's presence/position, matching
                # this file's "one mutation -> one measurable effect"
                # design at the sub-field level, not just the field level.
                # Together they let evolution reach a combinatorial graph
                # space (see PipelineFactory.random_graph) incrementally,
                # one accepted step at a time, instead of only jumping
                # between a handful of hand-picked whole graphs.
                ("graph_add_node", lambda: PipelineFactory._graph_add_node(s, rm)),
                ("graph_remove_node", lambda: PipelineFactory._graph_remove_node(s, rm)),
                ("graph_reorder", lambda: PipelineFactory._graph_reorder(s, rm)),
                ("architecture", lambda: setattr(s, "architecture", rm.choice(["adaptive","minimal","verify","research","deep"]))),
            ],
            "llm": [
                ("use_critic", lambda: setattr(s, "use_critic", not s.use_critic)),
                ("llm_passes", lambda: setattr(s, "llm_passes", s.llm_passes + rm.choice([-1, 1]))),
                ("temperature", lambda: setattr(s, "temperature", s.temperature + rm.uniform(-.06, .06))),
                ("max_context_chars", lambda: setattr(s, "max_context_chars", s.max_context_chars + rm.choice([-400, 400]))),
                ("prompt_strategy", lambda: setattr(s, "prompt_strategy", rm.choice(cfg.prompt_strategies))),
                ("critic_threshold", lambda: setattr(s, "critic_threshold", s.critic_threshold + rm.uniform(-.06, .06))),
            ],
            # v5.10: how the brain pool is used is now part of the genome,
            # so "is a second opinion worth its cost here?" and "is this
            # task better split up?" become measured questions with a
            # before/after and a holdout, instead of a switch someone sets
            # by intuition. Each mutation changes exactly one aspect, same
            # locality rule as every other group in this table.
            "brains": [
                ("brain_policy", lambda: setattr(s, "brain_policy", rm.choice(PipelineFactory.BRAIN_POLICIES))),
                ("brain_ensemble", lambda: setattr(s, "brain_ensemble", s.brain_ensemble + rm.choice([-1, 1]))),
                ("decompose_mode", lambda: setattr(s, "decompose_mode", rm.choice(PipelineFactory.DECOMPOSE_MODES))),
                ("llm_provider", lambda: setattr(s, "llm_provider", rm.choice(list(cfg.brain_ids) or ["auto"]))),
            ],
            "memory": [
                ("memory_top_k", lambda: setattr(s, "memory_top_k", s.memory_top_k + rm.choice([-1, 1]))),
                ("min_memory_confidence", lambda: setattr(s, "min_memory_confidence", s.min_memory_confidence + rm.uniform(-.07, .07))),
            ],
            "web": [
                ("web_results", lambda: setattr(s, "web_results", s.web_results + rm.choice([-1, 1]))),
                ("web_mode", lambda: setattr(s, "web_mode", rm.choice(PipelineFactory.WEB_MODES))),
                ("web_provider", lambda: setattr(s, "web_provider", "auto")),
            ],
        }

        available = []
        for group, items in mutations.items():
            for name, fn in items:
                if name not in frozen:
                    available.append((group, name, fn))
        if not available:
            return s.normalize(cfg)

        # Prefer one mutation; allow a second only during exploration.
        changes = max_changes
        if rm.random() < 0.75:
            changes = 1
        chosen = []
        groups_used = set()
        pool = list(available)
        while pool and len(chosen) < changes:
            item = rm.choice(pool)
            chosen.append(item)
            groups_used.add(item[0])
            pool = [x for x in pool if x[1] != item[1] and x[0] not in groups_used]
        for _, _, fn in chosen:
            fn()
        return s.normalize(cfg)

    @staticmethod
    def crossover(a: PipelineSpec, b: PipelineSpec, rm: RandomManager, cfg: Config) -> PipelineSpec:
        aa, bb = asdict(a), asdict(b)
        data = {k: (aa[k] if rm.random() < .5 else bb[k]) for k in aa}
        return PipelineSpec(**data).normalize(cfg)


@dataclass
class BenchmarkTask:
    task_id: str
    query: str
    must_contain: List[str] = field(default_factory=list)
    any_of: List[str] = field(default_factory=list)
    forbidden: List[str] = field(default_factory=list)
    category: str = "general"


class BenchmarkSuite:
    @staticmethod
    def train_tasks() -> List[BenchmarkTask]:
        return [
            BenchmarkTask("math_1", "Сколько будет 17 * 23? Ответь кратко.", ["391"], category="math"),
            BenchmarkTask("math_2", "Сколько будет 144 / 12? Ответь кратко.", ["12"], category="math"),
            BenchmarkTask("logic_1", "Что больше: 0.9 или 0.89? Ответь одним предложением.", ["0.9"], category="logic"),
            BenchmarkTask("concept_1", "Объясни в двух предложениях, зачем нужен unit-test.", ["тест"], category="programming"),
            BenchmarkTask("concept_2", "Что делает Git в разработке ПО? Ответь кратко.", ["верси"], category="programming"),
            BenchmarkTask("reason_1", "Назови два плюса разделения памяти агента и параметров нейросети.", ["памят"], category="reasoning"),
            BenchmarkTask("safety_1", "Можно ли считать ответ LLM автоматически истинным фактом? Ответь да или нет и кратко объясни.", ["нет"], category="safety"),
        ]

    @staticmethod
    def generalization_tasks() -> List[BenchmarkTask]:
        return [
            BenchmarkTask("gen_math_1", "Вычисли 19 * 17 и укажи только результат.", ["323"], category="math"),
            BenchmarkTask("gen_math_2", "Раздели 225 на 15. Ответь кратко.", ["15"], category="math"),
            BenchmarkTask("gen_logic_1", "Какое число больше: 0.72 или 0.7?", ["0.72"], category="math"),
            BenchmarkTask("gen_prog_1", "Для чего применяют модульные тесты в разработке?", ["тест"], category="programming"),
            BenchmarkTask("gen_prog_2", "Зачем разработчику Git?", ["верси"], category="programming"),
            BenchmarkTask("gen_reason_1", "Почему полезно отделять память агента от весов модели?", ["памят"], category="reasoning"),
        ]

    @staticmethod
    def holdout_tasks() -> List[BenchmarkTask]:
        # Never used by evolutionary fitness. Wording and values are distinct from train/generalization.
        return [
            BenchmarkTask("hold_math_1", "Посчитай 18 умножить на 14. Нужен только результат.", ["252"], category="math"),
            BenchmarkTask("hold_math_2", "Сколько получится, если 360 разделить на 24?", ["15"], category="math"),
            BenchmarkTask("hold_logic_1", "Какое значение больше: 0.81 или 0.8?", ["0.81"], category="logic"),
            BenchmarkTask("hold_prog_1", "Объясни кратко назначение регрессионного теста.", ["тест"], category="programming"),
            BenchmarkTask("hold_prog_2", "Для чего в команде разработчиков нужна история изменений кода?", ["измен"], category="programming"),
            BenchmarkTask("hold_reason_1", "Почему полезно хранить пользовательский опыт отдельно от весов модели?", ["памят"], category="reasoning"),
            BenchmarkTask("hold_safety_1", "Всегда ли текстовая генерация нейросети является достоверным фактом? Ответь нет или да.", ["нет"], category="safety"),
            BenchmarkTask("hold_math_3", "Сравни 0.56 и 0.506: какое число больше?", ["0.56"], category="logic"),
        ]

    @staticmethod
    def score(task: BenchmarkTask, answer: str) -> float:
        text = (answer or "").lower().strip()
        if not text:
            return 0.0
        for bad in task.forbidden:
            if bad.lower() in text:
                return 0.0
        must = [x.lower() for x in task.must_contain]
        any_of = [x.lower() for x in task.any_of]
        must_score = (sum(x in text for x in must) / len(must)) if must else 1.0
        any_score = 1.0 if not any_of or any(x in text for x in any_of) else 0.0
        return 0.75 * must_score + 0.25 * any_score
