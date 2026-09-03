"""
mana.cli — argparse entry point wiring Config -> ManaAgent -> (batch/interactive/voice) run modes.
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

from .config import Config
from .version import PRODUCT_VERSION, format_version_report, version_report
from .agent import ManaAgent
from .voice import VoiceInterface
from .optional_deps import HAS_REQUESTS, HAS_WEB, HAS_SOUNDDEVICE, sd
from . import events, paths

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.2"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.max_pipeline_evaluations_per_cycle = max(8, args.population * max(1, args.generations))
    cfg.strategy_generations = max(1, args.generations)
    cfg.strategy_population = max(2, args.population)
    cfg.enable_web = not args.no_web and HAS_WEB
    cfg.enable_llm = not args.no_llm and HAS_REQUESTS
    cfg.voice_enabled = args.voice
    cfg.verbose_logging = args.verbose
    cfg.evolution_workers = max(1, min(4, args.evolution_workers))
    cfg.voice_whisper_model = args.voice_model
    cfg.voice_seconds = args.voice_seconds
    cfg.voice_language = args.voice_language
    cfg.voice_tts_backend = args.voice_tts_backend
    cfg.voice_silero_speaker = args.voice_speaker
    cfg.voice_output_device = args.voice_output_device
    cfg.memory_session_id = getattr(args, "session_id", "default") or "default"
    cfg.local_exec_enabled = bool(getattr(args, "enable_local_exec", False))
    cfg.local_exec_timeout = float(getattr(args, "exec_timeout", cfg.local_exec_timeout))
    if getattr(args, "web_max_retries", None) is not None:
        cfg.web_max_retries = max(0, int(args.web_max_retries))
    if getattr(args, "web_retry_delay", None) is not None:
        cfg.web_retry_delay_seconds = max(0.0, float(args.web_retry_delay))
    if getattr(args, "web_parallel", False):
        cfg.web_serialized_requests = False
    if getattr(args, "llm_model", None):
        cfg.ollama_model = args.llm_model
    if getattr(args, "llm_url", None):
        cfg.ollama_url = args.llm_url
    cfg.hardware_auto_adapt = not getattr(args, "no_hardware_adapt", False)
    # --- brain pool ---------------------------------------------------
    # NOTE: --no-llm keeps its original meaning (no local backend, see
    # BrainPool.usable) and additionally disables remote brains, because a
    # user typing --no-llm to run offline must get an offline run, not a
    # run that quietly reaches four cloud APIs instead of one local model.
    if getattr(args, "no_llm", False):
        cfg.brain_external_enabled = False
    if getattr(args, "no_external_brains", False):
        cfg.brain_external_enabled = False
    if getattr(args, "brain_policy", None):
        cfg.brain_policy = args.brain_policy
    if getattr(args, "brains_file", None):
        cfg.brains_file = args.brains_file
    if getattr(args, "allow_paid_brains", False):
        cfg.brain_allow_paid = True
    if getattr(args, "consensus", None):
        cfg.brain_consensus_n = max(2, int(args.consensus))
    return cfg


def format_brains(status: Dict[str, Any]) -> str:
    """Human-readable brain table for --list-brains.

    A brain that is configured but not ready is the interesting case (no
    key, cooling down, quota spent), so the reason is shown in-line rather
    than making the user diff two JSON blobs to find it.
    """
    lines = [f"policy={status['policy']}  готовы: {len(status['available'])}/{len(status['brains'])}", ""]
    header = f"{'BRAIN':<18}{'MODEL':<34}{'TIER':<8}{'READY':<7}{'CALLS':<7}{'LAT':<8}{'Q':<6}СТАТУС"
    lines.append(header)
    lines.append("-" * len(header))
    for b in status["brains"]:
        if b["ready"]:
            note = "ok"
        elif not b["enabled"]:
            note = "выключен"
        elif not b["key_present"]:
            note = f"нет ключа ({b['api_key_env']})"
        elif b["cooldown_for"] > 0:
            note = f"кулдаун {b['cooldown_for']:.0f}с: {b['last_error'][:40]}"
        elif b["rpd"] and b["day_count"] >= b["rpd"]:
            note = f"исчерпан дневной лимит ({b['rpd']})"
        elif not b["usable"]:
            note = "недоступен (сеть/режим)"
        else:
            note = "лимит запросов в минуту"
        lines.append(f"{b['brain_id']:<18}{b['model'][:33]:<34}{b['tier']:<8}"
                     f"{('да' if b['ready'] else 'нет'):<7}{b['calls']:<7}"
                     f"{b['ewma_latency']:<8.2f}{b['ewma_quality']:<6.2f}{note}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Argument parser, split out of main() so tests can construct it
    without running the agent."""
    parser = argparse.ArgumentParser(description=f"MANA {PRODUCT_VERSION}")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--self-improve", type=int, default=0)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--evolution-workers", type=int, default=2, choices=range(1, 5), metavar="N")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--routing-benchmark", action="store_true", help="Проверить AUTO/LOCAL/WEB/MIXED routing и реальное выполнение web")
    parser.add_argument("--routing-holdout", action="store_true", help="Запустить изолированный routing holdout")
    parser.add_argument("--adaptive-benchmark", action="store_true", help="Проверить adaptive compute / confidence / architecture")
    parser.add_argument("--adaptive-holdout", action="store_true", help="Запустить изолированный adaptive-compute holdout")
    parser.add_argument("--adaptive-repetitions", type=int, default=None, help="Количество повторов adaptive benchmark")
    parser.add_argument("--adaptive-holdout-repetitions", type=int, default=None, help="Количество повторов adaptive holdout")
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--enable-local-exec", action="store_true", help="Разрешить запуск небольших локальных Python-проверок в sandbox; без повышения прав")
    parser.add_argument("--exec-timeout", type=float, default=3.0, help="Макс. время локальной проверки, секунд")
    parser.add_argument("--web-max-retries", type=int, default=None, help="Макс. повторов Web-запроса при transport error")
    parser.add_argument("--web-retry-delay", type=float, default=None, help="Задержка между повторными Web-запросами")
    parser.add_argument("--web-parallel", action="store_true", help="Разрешить параллельные Web-запросы в evolution (по умолчанию запросы сериализуются)")
    parser.add_argument("--verify", metavar="EXPR", default=None, help="Проверить арифметическое выражение локально")
    parser.add_argument("--run-code", metavar="CODE_OR_FILE", default=None, help="Запустить Python-файл или inline-код в sandbox (требует --enable-local-exec)")
    parser.add_argument("--session-id", default="default", help="Идентификатор постоянной сессии диалога")
    parser.add_argument("--memory-search", metavar="QUERY", default=None, help="Семантический поиск по долговременной памяти")
    parser.add_argument("--memory-status", action="store_true", help="Показать состояние persistent memory")
    parser.add_argument("--learn", metavar="PATH_OR_TOPIC", default=None, help="Изучить файл, каталог или тему через Web")
    parser.add_argument("--learn-topic", metavar="TOPIC", default=None, help="Изучить тему через Web")
    parser.add_argument("--learn-domain", default="", help="Домен знания, например 1c")
    parser.add_argument("--learn-nonrecursive", action="store_true", help="Не обходить подкаталоги")
    parser.add_argument("--llm-model", default=None, metavar="NAME",
                        help="Имя модели LLM-бэкенда (для ollama — то, что показывает `ollama list`, "
                             "например 'mana' или 'qwen2.5:7b-instruct-q4_K_M'). "
                             "По умолчанию берётся Config.ollama_model.")
    parser.add_argument("--llm-url", default=None, metavar="URL",
                        help="Endpoint ollama (по умолчанию http://localhost:11434/api/generate).")
    parser.add_argument("--version", action="store_true", help="Показать версию продукта и всех подсистем")
    parser.add_argument("--knowledge-status", action="store_true", help="Показать статистику базы знаний")
    parser.add_argument("--hardware-status", action="store_true", help="Показать определённый профиль машины и что было адаптировано")
    parser.add_argument("--no-hardware-adapt", action="store_true", help="Не подстраивать Config под текущую машину")
    parser.add_argument("--list-tools", action="store_true", help="Показать зарегистрированные инструменты агента")
    # --- brain pool (mana.brains) ---
    parser.add_argument("--paths-status", action="store_true",
                        help="Где MANA ищет состояние, песочницу и собственный код "
                             "(первое, что нужно смотреть, если память 'потерялась')")
    parser.add_argument("--list-brains", action="store_true",
                        help="Показать все мозги: какие настроены, готовы, в кулдауне или исчерпали free-tier")
    parser.add_argument("--brains-status", action="store_true",
                        help="Полный JSON-статус пула мозгов (замеренные латентность и качество)")
    parser.add_argument("--brain-policy", default=None,
                        choices=["capability_first", "fastest", "cheapest", "least_loaded", "strongest", "round_robin"],
                        help="Как пул ранжирует мозги при выборе")
    parser.add_argument("--brains-file", default=None, metavar="PATH",
                        help="JSON-файл, дополняющий/переопределяющий каталог мозгов")
    parser.add_argument("--no-external-brains", action="store_true",
                        help="Только локальные мозги: ни один внешний API не вызывается")
    parser.add_argument("--allow-paid-brains", action="store_true",
                        help="Разрешить платные мозги (по умолчанию используются только бесплатные и локальные)")
    parser.add_argument("--ask", metavar="TASK", default=None,
                        help="Задать один вопрос и выйти (обычный путь solve_task)")
    parser.add_argument("--consensus", type=int, default=None, metavar="N",
                        help="Спросить N мозгов параллельно и показать ответ вместе со степенью их согласия")
    parser.add_argument("--decompose", metavar="TASK", default=None,
                        help="Разложить задачу на подзадачи, решить их на разных мозгах и собрать ответ")
    parser.add_argument("--list-code-targets", action="store_true", help="Показать список whitelisted целей для self-improve-code")
    parser.add_argument("--self-improve-code", metavar="TARGET_ID", default=None, help="Предложить и (при принятии gate'ом) применить патч кода для указанной цели")
    parser.add_argument("--code-instruction", default="", help="Инструкция для LLM при --self-improve-code")
    parser.add_argument("--code-history", nargs="?", const="__all__", default=None, metavar="TARGET_ID", help="Показать журнал применённых code-патчей (для всех целей или указанной)")
    parser.add_argument("--rollback-code", metavar="TARGET_ID", default=None, help="Откатить последний применённый патч для указанной цели")
    parser.add_argument("--graph-memory-status", action="store_true", help="Показать статистику графовой памяти (узлы/рёбра) текущей сессии")
    parser.add_argument("--graph-memory-search", metavar="QUERY", default=None, help="Найти контекст через обход графа памяти (не просто top-k)")
    parser.add_argument("--voice", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Подробный лог эволюции и benchmark")
    parser.add_argument("--voice-model", default="small")
    parser.add_argument("--voice-seconds", type=float, default=30.0)
    parser.add_argument("--voice-language", default="ru")
    parser.add_argument("--voice-tts-backend", choices=["auto", "silero", "pyttsx3"], default="auto")
    parser.add_argument("--voice-speaker", default="xenia")
    parser.add_argument("--voice-output-device", type=int, default=None)
    parser.add_argument("--list-audio-devices", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # The console is this process's output device, so say so once. Library
    # code emits events (see mana/events.py) and no longer assumes a
    # terminal exists; without a sink installed those events go nowhere,
    # which is exactly what the windowed app wants and exactly what a CLI
    # run must not do.
    events.install_console_sink()

    if args.paths_status:
        print(json.dumps(paths.status(), ensure_ascii=False, indent=2))
        return 0

    # Handled before any agent is constructed: reporting the version must
    # not require loading embedding models or opening databases.
    if args.version:
        print(format_version_report())
        return 0

    if args.list_audio_devices:
        if not HAS_SOUNDDEVICE:
            print("sounddevice не установлен"); return 1
        print(sd.query_devices()); return 0

    cfg = build_config(args)
    agent = ManaAgent(cfg)
    if args.learn_topic is not None:
        print(json.dumps(agent.acquire_topic_from_web(args.learn_topic, domain=args.learn_domain or args.learn_topic), ensure_ascii=False, indent=2)); return 0
    if args.learn is not None:
        print(json.dumps(agent.acquire_knowledge(args.learn, domain=args.learn_domain, recursive=not args.learn_nonrecursive), ensure_ascii=False, indent=2)); return 0
    if args.knowledge_status:
        print(json.dumps(agent.knowledge_status(), ensure_ascii=False, indent=2)); return 0
    if args.hardware_status:
        print(json.dumps(agent._json_safe(agent.hardware_status()), ensure_ascii=False, indent=2)); return 0
    if args.list_tools:
        print(json.dumps(agent.tools_status(), ensure_ascii=False, indent=2)); return 0
    if args.list_brains:
        print(format_brains(agent.brains_status())); return 0
    if args.brains_status:
        print(json.dumps(agent._json_safe(agent.brains_status()), ensure_ascii=False, indent=2)); return 0
    if args.decompose is not None:
        print(json.dumps(agent._json_safe(agent.solve_decomposed(args.decompose)),
                          ensure_ascii=False, indent=2)); return 0
    if args.consensus is not None and args.ask is not None:
        print(json.dumps(agent._json_safe(agent.ask_consensus(args.ask, n=int(args.consensus))),
                          ensure_ascii=False, indent=2)); return 0
    if args.ask is not None:
        print(json.dumps(agent._json_safe(agent.solve_task(args.ask)), ensure_ascii=False, indent=2)); return 0
    if args.list_code_targets:
        print(json.dumps(agent.list_code_targets(), ensure_ascii=False, indent=2)); return 0
    if args.code_history is not None:
        target_id = None if args.code_history == "__all__" else args.code_history
        print(json.dumps(agent.code_history(target_id), ensure_ascii=False, indent=2)); return 0
    if args.rollback_code is not None:
        print(json.dumps(agent.rollback_code(args.rollback_code), ensure_ascii=False, indent=2)); return 0
    if args.self_improve_code is not None:
        report = agent.self_improve_code(args.self_improve_code, args.code_instruction)
        print(json.dumps(agent._json_safe(report), ensure_ascii=False, indent=2)); return 0
    if args.graph_memory_status:
        print(json.dumps(agent.graph_memory_status(), ensure_ascii=False, indent=2)); return 0
    if args.graph_memory_search is not None:
        print(json.dumps(agent.graph_memory_search(args.graph_memory_search), ensure_ascii=False, indent=2)); return 0
    if args.memory_status:
        print(json.dumps({"version": agent.VERSION, "session_id": agent.session_id, "db": cfg.memory_db_path, "session": agent.persistent_memory.get_session(agent.session_id), "events": agent._memory_event_count(agent.session_id), "knowledge": agent.knowledge_status()}, ensure_ascii=False, indent=2)); return 0
    if args.memory_search is not None:
        print(json.dumps(agent._json_safe(agent.persistent_memory.safe_search_global(args.memory_search, cfg.memory_semantic_top_k)), ensure_ascii=False, indent=2))
        return 0
    if args.verify is not None:
        print(json.dumps(agent.tools.call("verify_arithmetic", expression=args.verify).output, ensure_ascii=False, indent=2)); return 0
    if args.run_code is not None:
        raw = str(args.run_code)
        path = Path(raw)
        if path.exists() and path.is_file():
            code = path.read_text(encoding="utf-8")
            result = agent.tools.call("run_code", code=code).output
            result["source"] = str(path.resolve())
        else:
            result = agent.tools.call("run_code", code=raw).output
            result["source"] = "inline"
        print(json.dumps(result, ensure_ascii=False, indent=2)); return 0
    if args.routing_benchmark:
        agent.routing_benchmark(); return 0
    if args.routing_holdout:
        print(json.dumps(agent.routing_holdout(), ensure_ascii=False, indent=2)); return 0
    if args.adaptive_benchmark:
        print(json.dumps(agent.adaptive_benchmark(False, repetitions=getattr(args, "adaptive_repetitions", None)), ensure_ascii=False, indent=2)); return 0
    if args.adaptive_holdout:
        print(json.dumps(agent.adaptive_benchmark(True, repetitions=getattr(args, "adaptive_holdout_repetitions", None)), ensure_ascii=False, indent=2)); return 0
    if args.benchmark:
        agent.benchmark(); return 0
    config_only_flags = (args.web_parallel or args.web_max_retries is not None or args.web_retry_delay is not None) and args.self_improve == 0 and not args.interactive and not args.voice and not args.routing_benchmark and not args.routing_holdout and not args.adaptive_benchmark and not args.adaptive_holdout and not args.run_code and args.verify is None and args.learn is None and args.learn_topic is None and not args.knowledge_status
    if config_only_flags:
        print(json.dumps({
            "version": agent.VERSION, "config_only": True,
            "web": {
                "enabled": bool(agent.web.enabled),
                "serialized_requests": bool(agent.config.web_serialized_requests),
                "max_retries": int(agent.config.web_max_retries),
                "retry_delay_seconds": float(agent.config.web_retry_delay_seconds),
            }
        }, ensure_ascii=False, indent=2)); return 0
    if args.self_improve > 0:
        agent.run_cycles(args.self_improve)
    elif args.interactive:
        agent.interactive()
    elif not args.voice:
        agent.run_cycles(args.cycles)

    if args.voice:
        try:
            VoiceInterface(agent, cfg).run()
        except Exception as exc:
            print(f"❌ Voice mode unavailable: {exc}")

    print("\nПровайдеры:")
    print(json.dumps(agent.llm.status(), ensure_ascii=False))
    print("\nИтог:")
    print(f"version={agent.VERSION}")
    print(f"cycle={agent.cycle}")
    print(f"best_pipeline_fitness={agent.best_pipeline_fitness:.3f}")
    print(f"pipeline={agent.pipeline.pretty()}")
    print(f"memory_entries={len(agent.memory.entries)}")
    print(f"persistent_memory_events={agent._memory_event_count(agent.session_id)}")
    print(f"experience_records={agent.experience.count()}")
    if agent.evolution_reports:
        r = agent.evolution_reports[-1]
        e = r["comparison"]["effect"]
        print("last_cycle_effect=")
        print(json.dumps({
            "cycle": r["cycle"], "verdict": r["verdict"], "accepted": r["accepted"],
            "train_quality": e["train_quality"], "holdout_quality": e["holdout_quality"],
            "train_p50_latency": e["train_p50_latency"], "train_p95_latency": e["train_p95_latency"],
            "train_timeout_rate": e["train_timeout_rate"], "train_fallback_rate": e["train_fallback_rate"],
            "decision": r.get("decision", {}),
            "pipeline_changes": r.get("pipeline_changes", {}),
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
