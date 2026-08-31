#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_diagnostics.py — run everything needed to give Claude a full,
structured picture of how MANA behaves on YOUR machine: the offline test
suite (safety net that was missing until now), a hardware profile of this
specific machine, and an at-scale memory benchmark (the concrete numbers
behind the "Memory Engine doesn't scale" checklist item).

Usage:
    python scripts/run_diagnostics.py                 # offline tests + hardware + memory benchmark
    MANA_TEST_LLM=1 python scripts/run_diagnostics.py  # also run LLM-dependent tests (needs Ollama etc. running)
    MANA_TEST_WEB=1 python scripts/run_diagnostics.py  # also run web-dependent tests (needs network)
    MANA_TEST_LLM=1 MANA_TEST_LLM_CODE_EVOLUTION=1 python scripts/run_diagnostics.py
        # also exercises self_improve_code's real-LLM path (always rolls back, see
        # tests/test_llm_dependent.py's docstring for exactly what that means)

Output: prints a human-readable summary to stdout AND writes
mana_diagnostic_report.json (structured) + mana_diagnostic_report.txt
(raw pytest output) next to this script's parent directory. Send both
files (or paste their content) back for analysis.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_pytest_available() -> bool:
    try:
        import pytest  # noqa: F401
        return True
    except ImportError:
        return False


def run_pytest_suite(report: dict) -> None:
    _section("1/3 — Test suite (the persisted safety net)")
    include_llm = os.environ.get("MANA_TEST_LLM") == "1"
    include_web = os.environ.get("MANA_TEST_WEB") == "1"
    parts = []
    if not include_llm:
        parts.append("not llm")
    if not include_web:
        parts.append("not web")
    marker_expr = " and ".join(parts) if parts else ""

    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
    if marker_expr:
        cmd += ["-m", marker_expr]
    print(f"Running: {' '.join(cmd)}")
    if include_llm:
        print("(MANA_TEST_LLM=1 -- includes real LLM calls, requires a reachable backend)")
    if include_web:
        print("(MANA_TEST_WEB=1 -- includes real network calls)")

    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900)
    elapsed = time.perf_counter() - started
    output = proc.stdout + "\n" + proc.stderr
    print(output[-4000:])  # last chunk to stdout; full output goes to the .txt report
    report["pytest"] = {
        "returncode": proc.returncode,
        "elapsed_seconds": round(elapsed, 2),
        "included_llm_tests": include_llm,
        "included_web_tests": include_web,
        "full_output_saved_to": "mana_diagnostic_report.txt",
    }
    (REPO_ROOT / "mana_diagnostic_report.txt").write_text(output, encoding="utf-8")


def run_hardware_profile(report: dict) -> None:
    _section("2/3 — Hardware profile of THIS machine")
    from mana.hardware import detect_hardware, apply_hardware_profile
    from mana.config import Config
    from dataclasses import asdict

    profile = detect_hardware()
    print(json.dumps(asdict(profile), ensure_ascii=False, indent=2))
    cfg = Config()
    changes = apply_hardware_profile(cfg, profile)
    print("\nWould adapt (if hardware_auto_adapt=True, the default):")
    print(json.dumps(changes, ensure_ascii=False, indent=2))
    report["hardware"] = {"profile": asdict(profile), "would_adapt": changes,
                           "python_version": platform.python_version(),
                           "platform": platform.platform()}


def run_memory_scale_benchmark(report: dict) -> None:
    """The concrete numbers behind checklist item 'Memory Engine doesn't
    scale': how semantic_search / graph_context / FTS rebuild cost grow
    as event count grows. This is a benchmark, not a pass/fail test."""
    _section("3/3 — Memory-at-scale benchmark (informs Memory Engine 2.0 priority)")
    import tempfile
    from mana.config import Config
    from mana.memory import MemoryManager
    from mana import graph_memory as gm

    results = []
    sizes = [100, 500, 2000]
    for n in sizes:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bench"
            cfg = Config(memory_root=str(root), memory_db_path=str(root / "m.sqlite3"))
            cfg.ensure_dirs()

            t0 = time.perf_counter()
            mm = MemoryManager(cfg)  # includes the unconditional FTS rebuild
            init_time = time.perf_counter() - t0

            store = gm.GraphMemoryStore(mm)
            t0 = time.perf_counter()
            for i in range(n):
                store.record_turn("bench", f"Вопрос номер {i} про тему {i % 20}",
                                   f"Ответ номер {i} с деталями {i % 20} и данными.")
            write_time = time.perf_counter() - t0
            mm.close()  # NOTE (bugfix, caught on real Windows hardware -- the sandbox
            # this script was authored in is Linux, where an open sqlite3 handle
            # doesn't block deleting the file, so this was never exercised there).
            # Windows refuses to delete a file a process still has open; without
            # this the TemporaryDirectory cleanup below raised PermissionError
            # (WinError 32) on every run.

            # Re-open (simulates a real restart with n events already present)
            t0 = time.perf_counter()
            mm2 = MemoryManager(cfg)
            reopen_fts_rebuild_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            mm2.semantic_search("тема 5 данные", limit=8, session_id="bench", cross_session=True)
            search_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            gm.GraphMemoryStore(mm2).graph_context("bench", "тема 5 данные", depth=2, limit=8)
            graph_time = time.perf_counter() - t0
            mm2.close()

            row = {"events": n, "initial_init_seconds": round(init_time, 4),
                   "write_seconds_total": round(write_time, 4),
                   "reopen_fts_rebuild_seconds": round(reopen_fts_rebuild_time, 4),
                   "semantic_search_seconds": round(search_time, 4),
                   "graph_context_seconds": round(graph_time, 4)}
            print(json.dumps(row, ensure_ascii=False))
            results.append(row)
    report["memory_scale_benchmark"] = results


def main() -> int:
    report: dict = {"timestamp": time.time(), "repo_root": str(REPO_ROOT)}

    if not check_pytest_available():
        print("pytest is not installed. Install it first:\n\n    pip install pytest\n")
        return 1

    run_pytest_suite(report)
    run_hardware_profile(report)
    run_memory_scale_benchmark(report)

    out_path = REPO_ROOT / "mana_diagnostic_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _section("Done")
    print(f"Structured report: {out_path}")
    print(f"Full pytest output: {REPO_ROOT / 'mana_diagnostic_report.txt'}")
    print("Send both files (or paste their content) back for analysis.")
    return 0 if report["pytest"]["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
