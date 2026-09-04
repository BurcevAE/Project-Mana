#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_research_cycle.py — let the cycle run itself against real
brains, and print what it decided.

Nothing here tells it what to do. It gets a budget, a way to ask a brain
a question, and a way to have the answer graded by the immutable core;
every choice after that -- measure this, experiment on that, the
vocabulary is the problem -- comes out of `mana.cognition.research`
reading its own evidence.

Usage:
    python scripts/run_research_cycle.py --budget 300 --steps 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mana import Config                                  # noqa: E402
from mana.brains import BrainPool                       # noqa: E402
from mana.cognition.research import ResearchCycle       # noqa: E402
from mana.cognition.self_model import SelfModel         # noqa: E402
from mana.core import oracle                            # noqa: E402

#: The answer format the oracle grades against. Stated to the brain
#: verbatim, because an ungradable answer is recorded separately from a
#: wrong one and a run full of them measures the prompt, not the system.
FORMAT = ("Ответь только итоговым значением, без пояснений, без единиц "
          "измерения и без знаков препинания вокруг него.")


def make_runner(pool: BrainPool, texts: dict):
    """A lesson runner over live brains, graded by core.oracle."""
    def run(task):
        texts[task.task_id] = task.prompt
        answer = pool.ask(f"{task.prompt}\n\n{FORMAT}",
                          kind=task.domain, difficulty=task.difficulty,
                          context_tag="research")
        graded = oracle.grade(task, answer.get("text") or "")
        return graded.correct, answer.get("brain", ""), 1
    return run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--out", default="mana_research_cycle.json")
    args = parser.parse_args()

    pool = BrainPool(Config())
    available = pool.available()
    print(f"мозги: {', '.join(available) or 'нет'}")
    if not available:
        print("нечего запускать: ни один мозг не доступен")
        return 1

    model = SelfModel()
    texts: dict = {}
    cycle = ResearchCycle(model, task_texts=texts,
                          budget_calls=args.budget, max_steps=args.steps)

    started = time.perf_counter()
    report = cycle.run(make_runner(pool, texts))
    elapsed = time.perf_counter() - started

    print(f"\nшагов: {report['steps']}  вызовов: {report['calls_used']}  "
          f"время: {elapsed:.0f} с")
    print(f"остановка: {report['stop_reason']}")
    print(f"снято неопределённости: {report['total_resolution']}\n")

    for i, step in enumerate(report["history"], 1):
        print(f"  {i}. [{step['activity']}] {step['description']}")
        print(f"     → {step['outcome']}  (снято {step['resolution']:.3f})")

    if report["failure_clusters"]:
        print("\nструктура отказов:")
        for c in report["failure_clusters"]:
            print(f"  {c['field_name']}={c['value']}: {c['failures']} отказов / "
                  f"{c['successes']} успехов, избыток {c['excess']:+.2f}")

    if report["representation_findings"]:
        print("\nсловарь описания задач недостаточен, что помогло бы:")
        for f in report["representation_findings"]:
            print(f"  {f['field_name']}: разделяет {f['separates_pairs']} "
                  f"противоречащих пар, осталось бы {f['remaining_pairs']}")

    # Что прогон стоил на самом деле. До фазы 14 здесь можно было
    # напечатать только число вызовов, в котором обращение к 120B и к
    # локальной 7B — одна и та же единица.
    from mana.core.cost import efficiency
    spent = pool.total_cost()
    gain = float(report["total_resolution"])
    print("\nстоимость прогона в реальных единицах:")
    print(f"  {spent.describe()}")
    print(f"\nприрост {gain:.3f} на единицу стоимости:")
    for name, value in efficiency(gain, spent).items():
        shown = "не измерено" if value is None else f"{value:.5f}"
        print(f"  {name:14s} {shown}")

    print("\nчто система теперь о себе знает:")
    for key, cap in sorted(model.capabilities().items()):
        if cap.band == "all":
            continue
        lo, hi = cap.confidence_interval
        print(f"  {key:24s} {cap.score:.2f} [{lo:.2f}-{hi:.2f}]  n={cap.observations}")

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nотчёт: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
