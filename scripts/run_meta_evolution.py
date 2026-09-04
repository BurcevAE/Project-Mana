#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_meta_evolution.py — run the meta layer against real brains
and let it say what it actually costs.

This script is not expected to reach a meta-conclusion. An episode is a
whole search run and the gate wants `MIN_PAIRED_TRIALS` of them per arm;
at a few hundred calls each that is tens of thousands of calls. What it
does show, on real episodes rather than on an estimate, is the cost per
episode and the gate refusing to conclude from too few of them -- which
is the honest state of this layer, and more useful than a demonstration
tuned until it passed.

Usage:
    python scripts/run_meta_evolution.py --seeds 4 --episode-budget 60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mana import Config                                        # noqa: E402
from mana.brains import BrainPool                              # noqa: E402
from mana.cognition.meta import (EPISODE_BAR, EpisodeResult,    # noqa: E402
                                 MetaEvolution, judge, run_episodes,
                                 yield_report)
from mana.cognition.research import ResearchCycle              # noqa: E402
from mana.cognition.self_model import Observation, SelfModel    # noqa: E402
from mana.core import gates, oracle, tasks as core_tasks       # noqa: E402

FORMAT = ("Ответь только итоговым значением, без пояснений, без единиц "
          "измерения и без знаков препинания вокруг него.")


def make_episode_runner(pool: BrainPool, budget: int, steps: int):
    """One episode: a seeded research run under one search policy.

    The policy is applied by writing the weight into the module the
    search reads it from -- which is the only way to test a weight that
    is actually in force, and is why this script restores it afterwards.
    """
    from mana.cognition import gaps

    def run(seed: int, policy) -> EpisodeResult:
        original = dict(gaps.PRIORITY_WEIGHTS)
        for name, value in policy.items():
            key = name.split(".", 1)[1]
            if key in gaps.PRIORITY_WEIGHTS:
                gaps.PRIORITY_WEIGHTS[key] = value
        try:
            model = SelfModel()
            for task in core_tasks.generate("arithmetic", 6, seed=seed):
                reply = pool.ask(f"{task.prompt}\n\n{FORMAT}", kind=task.domain,
                                 difficulty=task.difficulty, context_tag="meta")
                graded = oracle.grade(task, reply.get("text") or "")
                model.record(Observation(task.task_id, task.domain, task.difficulty,
                                         graded.correct,
                                         reason="" if graded.correct else graded.reason,
                                         calls=1))
            cycle = ResearchCycle(model, budget_calls=budget, max_steps=steps)

            def lesson_runner(task):
                reply = pool.ask(f"{task.prompt}\n\n{FORMAT}", kind=task.domain,
                                 difficulty=task.difficulty, context_tag="meta")
                return oracle.grade(task, reply.get("text") or "").correct, "pool", 1

            report = cycle.run(lesson_runner)
            return EpisodeResult(
                seed=seed, policy_id=str(sorted(policy.items())),
                resolved=float(report["total_resolution"]),
                capability_gain=0.0,
                calls_used=int(report["calls_used"]) + 6,
                accepted_claims=len(report["adopted_capabilities"]))
        finally:
            gaps.PRIORITY_WEIGHTS.clear()
            gaps.PRIORITY_WEIGHTS.update(original)
    return run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--episode-budget", type=int, default=60)
    parser.add_argument("--episode-steps", type=int, default=3)
    parser.add_argument("--out", default="mana_meta.json")
    args = parser.parse_args()

    pool = BrainPool(Config())
    if not pool.available():
        print("нечего запускать: ни один мозг не доступен")
        return 1
    print(f"мозги: {', '.join(pool.available())}")

    m = MetaEvolution()
    print(f"\nнастраиваемых параметров: {len(m.parameters)}")
    for name, value in sorted(m.policy().items()):
        print(f"  {name:32s} {value:+.3f}")

    print("\nчего мета-слой коснуться не может:")
    from mana.cognition.meta import MetaError, MetaParameter, check_tunable
    for name, module, key in (("gates.alpha", "mana.core.gates", "ALPHA"),
                              ("oracle", "mana.core.oracle", "grade"),
                              ("curriculum.pass_bar", "mana.cognition.curriculum",
                               "PASS_BAR"),
                              ("meta.bar", "mana.cognition.meta", "EPISODE_BAR")):
        try:
            check_tunable(MetaParameter(name, module, key, 0.0, 0.0, 1.0, ""))
            print(f"  {name}: ПРОПУЩЕН — это дефект")
        except MetaError as exc:
            print(f"  {name}: отказ — {exc}")

    target = "gap.cost"
    proposal = m.propose(target, m.policy()[target] - 0.4,
                         "меньше штраф за стоимость — глубже поиск")
    print(f"\nпредложение: {proposal.describe()}")

    seeds = list(range(1, args.seeds + 1))
    print(f"гоняю {len(seeds)} пар эпизодов "
          f"(гейту нужно {gates.MIN_PAIRED_TRIALS})...")
    started = time.perf_counter()
    run_episodes(proposal, seeds,
                 make_episode_runner(pool, args.episode_budget, args.episode_steps))
    verdict = judge(proposal, hidden=None, counterexamples=(0, 0))
    elapsed = time.perf_counter() - started

    report = yield_report(proposal)
    per_episode = report["calls"] / max(1, 2 * len(seeds))
    needed = 2 * gates.MIN_PAIRED_TRIALS * per_episode

    print(f"\nэпизодов: {report['episodes']} пар, время {elapsed:.0f} с")
    print(f"  базовая политика:  ценность {report['baseline_value']:.3f}")
    print(f"  кандидат:          ценность {report['candidate_value']:.3f}")
    print(f"  вызовов потрачено: {report['calls']} ({per_episode:.0f} на эпизод)")
    print(f"\nвердикт: {'ПРИНЯТО' if verdict.accepted else 'ОТКЛОНЕНО'} — {verdict.reason}")
    print(f"не прошли гейты: {', '.join(verdict.failed_gates) or '—'}")
    print(f"\nстоимость одного мета-вывода при этой цене эпизода: "
          f"≈{needed:,.0f} вызовов")

    payload = {"policy": m.policy(), "proposal": proposal.as_dict(),
               "yield": report, "calls_per_episode": per_episode,
               "calls_for_one_conclusion": needed}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nотчёт: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
