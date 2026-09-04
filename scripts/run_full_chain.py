#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_full_chain.py — the whole chain against live brains, with
nothing directing it between the steps.

Weakness → hypothesis → experiment → discovery → independent confirmation
on fresh tasks → a capability installed in the genome → the compiler
choosing it for the next task of that kind.

The script supplies three things and no decisions: a way to run an
operator chain through the brain pool, a way to score a chain against the
hidden holdout, and a counterexample search. Which slice to attack, what
to hypothesise, whether the evidence carries -- none of that is here.

Usage:
    python scripts/run_full_chain.py --budget 900 --steps 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mana import Config                                     # noqa: E402
from mana.brains import BrainPool                           # noqa: E402
from mana.cognition.genome import CognitiveGenome           # noqa: E402
from mana.cognition.research import ResearchCycle           # noqa: E402
from mana.cognition.self_model import Observation, SelfModel  # noqa: E402
from mana.core import oracle, splits, tasks as core_tasks   # noqa: E402

FORMAT = ("Ответь только итоговым значением, без пояснений, без единиц "
          "измерения и без знаков препинания вокруг него.")

#: How each operator turns into a prompt. Not an interpreter -- the real
#: one lives in cognition/runtime.py and needs a full agent. This is the
#: smallest thing that makes a chain genuinely different from one call,
#: so that a measured difference is a difference in the chain.
STEP_PROMPT = {
    "CRITIQUE": "Проверь решение выше. Есть ли ошибка? Ответь: ошибка или верно.",
    "REPAIR": f"Учитывая замечание, дай окончательный ответ. {FORMAT}",
    "VERIFY": f"Пересчитай независимо и дай ответ. {FORMAT}",
    "DECOMPOSE": "Разбей на шаги и посчитай по шагам.",
    "SYNTHESIZE": f"Собери итог из шагов выше. {FORMAT}",
    "RETRIEVE": "Какое правило порядка операций здесь применимо? Кратко.",
}


def run_chain(pool: BrainPool, steps: Sequence[str], prompt: str,
              domain: str, difficulty: float) -> Tuple[str, int]:
    """Execute an operator chain and return (final answer, calls used)."""
    transcript = f"{prompt}\n\n{FORMAT}"
    answer = ""
    calls = 0
    for step in steps:
        if step in ("OBSERVE", "ANSWER"):
            continue
        instruction = STEP_PROMPT.get(step, "")
        ask = f"{transcript}\n\n{instruction}".strip() if instruction else transcript
        reply = pool.ask(ask, kind=domain, difficulty=difficulty,
                         context_tag="chain")
        calls += 1
        text = (reply.get("text") or "").strip()
        transcript = f"{ask}\n\n{text}"
        if step in ("GENERATE", "REPAIR", "VERIFY", "SYNTHESIZE"):
            answer = text or answer
    return answer, calls


def make_trial_runner(pool: BrainPool):
    def run(steps: Sequence[str], task: Any) -> Tuple[bool, int]:
        answer, calls = run_chain(pool, steps, task.prompt, task.domain,
                                  task.difficulty)
        return oracle.grade(task, answer).correct, calls
    return run


def make_hidden_fn(pool: BrainPool, per_domain: int, log: list):
    """Score one chain on the hidden holdout.

    The chain never sees a Task and the caller never sees the tasks --
    `hidden_score` hands out public dicts and returns only an aggregate.
    That is the whole reason there is no `hidden_tasks()` to import.
    """
    def score(steps: Sequence[str]) -> float:
        def answer_fn(public: Dict[str, Any]) -> str:
            text, _calls = run_chain(pool, steps, public["prompt"],
                                     public["domain"], public["difficulty"])
            return text
        result = splits.hidden_score(answer_fn, per_domain=per_domain,
                                     label="->".join(steps))
        log.append({"steps": list(steps), "score": round(result.accuracy, 3)})
        return result.accuracy
    return score


def make_counterexample_fn(pool: BrainPool, probes: int):
    """Look for where the mechanism breaks, in the adjacent band.

    A search that only ever probes where the effect was found cannot
    fail, which makes it decoration rather than evidence.
    """
    def search(hypothesis: Any) -> Tuple[int, int]:
        found = 0
        edge = core_tasks.generate(hypothesis.domain, probes, seed=31337,
                                   difficulty_range=(0.35, 0.65))
        for task in edge:
            base, _ = run_chain(pool, hypothesis.baseline_steps, task.prompt,
                                task.domain, task.difficulty)
            cand, _ = run_chain(pool, hypothesis.candidate_steps, task.prompt,
                                task.domain, task.difficulty)
            if oracle.grade(task, base).correct and not oracle.grade(task, cand).correct:
                found += 1          # the candidate broke what the baseline had
        return probes, found
    return search


def measure_starting_shape(pool: BrainPool, model: SelfModel, per_slice: int) -> None:
    """Give the cycle a system that already knows its own shape.

    Starting from nothing it would spend its first fifteen steps
    measuring -- the right order, but not what this script is here to
    show.
    """
    bands = (("easy", (0.0, 0.35)), ("medium", (0.35, 0.65)), ("hard", (0.65, 1.01)))
    # Every domain, not a couple: an untouched slice offers the widest
    # interval there is, so the cycle would rightly spend all its steps
    # measuring the ones this script skipped.
    for domain in core_tasks.DOMAINS:
        for _band, rng in bands:
            for task in core_tasks.generate(domain, per_slice, seed=555,
                                            difficulty_range=rng):
                answer, calls = run_chain(pool, ("OBSERVE", "GENERATE", "ANSWER"),
                                          task.prompt, task.domain, task.difficulty)
                graded = oracle.grade(task, answer)
                model.record(Observation(task.task_id, task.domain, task.difficulty,
                                         graded.correct,
                                         reason="" if graded.correct else graded.reason,
                                         calls=calls))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=900)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--per-slice", type=int, default=6)
    parser.add_argument("--hidden-per-domain", type=int, default=2)
    parser.add_argument("--probes", type=int, default=3)
    parser.add_argument("--out", default="mana_full_chain.json")
    args = parser.parse_args()

    pool = BrainPool(Config())
    available = pool.available()
    print(f"мозги: {', '.join(available) or 'нет'}")
    if not available:
        return 1

    model = SelfModel()
    started = time.perf_counter()
    print("\nизмеряю исходную форму...")
    measure_starting_shape(pool, model, args.per_slice)
    for key, cap in sorted(model.capabilities().items()):
        if cap.band != "all":
            lo, hi = cap.confidence_interval
            print(f"  {key:22s} {cap.score:.2f} [{lo:.2f}-{hi:.2f}]  n={cap.observations}")

    hidden_log: list = []
    base_genome = CognitiveGenome()
    cycle = ResearchCycle(
        model, budget_calls=args.budget, max_steps=args.steps, genome=base_genome,
        hidden_fn=make_hidden_fn(pool, args.hidden_per_domain, hidden_log),
        counterexample_fn=make_counterexample_fn(pool, args.probes))

    def lesson_runner(task):
        answer, calls = run_chain(pool, ("OBSERVE", "GENERATE", "ANSWER"),
                                  task.prompt, task.domain, task.difficulty)
        return oracle.grade(task, answer).correct, "pool", calls

    def task_source(domain, n):
        return core_tasks.generate(domain, n, seed=int(time.time()) % 90000,
                                   difficulty_range=(0.65, 1.01))

    print("\nзапускаю цикл...")
    report = cycle.run(lesson_runner, make_trial_runner(pool), task_source)
    elapsed = time.perf_counter() - started

    print(f"\nшагов: {report['steps']}  вызовов: {report['calls_used']}  "
          f"время: {elapsed:.0f} с")
    print(f"остановка: {report['stop_reason']}")
    for i, step in enumerate(report["history"], 1):
        print(f"  {i}. [{step['activity']}] {step['description']}")
        print(f"     → {step['outcome']}")

    if hidden_log:
        print("\nскрытая выборка (цепочка никогда не видела эти задачи):")
        for row in hidden_log:
            print(f"  {' → '.join(row['steps'])}: {row['score']:.2f}")

    adopted = report["adopted_capabilities"]
    print(f"\nпринятые способности: {', '.join(adopted) or 'ни одной'}")
    if adopted:
        from mana.cognition.compiler import Capabilities, compile_program
        from mana.cognition.programs import Budget
        task = core_tasks.generate("arithmetic", 1, seed=777,
                                   difficulty_range=(0.65, 1.01))[0]
        caps = Capabilities(brains=len(available), has_memory=True,
                            has_web=True, has_sandbox=True)
        before = compile_program(task.prompt, base_genome, caps, Budget(calls=12),
                                 difficulty=task.difficulty)
        after = compile_program(task.prompt, cycle.synthesizer.genome, caps,
                                Budget(calls=12), difficulty=task.difficulty)
        print("\nчто компилятор выбирает для сложной арифметики:")
        print(f"  было:  {before.template if before else '—'}")
        print(f"  стало: {after.template if after else '—'}")

    report["hidden_log"] = hidden_log
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nотчёт: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
