#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_algorithmic_brain.py — прототип G: первый не-LLM мозг,
принимаемый или отклоняемый теми же гейтами, что и всё остальное.

Ничего не обучается. Всё нужное уже существовало: точный вычислитель,
генератор задач, оракул, скрытая выборка, гейты. Прототип только
регистрирует вычислитель как мозг и просит ядро вынести вердикт.

Соблазн здесь очевиден: арифметический мозг на арифметике даст почти
идеальную точность, и это будет выглядеть триумфом. Само по себе это
ничего не доказывает. Доказывает то, что вокруг:

  * мозг обязан ОТКАЗЫВАТЬСЯ вне применимости, а не выдавать мусор;
  * на text_ops он НЕ должен давать улучшения;
  * заявленная применимость обязана совпасть с проверенными срезами.

Поэтому поиск контрпримеров ищет именно то, что убило бы идею: случай,
когда мозг вернул уверенный НЕВЕРНЫЙ ответ вместо отказа.

Usage:
    python scripts/run_algorithmic_brain.py --trials 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mana import Config                                          # noqa: E402
from mana.brains import BrainPool                                # noqa: E402
from mana.core import gates, oracle, splits, tasks as core_tasks  # noqa: E402
from mana.core.cost import CostVector, efficiency                # noqa: E402
from mana.core.gates import Claim, Evidence, PairedOutcome       # noqa: E402

FORMAT = ("Ответь только итоговым значением, без пояснений, без единиц "
          "измерения и без знаков препинания вокруг него.")

#: Домены, на которых мозг НЕ должен помогать. Проверка того, что
#: измеряется механизм, а не удача: улучшение здесь означало бы, что
#: измеряется что-то другое.
FOREIGN_DOMAINS = ("text_ops", "logic")


def ask(pool: BrainPool, brain: str, prompt: str, domain: str,
        difficulty: float) -> Tuple[str, CostVector]:
    reply = pool.ask_brain(brain, f"{prompt}\n\n{FORMAT}") if brain else {}
    cost = reply.get("cost") or {}
    return (reply.get("text") or ""), CostVector(
        calls=cost.get("calls", 0), wall_seconds=cost.get("wall_seconds", 0.0),
        tokens_in=cost.get("tokens_in", 0), tokens_out=cost.get("tokens_out", 0),
        unmeasured_token_calls=cost.get("unmeasured_token_calls", 0),
        by_substrate=cost.get("by_substrate", {}))


def ask_model(pool: BrainPool, prompt: str, domain: str,
              difficulty: float) -> Tuple[str, CostVector]:
    reply = pool.ask(f"{prompt}\n\n{FORMAT}", kind=domain, difficulty=difficulty,
                     context_tag="baseline",
                     exclude_substrates=("algorithmic",))
    cost = reply.get("cost") or {}
    return (reply.get("text") or ""), CostVector(
        calls=cost.get("calls", 0), wall_seconds=cost.get("wall_seconds", 0.0),
        tokens_in=cost.get("tokens_in", 0), tokens_out=cost.get("tokens_out", 0),
        unmeasured_token_calls=cost.get("unmeasured_token_calls", 0),
        by_substrate=cost.get("by_substrate", {}))


def paired_run(pool: BrainPool, tasks, log: Dict[str, Any]):
    """Обе руки на ОДНИХ задачах. Пары — то, что читает McNemar."""
    outcomes: List[PairedOutcome] = []
    base_cost = CostVector()
    cand_cost = CostVector()
    refusals = 0
    for task in tasks:
        base_text, bc = ask_model(pool, task.prompt, task.domain, task.difficulty)
        cand_text, cc = ask(pool, "arithmetic", task.prompt, task.domain, task.difficulty)
        base_cost = base_cost.add(bc)
        cand_cost = cand_cost.add(cc)
        if not cand_text:
            refusals += 1
        outcomes.append(PairedOutcome(
            task_id=task.task_id, domain=task.domain,
            baseline_correct=oracle.grade(task, base_text).correct,
            candidate_correct=oracle.grade(task, cand_text).correct))
    log["refusals_in_domain"] = refusals
    return outcomes, base_cost, cand_cost


def hidden_arms(pool: BrainPool, per_domain: int):
    """Скрытая выборка для обеих рук. Задачи наружу не выходят."""
    def model_answer(public: Dict[str, Any]) -> str:
        text, _ = ask_model(pool, public["prompt"], public["domain"],
                            public["difficulty"])
        return text

    def brain_answer(public: Dict[str, Any]) -> str:
        text, _ = ask(pool, "arithmetic", public["prompt"], public["domain"],
                      public["difficulty"])
        return text

    base = splits.hidden_score(model_answer, per_domain=per_domain, label="baseline")
    cand = splits.hidden_score(brain_answer, per_domain=per_domain, label="algorithmic")
    return base, cand


def counterexample_search(pool: BrainPool, probes: int) -> Tuple[int, int, List[str]]:
    """Ищем то, что убило бы идею: уверенный неверный ответ вместо отказа.

    Молчаливый отказ — это правильное поведение и НЕ контрпример.
    Контрпример — это когда мозг что-то ответил и ошибся.
    """
    sought = 0
    found = 0
    examples: List[str] = []
    for domain in FOREIGN_DOMAINS:
        for task in core_tasks.generate(domain, probes, seed=31337):
            sought += 1
            text, _ = ask(pool, "arithmetic", task.prompt, task.domain, task.difficulty)
            if not text:
                continue                       # отказался — так и надо
            if not oracle.grade(task, text).correct:
                found += 1
                examples.append(f"{task.task_id}: ответил {text!r} и ошибся")
    return sought, found, examples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--hidden-per-domain", type=int, default=3)
    parser.add_argument("--probes", type=int, default=6)
    parser.add_argument("--out", default="mana_algorithmic_brain.json")
    args = parser.parse_args()

    pool = BrainPool(Config())
    models = pool.language_models()
    print(f"мозги: {', '.join(pool.available())}")
    print(f"из них языковые модели: {', '.join(models) or 'нет'}")
    if not models:
        print("для базовой руки нужна хотя бы одна языковая модель")
        return 1

    log: Dict[str, Any] = {}
    started = time.perf_counter()

    tasks = core_tasks.generate("arithmetic", args.trials, seed=int(time.time()) % 90000)
    print(f"\nпарный прогон на {len(tasks)} задачах arithmetic...")
    outcomes, base_cost, cand_cost = paired_run(pool, tasks, log)

    print("скрытая выборка для обеих рук...")
    base_res, cand_res = hidden_arms(pool, args.hidden_per_domain)
    base_hidden, cand_hidden = base_res.accuracy, cand_res.accuracy

    print(f"поиск контрпримеров в {', '.join(FOREIGN_DOMAINS)}...")
    sought, found, examples = counterexample_search(pool, args.probes)

    claim = Claim(
        claim_id="algorithmic-arithmetic-brain", kind="program",
        description="алгоритмический мозг точнее языковой модели на арифметике",
        asserts_domains=("arithmetic",))
    evidence = Evidence(
        paired_dev=outcomes,
        baseline_hidden=base_hidden, candidate_hidden=cand_hidden,
        counterexamples_sought=sought, counterexamples_found=found,
        cost=base_cost.add(cand_cost))
    verdict = gates.judge(claim, evidence)
    elapsed = time.perf_counter() - started

    base_acc = gates.accuracy(outcomes, "baseline")
    cand_acc = gates.accuracy(outcomes, "candidate")

    print(f"\nпарный прогон, {len(outcomes)} задач, {elapsed:.0f} с")
    print(f"  языковая модель:      {base_acc:.2f}   {base_cost.describe()}")
    print(f"  алгоритмический мозг: {cand_acc:.2f}   {cand_cost.describe()}")
    print(f"  отказов внутри домена: {log['refusals_in_domain']}")

    print(f"\nскрытая выборка (задачи не видел никто):")
    print(f"  языковая модель:      {base_hidden:.2f}  "
          f"оценено {base_res.graded}, не оценено {base_res.ungradable}")
    print(f"  алгоритмический мозг: {cand_hidden:.2f}  "
          f"оценено {cand_res.graded}, не оценено {cand_res.ungradable}")
    print(f"  по доменам: модель {base_res.by_domain} | мозг {cand_res.by_domain}")
    if cand_res.graded != base_res.graded:
        # Обязательная оговорка, а не сноска. Отказ даёт пустой ответ,
        # оракул помечает его «не оценено», и он ВЫПАДАЕТ из знаменателя.
        # Отказывающийся мозг оценивается на меньшем числе задач, чем
        # модель, — и именно эти два числа сравнивает гейт
        # hidden_confirms, то есть сравнивает несравнимое.
        print(f"  ВНИМАНИЕ: знаменатели разные ({cand_res.graded} против "
              f"{base_res.graded}) — эти числа не сравнимы напрямую,")
        print(f"  а вердикт hidden_confirms опирается именно на их сравнение.")

    print(f"\nконтрпримеры: {found} из {sought} проб в чужих доменах")
    for example in examples[:5]:
        print(f"  {example}")
    if not found:
        print("  уверенных неверных ответов нет — вне применимости молчит")

    print(f"\nвердикт гейтов: {'ПРИНЯТО' if verdict.accepted else 'ОТКЛОНЕНО'}")
    print(f"  {verdict.reason}")
    if verdict.failed_gates:
        print(f"  не прошли: {', '.join(verdict.failed_gates)}")

    if cand_acc > base_acc:
        gain = cand_acc - base_acc
        print(f"\nприрост {gain:.3f} на единицу стоимости кандидата:")
        for name, value in efficiency(gain, cand_cost).items():
            shown = "не измерено" if value is None else f"{value:.5f}"
            print(f"  {name:14s} {shown}")

    payload = {"verdict": verdict.as_dict(),
               "baseline_accuracy": base_acc, "candidate_accuracy": cand_acc,
               "baseline_hidden": base_hidden, "candidate_hidden": cand_hidden,
               "hidden_detail": {"baseline": base_res.as_dict(),
                                 "candidate": cand_res.as_dict()},
               "counterexamples": {"sought": sought, "found": found,
                                   "examples": examples},
               "baseline_cost": base_cost.as_dict(),
               "candidate_cost": cand_cost.as_dict(),
               "refusals_in_domain": log["refusals_in_domain"],
               "elapsed_seconds": round(elapsed, 1)}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nотчёт: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
