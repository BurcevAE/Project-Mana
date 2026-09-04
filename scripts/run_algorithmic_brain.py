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
from mana.cognition.brain_factory import choose_mechanism        # noqa: E402

FORMAT = ("Ответь только итоговым значением, без пояснений, без единиц "
          "измерения и без знаков препинания вокруг него.")

#: Домены, на которых мозг НЕ должен помогать. Проверка того, что
#: измеряется механизм, а не удача: улучшение здесь означало бы, что
#: измеряется что-то другое.
#: Какой алгоритмический мозг отвечает за какой домен. Правило выбора
#: механизма (brain_factory.choose_mechanism) отвечает "algorithmic" для
#: всех четырёх: ответ в каждом из них вычислим точно.
DOMAIN_BRAIN = {"arithmetic": "arithmetic", "sequence": "sequence-solver",
                "text_ops": "text-ops", "logic": "order-logic"}


def ask(pool: BrainPool, brain: str, prompt: str, domain: str,
        difficulty: float) -> Tuple[str, CostVector]:
    reply = pool.ask_brain(brain, f"{prompt}\n\n{FORMAT}") if brain else {}
    cost = reply.get("cost") or {}
    return (reply.get("text") or ""), CostVector(
        calls=cost.get("calls", 0), wall_seconds=cost.get("wall_seconds", 0.0),
        tokens_in=cost.get("tokens_in", 0), tokens_out=cost.get("tokens_out", 0),
        unmeasured_token_calls=cost.get("unmeasured_token_calls", 0),
        by_substrate=cost.get("by_substrate", {}))


def ask_cascade(pool: BrainPool, prompt: str, domain: str,
                difficulty: float) -> Tuple[str, CostVector]:
    """Система как она будет работать: дешёвый субстрат первым, отказ
    проваливается к модели.

    Это и есть кандидат. Первая версия сравнивала голый мозг с моделью
    на ВСЕХ доменах, и гейт справедливо отклонил: такая замена
    уничтожает sequence, потому что мозг там просто молчит. Но никто не
    предлагает заменить модель мозгом везде — предлагается направить к
    нему арифметику. Сравнивать надо то, что предлагается.
    """
    reply = pool.ask(prompt + "\n\n" + FORMAT, kind=domain, difficulty=difficulty,
                     context_tag="cascade")
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
        cand_text, cc = ask_cascade(pool, task.prompt, task.domain, task.difficulty)
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

    def cascade_answer(public: Dict[str, Any]) -> str:
        text, _ = ask_cascade(pool, public["prompt"], public["domain"],
                              public["difficulty"])
        return text

    base = splits.hidden_score(model_answer, per_domain=per_domain, label="baseline")
    cand = splits.hidden_score(cascade_answer, per_domain=per_domain, label="cascade")
    return base, cand


def counterexample_search(pool: BrainPool, probes: int, brain: str,
                          foreign: Tuple[str, ...]) -> Tuple[int, int, List[str]]:
    """Ищем то, что убило бы идею: уверенный неверный ответ вместо отказа.

    Молчаливый отказ — это правильное поведение и НЕ контрпример.
    Контрпример — это когда мозг что-то ответил и ошибся.
    """
    sought = 0
    found = 0
    examples: List[str] = []
    for domain in foreign:
        for task in core_tasks.generate(domain, probes, seed=31337):
            sought += 1
            text, _ = ask(pool, brain, task.prompt, task.domain, task.difficulty)
            if not text:
                continue                       # отказался — так и надо
            if not oracle.grade(task, text).correct:
                found += 1
                examples.append(f"{task.task_id}: ответил {text!r} и ошибся")
    return sought, found, examples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="arithmetic",
                        choices=sorted(DOMAIN_BRAIN))
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

    domain = args.domain
    brain = DOMAIN_BRAIN[domain]
    foreign = tuple(d for d in DOMAIN_BRAIN if d != domain)[:2]
    tasks = core_tasks.generate(domain, args.trials, seed=int(time.time()) % 90000)
    print(f"\nмеханизм для {domain}: "
          f"{choose_mechanism(domain, exactly_computable=True).mechanism}")
    print(f"парный прогон на {len(tasks)} задачах {domain}...")
    outcomes, base_cost, cand_cost = paired_run(pool, tasks, log)

    print("скрытая выборка для обеих рук...")
    base_res, cand_res = hidden_arms(pool, args.hidden_per_domain)
    base_hidden, cand_hidden = base_res.strict_accuracy, cand_res.strict_accuracy

    print(f"поиск контрпримеров в {', '.join(foreign)}...")
    sought, found, examples = counterexample_search(pool, args.probes, brain, foreign)

    claim = Claim(
        claim_id=f"algorithmic-{brain}", kind="program",
        description=f"каскад с мозгом {brain} точнее одной модели на {domain}",
        asserts_domains=(domain,))
    # with_hidden берёт СТРОГУЮ точность и разбивку по доменам: гейт
    # сам сузит подтверждение до доменов заявки и проверит, не обвалился
    # ли какой-то другой. Передавать .accuracy руками — это и есть способ
    # сравнить отказывающийся мозг с угадывающей моделью по разным
    # знаменателям.
    evidence = Evidence(
        paired_dev=outcomes,
        counterexamples_sought=sought, counterexamples_found=found,
        cost=base_cost.add(cand_cost)).with_hidden(base_res, cand_res)
    verdict = gates.judge(claim, evidence)
    elapsed = time.perf_counter() - started

    base_acc = gates.accuracy(outcomes, "baseline")
    cand_acc = gates.accuracy(outcomes, "candidate")

    print(f"\nпарный прогон, {len(outcomes)} задач, {elapsed:.0f} с")
    print(f"  языковая модель:      {base_acc:.2f}   {base_cost.describe()}")
    print(f"  каскад с мозгом:      {cand_acc:.2f}   {cand_cost.describe()}")
    print(f"  отказов внутри домена: {log['refusals_in_domain']}")

    print(f"\nскрытая выборка (задачи не видел никто):")
    print(f"  языковая модель:      строго {base_hidden:.2f} "
          f"(мягко {base_res.accuracy:.2f}), из {base_res.attempted} задач")
    print(f"  каскад:               строго {cand_hidden:.2f} "
          f"(мягко {cand_res.accuracy:.2f}), из {cand_res.attempted} задач")
    print(f"  по доменам строго: модель {base_res.strict_by_domain}")
    print(f"                     мозг   {cand_res.strict_by_domain}")
    scope = verdict.measurements.get("hidden_scope")
    print(f"  гейт сузил подтверждение до: {scope}")
    if "hidden_scoped_margin" in verdict.measurements:
        print(f"  в доменах заявки: "
              f"{verdict.measurements['hidden_scoped_baseline']:.2f} -> "
              f"{verdict.measurements['hidden_scoped_candidate']:.2f}")
    if verdict.measurements.get("hidden_collapsed_domains"):
        print(f"  обвалились вне заявки: "
              f"{verdict.measurements['hidden_collapsed_domains']}")

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
