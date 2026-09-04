#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_acceptance.py — приёмочный эксперимент §37: полная цепочка
с НАСТОЯЩИМ перезапуском процесса.

Ключевое слово — «настоящим». Проверка в одном процессе не доказывает
ничего о перезапуске: объект остаётся в памяти, и «способность жива»
означает только «переменная не обнулилась». Поэтому скрипт работает в
два прохода, и второй запускается отдельным процессом:

    --phase adopt    измерить слабость, доказать механизм, принять,
                     сохранить геном и выйти
    --phase verify   стартовать с нуля, загрузить геном с диска и
                     проверить, что способность на месте И РАБОТАЕТ

Второе условие важнее первого. Найти запись в файле — это не
способность; способность — это когда система после перезапуска решает
задачу тем механизмом, который приняла.

Usage:
    python scripts/run_acceptance.py --phase adopt
    python scripts/run_acceptance.py --phase verify
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
from mana.cognition import brain_factory as bf                   # noqa: E402
from mana.cognition import genome as genome_mod                  # noqa: E402
from mana.cognition.brain_factory import BrainCandidate, MechanismChoice  # noqa: E402
from mana.cognition.genome import CognitiveGenome                # noqa: E402
from mana.core import gates, oracle, splits, tasks as core_tasks  # noqa: E402
from mana.core.cost import CostVector                            # noqa: E402
from mana.core.gates import PairedOutcome                        # noqa: E402

FORMAT = ("Ответь только итоговым значением, без пояснений, без единиц "
          "измерения и без знаков препинания вокруг него.")

#: Срез, на котором ведётся эксперимент. Арифметика выбрана потому, что
#: её решатель — единственный, переживший все три поверхности; принимать
#: то, что уже известно как привязанное к разметке, было бы принятием
#: заведомо непереносимого.
DOMAIN = "arithmetic"
BRAIN = "arithmetic"


def _cost(reply: Dict[str, Any]) -> CostVector:
    raw = reply.get("cost") or {}
    return CostVector(
        calls=raw.get("calls", 0), wall_seconds=raw.get("wall_seconds", 0.0),
        tokens_in=raw.get("tokens_in", 0), tokens_out=raw.get("tokens_out", 0),
        unmeasured_token_calls=raw.get("unmeasured_token_calls", 0),
        by_substrate=raw.get("by_substrate", {}))


def ask_model(pool: BrainPool, prompt: str, difficulty: float):
    reply = pool.ask(prompt + "\n\n" + FORMAT, kind=DOMAIN, difficulty=difficulty,
                     context_tag="baseline", exclude_substrates=("algorithmic",))
    return (reply.get("text") or ""), _cost(reply)


def ask_cascade(pool: BrainPool, prompt: str, difficulty: float):
    reply = pool.ask(prompt + "\n\n" + FORMAT, kind=DOMAIN, difficulty=difficulty,
                     context_tag="cascade")
    return (reply.get("text") or ""), _cost(reply)


def adopt_phase(pool: BrainPool, path: Path, trials: int, hidden_per: int,
                probes: int) -> int:
    print(f"ПРОХОД 1 — измерить, доказать, принять, сохранить\n")

    tasks = core_tasks.generate(DOMAIN, trials, seed=int(time.time()) % 90000)
    outcomes: List[PairedOutcome] = []
    base_cost = CostVector()
    cand_cost = CostVector()
    for task in tasks:
        base_text, bc = ask_model(pool, task.prompt, task.difficulty)
        cand_text, cc = ask_cascade(pool, task.prompt, task.difficulty)
        base_cost, cand_cost = base_cost.add(bc), cand_cost.add(cc)
        outcomes.append(PairedOutcome(
            task_id=task.task_id, domain=task.domain,
            baseline_correct=oracle.grade(task, base_text).correct,
            candidate_correct=oracle.grade(task, cand_text).correct))

    base_acc = gates.accuracy(outcomes, "baseline")
    cand_acc = gates.accuracy(outcomes, "candidate")
    print(f"  парно на {len(outcomes)} задачах: модель {base_acc:.2f}, "
          f"каскад {cand_acc:.2f}")

    holdout = splits.HOLDOUT_V1
    base_hidden = splits.hidden_score(
        lambda p: ask_model(pool, p["prompt"], p["difficulty"])[0],
        per_domain=hidden_per, label="acc-baseline", holdout=holdout)
    cand_hidden = splits.hidden_score(
        lambda p: ask_cascade(pool, p["prompt"], p["difficulty"])[0],
        per_domain=hidden_per, label="acc-cascade", holdout=holdout)
    print(f"  скрытая {holdout.name}, домен заявки: "
          f"{base_hidden.strict_by_domain.get(DOMAIN, 0):.2f} -> "
          f"{cand_hidden.strict_by_domain.get(DOMAIN, 0):.2f}")

    sought = found = 0
    for domain in ("logic", "text_ops"):
        for task in core_tasks.generate(domain, probes, seed=31337):
            sought += 1
            reply = pool.ask_brain(BRAIN, task.prompt)
            text = reply.get("text") or ""
            if text and not oracle.grade(task, text).correct:
                found += 1
    print(f"  контрпримеры: {found} из {sought}")

    candidate = BrainCandidate(
        brain_id=BRAIN, substrate=bf.ALGORITHMIC, domain=DOMAIN, band="hard",
        mechanism=bf.choose_mechanism(DOMAIN, exactly_computable=True),
        measured_cost=cand_cost.as_dict())
    verdict = bf.evaluate(candidate, outcomes,
                          baseline_hidden=base_hidden, candidate_hidden=cand_hidden,
                          counterexamples=(sought, found),
                          cost=base_cost.add(cand_cost))
    print(f"\n  вердикт гейтов: {verdict.status} — {verdict.reason}")
    if not verdict.accepted:
        print("  принимать нечего; геном не тронут")
        return 1

    before = CognitiveGenome()
    adopted = bf.adopt(candidate, verdict, before)
    genome_mod.save(adopted, path)
    print(f"\n  принято и сохранено: {path}")
    print(f"  геном {adopted.genome_id}, мозгов {len(adopted.brains)}, "
          f"родитель {adopted.parent_id}")
    print(f"\nТеперь запустите ОТДЕЛЬНЫМ процессом:")
    print(f"  python scripts/run_acceptance.py --phase verify")
    return 0


def verify_phase(pool: BrainPool, path: Path, checks: int) -> int:
    print("ПРОХОД 2 — новый процесс, ничего не помнит\n")

    restored, note = genome_mod.load_report(path)
    print(f"  загрузка: {note}")
    if restored is None:
        print("  ПРОВАЛ: после перезапуска генома нет")
        return 1

    gene = restored.brains.get(BRAIN)
    if gene is None:
        print(f"  ПРОВАЛ: мозг {BRAIN} не пережил перезапуск")
        return 1
    print(f"  мозг на месте: {gene.brain_id} / {gene.substrate}, "
          f"применимость {gene.applicability}, родословная {gene.parent_ids or '—'}")
    print(f"  родословная генома: {restored.genome_id} <- {restored.parent_id} "
          f"({restored.mutation})")

    # Запись в файле — ещё не способность. Способность — это когда
    # система РЕШАЕТ задачу тем механизмом, который приняла.
    right = wrong = 0
    spent = CostVector()
    for task in core_tasks.generate(DOMAIN, checks, seed=4242):
        reply = pool.ask(task.prompt + "\n\n" + FORMAT, kind=DOMAIN,
                         difficulty=task.difficulty, context_tag="verify")
        spent = spent.add(_cost(reply))
        if reply.get("brain") != BRAIN:
            print(f"  ПРОВАЛ: задачу взял {reply.get('brain')!r}, а не принятый мозг")
            return 1
        if oracle.grade(task, reply.get("text") or "").correct:
            right += 1
        else:
            wrong += 1

    print(f"\n  после перезапуска решено {right}/{checks}, неверно {wrong}")
    print(f"  стоимость: {spent.describe()}")
    if wrong:
        print("  ПРОВАЛ: способность на месте, но работает хуже, чем при приёмке")
        return 1
    print("\n  ПРОЙДЕНО: способность пережила перезапуск и работает")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("adopt", "verify"), required=True)
    parser.add_argument("--genome", default="mana_state/genome.json")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--hidden-per-domain", type=int, default=2)
    parser.add_argument("--probes", type=int, default=4)
    parser.add_argument("--checks", type=int, default=10)
    args = parser.parse_args()

    pool = BrainPool(Config())
    if args.phase == "adopt" and not pool.language_models():
        print("для базовой руки нужна хотя бы одна языковая модель")
        return 1

    path = Path(args.genome)
    return (adopt_phase(pool, path, args.trials, args.hidden_per_domain, args.probes)
            if args.phase == "adopt" else verify_phase(pool, path, args.checks))


if __name__ == "__main__":
    raise SystemExit(main())
