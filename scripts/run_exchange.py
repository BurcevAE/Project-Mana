#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_exchange.py — выгрузить и загрузить пакет обмена.

Мост между установками, без сервера: два файла и две команды. Этого
достаточно, чтобы две MANA начали делиться гипотезами, и достаточно, чтобы
проверить, что обмен делает то, что обещает, — а сервер к этому ничего не
добавляет, кроме того, что его надо где-то держать.

Через мост едут ГИПОТЕЗЫ, а не вердикты. Импорт кладёт чужие гипотезы в
очередь; принимает их по-прежнему только локальный приёмочный аппарат, на
локальной скрытой выборке. Поэтому испорченный или враждебный экземпляр
ничего сломать не может: плохая гипотеза не проходит ворота, как любая
другая.

    python scripts/run_exchange.py --show
    python scripts/run_exchange.py --export обмен.json
    python scripts/run_exchange.py --import обмен-от-коллеги.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mana.cognition import exchange              # noqa: E402
from mana.core import identity                   # noqa: E402


def queue_path() -> Path:
    from mana import paths
    return Path(paths.data_root()) / "exchange_queue.json"


def load_queue() -> dict:
    path = queue_path()
    if not path.is_file():
        return {"hypotheses": [], "reports": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_queue(queue: dict) -> None:
    path = queue_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(queue, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def show(queue: dict) -> int:
    print(f"экземпляр: {identity.fingerprint()}")
    print(f"очередь:   {queue_path()}")
    print()
    print(f"гипотез: {len(queue['hypotheses'])}, отчётов: {len(queue['reports'])}")

    reports = []
    for record in queue["reports"]:
        try:
            reports.append(exchange.Report(**record))
        except Exception:
            continue
    if not reports:
        print()
        print("воспроизведений пока нет: отчётов ни от кого не поступало")
        return 0

    print()
    print("что где воспроизвелось:")
    print(f"  {'гипотеза':18s} {'сред':>5} {'судили':>7} {'принято':>8} "
          f"{'отклонено':>10} {'без силы':>9} {'выборок':>8}")
    for hypothesis_id, row in sorted(exchange.replication(reports).items()):
        print(f"  {hypothesis_id:18s} {row['instances']:5d} {row['ruled']:7d} "
              f"{row['accepted']:8d} {row['rejected']:10d} "
              f"{row['not_evaluated']:9d} {row['distinct_holdouts']:8d}")
    print()
    print("«без силы» — NOT_EVALUATED: там не хватило наблюдений, чтобы судить.")
    print("Это не несогласие, и делить надо на «судили», а не на «сред».")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", metavar="ФАЙЛ")
    parser.add_argument("--import", dest="import_", metavar="ФАЙЛ")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    queue = load_queue()

    if args.import_:
        known = {h.get("hypothesis_id") for h in queue["hypotheses"]}
        try:
            result = exchange.import_bundle(args.import_, known=known)
        except exchange.ExchangeError as exc:
            print(f"пакет не принят: {exc}")
            return 1
        for hypothesis in result.hypotheses:
            queue["hypotheses"].append(hypothesis.as_dict())
        for report in result.reports:
            queue["reports"].append(report.as_dict())
        save_queue(queue)

        print(f"от экземпляра {result.instance or '(не указан)'}:")
        print(f"  новых гипотез: {len(result.hypotheses)}")
        print(f"  отчётов:       {len(result.reports)}")
        if result.refused:
            print(f"  отклонено:     {len(result.refused)}")
            for refusal in result.refused[:5]:
                print(f"    [{refusal['kind']}] {refusal['reason'][:90]}")
        print()
        print("Вердикты из пакета НЕ приняты как истина — они лишь показывают,")
        print("что у кого получилось. Принять гипотезу здесь может только")
        print("локальный эксперимент на локальной скрытой выборке.")
        return 0

    if args.export:
        hypotheses, reports = [], []
        for record in queue["hypotheses"]:
            try:
                hypotheses.append(exchange.Hypothesis(
                    mutation=record["mutation"], params=record.get("params") or {},
                    note=record.get("note", "")))
            except exchange.ExchangeError as exc:
                print(f"не выгружена гипотеза: {exc}")
        for record in queue["reports"]:
            try:
                reports.append(exchange.Report(**record))
            except Exception:
                pass
        written = exchange.export_bundle(args.export, hypotheses, reports)
        print(f"записано: {written['path']}")
        print(f"  гипотез: {written['hypotheses']}, отчётов: {written['reports']}")
        print(f"  от экземпляра: {written['instance']}")
        return 0

    return show(queue)


if __name__ == "__main__":
    raise SystemExit(main())
