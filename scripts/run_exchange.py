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


def show(queue: exchange.Queue) -> int:
    stats = queue.stats()
    print(f"экземпляр: {identity.fingerprint()}")
    print(f"очередь:   {queue.path}")
    print()
    print(f"гипотез: {stats['hypotheses']} "
          f"(передаваемых {stats['shareable']}, "
          f"непередаваемых {stats['not_shareable']}), "
          f"отчётов: {stats['reports']}")

    reports = queue.reports()
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

    queue = exchange.Queue(queue_path())

    if args.import_:
        try:
            result = exchange.import_bundle(args.import_,
                                            known=queue.known_ids())
        except exchange.ExchangeError as exc:
            print(f"пакет не принят: {exc}")
            return 1
        queue.absorb(result)

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
        # shareable_only: a hypothesis that could not be vetted is still
        # kept locally, and must not leave by accident.
        hypotheses = queue.hypotheses(shareable_only=True)
        written = exchange.export_bundle(args.export, hypotheses,
                                         queue.reports())
        print(f"записано: {written['path']}")
        print(f"  гипотез: {written['hypotheses']}, отчётов: {written['reports']}")
        print(f"  от экземпляра: {written['instance']}")
        return 0

    return show(queue)


if __name__ == "__main__":
    raise SystemExit(main())
