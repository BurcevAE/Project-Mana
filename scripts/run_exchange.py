#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_exchange.py — выгрузить и загрузить пакет обмена.

Обёртка над `mana.cognition.exchange.command_line`. Сама команда живёт в
модуле, потому что она нужна и установленному приложению: у собранной MANA
не было способа ни выгрузить, ни загрузить пакет, и всякая установка, не
являющаяся заодно клоном репозитория, не могла участвовать в обмене.

Через мост едут ГИПОТЕЗЫ, а не вердикты. Импорт кладёт чужие гипотезы в
очередь; принять их может только локальный эксперимент.

    python scripts/run_exchange.py
    python scripts/run_exchange.py export пакет.json
    python scripts/run_exchange.py import пакет-от-коллеги.json

То же самое из установленного приложения:

    MANA.exe --exchange show
    MANA.exe --exchange export пакет.json
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mana.cognition.exchange import command_line   # noqa: E402

if __name__ == "__main__":
    raise SystemExit(command_line(sys.argv[1:]))
