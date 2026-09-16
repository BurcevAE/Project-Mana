"""Gate for widening from the model's state (docs/РАСШИРЕНИЕ_ПРОСТРАНСТВА.md, 10):
without the estimate, the I and D rows of the doubt probe repeat field for
field. The device trials are checked by run_birth_gate.py.

    python -u -X utf8 scripts/run_widen_gate.py [workers] doubt_results.jsonl
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_birth as B  # noqa: E402


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    old = [json.loads(l) for l in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
           if l.strip()]
    keys = [(r["policy"], r["law"], r["seed"]) for r in old]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        now = list(pool.map(B.job, keys, chunksize=16))
    differ = [k for k, a, b in zip(keys, old, now) if a != b]
    print(f"строки I и D поле в поле: {len(keys) - len(differ)} из {len(keys)}")
    for k in differ[:10]:
        print(f"  расходится: {k}")


if __name__ == "__main__":
    main()
