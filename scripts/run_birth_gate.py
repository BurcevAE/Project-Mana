"""Gate of the hypothesis-birth joint (docs/РОЖДЕНИЕ_ГИПОТЕЗ.md, 4 and 10.5).

Without the explain hook nothing in inquiry may change. The record is taken
on the code before the joint is added, and checked after, field for field:
every trial of the device policies the inquiry experiment already runs --
counts, actions, to-settle, to-correct, verdicts -- and every step of the
report: what was pressed, at what cost, with what gain, what came of it,
who led.

    python -u -X utf8 scripts/run_birth_gate.py записать|сверить [path.json]
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mana.cognition import inquiry  # noqa: E402
from mana.world import device as dev  # noqa: E402

SEEDS = 20
BUDGET = 250
VARIANTS = {
    "A": dict(noise=0.0),
    "B": dict(noise=0.1),
    "C": dict(needs=2),
    "D": dict(needs=3),
    "E": dict(hidden_pair=True),
}
POLICIES = {
    "inquiry": dict(),
    "E-geometric": dict(model_check=True, vulnerability=inquiry.GEOMETRIC),
    "E-claims": dict(model_check=True, vulnerability=inquiry.CLAIMS),
    "E-claimlevel": dict(model_check=True, vulnerability=inquiry.CLAIMS, claim_level=True),
    "claims-economy": dict(stakes=inquiry.Stakes(error_cost=12.0), boundary=inquiry.BOUNDARY_CLAIMS),
}


def _plain(value):
    return json.loads(json.dumps(value, default=lambda v: list(v) if isinstance(v, (tuple, set))
                                 else repr(v)))


def record():
    out = {}
    for variant, options in VARIANTS.items():
        for seed in range(SEEDS):
            key = f"{variant}/{seed}/enumeration"
            out[key] = _plain(asdict(dev.run_enumeration(dev.random_device(seed, **options),
                                                         BUDGET)))
            for name, kwargs in POLICIES.items():
                trial, report = dev.run_inquiry(dev.random_device(seed, **options), BUDGET,
                                                **kwargs)
                out[f"{variant}/{seed}/{name}"] = _plain({
                    "trial": asdict(trial), "spent": report.spent, "stopped": report.stopped,
                    "answers": report.answers, "open": report.open_questions,
                    "steps": [asdict(step) for step in report.steps]})
    return out


def main() -> None:
    mode = sys.argv[1]
    path = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "birth_gate.json")
    now = record()
    if mode == "записать":
        path.write_text(json.dumps(now, ensure_ascii=False), encoding="utf-8")
        print(f"записано испытаний: {len(now)} -> {path}")
        return
    before = json.loads(path.read_text(encoding="utf-8"))
    differ = [k for k in before if before[k] != now.get(k)]
    print(f"совпали поле в поле: {len(before) - len(differ)} из {len(before)}")
    for key in differ[:20]:
        print(f"  расходится: {key}")


if __name__ == "__main__":
    main()
