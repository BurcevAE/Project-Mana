"""MANA goes through the cycle of a new explanation on one task, and its own
record is written out (docs/РОЖДЕНИЕ_ГИПОТЕЗ.md, 13). Not a comparison and
not a metric: what it believed, what it doubted, what explanations were
born, which experiment it chose and why, what came of it.

    python -u -X utf8 scripts/run_one_task.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mana.cognition import explain, inquiry  # noqa: E402
from mana.world import device as dev  # noqa: E402

NAMES = ["s0", "s1", "s2"]
TABLE = tuple(sum(bool(i >> k & 1) for k in range(3)) >= 2 for i in range(8))
START = [(True, True, False), (False, False, False)]
BUDGET = 40


def bits(config):
    return "".join("1" if v else "0" for v in config)


def main() -> None:
    device = dev.Device(3, {"b": dev.Rule("table", table=TABLE)}, noise=0.0, seed=0)
    knowledge = dev.knowledge_for(device)
    h = knowledge["b"]
    probes = dev.DeviceProbes(device)
    events = []

    for config in START:
        outcome = device.rules["b"].holds(config)
        h.update(probes.params("b", config), outcome)
        events.append(f"дано наблюдение: {bits(config)} → {'работает' if outcome else 'нет'}")

    inner = explain.from_history(NAMES)
    counts = {"other": 0, "doubt": 0}

    def hook(hs):
        why = ("ни одно объяснение не подходит (OTHER лидирует)"
               if hs.explained > counts["other"] else "сомнение перед принятием лидера")
        counts["other"], counts["doubt"] = hs.explained, hs.doubted
        leader, weight = hs.leader()
        born, _ = inner(hs)
        events.append(f"\n  ▸ спрашивает discovery: {why}; лидер сейчас «{leader}» ({weight:.2f}); "
                      f"наблюдений {len(hs.history)}")
        for b in born:
            row = "".join("1" if b.predict(probes.params("b", c))[True] > 0.5 else "0"
                          for c in device.reachable())
            events.append(f"      родилось: {b.name}   [предсказания на 000…111: {row}]")
        if not born:
            events.append("      родилось: ничего точного на этой истории")
        return born, _

    class Logged(dev.DeviceProbes):
        def act(self, spec):
            before = h.leader()
            outcome = super().act(spec)
            events.append(f"  проба {bits(spec.params['config'])} (цена {spec.cost}): "
                          f"{'работает' if outcome else 'нет'}   — перед ней лидер «{before[0]}» "
                          f"({before[1]:.2f})")
            return outcome

    report = inquiry.inquire(lambda: inquiry.unsettled(knowledge), Logged(device), BUDGET,
                             challenge=dev.challenge_for(device), explain=hook, doubt=True)

    print("=== Задача: кнопка работает, когда включены хотя бы два из трёх переключателей")
    print("    (закон MANA не известен; оценка — только в конце)\n")
    step = 0
    for line in events:
        print(line)
    print("\n=== Выбор проб и что он давал")
    for st in report.steps:
        step += 1
        print(f"  {step}. {st.action}, цена {st.cost}, ожидалось бит {st.gain:.2f}, "
              f"исход {'работает' if st.outcome else 'нет'}, лидер после «{st.leader}» ({st.weight:.2f})")
    print(f"\n=== Итог: {report.stopped}, потрачено действий {device.actions}")
    print(f"  ответ: {h.settled}")
    print(f"  вызовов discovery: по OTHER {h.explained}, сомнений {h.doubted}, "
          f"сомнение открыло вопрос {h.doubt_reopened} раз")
    seen = {tuple(p['config']) for p, _ in h.history}
    print(f"  увидено конфигураций {len(seen)} из 8")
    print(f"  оценка по скрытому закону: {dev.verdicts(device, knowledge)['b']}")


if __name__ == "__main__":
    main()
