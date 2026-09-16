"""Can MANA find by itself that its space of explanations must widen?
(docs/РАСШИРЕНИЕ_ПРОСТРАНСТВА.md, 6 and 10.) Two declared tasks, no comparison,
and no external doubt flag anywhere.

    1  two of three -- does doubt arise from the model's own state, and does
       that state call discovery? If not, the experiment does not count.
    2  depends on s1 -- a law inside the family: is the space widened when
       the right explanation is already there?

    python -u -X utf8 scripts/run_self_doubt.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mana.cognition import doubt, explain, inquiry  # noqa: E402
from mana.world import device as dev  # noqa: E402

NAMES = ["s0", "s1", "s2"]
START = [(True, True, False), (False, False, False)]
BUDGET = 40
TASKS = [
    ("1. хотя бы два из трёх", lambda i: sum(bool(i >> k & 1) for k in range(3)) >= 2),
    ("2. зависит от s1", lambda i: bool(i >> 1 & 1)),
]


def bits(config):
    return "".join("1" if v else "0" for v in config)


def run(title, law):
    table = tuple(law(i) for i in range(8))
    device = dev.Device(3, {"b": dev.Rule("table", table=table)}, noise=0.0, seed=0)
    knowledge = dev.knowledge_for(device)
    h = knowledge["b"]
    probes = dev.DeviceProbes(device)
    for c in START:
        h.update(probes.params("b", c), device.rules["b"].holds(c))
    events = []
    inner = explain.from_history(NAMES)
    counters = {"other": 0, "widened": 0}
    by_state = []

    def hook(hs):
        if hs.widened > counters["widened"]:
            why = "состояние модели: уверенности над языком недостаточно"
            by_state.append(len(hs.history))
        else:
            why = "ни одно объяснение не подходит (OTHER)"
        counters["other"], counters["widened"] = hs.explained, hs.widened
        born, mass = inner(hs)
        events.append(f"  ▸ discovery вызван: {why}; наблюдений {len(hs.history)}; "
                      f"родилось {len(born)}")
        return born, mass

    class Logged(dev.DeviceProbes):
        def act(self, spec):
            before = h.leader()
            outcome = super().act(spec)
            events.append(f"  проба {bits(spec.params['config'])} (цена {spec.cost}): "
                          f"{'работает' if outcome else 'нет'} — перед ней лидер «{before[0]}» "
                          f"({before[1]:.2f})")
            return outcome

    report = inquiry.inquire(lambda: inquiry.unsettled(knowledge), Logged(device), BUDGET,
                             challenge=dev.challenge_for(device), explain=hook,
                             language=doubt.over_language(NAMES))
    print(f"\n=== Задача {title}")
    print("  дано: " + "; ".join(f"{bits(c)} → {'работает' if device.rules['b'].holds(c) else 'нет'}"
                                 for c in START))
    for line in events:
        print(line)
    print("  оценки уверенности над языком (в момент принятия):")
    for grown, leader, est in h.space_estimates:
        belief = "—" if est.belief is None else f"{est.belief:.3f}"
        print(f"    наблюдений {grown}, лидер «{leader}»: над языком {belief} "
              f"[{est.low:.3f}; {est.high:.3f}], согласных программ {est.consistent} из "
              f"{est.drawn}, {'достаточно' if est.enough else 'НЕДОСТАТОЧНО'}"
              f"{'' if est.decided else ' (не решено по вычислениям)'}, "
              f"непредставленных поведений {len(est.rivals)}")
    print(f"  итог: {report.stopped}; ответ {h.settled}; действий {device.actions}")
    print(f"  discovery по OTHER {h.explained}, расширений по состоянию {h.widened}; "
          f"«языка недостаточно»: {h.language_short}")
    print(f"  оценка по скрытому закону: {dev.verdicts(device, knowledge)['b']}")
    return by_state, h


def main() -> None:
    by_state, _ = run(*TASKS[0])
    counted = bool(by_state)
    print(f"\n  КОНТРАКТ: discovery вызван состоянием модели на «2 из 3» — "
          f"{'да, эксперимент засчитан' if counted else 'нет, эксперимент НЕ засчитан'}")
    run(*TASKS[1])


if __name__ == "__main__":
    main()
