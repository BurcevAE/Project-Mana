"""Birth of hypotheses: the inquiry-discovery joint (docs/РОЖДЕНИЕ_ГИПОТЕЗ.md).

Three switches, one button, every one of the 256 laws, 10 seeds each. The
law reaches neither inquiry nor discovery; only the scoring reads it.

    I  inquiry, probes by bits per cost, with the explain hook
    R  the same learner and hook, probes chosen uniformly among those offered
    N  inquiry without the hook
    P  passive discovery: observations in Gray-code order, discovery after
       each, no question to steer it

    python -u -X utf8 scripts/run_birth.py [workers] [results.jsonl]
"""
from __future__ import annotations

import itertools
import json
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mana.cognition import explain as EX  # noqa: E402
from mana.cognition import inquiry  # noqa: E402
from mana.discovery import policy as P  # noqa: E402
from mana.discovery.language import Evaluator  # noqa: E402
from mana.world import device as dev  # noqa: E402

LAWS = range(256)
SEEDS = range(10)
BUDGET = 40
INITIAL = 2
NAMES = ["s0", "s1", "s2"]
POLICIES = ("I", "R", "N", "P")


def table_of(law):
    return tuple(bool(law >> i & 1) for i in range(8))


def _index(config):
    return sum(1 << i for i, v in enumerate(config) if v)


def stratum(law) -> str:
    """S1 inside inquiry's base family; S2 one pairwise exception away; S3 the
    rest -- read off inquiry's own family, not off a judgement of difficulty."""
    table = table_of(law)
    base = [tuple(bool(i >> k & 1) for i in range(8)) for k in range(3)]
    base += [tuple([True] * 8), tuple([False] * 8)]
    if table in base:
        return "S1"
    for b in base:
        for d1, d2 in itertools.combinations(range(3), 2):
            for x, y in itertools.product((0, 1), repeat=2):
                flipped = tuple(v != ((i >> d1 & 1) == x and (i >> d2 & 1) == y)
                                for i, v in enumerate(b))
                if flipped == table:
                    return "S2"
    return "S3"


def _setup(law, seed):
    device = dev.Device(3, {"b": dev.Rule("table", table=table_of(law))}, noise=0.0, seed=seed)
    start = random.Random(1000 + seed).sample(device.reachable(), INITIAL)
    return device, start


def _behaves(predict, probes, device):
    return all((predict(probes.params("b", c)).get(True, 0.0) > 0.5)
               == device.rules["b"].holds(c) for c in device.reachable())


class _Recorder:
    """The hook, counting its calls and noting whether the first call's most
    likely explanation was already the law (success of kind A)."""

    def __init__(self, device):
        self.inner = EX.from_history(NAMES)
        self.device = device
        self.calls = 0
        self.first_right = None

    def __call__(self, hypotheses):
        born, mass = self.inner(hypotheses)
        self.calls += 1
        if self.calls == 1 and born:
            top = born[max(range(len(born)), key=lambda i: (mass[i], -i))]
            self.first_right = _behaves(top.predict, dev.DeviceProbes(self.device), self.device)
        return born, mass


class _Logged(dev._Watched):
    def __init__(self, device, knowledge, progress):
        super().__init__(device, knowledge, progress)
        self.configs = []
        self.probes = 0

    def act(self, spec):
        self.configs.append(tuple(spec.params["config"]))
        self.probes += 1
        return super().act(spec)


def _answer(device, knowledge):
    h = knowledge["b"]
    if not h.settled or h.settled[0] != inquiry.ANSWERED:
        return None
    members = [h.get(n) for n in h.settled[1]]
    return max((m for m in members if m is not None), key=lambda m: m.weight)


def _holdout(device, answer, observed):
    hidden = [c for c in device.reachable() if c not in observed]
    if answer is None:
        return len(hidden), None
    probes = dev.DeviceProbes(device)
    right = sum((answer.predict(probes.params("b", c)).get(True, 0.0) > 0.5)
                == device.rules["b"].holds(c) for c in hidden)
    return len(hidden), right


def run_learner(policy, law, seed):
    device, start = _setup(law, seed)
    knowledge = dev.knowledge_for(device)
    probes = dev.DeviceProbes(device)
    for c in start:
        knowledge["b"].update(probes.params("b", c), device.rules["b"].holds(c))
    progress = dev._Progress(device, knowledge)
    world = _Logged(device, knowledge, progress)
    hook = _Recorder(device) if policy in ("I", "R", "D") else None
    challenge = dev.challenge_for(device)
    if policy in ("I", "N", "D"):
        report = inquiry.inquire(lambda: inquiry.unsettled(knowledge), world, BUDGET,
                                 challenge=challenge, explain=hook, doubt=policy == "D")
        stopped = report.stopped
    else:
        rng = random.Random(2000 + seed)
        hypotheses, space = knowledge["b"], probes.space("b")
        stopped = inquiry.BUDGET
        while True:
            if inquiry.settle(hypotheses, space, challenge, explain=hook):
                stopped = inquiry.NO_QUESTIONS
                break
            question = inquiry.Question("b", "", hypotheses, {"action": "b"})
            offered = [s for s in probes.probes_for(question)
                       if device.actions + max(1, int(s.cost)) <= BUDGET]
            if not offered:
                break
            spec = rng.choice(offered)
            hypotheses.update(spec.params, world.act(spec))
    trial = dev._trial(policy, device, knowledge, world.non_discriminating, progress, stopped,
                       BUDGET)
    observed = set(start) | set(world.configs)
    hidden, right = _holdout(device, _answer(device, knowledge), observed)
    row = _row(policy, law, seed, trial, stopped, hook, world, hidden, right)
    if policy == "D":
        row["doubts"] = knowledge["b"].doubted
        row["doubt_reopened"] = knowledge["b"].doubt_reopened
        row["explained"] = knowledge["b"].explained
    return row


def _row(policy, law, seed, trial, stopped, hook, world, hidden, right):
    return {"policy": policy, "law": law, "seed": seed, "stratum": stratum(law),
            "verdict": trial.verdicts.get("b"), "to_correct": trial.to_correct,
            "to_settle": trial.to_settle, "actions": trial.actions, "stopped": stopped,
            "calls": hook.calls if hook else 0, "first_right": hook.first_right if hook else None,
            "non_discriminating": world.non_discriminating, "probes": world.probes,
            "hidden": hidden, "hidden_right": right, "observed_all": hidden == 0}


def run_passive(law, seed):
    """Discovery on observations that arrive in Gray-code order; no question."""
    device, start = _setup(law, seed)
    history = [(c, device.rules["b"].holds(c)) for c in start]
    probes = dev.DeviceProbes(device)

    generate = EX.from_history(NAMES)

    class _History:
        """What the generator reads: the observations, nothing else."""
        def __init__(self, rows):
            self.history = [(probes.params("b", c), o) for c, o in rows]

    def right_now():
        # The answer is chosen by the rule the generator uses for I and R --
        # the shortest program exact on the history -- so that what differs
        # between the policies is only which observations were made
        # (docs/РОЖДЕНИЕ_ГИПОТЕЗ.md, 10.8).
        born, mass = generate(_History(history))
        if not born:
            return False
        top = born[max(range(len(born)), key=lambda i: (mass[i], -i))]
        return _behaves(top.predict, probes, device)

    to_correct = 0 if right_now() else None
    step, seen = 0, set(start)
    while device.actions < BUDGET and len(seen) < 8:
        outcome = device.press("b")
        history.append((tuple(device.state), outcome))
        seen.add(tuple(device.state))
        if right_now():
            to_correct = device.actions if to_correct is None else to_correct
        else:
            to_correct = None
        if device.actions >= BUDGET:
            break
        step += 1
        device.toggle(((step & -step).bit_length() - 1) % 3)
    final = right_now()
    return {"policy": "P", "law": law, "seed": seed, "stratum": stratum(law),
            "verdict": dev.CORRECT if final else dev.WRONG,
            "to_correct": to_correct if final else None, "to_settle": None,
            "actions": device.actions, "stopped": "passive", "calls": 0, "first_right": None,
            "non_discriminating": 0, "probes": device.presses, "hidden": 8 - len(seen),
            "hidden_right": None, "observed_all": len(seen) == 8}


def job(args):
    policy, law, seed = args
    return run_passive(law, seed) if policy == "P" else run_learner(policy, law, seed)


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def report(rows):
    print(f"\n=== политики по разрезам (законов × сидов)")
    print("  политика разрез  n     верно  не объясн. открыто  до закрытия(мед)  до верной(мед)  вызовов(мед)")
    for policy in POLICIES:
        for s in ("S1", "S2", "S3", "все"):
            mine = [r for r in rows if r["policy"] == policy and (s == "все" or r["stratum"] == s)]
            if not mine:
                continue
            n = len(mine)
            share = lambda v: 100 * sum(1 for r in mine if r["verdict"] == v) / n  # noqa: E731
            print(f"  {policy:<8} {s:<6} {n:>5}  {share(dev.CORRECT):>5.1f}%  "
                  f"{share(dev.UNEXPLAINED):>8.1f}%  {share(dev.OPEN):>6.1f}%  "
                  f"{str(_median(r['to_settle'] for r in mine)):>16}  "
                  f"{str(_median(r['to_correct'] for r in mine)):>14}  "
                  f"{str(_median(r['calls'] for r in mine)):>12}")

    key = lambda r: (r["law"], r["seed"])  # noqa: E731
    by = {p: {key(r): r for r in rows if r["policy"] == p} for p in POLICIES}

    i3 = [r for r in by["I"].values() if r["stratum"] == "S3" and r["calls"] > 0]
    a = sum(1 for r in i3 if r["first_right"])
    print(f"\n=== A: первый вызов сразу дал закон (I, S3, где был вызов): {a} из {len(i3)} "
          f"({100 * a / max(1, len(i3)):.1f}%)")

    def paired(p, q, field, keep=lambda r: True):
        wins = losses = ties = 0
        for k, x in by[p].items():
            y = by[q].get(k)
            if y is None or not keep(x):
                continue
            vx = x[field] if x[field] is not None else BUDGET + 1
            vy = y[field] if y[field] is not None else BUDGET + 1
            wins += vx < vy
            losses += vx > vy
            ties += vx == vy
        return wins, losses, ties, dev.sign_test(wins, losses)

    print("\n=== B: I против R, цена до закрытия (меньше — лучше)")
    for title, keep in (("все", lambda r: True), ("S3", lambda r: r["stratum"] == "S3"),
                        ("S3 без A", lambda r: r["stratum"] == "S3" and r["first_right"] is False)):
        w, l, t, p = paired("I", "R", "to_settle", keep)
        print(f"  {title:<9} I лучше {w}, хуже {l}, поровну {t}; p = {p:.2e}")
    for policy in ("I", "R"):
        mine = list(by[policy].values())
        probes = sum(r["probes"] for r in mine)
        nd = sum(r["non_discriminating"] for r in mine)
        print(f"  различающих проб у {policy}: {100 * (probes - nd) / max(1, probes):.1f}% из {probes}")

    print("\n=== I против P, цена до верной модели")
    for title, keep in (("все", lambda r: True), ("S3", lambda r: r["stratum"] == "S3"),
                        ("I не видела всех 8", lambda r: not r["observed_all"]),
                        ("I видела все 8", lambda r: r["observed_all"])):
        w, l, t, p = paired("I", "P", "to_correct", keep)
        print(f"  {title:<18} I лучше {w}, хуже {l}, поровну {t}; p = {p:.2e}")

    answered = [r for r in by["I"].values() if r["hidden_right"] is not None]
    empty = sum(1 for r in answered if r["hidden"] == 0)
    with_hidden = [r for r in answered if r["hidden"] > 0]
    acc = sum(r["hidden_right"] for r in with_hidden) / max(1, sum(r["hidden"] for r in with_hidden))
    print(f"\n=== скрытые состояния (I, отвечено): без скрытых {empty} из {len(answered)}; "
          f"точность на скрытых {100 * acc:.1f}%")


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    path = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "birth_results.jsonl")
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row["policy"], row["law"], row["seed"])] = row
    jobs = [(p, law, seed) for p in POLICIES for law in LAWS for seed in SEEDS
            if (p, law, seed) not in done]
    started = time.time()
    print(f"испытаний сделано {len(done)}, осталось {len(jobs)}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, open(path, "a", encoding="utf-8") as fh:
        futures = [pool.submit(job, j) for j in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            row = future.result()
            done[(row["policy"], row["law"], row["seed"])] = row
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 500 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)}, {time.time() - started:.0f}с", flush=True)
    report(list(done.values()))


if __name__ == "__main__":
    main()
