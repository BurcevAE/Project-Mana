"""
mana.world.device — an unknown device that does nothing until touched.

What it is for
---------------
The test environment for `cognition/inquiry.py`. It has one property that
matters: **it produces no events.** Nothing happens unless MANA acts, so a
learner that waits learns nothing, and a learner that acts learns exactly
as much as its choice of actions buys. That is the situation in which the
passive loop stopped developing, made small enough to know the truth.

The device
-----------
Switches that can be toggled -- one may be locked -- and buttons that
work or fail. Each button's rule is hidden: it works when one switch is
on, or always, or never. In two variants one button is harder: it needs
two switches at once, or three. In a third it works when one switch is
on unless two others are both on -- an interaction of two conditions a
model reading the first switch ignores. Pressing may be noisy.

The rules are written here so that what a learner concludes can be scored
against them, the same arrangement as `world/universe.py`. No learner
reads them: the adapter exposes only switch states, actions and outcomes.
An answer is scored by behaviour -- does it predict the same outcome as
the hidden rule in every configuration the device can reach -- because
two differently named rules that behave identically there are the same
answer in this world.

Three policies, one learner
----------------------------
    passive      waits for events; there are none
    enumeration  walks every configuration in Gray-code order and presses
                 every button not yet answered
    inquiry      `inquiry.inquire`: the probe with the most expected bits
                 per action

Enumeration and inquiry share the learner -- the same hypotheses, the same
challenge before acceptance, the same test for when a question is settled
-- so what differs between them is only which action is taken next.
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..cognition import inquiry

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.10"

CORRECT, WRONG, UNEXPLAINED, OPEN = "correct", "wrong", "unexplained", "open"


@dataclass(frozen=True)
class Rule:
    kind: str                          # switch | always | never | all_of | unless_both
    switches: Tuple[int, ...] = ()

    def holds(self, state: Sequence[bool]) -> bool:
        if self.kind == "always":
            return True
        if self.kind == "never":
            return False
        if self.kind == "unless_both":
            a, e, f = self.switches
            return bool(state[a]) and not (state[e] and state[f])
        return all(state[i] for i in self.switches)


class Device:
    def __init__(self, switches: int, rules: Dict[str, Rule],
                 locked: Optional[Dict[int, bool]] = None, noise: float = 0.0,
                 seed: int = 0, initial: Optional[Sequence[bool]] = None) -> None:
        self.rng = random.Random(seed)
        self.state = (list(initial) if initial is not None
                      else [self.rng.random() < 0.5 for _ in range(switches)])
        self.locked = dict(locked or {})
        for index, value in self.locked.items():
            self.state[index] = value
        self.rules = dict(rules)
        self.noise = noise
        self.toggles = 0
        self.presses = 0

    @property
    def actions(self) -> int:
        return self.toggles + self.presses

    @property
    def buttons(self) -> List[str]:
        return sorted(self.rules)

    @property
    def free(self) -> List[int]:
        return [i for i in range(len(self.state)) if i not in self.locked]

    def events(self) -> list:
        """What the device does on its own: nothing."""
        return []

    def conditions(self, config: Sequence[bool]) -> Dict[str, bool]:
        return {f"s{i}": bool(v) for i, v in enumerate(config)}

    def toggle(self, index: int) -> None:
        if index in self.locked:
            raise ValueError(f"s{index} заблокирован")
        self.state[index] = not self.state[index]
        self.toggles += 1

    def press(self, button: str) -> bool:
        self.presses += 1
        works = self.rules[button].holds(self.state)
        if self.noise and self.rng.random() < self.noise:
            works = not works
        return works

    def reachable(self) -> List[Tuple[bool, ...]]:
        """Every configuration the free switches can make."""
        out = []
        for bits in itertools.product((False, True), repeat=len(self.free)):
            config = list(self.state)
            for index, bit in zip(self.free, bits):
                config[index] = bit
            out.append(tuple(config))
        return sorted(out)


class DeviceProbes:
    """The device as a `ProbeProvider`: set the switches, then press.

    A toggle on its own tells nothing -- no button was pressed -- so a
    probe is "bring the conditions to this configuration, then observe",
    the way any experiment is run. It costs the toggles plus the press.
    """

    def __init__(self, device: Device) -> None:
        self.device = device

    def params(self, button: str, config: Tuple[bool, ...]) -> Dict:
        return {"button": button, "config": config, "key": config,
                "conditions": self.device.conditions(config)}

    def space(self, button: str) -> List[Dict]:
        return [self.params(button, c) for c in self.device.reachable()]

    def probes_for(self, question: inquiry.Question) -> List[inquiry.ProbeSpec]:
        button = question.source["action"]
        here = tuple(self.device.state)
        out = []
        for config in self.device.reachable():
            flips = sum(1 for a, b in zip(here, config) if a != b)
            out.append(inquiry.ProbeSpec(action=f"press {button}",
                                         params=self.params(button, config),
                                         cost=flips + 1, safety=inquiry.REVERSIBLE))
        return out

    def act(self, spec: inquiry.ProbeSpec) -> bool:
        for index, (now, wanted) in enumerate(zip(list(self.device.state),
                                                  spec.params["config"])):
            if now != wanted:
                self.device.toggle(index)
        return self.device.press(spec.params["button"])


def knowledge_for(device: Device) -> Dict[str, inquiry.HypothesisSet]:
    """What MANA believes before touching anything: every hypothesis equal.

    The noise level is given to the hypotheses, not learned. Estimating it
    is a question of its own and outside this slice.
    """
    names = [f"s{i}" for i in range(len(device.state))]
    return {button: inquiry.condition_hypotheses(names, noise=device.noise)
            for button in device.buttons}


def challenge_for(device: Device):
    return inquiry.pairwise_rivals([f"s{i}" for i in range(len(device.state))])


def verdicts(device: Device, knowledge: Dict[str, inquiry.HypothesisSet]) -> Dict[str, str]:
    """Each button's answer, scored by behaviour against the hidden rule."""
    probes = DeviceProbes(device)
    out = {}
    for button in device.buttons:
        hypotheses = knowledge[button]
        if not hypotheses.settled:
            out[button] = OPEN
            continue
        kind, names = hypotheses.settled
        if kind == inquiry.UNEXPLAINED:
            out[button] = UNEXPLAINED
            continue
        members = [hypotheses.get(n) for n in names]
        answer = max((m for m in members if m is not None), key=lambda h: h.weight)
        rule = device.rules[button]
        same = all((answer.predict(probes.params(button, c)).get(True, 0.0) > 0.5)
                   == rule.holds(c) for c in device.reachable())
        out[button] = CORRECT if same else WRONG
    return out


@dataclass
class Trial:
    policy: str
    actions: int
    toggles: int
    presses: int
    non_discriminating: int
    correct: int
    wrong: int
    unexplained: int
    open: int
    to_settle: Optional[int]
    to_correct: Optional[int]
    stopped: str
    budget_left: int
    checks: int = 0
    verdicts: Dict[str, str] = field(default_factory=dict)


class _Progress:
    """When every question was first closed, and when every answer was
    first right -- checked on the way through, before each action."""

    def __init__(self, device, knowledge) -> None:
        self.device = device
        self.knowledge = knowledge
        self.to_settle: Optional[int] = None
        self.to_correct: Optional[int] = None

    def check(self) -> None:
        if self.to_settle is None and all(h.settled for h in self.knowledge.values()):
            self.to_settle = self.device.actions
        if self.to_correct is None and all(
                v == CORRECT for v in verdicts(self.device, self.knowledge).values()):
            self.to_correct = self.device.actions


def _trial(policy, device, knowledge, non_discriminating, progress, stopped, budget,
           checks: int = 0) -> Trial:
    progress.check()
    judged = verdicts(device, knowledge)
    counts = {k: 0 for k in (CORRECT, WRONG, UNEXPLAINED, OPEN)}
    for verdict in judged.values():
        counts[verdict] += 1
    return Trial(policy, device.actions, device.toggles, device.presses,
                 non_discriminating, counts[CORRECT], counts[WRONG],
                 counts[UNEXPLAINED], counts[OPEN], progress.to_settle,
                 progress.to_correct, stopped, max(0, budget - device.actions),
                 checks, judged)


def run_passive(device: Device, budget: int) -> Trial:
    knowledge = knowledge_for(device)
    for _ in range(budget):
        for _event in device.events():      # the whole of what waiting yields
            pass
    return _trial("passive", device, knowledge, 0, _Progress(device, knowledge),
                  "ждала событий: их нет", budget)


def run_enumeration(device: Device, budget: int) -> Trial:
    knowledge = knowledge_for(device)
    challenge = challenge_for(device)
    probes = DeviceProbes(device)
    spaces = {b: probes.space(b) for b in device.buttons}
    progress = _Progress(device, knowledge)
    non_discriminating = 0
    stopped = inquiry.BUDGET
    free = device.free
    step = 0
    while device.actions < budget:
        for button in device.buttons:
            hypotheses = knowledge[button]
            if inquiry.settle(hypotheses, spaces[button], challenge):
                continue
            if device.actions >= budget:
                break
            progress.check()
            params = probes.params(button, tuple(device.state))
            if not hypotheses.discriminates(params):
                non_discriminating += 1
            hypotheses.update(params, device.press(button))
            inquiry.settle(hypotheses, spaces[button], challenge)
        if all(h.settled for h in knowledge.values()):
            stopped = inquiry.NO_QUESTIONS
            break
        if not free or device.actions >= budget:
            break
        step += 1
        device.toggle(free[((step & -step).bit_length() - 1) % len(free)])
    return _trial("enumeration", device, knowledge, non_discriminating, progress,
                  stopped, budget)


class _Watched:
    """The provider, counting on the way through whether each press could
    tell the explanations in play apart."""

    def __init__(self, device, knowledge, progress) -> None:
        self.inner = DeviceProbes(device)
        self.knowledge = knowledge
        self.progress = progress
        self.non_discriminating = 0

    def probes_for(self, question):
        return self.inner.probes_for(question)

    def act(self, spec):
        self.progress.check()
        if not self.knowledge[spec.params["button"]].discriminates(spec.params):
            self.non_discriminating += 1
        return self.inner.act(spec)


def run_inquiry(device: Device, budget: int, model_check: bool = False,
                vulnerability: str = inquiry.GEOMETRIC,
                claim_level: bool = False,
                checks_to_close: int = 0,
                error_cost: Optional[float] = None,
                stakes: Optional[inquiry.Stakes] = None,
                graded: bool = False,
                boundary: Optional[str] = None) -> Tuple[Trial, inquiry.Report]:
    knowledge = knowledge_for(device)
    progress = _Progress(device, knowledge)
    world = _Watched(device, knowledge, progress)
    report = inquiry.inquire(lambda: inquiry.unsettled(knowledge), world, budget,
                             challenge=challenge_for(device), model_check=model_check,
                             vulnerability=vulnerability, claim_level=claim_level,
                             checks_to_close=checks_to_close, error_cost=error_cost,
                             stakes=stakes, graded=graded, boundary=boundary)
    checks = sum(1 for step in report.steps if step.motive == inquiry.MODEL_CHECK)
    name = ("claims-economy" if stakes is not None
            else ("E-claim-level" if claim_level else f"E-{vulnerability}") if model_check
            else "inquiry")
    return (_trial(name, device, knowledge, world.non_discriminating, progress,
                   report.stopped, budget, checks), report)


def random_device(seed: int, switches: int = 6, buttons: int = 5, locked: bool = True,
                  noise: float = 0.0, needs: int = 1, hidden_pair: bool = False) -> Device:
    """A random device. `needs` > 1 makes button b0 need that many switches
    at once -- a rule no single-condition hypothesis states. `hidden_pair`
    makes b0 work when switch a is on unless e and f are both on."""
    rng = random.Random(seed)
    lock = {rng.randrange(switches): rng.random() < 0.5} if locked else {}
    free = [i for i in range(switches) if i not in lock]
    rules: Dict[str, Rule] = {}
    for k in range(buttons):
        roll = rng.random()
        rules[f"b{k}"] = (Rule("switch", (rng.randrange(switches),)) if roll < 0.7
                          else Rule("always") if roll < 0.85 else Rule("never"))
    if needs > 1:
        rules["b0"] = Rule("all_of", tuple(sorted(rng.sample(free, needs))))
    elif hidden_pair:
        rules["b0"] = Rule("unless_both", tuple(rng.sample(free, 3)))
    return Device(switches, rules, locked=lock, noise=noise, seed=seed + 1000)


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test; ties are dropped before calling."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)
