"""
mana.research.adapters.device — the synthetic device as a research world.

The same probes `world/device.DeviceProbes` offers -- bring the switches to
a configuration, then press; the price is the toggles plus the press --
built on `Device` alone, so that the research loop does not go through the
lab's types. The truth is used in one place only, `wrong_situations`, which
is the oracle's count for `core/standing.audit` and is never read by the
loop.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from ...world.device import Device
from .. import contract
from ..loop import Explanations, condition_explanations, pairwise_rivals

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"


class DeviceWorld:
    """The device as a `World`: set the switches, then press."""

    def __init__(self, device: Device) -> None:
        self.device = device

    def params(self, button: str, config: Tuple[bool, ...]) -> Dict[str, Any]:
        return {"button": button, "config": config, "key": config,
                "conditions": self.device.conditions(config)}

    def probes_for(self, question: contract.Question) -> List[contract.Probe]:
        button = question.source["action"]
        here = tuple(self.device.state)
        out = []
        for config in self.device.reachable():
            flips = sum(1 for a, b in zip(here, config) if a != b)
            out.append(contract.Probe(action=f"press {button}",
                                      params=self.params(button, config),
                                      cost=flips + 1, safety=contract.REVERSIBLE))
        return out

    def act(self, probe: contract.Probe) -> bool:
        for index, (now, wanted) in enumerate(zip(list(self.device.state),
                                                  probe.params["config"])):
            if now != wanted:
                self.device.toggle(index)
        return self.device.press(probe.params["button"])


def knowledge_for(device: Device) -> Dict[str, Explanations]:
    """Before touching anything: for every button, every explanation equal."""
    names = [f"s{i}" for i in range(len(device.state))]
    return {button: condition_explanations(names, noise=device.noise)
            for button in device.buttons}


def challenge_for(device: Device) -> Callable[[contract.Explanation], List[contract.Explanation]]:
    return pairwise_rivals([f"s{i}" for i in range(len(device.state))])


def wrong_situations(device: Device, button: str, explanation: contract.Explanation) -> set:
    """Where `explanation` is wrong about `button` -- the oracle's count, for
    `core/standing.audit`. Reads the hidden rule; the loop never calls it."""
    world = DeviceWorld(device)
    rule = device.rules[button]
    return {config for config in device.reachable()
            if (explanation.predict(world.params(button, config)).get(True, 0.0) > 0.5)
            != rule.holds(config)}
