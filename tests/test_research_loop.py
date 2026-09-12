"""
mana/research: the research loop outside core. First what it must not
carry, then that it does what the lab loop (`cognition/inquiry.py` 1.14,
boundary mode) did -- action for action -- and that its answers go
through `core/standing.py`.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from mana.cognition import inquiry
from mana.core import standing
from mana.research import contract, loop
from mana.research.adapters import device as adapter
from mana.world import device as dev

RESEARCH = pathlib.Path(loop.__file__).parent


def _sources():
    return {path: path.read_text(encoding="utf-8") for path in RESEARCH.rglob("*.py")}


def test_the_research_package_carries_none_of_the_closed_mechanisms():
    closed = re.compile(r"\bmisfit\b|MISFIT_PRIOR|DEVIATION|FAMILY_MISS|CHECK_STRENGTH|"
                        r"checks_to_close|claim_level|claims_graded|GEOMETRIC")
    found = {str(p.name): closed.findall(src) for p, src in _sources().items() if closed.search(src)}
    assert not found, f"closed mechanisms in research/: {found}"


def test_the_research_package_does_not_import_the_lab():
    for path, src in _sources().items():
        for line in src.splitlines():
            if line.lstrip().startswith(("import ", "from ")):
                assert "inquiry" not in line, f"{path.name}: {line}"


def test_a_task_is_not_offered_the_rule_measured_dishonest():
    with pytest.raises(ValueError):
        contract.Stakes(error_cost=8.0, rule=standing.PATTERN)
    assert contract.Stakes(error_cost=8.0).rule == standing.CLAIMS


def _half(device, seed):
    j = device.free[seed % len(device.free)]
    home = device.state[j]
    return lambda params: 1.0 if params["config"][j] == home else 0.0


def reproduce(seed, options, cost, half=False):
    """The lab loop and the research loop on two copies of one device: what
    each did, and how each answer stood."""
    lab_device, new_device = dev.random_device(seed, **options), dev.random_device(seed, **options)
    lab_acts, new_acts = [], []

    class LabWorld(dev.DeviceProbes):
        def act(self, spec):
            out = super().act(spec)
            lab_acts.append((spec.params["button"], spec.params["config"], out))
            return out

    class NewWorld(adapter.DeviceWorld):
        def act(self, probe):
            out = super().act(probe)
            new_acts.append((probe.params["button"], probe.params["config"], out))
            return out

    lab_weight = _half(lab_device, seed) if half else (lambda params: 1.0)
    new_weight = _half(new_device, seed) if half else (lambda params: 1.0)
    lab = dev.knowledge_for(lab_device)
    lab_report = inquiry.inquire(lambda: inquiry.unsettled(lab), LabWorld(lab_device), 250,
                                 challenge=dev.challenge_for(lab_device),
                                 stakes=inquiry.Stakes(error_cost=cost, weight=lab_weight),
                                 boundary=standing.CLAIMS)
    new = adapter.knowledge_for(new_device)
    new_report = loop.inquire(lambda: loop.unsettled(new), NewWorld(new_device), 250,
                              contract.Stakes(error_cost=cost, weight=new_weight),
                              challenge=adapter.challenge_for(new_device),
                              evidence=loop.MEMBER)

    def lab_standing(hs):
        if not hs.settled:
            return ("open", ())
        if hs.settled[0] == inquiry.UNEXPLAINED:
            return (standing.UNEXPLAINED, ())
        return (hs.applicability["status"], tuple(hs.applicability["uncovered"]))

    def new_standing(e):
        if e.standing is None:
            return ("open", ())
        return (e.standing.status, e.standing.uncovered)

    return {"acts": (lab_acts, new_acts),
            "actions": (lab_device.actions, new_device.actions),
            "stopped": (lab_report.stopped, new_report.stopped),
            "standing": ({b: lab_standing(hs) for b, hs in lab.items()},
                         {b: new_standing(e) for b, e in new.items()})}


@pytest.mark.parametrize("options", [{}, {"needs": 2}, {"needs": 3}, {"hidden_pair": True}],
                         ids=["A", "C", "D", "E"])
def test_the_loop_does_what_the_lab_did(options):
    for seed in range(2):
        for cost in (16.0, 64.0):
            result = reproduce(seed, options, cost)
            for what, (lab, new) in result.items():
                assert lab == new, (options, seed, cost, what)


def test_the_loop_does_what_the_lab_did_on_a_task_that_relies_on_half():
    result = reproduce(0, {"needs": 3}, 64.0, half=True)
    for what, (lab, new) in result.items():
        assert lab == new, what


def _tie_dependent(evidence, worlds, seeds, cost=64.0):
    """Closed answers whose standing would differ had another member of the
    answering class won the tie of weights."""
    dependent = answered = 0
    for options in worlds:
        for seed in seeds:
            device = dev.random_device(seed, **options)
            knowledge = adapter.knowledge_for(device)
            loop.inquire(lambda: loop.unsettled(knowledge), adapter.DeviceWorld(device), 250,
                         contract.Stakes(error_cost=cost),
                         challenge=adapter.challenge_for(device), evidence=evidence)
            for e in knowledge.values():
                if not (e.settled and e.settled[0] == loop.ANSWERED):
                    continue
                answered += 1
                standings = {(a.inside, a.status, a.uncovered)
                             for a in (e.answer_for(e.get(n)) for n in e.settled[1])}
                dependent += len(standings) > 1
    return dependent, answered


def test_the_boundary_does_not_depend_on_who_won_the_tie():
    """Counted for the member, the same knowledge had different boundaries
    depending on which of several indistinguishable explanations led;
    counted for the class, it has one."""
    worlds = ({}, {"needs": 3}, {"hidden_pair": True})
    member, answered = _tie_dependent(loop.MEMBER, worlds[:1], range(10))
    assert answered and member > 0, "the defect must be visible where it was found"
    for options in worlds:
        dependent, answered = _tie_dependent(loop.CLASS, (options,), range(4))
        assert answered and dependent == 0, options


def test_evidence_counts_for_the_class_unless_told_otherwise():
    assert loop.Explanations([contract.Explanation("x", lambda p: {True: 1.0})]).evidence_for \
        == loop.CLASS
    with pytest.raises(ValueError):
        loop.inquire(lambda: [], None, 1, contract.Stakes(error_cost=1.0), evidence="whoever")


class Flicker:
    """Two situations; the second answers differently the second time it
    is seen -- chance a single look cannot show."""

    def __init__(self):
        self.seen = {}

    @staticmethod
    def params(a):
        return {"key": (a,), "conditions": {"a": a}}

    def probes_for(self, question):
        return [contract.Probe("look", self.params(a), cost=1) for a in (False, True)]

    def act(self, probe):
        key = probe.params["key"]
        self.seen[key] = self.seen.get(key, 0) + 1
        return not (key == (True,) and self.seen[key] == 2)


def _flicker(assume_repeatable):
    knowledge = {"q": loop.condition_explanations(["a"])}
    loop.inquire(lambda: loop.unsettled(knowledge), Flicker(), 50,
                 contract.Stakes(error_cost=16.0, assume_repeatable=assume_repeatable),
                 challenge=loop.pairwise_rivals(["a"]))
    return knowledge["q"].standing


def test_repeatability_assumed_calls_a_flicker_verified_and_says_it_assumed():
    answer = _flicker(True)
    assert answer.status == standing.VERIFIED_FOR_TASK
    assert answer.repeatability == standing.REPEATABLE_ASSUMED


def test_repeatability_checked_catches_the_flicker():
    """Checked, the situation seen once is repeated, answers differently, and
    no explanation that predicts a single outcome survives."""
    answer = _flicker(False)
    assert answer.status == standing.UNEXPLAINED


def test_every_answer_carries_its_standing_and_assumption():
    device = dev.random_device(0)
    knowledge = adapter.knowledge_for(device)
    report = loop.inquire(lambda: loop.unsettled(knowledge), adapter.DeviceWorld(device), 250,
                          contract.Stakes(error_cost=64.0),
                          challenge=adapter.challenge_for(device))
    assert report.stopped == loop.NO_QUESTIONS and set(report.answers) == set(knowledge)
    for answer in report.answers.values():
        assert answer.status in (standing.VERIFIED_FOR_TASK, standing.CONDITIONAL)
        assert answer.rule == standing.CLAIMS
        assert answer.assumption == standing.ASSUMPTIONS[standing.CLAIMS]


def test_an_answer_is_audited_by_core_against_the_oracle():
    """World E, device 80: the answer the loop closes is dishonest exactly
    where two conditions the model ignores interact -- the case
    core/standing names."""
    device = dev.random_device(80, hidden_pair=True)
    knowledge = adapter.knowledge_for(device)
    loop.inquire(lambda: loop.unsettled(knowledge), adapter.DeviceWorld(device), 250,
                 contract.Stakes(error_cost=16.0), challenge=adapter.challenge_for(device))
    e = knowledge["b0"]
    _, _, lead = e.leading_class(e.space)
    result = standing.audit(e.standing, adapter.wrong_situations(device, "b0", lead))
    assert not result.honest
    a, f1, f2 = device.rules["b0"].switches
    assert all(c[a] and c[f1] and c[f2] for c in result.inside)
