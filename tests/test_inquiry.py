"""
The inquiry slice: Question -> Probe -> choose -> act -> observe, on a
device that produces no events. Unit tests for the contract first, then
the criteria the experiment is run for, on a handful of devices.
"""
from __future__ import annotations

import pytest

from mana.cognition import inquiry
from mana.world import device as dev


def _two(noise=0.0):
    """Two hypotheses a single probe can tell apart perfectly."""
    yes = inquiry.Hypothesis("да", lambda p: {True: 1.0 - noise, False: noise})
    no = inquiry.Hypothesis("нет", lambda p: {True: noise, False: 1.0 - noise})
    return inquiry.HypothesisSet([yes, no])


# --------------------------------------------------------------------------
# the contract
# --------------------------------------------------------------------------

def test_entropy_and_gain_are_in_bits():
    hs = _two()
    assert hs.entropy() == pytest.approx(1.0)
    assert hs.expected_gain({}) == pytest.approx(1.0)       # one perfect probe
    assert _two(noise=0.5).expected_gain({}) == pytest.approx(0.0)


def test_a_probe_every_hypothesis_agrees_on_is_worth_nothing():
    same = inquiry.HypothesisSet([
        inquiry.Hypothesis("a", lambda p: {True: 1.0}),
        inquiry.Hypothesis("b", lambda p: {True: 1.0})])
    assert same.expected_gain({}) == pytest.approx(0.0)
    assert not same.discriminates({})


def test_an_observation_no_hypothesis_allows_is_a_contradiction():
    hs = inquiry.HypothesisSet([inquiry.Hypothesis("a", lambda p: {True: 1.0}),
                                inquiry.Hypothesis("b", lambda p: {True: 1.0})])
    before = [h.weight for h in hs.hypotheses]
    assert hs.update({}, False) is False
    assert [h.weight for h in hs.hypotheses] == before
    [question] = inquiry.unsettled({"x": hs})
    assert question.why_open == inquiry.CONTRADICTION


def test_a_rival_added_later_is_weighed_against_the_whole_history():
    hs = inquiry.condition_hypotheses(["s0", "s1"])
    for _ in range(3):
        hs.update({"conditions": {"s0": True, "s1": False}}, True)
    refuted = inquiry.Hypothesis("никогда-2", lambda p: {True: 0.0, False: 1.0})
    agrees = inquiry.Hypothesis("s0-2", lambda p: {True: float(p["conditions"]["s0"]),
                                                    False: float(not p["conditions"]["s0"])})
    hs.add([refuted, agrees], prior_each=0.1)
    assert hs.get("никогда-2").weight == 0.0
    assert hs.get("s0-2").weight > 0.0


def test_a_question_the_world_cannot_split_is_answered_as_a_class():
    """A button run by a locked switch behaves exactly like 'always'. No
    probe separates them, so the honest answer is the class -- and the
    loop stops there instead of asking forever."""
    device = dev.Device(3, {"b": dev.Rule("switch", (0,))}, locked={0: True},
                        initial=[True, False, False])
    trial, report = dev.run_inquiry(device, budget=500)
    assert report.stopped == inquiry.NO_QUESTIONS
    kind, names = report.answers["b"]
    assert kind == inquiry.ANSWERED and len(names) >= 2
    assert inquiry.ALWAYS in names and "зависит от s0" in names
    assert trial.correct == 1 and trial.budget_left > 400


def test_an_irreversible_probe_is_never_taken_unless_allowed():
    class World:
        def probes_for(self, question):
            return [inquiry.ProbeSpec("сжечь", {}, safety=inquiry.IRREVERSIBLE)]

        def act(self, spec):
            raise AssertionError("не должно было вызваться")

    knowledge = {"x": _two()}
    report = inquiry.inquire(lambda: inquiry.unsettled(knowledge), World(), budget=10)
    assert report.stopped == inquiry.NOTHING_WORTH_ASKING
    assert report.steps == []


def test_a_question_that_cannot_finish_in_the_budget_is_deferred():
    """Stopping is in the question's own units: bits still needed against
    the best rate on offer, not a floor borrowed from elsewhere."""
    class Weak:
        def probes_for(self, question):
            return [inquiry.ProbeSpec("шепнуть", {"key": 1}, cost=1)]

        def act(self, spec):
            return True

    knowledge = {"x": _two(noise=0.49)}
    report = inquiry.inquire(lambda: inquiry.unsettled(knowledge), Weak(), budget=20)
    assert report.stopped == inquiry.BUDGET
    assert report.steps == []


def test_bits_no_probe_can_deliver_are_not_owed():
    """Three answers no probe tells apart, one that differs. 2 bits between
    hypotheses, 0.81 between classes; one probe gives 0.41. Owing the 2 bits
    says 4.2 actions and defers within a budget of 3 -- found when closing
    that waited for anything else made the loop quit with most of its
    budget unspent. Owing the 0.81 takes the one probe and answers."""
    yes = [inquiry.Hypothesis(f"да-{i}", lambda p: {True: 0.9, False: 0.1}) for i in range(3)]
    no = inquiry.Hypothesis("нет", lambda p: {True: 0.1, False: 0.9})
    hs = inquiry.HypothesisSet(yes + [no])
    space = [{"key": 1}]
    assert hs.entropy() == pytest.approx(2.0)
    assert hs.class_entropy(space) == pytest.approx(0.8113, abs=1e-4)

    class One:
        def probes_for(self, question):
            return [inquiry.ProbeSpec("нажать", {"key": 1}, cost=1)]

        def act(self, spec):
            return True

    knowledge = {"x": hs}
    report = inquiry.inquire(lambda: inquiry.unsettled(knowledge), One(), budget=3)
    assert report.stopped == inquiry.NO_QUESTIONS
    assert len(report.steps) == 1
    kind, names = report.answers["x"]
    assert kind == inquiry.ANSWERED and set(names) == {"да-0", "да-1", "да-2"}


# --------------------------------------------------------------------------
# the device and the three policies
# --------------------------------------------------------------------------

def test_the_device_produces_nothing_on_its_own_so_waiting_learns_nothing():
    trial = dev.run_passive(dev.random_device(1), budget=200)
    assert trial.actions == 0
    assert trial.correct == 0 and trial.open == 5


def test_inquiry_learns_every_rule_from_nothing_and_never_guesses_wrong():
    for seed in range(8):
        trial, report = dev.run_inquiry(dev.random_device(seed), budget=250)
        assert trial.correct == 5 and trial.wrong == 0, (seed, trial)
        assert report.steps[0].why_open == inquiry.NOT_ENOUGH_DATA
        assert report.stopped == inquiry.NO_QUESTIONS
        assert trial.budget_left > 0


def test_inquiry_needs_fewer_actions_than_enumeration():
    faster = slower = 0
    for seed in range(10):
        inq = dev.run_inquiry(dev.random_device(seed), budget=250)[0]
        enum = dev.run_enumeration(dev.random_device(seed), budget=250)
        a = inq.to_correct if inq.to_correct is not None else 10 ** 6
        b = enum.to_correct if enum.to_correct is not None else 10 ** 6
        faster += a < b
        slower += a > b
        assert inq.non_discriminating <= enum.non_discriminating
    assert faster > slower


def test_a_rule_outside_the_first_hypotheses_is_not_explained_away():
    """b0 needs two switches at once. No starting hypothesis says so. The
    challenge before acceptance must catch the nearest wrong one."""
    for seed in range(6):
        trial, _ = dev.run_inquiry(dev.random_device(seed, needs=2), budget=250)
        assert trial.wrong == 0, seed


# --------------------------------------------------------------------------
# MODEL_CHECK: a second reason to act
# --------------------------------------------------------------------------

def test_model_check_is_worth_nothing_while_it_is_off():
    hs = inquiry.condition_hypotheses(["s0", "s1"])
    assert hs.model_check_gain({"conditions": {"s0": True, "s1": True}}) == 0.0


def test_a_check_far_from_the_evidence_is_worth_more_than_one_near_it():
    hs = inquiry.condition_hypotheses(["s0", "s1", "s2"])
    hs.misfit = inquiry.MISFIT_PRIOR
    seen = {"conditions": {"s0": True, "s1": False, "s2": False}}
    hs.update(seen, True)
    near = {"conditions": {"s0": True, "s1": True, "s2": False}}
    far = {"conditions": {"s0": True, "s1": True, "s2": True}}   # the model agrees here too
    assert hs.model_check_gain(far) > hs.model_check_gain(near) > 0.0


def test_a_claim_is_untested_until_its_pattern_was_seen_with_that_value():
    """Leader "depends on s0" commits, in each value of s0, to not caring
    about s1. After (s0=1,s1=0) and (s0=0,s1=0), the commitments left
    untested are exactly those with s1=1."""
    hs = inquiry.condition_hypotheses(["s0", "s1"])
    hs.space = [{"conditions": {"s0": a, "s1": b}, "key": (a, b)}
                for a in (False, True) for b in (False, True)]
    hs.misfit, hs.vulnerability = inquiry.MISFIT_PRIOR, inquiry.CLAIMS
    for params, outcome in ((({"s0": True, "s1": False}), True),
                            (({"s0": False, "s1": False}), False)):
        probe = {"conditions": params, "key": (params["s0"], params["s1"])}
        hs.update_misfit(probe, outcome)      # "depends on s0" led both times
        hs.update(probe, outcome)
    assert hs.leader()[0] == "зависит от s0"
    assert hs.untested_claims({"conditions": {"s0": True, "s1": False}}) == 0
    assert hs.untested_claims({"conditions": {"s0": True, "s1": True}}) == 1
    assert hs.untested_claims({"conditions": {"s0": False, "s1": True}}) == 1


def _rule(name, fn):
    return inquiry.Hypothesis(name, lambda p, fn=fn: {True: float(fn(p["conditions"])),
                                                       False: float(not fn(p["conditions"]))})


def _replayed(hypotheses, observations, space):
    hs = inquiry.HypothesisSet(hypotheses)
    hs.misfit, hs.vulnerability, hs.space = inquiry.MISFIT_PRIOR, inquiry.CLAIMS, space
    for params, outcome in observations:
        hs.update_misfit(params, outcome)
        hs.update(params, outcome)
    return hs


def test_evidence_for_an_abandoned_model_does_not_vouch_for_its_successor():
    """A leads and its commitments pass; one observation refutes A and
    leaves B. The four observations A and B agreed on were commitments the
    whole family made -- they vouch for B too. The one that refuted A is
    the one B was chosen on: it cannot also confirm B."""
    names = ["s0", "s1", "s2"]
    space = [{"conditions": dict(zip(names, bits)), "key": bits}
             for bits in __import__("itertools").product((False, True), repeat=3)]
    at = lambda a, b, c: {"conditions": {"s0": a, "s1": b, "s2": c}, "key": (a, b, c)}
    observations = [(at(True, True, False), True), (at(False, True, True), False),
                    (at(False, False, False), False), (at(True, True, True), True)]
    refuting = (at(True, False, False), False)          # A says True, B says False

    def a_and_b():
        return [_rule("A: зависит от s0", lambda c: c["s0"]),
                _rule("B: s0 и s1", lambda c: c["s0"] and c["s1"]),
                inquiry.Hypothesis(inquiry.OTHER, lambda p: {True: 0.5, False: 0.5})]

    both = _replayed(a_and_b(), observations, space)
    for h, w in zip(both.hypotheses, (0.6, 0.3, 0.1)):
        h.weight = w                                        # A leads
    both._misfit_model = None
    earned_by_a = both.current_misfit()
    assert both.leader()[0].startswith("A")
    assert earned_by_a < inquiry.MISFIT_PRIOR               # A's claims passed

    both.update_misfit(*refuting)
    both.update(*refuting)
    assert both.leader()[0].startswith("B")

    def b_only():
        return [_rule("B: s0 и s1", lambda c: c["s0"] and c["s1"]),
                inquiry.Hypothesis(inquiry.OTHER, lambda p: {True: 0.5, False: 0.5})]

    shared_only = _replayed(b_only(), observations, space)
    hindsight = _replayed(b_only(), observations + [refuting], space)
    assert both.current_misfit() == pytest.approx(shared_only.current_misfit())
    assert both.current_misfit() > hindsight.current_misfit()   # the selecting one is left out


def test_a_model_chosen_only_by_splitting_observations_starts_unconfirmed():
    """Every observation split the family, so the leader fits all of them
    and was chosen on all of them: it has no independent confirmation yet.
    Its first prediction made as leader is the first thing that counts."""
    names = ["s0", "s1"]
    space = [{"conditions": dict(zip(names, bits)), "key": bits}
             for bits in __import__("itertools").product((False, True), repeat=2)]
    at = lambda a, b: {"conditions": {"s0": a, "s1": b}, "key": (a, b)}
    hs = _replayed([_rule("A: зависит от s0", lambda c: c["s0"]),
                    _rule("B: зависит от s1", lambda c: c["s1"]),
                    inquiry.Hypothesis(inquiry.OTHER, lambda p: {True: 0.5, False: 0.5})],
                   [(at(True, False), False)], space)
    assert hs.leader()[0].startswith("B")
    assert hs.current_misfit() == pytest.approx(inquiry.MISFIT_PRIOR)
    hs.update_misfit(at(False, True), True)
    hs.update(at(False, True), True)
    assert hs.current_misfit() < inquiry.MISFIT_PRIOR


def test_coverage_and_evidence_are_different_facts_about_one_observation():
    """One observation chose B over A: for B it is coverage (that
    configuration is known), selection (it made B leader), and not
    evidence (B was not committed to it in advance). A probe back at that
    configuration risks nothing; B's obligation there is still owed, and
    is tested at a configuration not yet seen."""
    names = ["s0", "s1", "s2"]
    space = [{"conditions": dict(zip(names, bits)), "key": bits}
             for bits in __import__("itertools").product((False, True), repeat=3)]
    at = lambda a, b, c: {"conditions": {"s0": a, "s1": b, "s2": c}, "key": (a, b, c)}
    hs = _replayed([_rule("A: зависит от s0", lambda c: c["s0"]),
                    _rule("B: зависит от s1", lambda c: c["s1"]),
                    inquiry.Hypothesis(inquiry.OTHER, lambda p: {True: 0.5, False: 0.5})],
                   [(at(True, False, False), False)], space)
    chooser = at(True, False, False)
    lead = hs.get("B: зависит от s1")
    assert hs.leader()[0] == lead.name
    assert hs.covered(chooser)                                         # coverage: yes
    params, outcome = hs.history[0]
    assert not hs._vouches(hs.evidence[0], params, outcome, lead)      # evidence: no
    assert hs._dist(lead, params) != hs._dist(hs.get("A: зависит от s0"), params)  # selection: yes

    assert hs.untested_claims(chooser) == 0                   # known outcome: no risk
    same_claim_elsewhere = at(True, False, True)              # s1=0 again, not yet seen
    assert not hs.covered(same_claim_elsewhere)
    assert hs.untested_claims(same_claim_elsewhere) > 0       # still owed, still testable
    assert hs.current_misfit() == pytest.approx(inquiry.MISFIT_PRIOR)   # nothing free


def test_a_distinguishing_observation_confirms_only_what_the_rival_disputed():
    """B leads; a rival that reads s2 disagrees at the probe. Of B's first-time
    obligations there, only the one about s2 was at risk -- the rest were
    touched, not tested."""
    names = ["s0", "s1", "s2"]
    space = [{"conditions": dict(zip(names, bits)), "key": bits}
             for bits in __import__("itertools").product((False, True), repeat=3)]
    hs = inquiry.HypothesisSet([_rule("B: зависит от s0", lambda c: c["s0"]),
                                _rule("R: s0 и s2", lambda c: c["s0"] and c["s2"]),
                                inquiry.Hypothesis(inquiry.OTHER, lambda p: {True: 0.5, False: 0.5})])
    hs.get("B: зависит от s0").weight, hs.get("R: s0 и s2").weight = 0.6, 0.3
    hs.misfit, hs.vulnerability, hs.space = inquiry.MISFIT_PRIOR, inquiry.CLAIMS, space
    probe = {"conditions": {"s0": True, "s1": True, "s2": False}, "key": (True, True, False)}
    lead = hs.get("B: зависит от s0")
    checked = hs._claims_at(lead, probe)
    at_risk = hs._at_risk(lead, probe, checked)
    assert len(checked) == 2 and at_risk == {((True,), "s2", False)}

    touched = inquiry.HypothesisSet([_rule("B: зависит от s0", lambda c: c["s0"]),
                                     _rule("R: s0 и s2", lambda c: c["s0"] and c["s2"]),
                                     inquiry.Hypothesis(inquiry.OTHER, lambda p: {True: 0.5, False: 0.5})])
    touched.get("B: зависит от s0").weight, touched.get("R: s0 и s2").weight = 0.6, 0.3
    touched.misfit, touched.vulnerability, touched.space = inquiry.MISFIT_PRIOR, inquiry.CLAIMS, space
    hs.claim_level = True
    hs.update_misfit(probe, True)
    touched.update_misfit(probe, True)
    assert hs.misfit > touched.misfit          # one obligation at risk, not two touched


def test_the_two_measures_are_one_switch():
    hs = inquiry.condition_hypotheses(["s0", "s1"])
    hs.space = [{"conditions": {"s0": a, "s1": b}, "key": (a, b)}
                for a in (False, True) for b in (False, True)]
    hs.update({"conditions": {"s0": True, "s1": False}, "key": (True, False)}, True)
    probe = {"conditions": {"s0": False, "s1": True}}
    assert hs.exposure(probe) == hs.novelty(probe) == 2
    hs.vulnerability = inquiry.CLAIMS
    assert hs.exposure(probe) == hs.untested_claims(probe)


def test_the_claims_measure_raises_no_false_alarm_where_the_model_is_right():
    for seed in range(8):
        trial, _ = dev.run_inquiry(dev.random_device(seed), budget=250, model_check=True,
                                   vulnerability=inquiry.CLAIMS)
        assert trial.correct == 5 and trial.unexplained == 0, seed


def test_a_world_that_breaks_the_unanimous_model_raises_misfit():
    rule = inquiry.Hypothesis("зависит от s0",
                              lambda p: {True: float(p["conditions"]["s0"]),
                                         False: float(not p["conditions"]["s0"])})
    hs = inquiry.HypothesisSet([rule, inquiry.Hypothesis(
        inquiry.OTHER, lambda p: {True: 0.5, False: 0.5}, weight=0.05)])
    hs.misfit = inquiry.MISFIT_PRIOR
    hs.update({"conditions": {"s0": True, "s1": False}}, True)
    hs.update_misfit({"conditions": {"s0": True, "s1": True}}, False)
    assert hs.misfit == pytest.approx(1.0)
    assert hs.revise_from is not None and hs.revise_from.name == "зависит от s0"


def test_with_the_check_on_a_world_it_can_explain_costs_no_false_alarm():
    for seed in range(8):
        trial, _ = dev.run_inquiry(dev.random_device(seed), budget=250, model_check=True)
        assert trial.correct == 5 and trial.wrong == 0 and trial.unexplained == 0, seed


def test_noise_is_not_taken_for_a_wrong_model():
    for seed in range(4):
        trial, _ = dev.run_inquiry(dev.random_device(seed, noise=0.1), budget=500,
                                   model_check=True)
        assert trial.unexplained == 0 and trial.wrong == 0, seed


@pytest.mark.xfail(strict=True, reason=(
    "найдено экспериментом: при заявленных MISFIT_PRIOR=0.05 и DEVIATION=0.5 "
    "проверка модели обнаружила правило «И трёх» в 9 устройствах из 50; "
    "вера в несоответствие тает на обычных различающих зондах, не касавшихся "
    "области, где лидер неправ"))
def test_the_check_finds_a_rule_no_hypothesis_can_state_in_most_devices():
    found = 0
    for seed in range(10):
        trial, _ = dev.run_inquiry(dev.random_device(seed, needs=3), budget=250,
                                   model_check=True)
        found += trial.verdicts["b0"] == dev.UNEXPLAINED
    assert found > 5


def test_a_model_is_accepted_only_after_surviving_its_own_checks():
    """With k checks to close, every accepted leader survived k model checks
    of its own -- or nothing was left that could check it."""
    device = dev.random_device(0)
    knowledge = dev.knowledge_for(device)
    inquiry.inquire(lambda: inquiry.unsettled(knowledge), dev.DeviceProbes(device),
                    budget=250, challenge=dev.challenge_for(device), model_check=True,
                    vulnerability=inquiry.CLAIMS, checks_to_close=3)
    for button, hs in knowledge.items():
        assert hs.settled and hs.settled[0] == inquiry.ANSWERED, button
        lead = hs._lead()
        assert (hs._survived.get(id(lead), 0) >= 3
                or not any(hs.exposure(p) > 0 for p in hs.space)), button
    assert all(v == dev.CORRECT for v in dev.verdicts(device, knowledge).values())


def test_doubt_falls_only_for_what_the_leader_predicted_in_advance():
    """Doubt starts at the stated share and falls by the stated strength for
    each configuration not seen before that the leader called right. A
    repeat tells nothing; a model that did not lead then earns nothing."""
    s0 = _rule("зависит от s0", lambda c: c["s0"])
    s1 = _rule("зависит от s1", lambda c: c["s1"])
    hs = inquiry.HypothesisSet([s0, s1])
    s0.weight, s1.weight = 0.9, 0.1
    base = hs.doubt()
    assert base == pytest.approx(inquiry.FAMILY_MISS)
    here = {"key": (True, False), "conditions": {"s0": True, "s1": False}}
    hs.note_observation(here, True)
    hs.update(here, True)
    odds = lambda p: p / (1 - p)
    assert odds(hs.doubt()) == pytest.approx(odds(base) * (1 - inquiry.CHECK_STRENGTH))
    after_one = hs.doubt()
    hs.note_observation(here, True)                 # the same configuration again
    hs.update(here, True)
    assert hs.doubt() == pytest.approx(after_one)
    s0.weight, s1.weight = 0.1, 0.9                 # another model leads now
    assert hs.doubt() == pytest.approx(base)


def test_a_dear_error_buys_more_checks_than_a_cheap_one():
    """The same device, two prices of a wrong answer: the dearer one checks
    the model more before accepting it, and both still answer correctly."""
    runs = {}
    for cost in (10, 500):
        trial, report = dev.run_inquiry(dev.random_device(0), 250, model_check=True,
                                        vulnerability=inquiry.CLAIMS, error_cost=cost)
        assert trial.correct == 5, cost
        runs[cost] = sum(1 for s in report.steps if s.motive == inquiry.MODEL_CHECK)
    assert runs[500] > runs[10]


def _stakes_set(cost, weight):
    """Leader "depends on s0" over four configurations of s0, s1."""
    hs = inquiry.condition_hypotheses(["s0", "s1"])
    hs.space = [{"key": (a, b), "config": (a, b), "conditions": {"s0": a, "s1": b}}
                for a in (False, True) for b in (False, True)]
    hs.stakes = inquiry.Stakes(error_cost=cost, weight=weight)
    for h in hs.hypotheses:
        h.weight = 0.01
    hs.get("зависит от s0").weight = 0.9
    return hs


def test_a_claim_the_task_does_not_rest_on_protects_nothing():
    """The same claim, two tasks: worth the error it could prevent where the
    task relies on it, nothing where it does not -- and in proportion to
    what an error costs."""
    only_s1 = lambda p: 1.0 if p["conditions"]["s1"] else 0.0
    hs = _stakes_set(10.0, only_s1)
    unused = {"key": (True, False), "conditions": {"s0": True, "s1": False}}
    used = {"key": (True, True), "conditions": {"s0": True, "s1": True}}
    assert hs.claim_stake(unused) == 0.0
    assert hs.claim_stake(used) == pytest.approx(10.0 * 0.5)
    assert _stakes_set(20.0, only_s1).claim_stake(used) == pytest.approx(2 * hs.claim_stake(used))


def test_a_claim_seen_once_is_only_partly_tested_when_graded():
    """Leader "depends on s0" over s0, s1, s2. After (1,1,0), the claim "at
    s0=1 the outcome does not depend on s1=1" is seen at one of its two
    configurations. Binary: settled, it protects nothing at (1,1,1).
    Graded: half of it is still open."""
    def make(graded):
        hs = inquiry.condition_hypotheses(["s0", "s1", "s2"])
        hs.space = [{"key": c, "config": c, "conditions": {"s0": c[0], "s1": c[1], "s2": c[2]}}
                    for c in ((a, b, e) for a in (False, True) for b in (False, True)
                              for e in (False, True))]
        hs.stakes = inquiry.Stakes(error_cost=8.0)
        hs.claims_graded = graded
        for h in hs.hypotheses:
            h.weight = 0.01
        hs.get("зависит от s0").weight = 0.9
        seen = hs.space[6]                                  # (1, 1, 0)
        hs.note_claims(seen, True)
        hs.update(seen, True)
        return hs
    probe = {"key": (True, True, True), "conditions": {"s0": True, "s1": True, "s2": True}}
    # two claims at (1,1,1), each a quarter of the task: s1=1 half seen, s2=1 unseen
    assert make(False).claim_stake(probe) == pytest.approx(8.0 * 0.25)
    assert make(True).claim_stake(probe) == pytest.approx(8.0 * (0.25 * 0.5 + 0.25))


def test_three_inclusion_rules_reach_three_distances_from_what_was_seen():
    """Leader "depends on s0" over s0, s1, s2, one independent observation
    at (1,1,0): `observed` keeps that point, `claims` keeps it too (the rest
    of s0=1 needs s1=0 or s2=1 seen), `pattern` takes all of s0=1. A second
    observation at (1,0,1) completes the claims for all of s0=1."""
    def make(rule):
        hs = inquiry.condition_hypotheses(["s0", "s1", "s2"])
        hs.space = [{"key": c, "config": c, "conditions": {"s0": c[0], "s1": c[1], "s2": c[2]}}
                    for c in ((a, b, e) for a in (False, True) for b in (False, True)
                              for e in (False, True))]
        hs.stakes = inquiry.Stakes(error_cost=8.0)
        hs.boundary = rule
        for h in hs.hypotheses:
            h.weight = 0.01
        hs.get("зависит от s0").weight = 0.9
        return hs
    def observe(hs, index):
        seen = hs.space[index]
        hs.note_claims(seen, True)
        hs.update(seen, True)
    s0_on = {(True, b, e) for b in (False, True) for e in (False, True)}
    sizes = {}
    for rule in (inquiry.BOUNDARY_OBSERVED, inquiry.BOUNDARY_CLAIMS, inquiry.BOUNDARY_PATTERN):
        hs = make(rule)
        observe(hs, 6)                                      # (1, 1, 0)
        lead = hs.get("зависит от s0")
        sizes[rule] = hs._inside(lead)
        observe(hs, 5)                                      # (1, 0, 1)
        sizes[rule, 2] = hs._inside(lead)
    assert sizes[inquiry.BOUNDARY_OBSERVED] == {(True, True, False)}
    assert sizes[inquiry.BOUNDARY_CLAIMS] == {(True, True, False)}
    assert sizes[inquiry.BOUNDARY_PATTERN] == s0_on
    assert sizes[inquiry.BOUNDARY_CLAIMS, 2] == s0_on
    assert sizes[inquiry.BOUNDARY_OBSERVED, 2] == {(True, True, False), (True, False, True)}


def test_an_answer_closes_with_its_standing_for_the_task():
    """Every answer carries its standing: verified for the task, or
    conditional with the part the task relies on unverified; and nothing
    a verified answer covers is wrong on a world its family explains."""
    device = dev.random_device(0)
    knowledge = dev.knowledge_for(device)
    inquiry.inquire(lambda: inquiry.unsettled(knowledge), dev.DeviceProbes(device), 250,
                    challenge=dev.challenge_for(device),
                    stakes=inquiry.Stakes(error_cost=64.0), boundary=inquiry.BOUNDARY_CLAIMS)
    assert all(v == dev.CORRECT for v in dev.verdicts(device, knowledge).values())
    for hs in knowledge.values():
        standing = hs.applicability
        assert standing["status"] in (inquiry.VERIFIED_FOR_TASK, inquiry.CONDITIONAL)
        assert (standing["status"] == inquiry.CONDITIONAL) == bool(standing["uncovered"])


def test_the_claims_economy_needs_no_misfit():
    """Every answer right, the model checked, and no belief about the model
    as a whole anywhere."""
    device = dev.random_device(0)
    knowledge = dev.knowledge_for(device)
    report = inquiry.inquire(lambda: inquiry.unsettled(knowledge), dev.DeviceProbes(device),
                             250, challenge=dev.challenge_for(device),
                             stakes=inquiry.Stakes(error_cost=8.0))
    assert all(v == dev.CORRECT for v in dev.verdicts(device, knowledge).values())
    assert all(hs.misfit is None for hs in knowledge.values())
    assert any(s.motive == inquiry.MODEL_CHECK for s in report.steps)


def test_the_claims_economy_revises_once_before_calling_a_rule_unexplained():
    """Found on C devices 3, 41 and 43: once no explanation fit, OTHER led and
    the rule -- two switches at once, which the challenge can state -- was
    called unexplained. The refuted leader gets one revision first."""
    for seed in (3, 41, 43):
        trial, _ = dev.run_inquiry(dev.random_device(seed, needs=2), 250,
                                   stakes=inquiry.Stakes(error_cost=1.0))
        assert trial.verdicts["b0"] == dev.CORRECT, seed


def test_world_e_hides_an_interaction_of_two_conditions():
    """b0 works when a is on unless e and f are both on; every other button
    is the one the same seed gives on D."""
    rule = dev.Rule("unless_both", (0, 1, 2))
    assert rule.holds([True, False, True]) and rule.holds([True, True, False])
    assert not rule.holds([True, True, True]) and not rule.holds([False, False, False])
    e = dev.random_device(0, hidden_pair=True)
    d = dev.random_device(0, needs=3)
    assert e.rules["b0"].kind == "unless_both"
    assert {b: r for b, r in e.rules.items() if b != "b0"} == \
        {b: r for b, r in d.rules.items() if b != "b0"}


def test_under_noise_it_does_not_stop_while_it_can_still_finish():
    for seed in range(4):
        trial, report = dev.run_inquiry(dev.random_device(seed, noise=0.1), budget=500)
        assert report.stopped in (inquiry.NO_QUESTIONS, inquiry.BUDGET), seed
        assert trial.wrong == 0, seed
