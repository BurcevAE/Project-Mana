"""Widening the space of explanations from the model's own state
(docs/РАСШИРЕНИЕ_ПРОСТРАНСТВА.md): the prior is the description code itself;
belief over the language decides with the existing rule; not enough keeps
the question open; the estimate cannot see the world."""
import ast
import inspect
import random
from collections import Counter

from mana.cognition import doubt, inquiry
from mana.discovery import description, prior
from mana.discovery.language import const, get
from mana.world import device as dev

NAMES = ["s0", "s1", "s2"]


def _law(table_of):
    table = tuple(table_of(i) for i in range(8))
    return dev.Device(3, {"b": dev.Rule("table", table=table)}, noise=0.0, seed=0)


def _observe(device, h, configs):
    probes = dev.DeviceProbes(device)
    for c in configs:
        h.update(probes.params("b", c), device.rules["b"].holds(c))


def test_a_draw_has_the_probability_its_description_gives_it():
    """Programs of one node over one variable: the variable has 2^-log2(6) and
    all constants together the same, so each half; the constant 0 is the
    shortest integer code, one bit, so a quarter."""
    rng = random.Random(1)
    drawn = Counter(prior.sample(["x"], rng, most=1) for _ in range(40000))
    n = sum(drawn.values())
    assert abs(drawn[get("x")] / n - 0.5) < 0.02
    assert abs(drawn[const(0)] / n - 0.25) < 0.02
    ratio = (drawn[const(1)] / n) / (drawn[const(0)] / n)
    expected = 2.0 ** -(description.integer_bits(1) - description.integer_bits(0))
    assert abs(ratio - expected) < 0.05


def test_the_chains_agree_with_unbiased_rejection_from_the_prior():
    """Where rejection is cheap -- two observations, many consistent programs
    -- the posterior share of the leader's behaviour by the chains matches
    the unbiased share by drawing from the prior and rejecting."""
    device = _law(lambda i: bool(i >> 1 & 1))                      # s1
    h = dev.knowledge_for(device)["b"]
    _observe(device, h, [(True, True, False), (False, False, False)])
    space = dev.DeviceProbes(device).space("b")
    lead = h.get("зависит от s1")
    leader = tuple(lead.predict(p)[True] > 0.5 for p in space)
    agreeing, consistent = doubt.by_rejection(NAMES, h.history, leader, space, 150000, seed=5)
    reference = agreeing / consistent
    est = doubt.over_language(NAMES, cap=60000, batch=60000, seed=3)(h, lead, space)
    assert consistent > 3000
    assert abs(est.belief - reference) < 0.05, (est.belief, reference)


def test_an_autocorrelated_series_is_worth_fewer_samples():
    rng = random.Random(0)
    independent = [float(rng.random() < 0.5) for _ in range(4000)]
    sticky, state = [], 0.0
    for _ in range(4000):
        if rng.random() < 0.02:
            state = 1.0 - state
        sticky.append(state)
    assert doubt.effective_size(independent) > 2000
    assert doubt.effective_size(sticky) < 400
    assert doubt.effective_size([1.0] * 100) == 100


def test_the_estimate_cannot_see_the_world():
    for module in (doubt, prior):
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                named = [getattr(node, "module", "") or ""] + [a.name for a in node.names]
                assert not any("world" in n or "device" in n for n in named)


def test_with_every_configuration_seen_belief_over_the_language_is_whole():
    device = _law(lambda i: bool(i >> 1 & 1))                      # s1
    h = dev.knowledge_for(device)["b"]
    every = [tuple(bool(i >> k & 1) for k in range(3)) for i in range(8)]
    _observe(device, h, every)
    lead = h.get("зависит от s1")
    space = dev.DeviceProbes(device).space("b")
    est = doubt.over_language(NAMES, cap=20000)(h, lead, space)
    assert est.belief == 1.0 and est.enough and not est.rivals and not est.language_short


def test_with_nothing_seen_belief_over_the_language_is_not_enough():
    device = _law(lambda i: bool(i >> 1 & 1))
    h = dev.knowledge_for(device)["b"]
    lead = h.get("зависит от s1")
    space = dev.DeviceProbes(device).space("b")
    est = doubt.over_language(NAMES, cap=20000)(h, lead, space)
    assert not est.enough and est.rivals


def test_not_enough_keeps_the_question_open_even_with_nothing_to_add():
    device = _law(lambda i: bool(i >> 1 & 1))
    h = dev.knowledge_for(device)["b"]
    _observe(device, h, [(True, True, False), (False, False, False)])
    space = dev.DeviceProbes(device).space("b")
    never = lambda hs, lead, sp: doubt.Estimate(0.1, 0.0, 0.2, 10, 1, 100, enough=False,  # noqa: E731
                                                decided=True, language_short=False)
    h.get("зависит от s1").weight = 0.999
    for other in h.hypotheses:
        if other.name != "зависит от s1":
            other.weight = 0.001 / (len(h.hypotheses) - 1)
    assert inquiry.settle(h, space, None, language=never) is None
    assert h.widened == 1 and inquiry.settle(h, space, None, language=never) is None
    assert h.widened == 1                                  # once per history


def test_without_the_estimate_settle_is_what_it_was():
    device = _law(lambda i: bool(i >> 1 & 1))
    knowledge = dev.knowledge_for(device)
    _observe(device, knowledge["b"], [(True, True, False), (False, False, False)])
    inquiry.inquire(lambda: inquiry.unsettled(knowledge), dev.DeviceProbes(device), 40,
                    challenge=dev.challenge_for(device))
    h = knowledge["b"]
    assert h.widened == 0 and not h.space_estimates and h.settled[0] == inquiry.ANSWERED
