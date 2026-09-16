"""The joint of inquiry and discovery (docs/РОЖДЕНИЕ_ГИПОТЕЗ.md): OTHER calls
for new explanations instead of ending the question; the generator sees only
the history; without it inquiry is what it was."""
import ast
import inspect

from mana.cognition import explain, inquiry
from mana.world import device as dev

NAMES = ["s0", "s1", "s2"]


def _two_of_three():
    table = tuple(sum(bool(i >> k & 1) for k in range(3)) >= 2 for i in range(8))
    return dev.Device(3, {"b": dev.Rule("table", table=table)}, noise=0.0, seed=0)


def _observe(device, hypotheses, configs):
    probes = dev.DeviceProbes(device)
    for config in configs:
        hypotheses.update(probes.params("b", config), device.rules["b"].holds(config))


def test_a_table_rule_holds_by_its_table():
    table = tuple(i % 3 == 0 for i in range(8))
    rule = dev.Rule("table", table=table)
    for i in range(8):
        state = [bool(i >> k & 1) for k in range(3)]
        assert rule.holds(state) == table[i]


def test_the_generator_cannot_see_the_world():
    for node in ast.walk(ast.parse(inspect.getsource(explain))):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            named = [getattr(node, "module", "") or ""] + [a.name for a in node.names]
            assert not any("world" in n or "device" in n for n in named)


def test_the_generator_offers_programs_exact_on_the_history():
    device = _two_of_three()
    hypotheses = dev.knowledge_for(device)["b"]
    seen = [(False, False, False), (True, True, False), (True, False, False)]
    _observe(device, hypotheses, seen)
    born, mass = explain.from_history(NAMES)(hypotheses)
    probes = dev.DeviceProbes(device)
    assert 0 < len(born) <= explain.MOST and len(mass) == len(born)
    assert all(m > 0 for m in mass)
    for h in born:
        for config in seen:
            works = h.predict(probes.params("b", config))[True] > 0.5
            assert works == device.rules["b"].holds(config)


def test_without_the_hook_other_still_ends_the_question():
    device = _two_of_three()
    hypotheses = dev.knowledge_for(device)["b"]
    _observe(device, hypotheses, [(True, True, False), (True, False, False), (False, True, False),
                                  (False, False, True)])
    space = dev.DeviceProbes(device).space("b")
    assert inquiry.settle(hypotheses, space, None)[0] == inquiry.UNEXPLAINED


def test_with_the_hook_other_calls_for_explanations_sharing_its_prior():
    device = _two_of_three()
    hypotheses = dev.knowledge_for(device)["b"]
    _observe(device, hypotheses, [(True, True, False), (True, False, False), (False, True, False),
                                  (False, False, True)])
    space = dev.DeviceProbes(device).space("b")
    other = hypotheses.get(inquiry.OTHER).prior
    before = len(hypotheses.hypotheses)
    assert inquiry.settle(hypotheses, space, None, explain=explain.from_history(NAMES)) is None
    assert hypotheses.explained == 1
    born = hypotheses.hypotheses[before:]
    assert born and all(h.name.startswith("программа") for h in born)
    total = sum(h.prior for h in hypotheses.hypotheses)
    assert abs(sum(h.prior for h in born) / total - other / (1.0 + other)) < 1e-9


def test_the_calls_stop_at_the_declared_limit():
    """A generator that offers nothing keeps the question open only so many
    times; then OTHER ends it, as it did before the joint."""
    device = _two_of_three()
    hypotheses = dev.knowledge_for(device)["b"]
    _observe(device, hypotheses, [(True, True, False), (True, False, False), (False, True, False),
                                  (False, False, True)])
    space = dev.DeviceProbes(device).space("b")
    asked = []
    nothing = lambda h: (asked.append(1), ([], []))[1]  # noqa: E731
    for _ in range(inquiry.EXPLAIN_LIMIT):
        assert inquiry.settle(hypotheses, space, None, explain=nothing) is None
    assert len(asked) == inquiry.EXPLAIN_LIMIT == hypotheses.explained
    assert inquiry.settle(hypotheses, space, None, explain=nothing)[0] == inquiry.UNEXPLAINED
    assert len(asked) == inquiry.EXPLAIN_LIMIT


def test_in_the_loop_other_asks_the_generator_instead_of_ending():
    """Inside `inquire`, when OTHER leads the hook is asked and the loop goes
    on; without it the question ends unexplained. Which answer comes out is
    the probe's to measure, not a test's: a family can also accept a wrong
    explanation before OTHER ever leads (found before the probe, 10.7)."""
    table = tuple(bool(bin(i).count("1") % 2) for i in range(8))          # parity
    every = [tuple(bool(i >> k & 1) for k in range(3)) for i in range(8)]
    endings = {}
    for name, make in (("с", lambda: explain.from_history(NAMES)), ("без", lambda: None)):
        device = dev.Device(3, {"b": dev.Rule("table", table=table)}, noise=0.0, seed=0)
        knowledge = dev.knowledge_for(device)
        _observe(device, knowledge["b"], every)
        asked = []
        inner = make()
        hook = None if inner is None else (lambda h, inner=inner: (asked.append(1), inner(h))[1])
        inquiry.inquire(lambda: inquiry.unsettled(knowledge), dev.DeviceProbes(device), 40,
                        challenge=dev.challenge_for(device), explain=hook)
        endings[name] = (len(asked), knowledge["b"].settled)
    assert endings["с"][0] >= 1
    assert endings["без"] == (0, (inquiry.UNEXPLAINED, (inquiry.OTHER,)))
