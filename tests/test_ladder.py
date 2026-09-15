"""The ladder of necessary depth (H3, D0): an instrument, held to its spec."""
import ast
import inspect

import numpy as np
import pytest

from mana.discovery import ladder
from mana.discovery.language import CMP, IF, Evaluator, children, size


def _walk(p, parent=None, position=None):
    yield p, parent, position
    for i, kid in enumerate(children(p)):
        yield from _walk(kid, p, i)


def test_the_same_seed_gives_the_same_world_and_seeds_differ():
    for family in ladder.FAMILIES:
        assert ladder.truth(family, 3, 7) == ladder.truth(family, 3, 7)
        assert len({ladder.truth(family, 3, s) for s in range(10)}) > 5


def test_a_rung_is_larger_than_the_one_below_and_rung_one_is_flat():
    for family in ladder.FAMILIES:
        for seed in range(5):
            sizes = [size(ladder.truth(family, k, seed)) for k in ladder.RUNGS]
            assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
            assert sizes[0] == 6                      # one if(c, a, b): the flat control C0


def test_every_branch_of_a_case_holds_on_a_tenth_of_the_world():
    states = ladder._states()
    for seed in range(10):
        p = ladder.truth("S", 5, seed)
        region = np.ones(len(states["x"]), dtype=bool)
        while p[0] == IF:
            true = (ladder._vector(p[1], states) != 0) & region
            assert np.mean(true) >= ladder.MIN_BRANCH
            region &= ~true
            p = p[3]
        assert np.mean(region) >= ladder.MIN_BRANCH


def test_the_ladders_interpreter_agrees_with_the_language_on_every_state():
    states = ladder._states()
    evaluator = Evaluator(states)
    for family in ladder.FAMILIES + ladder.CONTROLS:
        for rung in ([1] if family in ladder.CONTROLS else ladder.RUNGS):
            p = ladder.truth(family, rung, 3)
            assert np.array_equal(ladder._vector(p, states), evaluator(p))
            w = ladder.world(family, rung, 3)
            sample = {v: int(states[v][1234]) for v in ladder.VARIABLES}
            assert w.rule(sample) == int(evaluator(p)[1234])


def test_a_comparison_appears_only_as_the_condition_of_an_if():
    for family in ladder.FAMILIES + ladder.CONTROLS:
        for rung in ([1] if family in ladder.CONTROLS else ladder.RUNGS):
            for seed in range(5):
                for node, parent, position in _walk(ladder.truth(family, rung, seed)):
                    if node[0] == CMP:
                        assert parent is not None and parent[0] == IF and position == 0


def test_the_out_of_catalogue_control_is_a_comparison_of_sums():
    p = ladder.truth("C1", 1, 0)
    assert p[0] == IF and p[1][0] == CMP and p[2] == ("const", 1) and p[3] == ("const", 0)
    with pytest.raises(ValueError):
        ladder.truth("C1", 2, 0)


def test_noise_is_a_declared_parameter_of_every_world():
    clean, noisy = ladder.world("S", 2, 0), ladder.world("S", 2, 0, noise=ladder.NOISE)
    assert clean.noise == 0.0 and noisy.noise == ladder.NOISE and clean.truth == noisy.truth


def test_no_learner_can_see_the_ladder():
    from mana.discovery import (description, language, policy, reflect, search,
                                selection)
    for module in (description, language, policy, reflect, search, selection):
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                named = [getattr(node, "module", "") or ""] + [a.name for a in node.names]
                assert not any("ladder" in name for name in named)
