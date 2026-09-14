"""
Discovery, step 1: a poor language, syntactic edits, and one currency --
bits. What is pinned: the arithmetic of the currency, that the search finds
the rule where one exists and does not buy back noise, and that no learner
module can see the world's truth.
"""
from __future__ import annotations

import ast
import inspect

import numpy as np

from mana.discovery import baselines, description
from mana.discovery import search as discovery
from mana.discovery.language import (EQUAL, LESS, Evaluator, cmp, const, get,
                                     if_, replace, show, size)
from mana.discovery.worlds import W0, W3


def test_a_program_is_a_value_that_evaluates_on_columns():
    rule = if_(cmp(LESS, const(5), get("x")), get("y"), get("z"))
    out = Evaluator({"x": [6, 2], "y": [1, 1], "z": [7, 7]})(rule)
    assert out.tolist() == [1, 7]
    assert show(rule) == "if((5 < x), y, z)" and size(rule) == 6
    assert replace(rule, (2,), get("x")) == if_(rule[1], get("y"), get("x"))
    assert rule == if_(cmp(LESS, const(5), get("x")), get("y"), get("z"))


def test_integers_are_written_in_elias_gamma_on_the_zigzag():
    assert description.integer_bits(0) == 1
    assert description.integer_bits(-1) == 3
    assert description.integer_bits(5) == 7


def _bits(split, program):
    on_train = Evaluator(split.train)
    alphabet = description.alphabet_of(split.train_outcomes)
    return (description.program_bits(program, 3)
            + description.error_bits(on_train(program), split.train_outcomes, alphabet))


def test_remembering_one_exception_costs_more_than_it_saves():
    """The currency, in one case: fixing a single noisy point with a
    condition that names it is a worse description, not a better one."""
    split = W3.split(200, 50, seed=0)
    rule = W3.truth
    on_train = Evaluator(split.train)
    wrong = np.nonzero(on_train(rule) != split.train_outcomes)[0]
    assert len(wrong)
    i = int(wrong[0])
    point = {name: int(split.train[name][i]) for name in ("x", "y", "z")}
    patched = if_(cmp(EQUAL, get("x"), const(point["x"])),
                  if_(cmp(EQUAL, get("y"), const(point["y"])),
                      if_(cmp(EQUAL, get("z"), const(point["z"])),
                          const(int(split.train_outcomes[i])), rule), rule), rule)
    assert int(np.count_nonzero(on_train(patched) != split.train_outcomes)) == len(wrong) - 1
    assert _bits(split, patched) > _bits(split, rule)


def test_a_table_of_the_training_states_costs_more_than_the_rule():
    split = W0.split(200, 50, seed=0)
    alphabet = description.alphabet_of(split.train_outcomes)
    assert _bits(split, W0.truth) < description.table_bits(200, 3, 10, alphabet)


def test_the_search_finds_the_rule_of_w0():
    split = W0.split(150, 100, seed=0)
    found = discovery.search(split.train, split.train_outcomes)
    assert W0.grade(lambda cols: discovery.predict(found.program, cols)) == 1.0
    assert found.bits <= _bits(split, W0.truth) + 1e-6


def test_on_noise_it_keeps_the_rule_and_buys_no_exceptions():
    split = W3.split(200, 100, seed=0)
    found = discovery.search(split.train, split.train_outcomes)
    assert W3.grade(lambda cols: discovery.predict(found.program, cols)) == 1.0
    assert size(found.program) <= size(W3.truth)


def test_the_budget_is_respected():
    split = W0.split(150, 50, seed=1)
    found = discovery.search(split.train, split.train_outcomes, budget=500)
    assert found.evaluations <= 500


def test_the_floor_is_what_it_says():
    split = W0.split(200, 300, seed=0)
    table = baselines.table_predict(split.train, split.train_outcomes, split.test)
    tree, nodes = baselines.tree_predict(split.train, split.train_outcomes, split.test)
    assert len(table) == len(tree) == 300 and nodes > 1


def test_no_learner_module_can_see_the_worlds():
    """A learner that peeks scores perfectly and means nothing."""
    from mana.discovery import language, search

    for module in (language, description, search, baselines):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                named = [node.module or ""] + [alias.name for alias in node.names]
            elif isinstance(node, ast.Import):
                named = [alias.name for alias in node.names]
            else:
                continue
            assert not any("worlds" in name for name in named), module.__name__


# --------------------------------------------------------------------------
# step 2: a variable nobody observed
# --------------------------------------------------------------------------

def test_a_transition_is_run_along_each_episode_from_its_start():
    from mana.discovery import invent
    from mana.discovery.language import SUB

    flips = if_(get("a"), (SUB, const(1), get(invent.PREV)), get(invent.PREV))
    a = np.array([[0, 1, 0, 1, 1], [1, 0, 0, 0, 0]])
    assert invent.run(flips, 0, {"a": a}).tolist() == [[0, 1, 1, 0, 1], [1, 1, 1, 1, 1]]


def test_the_hidden_switch_of_w2_is_invented_and_pays_for_itself():
    from mana.discovery import invent
    from mana.discovery.worlds import W2

    train, test = W2.split(20, 50, seed=0)
    found = invent.invent(train.columns, train.outcomes)
    assert found.accepted, found.note
    assert found.bits < found.base.bits
    predicted = invent.predict(found, test.columns)
    assert float(np.mean(predicted == test.clean)) >= 0.99
    value = invent.hidden(found, test.columns)
    recovered = max(np.mean(value == test.hidden), np.mean(value != test.hidden))
    assert recovered >= 0.99
    # Without the new variable the same search is near a coin between x and y.
    flat = {name: values.reshape(-1) for name, values in test.columns.items()}
    base = discovery.predict(found.base.program, flat)
    assert float(np.mean(base == test.clean.reshape(-1))) <= 0.65


def test_where_the_program_is_right_there_is_nothing_to_invent():
    from mana.discovery import invent

    split = W0.split(200, 50, seed=0)
    shaped = {name: values.reshape(10, 20) for name, values in split.train.items()}
    found = invent.invent(shaped, split.train_outcomes.reshape(10, 20))
    assert not found.accepted and "объяснять нечего" in found.note


def test_noise_does_not_earn_an_invented_cause():
    """The same currency that refuses a table of exceptions refuses a
    hidden variable made up to explain them."""
    from mana.discovery import invent

    split = W3.split(200, 50, seed=0)
    shaped = {name: values.reshape(10, 20) for name, values in split.train.items()}
    found = invent.invent(shaped, split.train_outcomes.reshape(10, 20))
    assert not found.accepted, found.describe()


def test_the_inventor_cannot_see_the_worlds_either():
    from mana.discovery import invent

    tree = ast.parse(inspect.getsource(invent))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            named = ([getattr(node, "module", "") or ""]
                     + [alias.name for alias in node.names])
            assert not any("worlds" in name for name in named)


# --------------------------------------------------------------------------
# step 3: the language grows a word when a word pays
# --------------------------------------------------------------------------

from mana.discovery.language import HOLE, Primitive, add, nodes, sub  # noqa: E402

#: What the search found for T1 and T3 on seed 0, written out.
_T1 = if_(cmp(LESS, get("y"), get("x")), sub(get("x"), get("y")), sub(get("y"), get("x")))
_T3 = add(get("y"), if_(cmp(LESS, get("x"), get("z")),
                        sub(get("z"), get("x")), sub(get("x"), get("z"))))


def test_two_pieces_generalise_to_a_template_with_holes():
    from mana.discovery import library

    template, arity = library.generalise(_T1, _T3[2])
    assert arity == 2
    assert show(template) == "if((#0 < #1), (#1 - #0), (#0 - #1))"
    assert sum(1 for _, node in nodes(template) if node[0] == HOLE) == 6


def test_rewriting_keeps_what_every_program_computes():
    from mana.discovery import library

    template, arity = library.generalise(_T1, _T3[2])
    word = Primitive("f1", template, arity)
    split = W0.split(200, 50, seed=0)
    for program in (_T1, _T3):
        rewritten = library.rewrite(program, word)
        assert rewritten != program
        assert (Evaluator(split.train, {"f1": word})(rewritten)
                == Evaluator(split.train)(program)).all()


def test_the_distance_is_found_as_a_word_and_pays_for_itself():
    from mana.discovery import library

    grown = library.compress([_T1, _T3], 3)
    assert [name for name, _, _ in grown.kept] == ["f1"]
    assert grown.library["f1"].arity == 2
    assert grown.bits_after < grown.bits_before
    assert [show(p) for p in grown.programs] == ["f1(y, x)", "(y + f1(x, z))"]


def test_a_word_used_once_does_not_pay_for_its_definition():
    from mana.discovery import library

    assert library.compress([_T1], 3).library == {}


def test_a_larger_language_costs_at_every_node():
    assert description.program_bits(_T1, 3, library=1) > description.program_bits(_T1, 3)


def test_with_the_word_a_question_over_other_names_is_solved():
    """Transfer: W4 is |q - r| + p. The starting language did not find it
    within the budget on this split; with the word learned from T1 and T3
    it is two edits away."""
    from mana.discovery import library
    from mana.discovery.worlds import W4

    grown = library.compress([_T1, _T3], 3)
    split = W4.split(200, 300, seed=0)
    found = discovery.search(split.train, split.train_outcomes, library=grown.library)
    assert W4.grade(lambda cols: discovery.predict(found.program, cols, grown.library)) == 1.0
    assert show(found.program) == "(p + f1(q, r))"


def test_a_word_naming_a_variable_the_world_lacks_is_not_offered_there():
    """Learned on seed 2 as |#0 - y|: offered on a world of p, q and r it
    crashed the search. There it simply does not exist."""
    from mana.discovery.worlds import W4

    template = if_(cmp(LESS, get("y"), (HOLE, 0)), sub((HOLE, 0), get("y")),
                   sub(get("y"), (HOLE, 0)))
    words = {"f1": Primitive("f1", template, 1)}
    split = W4.split(200, 50, seed=0)
    leaves, _ = discovery.vocabulary(split.train, split.train_outcomes, words)
    assert not any(leaf[0] == "prim" for leaf in leaves)
    found = discovery.search(split.train, split.train_outcomes, library=words,
                             budget=2000)
    assert "f1" not in show(found.program)


# --------------------------------------------------------------------------
# step 4: measuring the search, not changing it
# --------------------------------------------------------------------------

def test_the_search_says_why_it_stopped_and_what_it_found():
    split = W0.split(200, 50, seed=0)
    found = discovery.search(split.train, split.train_outcomes)
    assert found.termination == discovery.SEARCH_EXHAUSTED
    assert found.status == discovery.FOUND_EXACT and found.train_errors == 0
    assert found.seconds > 0
    cut = discovery.search(split.train, split.train_outcomes, budget=300)
    assert cut.termination == discovery.SEARCH_LIMIT


def test_the_ceiling_finds_the_smallest_exact_program():
    from mana.discovery import ceiling

    rng = np.random.default_rng(0)
    columns = {"x": rng.integers(0, 10, 60), "y": rng.integers(0, 10, 60)}
    leaves, _ = discovery.vocabulary(columns, columns["x"] + columns["y"])
    assert ceiling.smallest_fit(columns, columns["y"], leaves).smallest == 1
    found = ceiling.smallest_fit(columns, columns["x"] + columns["y"], leaves)
    assert found.smallest == 3 and found.stopped == ceiling.FOUND


def test_a_ceiling_that_stops_says_only_what_it_knows():
    from mana.discovery import ceiling

    split = W0.split(200, 50, seed=0)
    leaves, _ = discovery.vocabulary(split.train, split.train_outcomes)
    cut = ceiling.smallest_fit(split.train, split.train_outcomes, leaves,
                               max_classes=50)
    assert cut.smallest is None and cut.stopped == ceiling.CLASSES
    assert cut.lower_bound >= 1


def test_every_question_measured_has_a_certificate_inside_the_searched_space():
    """What makes a miss the search's and not the language's: the rule
    itself, written in the language, within the size the search explores."""
    from mana.discovery.worlds import FAMILY

    for world in [W0, W3] + list(FAMILY.values()):
        assert size(world.truth) <= discovery.MAX_SIZE, world.name
        assert world.grade(lambda cols: discovery.predict(world.truth, cols)) == 1.0


# --------------------------------------------------------------------------
# step 5.0: the whole budget curve from one run
# --------------------------------------------------------------------------

def test_one_profiled_run_gives_what_every_smaller_budget_would_have():
    """The claim that makes the budget curve affordable, checked against
    the real thing: a search with budget B returns exactly what the best of
    the first B programs of a longer run was."""
    from mana.discovery.worlds import FAMILY

    split = FAMILY["T1"].split(200, 50, seed=0)
    long = discovery.search(split.train, split.train_outcomes, budget=400000,
                            profile=True)
    for budget in (3000, 40000, 120000, 400000):
        short = discovery.search(split.train, split.train_outcomes, budget=budget)
        program, found_at, bits = discovery.at_budget(long, budget)
        assert program == short.program, budget
        assert found_at == short.found_at and abs(bits - short.bits) < 1e-9
        assert discovery.termination_at(long, budget) == short.termination, budget


def test_without_a_profile_nothing_is_recorded():
    split = W0.split(150, 50, seed=0)
    found = discovery.search(split.train, split.train_outcomes, budget=2000)
    assert found.anytime == []


# --------------------------------------------------------------------------
# cells: a world one region at a time (experiments A and B)
# --------------------------------------------------------------------------

def test_where_one_model_explains_everything_no_boundary_is_drawn():
    from mana.discovery import cells

    split = W0.split(200, 50, seed=0)
    grown = cells.adaptive(split.train, split.train_outcomes)
    assert not grown.tree.split
    assert W0.grade(lambda cols: discovery.predict(grown.program, cols)) == 1.0


def test_noise_earns_no_boundary():
    from mana.discovery import cells

    split = W3.split(200, 50, seed=0)
    grown = cells.adaptive(split.train, split.train_outcomes)
    assert not grown.tree.split, show(grown.program)


def test_a_rule_that_is_simple_piece_by_piece_is_found_in_pieces():
    """|x - y| is x - y on one side of x = y and y - x on the other."""
    from mana.discovery import cells
    from mana.discovery.worlds import FAMILY

    world = FAMILY["T1"]
    split = world.split(200, 50, seed=0)
    grown = cells.adaptive(split.train, split.train_outcomes)
    assert grown.tree.split and len(grown.tree.leaves()) >= 2
    assert world.grade(lambda cols: discovery.predict(grown.program, cols)) == 1.0


def test_a_split_is_kept_only_where_it_shortens_the_description():
    from mana.discovery import cells
    from mana.discovery.worlds import FAMILY

    split = FAMILY["T1"].split(200, 50, seed=0)
    grown = cells.adaptive(split.train, split.train_outcomes)

    def check(cell):
        if cell.split:
            # The cell's own model is kept beside its split: what it cost.
            assert cell.bits(3) < 1.0 + cell.found.bits
            check(cell.yes)
            check(cell.no)

    check(grown.tree)
    assert grown.evaluations > 0 and grown.searches >= 1


def test_a_grid_is_laid_down_before_looking():
    from mana.discovery import cells

    split = W0.split(200, 50, seed=0)
    grown = cells.grid(split.train, split.train_outcomes, bins=2, local_budget=3000)
    assert len(grown.tree.leaves()) == 8                    # 2 x 2 x 2
    stitched = grown.tree.program()
    whole = discovery.predict(stitched, split.train)
    for cell in grown.tree.leaves():
        assert cell.found is not None
    assert len(whole) == len(split.train_outcomes)


def test_cells_cannot_see_the_worlds():
    from mana.discovery import cells

    tree = ast.parse(inspect.getsource(cells))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            named = ([getattr(node, "module", "") or ""]
                     + [alias.name for alias in node.names])
            assert not any("worlds" in name for name in named)


# --------------------------------------------------------------------------
# step 6: moves of the search, learnt from how answers were reached
# --------------------------------------------------------------------------

def test_a_traced_search_finds_the_same_and_says_how_one_edit_at_a_time():
    split = W0.split(200, 50, seed=1)
    plain = discovery.search(split.train, split.train_outcomes, budget=20000)
    traced = discovery.search(split.train, split.train_outcomes, budget=20000, trace=True)
    assert (plain.program, plain.evaluations, plain.found_at) == \
        (traced.program, traced.evaluations, traced.found_at)
    leaves, conditions = discovery.vocabulary(split.train, split.train_outcomes)
    steps = traced.derivation
    assert steps[-1] == traced.program and steps[0] in leaves
    for p, q in zip(steps, steps[1:]):
        assert q in set(discovery.neighbours(p, leaves, conditions, discovery.MAX_SIZE))


def test_a_move_rewrites_any_node_with_leaves_of_the_vocabulary():
    from mana.discovery.language import hole

    move = discovery.Macro("m", sub(add(discovery.SELF, hole(0)), hole(1)), 2)
    x, y, z = get("x"), get("y"), get("z")
    made = set(discovery.macro_neighbours(if_(cmp(LESS, x, const(3)), z, y), [x, y, z],
                                          [move], discovery.MAX_SIZE))
    assert if_(cmp(LESS, sub(add(x, y), z), const(3)), z, y) in made


def _derivations():
    from mana.discovery import derive

    x, y, z = get("x"), get("y"), get("z")
    columns = {"x": [0, 5, 9], "y": [3, 1, 7], "z": [2, 8, 4]}
    leaves, conditions = discovery.vocabulary(columns, [1, 2, 3])
    chains = [
        [y, sub(x, y), if_(cmp(LESS, x, y), y, sub(x, y)),
         if_(cmp(LESS, x, y), sub(y, x), sub(x, y))],
        [z, add(z, x), add(sub(z, y), x), add(if_(cmp(LESS, z, y), y, sub(z, y)), x),
         add(if_(cmp(LESS, z, y), sub(y, z), sub(z, y)), x)],
        [x, add(x, y), add(sub(x, z), y), add(if_(cmp(LESS, x, z), z, sub(x, z)), y),
         add(if_(cmp(LESS, x, z), sub(z, x), sub(x, z)), y)],
    ]
    return [derive.Derivation(chain, leaves, conditions, discovery.MAX_SIZE, f"t{i}")
            for i, chain in enumerate(chains)]


def test_runs_of_steps_several_derivations_share_become_a_move_that_pays():
    from mana.discovery import derive

    derivations = _derivations()
    learned = derive.compress(derivations, 3)
    assert learned.macros and learned.bits_after < learned.bits_before
    assert all(any(part == discovery.SELF for _, part in nodes(m.template))
               for m in learned.macros)
    bits, used = derive.derivation_bits(derivations[1], learned.macros)
    assert used >= 1 and bits < derive.derivation_bits(derivations[1], [])[0]


def test_the_move_learner_cannot_see_the_worlds():
    from mana.discovery import derive

    for node in ast.walk(ast.parse(inspect.getsource(derive))):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            named = [getattr(node, "module", "") or ""] + [a.name for a in node.names]
            assert not any("worlds" in name for name in named)
