"""The research-question contract, held to its gate before any probe
(docs/ГЛУБИНА_ВОПРОС_ИССЛЕДОВАНИЯ.md, 6): the interpreter is blind to kinds of
question; the catalogue is data; through it D1b is D1b and no plan is the
flat search; a research question can be posed on its own result."""
import ast
import inspect
import re
from collections import Counter

import numpy as np

from mana.discovery import plans, problems, questions
from mana.discovery import policy as P
from mana.discovery.language import EQUAL, LESS, cmp, const, get, show, size
from mana.discovery.worlds import W0

FIELDS = ("program", "bits", "program_bits", "error_bits", "evaluations", "rounds",
          "history", "found_at", "termination", "train_errors", "anytime")
#: Names of kinds of question. None may appear in the interpreter: Russian
#: stems anywhere, English words as whole words ("cover" is in "discovery").
KIND_STEMS = ("остат", "покрыт", "развилк", "промах", "случа", "новый элемент")
KIND_WORDS = ("remainder", "residual", "cover", "covering", "split", "lookahead", "look",
              "misses", "cases", "new element")
NARROW = P.without(P.CURRENT, "add a condition").rules


def _sum_of_two(seed=0):
    split = W0.split(200, 50, seed=seed)
    cols = split.train
    target = (np.where(cols["x"] > 5, cols["y"], cols["z"])
              + (cols["y"] < 3).astype(np.int64))
    return cols, target


def _tree(solution):
    return [(n.problem.operator, n.problem.parent, n.problem.depends_on, n.budget,
             dict(n.cost), show(n.program), n.chosen, n.used, tuple(n.children))
            for n in solution.nodes]


# -- 1. blind to kinds of question -------------------------------------------

def test_the_interpreter_names_no_kind_of_question():
    source = inspect.getsource(questions).lower()
    assert not [k for k in KIND_STEMS if k in source]
    assert not [w for w in KIND_WORDS if re.search(rf"\b{re.escape(w)}\b", source)]


def test_the_interpreter_does_not_branch_on_the_form_of_a_value():
    tree = ast.parse(inspect.getsource(questions))
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
                and n.name in ("_Interpreter", "_Scope")):
        for call in (n for n in ast.walk(cls) if isinstance(n, ast.Call)):
            name = getattr(call.func, "id", getattr(call.func, "attr", ""))
            assert name not in ("isinstance", "type"), f"{cls.name} checks a form"


def test_the_interpreter_cannot_see_the_worlds():
    for module in (questions, plans):
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                named = [getattr(node, "module", "") or ""] + [a.name for a in node.names]
                assert not any("worlds" in n or "ladder" in n for n in named)


# -- 2. the catalogue is data --------------------------------------------------

def test_every_plan_of_the_catalogue_runs_through_the_one_interpreter():
    cols, target = _sum_of_two()
    for plan in plans.D1B + plans.DIAGNOSTIC + (plans.p2_step(4),):
        solved = questions.solve(problems.Problem.whole(cols, target), P.CURRENT, 80000,
                                 plans=(plan,), flat_share=0.5, env={"ahead": NARROW})
        posed = [r for r in solved.records if r.plan == plan.name]
        assert posed, plan.name
        asked = {n.problem.operator for n in solved.nodes[1:]}
        assert any(r.derived for r in posed) == bool(asked & {e.name for e in plan.entries})


def test_taking_a_plan_out_takes_its_questions_away():
    cols, target = _sum_of_two()
    whole = questions.solve(problems.Problem.whole(cols, target), P.CURRENT, 60000,
                            plans=plans.D1B, flat_share=0.5)
    fewer = questions.solve(problems.Problem.whole(cols, target), P.CURRENT, 60000,
                            plans=plans.D1B[:2], flat_share=0.5)
    gone = {e.name for e in plans.D1B[2].entries}
    assert gone & {n.problem.operator for n in whole.nodes}
    assert not gone & {n.problem.operator for n in fewer.nodes}
    assert plans.D1B[2].name not in {r.plan for r in fewer.records}


# -- 3. D1b through the contract, and the empty catalogue ----------------------

def test_with_no_plan_a_question_is_the_flat_search():
    split = W0.split(200, 50, seed=0)
    flat = P.run(P.with_budget(P.CURRENT, 30000), split.train, split.train_outcomes, profile=True)
    solved = questions.solve(problems.Problem.whole(split.train, split.train_outcomes),
                             P.CURRENT, 30000, profile=True)
    assert all(getattr(flat, f) == getattr(solved.found, f) for f in FIELDS)
    assert solved.ledger.spent[problems.SEARCH] == flat.evaluations
    assert sum(solved.ledger.spent.values()) == flat.evaluations
    assert len(solved.nodes) == 1 and not solved.records


def test_d1b_through_the_contract_is_d1b_field_for_field():
    cols, target = _sum_of_two()
    for share in (1.0, 0.5, 0.25):
        d1b = problems.solve(problems.Problem.whole(cols, target), P.CURRENT, 60000,
                             catalogue=problems.BRANCHES, flat_share=share, profile=True)
        mine = questions.solve(problems.Problem.whole(cols, target), P.CURRENT, 60000,
                               plans=plans.D1B, flat_share=share, profile=True)
        assert mine.program == d1b.program
        assert all(getattr(mine.found, f) == getattr(d1b.found, f) for f in FIELDS)
        assert mine.ledger.spent == d1b.ledger.spent
        assert _tree(mine) == _tree(d1b)
        assert mine.used_depth() == d1b.used_depth()


# -- 4. a research question on its own result ----------------------------------

def test_a_research_question_can_be_posed_on_what_research_found():
    """Q0 -> X -> Q1(X) -> research on Q1 -> Y: no other kind of node, no other path."""
    study = questions.Plan(
        "о X", ("own",),
        (questions.Entry("о X", ("question", ("vadd", ("eval", ("var", "X")), ("target",)),
                                 ("domain",)), ("var", "share")),),
        ("?", "X"))
    cols, target = _sum_of_two()
    solved = questions.solve(problems.Problem.whole(cols, target), P.CURRENT, 80000,
                             plans=(study,) + plans.D1B, flat_share=0.5)
    on_x = [n for n in solved.nodes if n.problem.operator == "о X"]
    assert on_x and any(n.children for n in on_x)
    deeper = [solved.nodes[i] for n in on_x for i in n.children]
    assert all(type(n) is problems.Node for n in solved.nodes)
    assert any(r.parent in {n.index for n in on_x} for r in solved.records)
    assert solved.used_depth() >= 3 and deeper


# -- plans by depth (D2-prep-2) --------------------------------------------------

def _depths(solution):
    depth = {}
    for n in solution.nodes:
        depth[n.index] = 1 if n.problem.parent is None else depth[n.problem.parent] + 1
    return depth


def test_the_same_plans_at_every_depth_by_schedule_is_the_default_path():
    cols, target = _sum_of_two()
    default = questions.solve(problems.Problem.whole(cols, target), P.CURRENT, 60000,
                              plans=plans.D1B, flat_share=0.5)
    scheduled = questions.solve(problems.Problem.whole(cols, target), P.CURRENT, 60000,
                                flat_share=0.5, schedule=[plans.D1B] * problems.MAX_DEPTH)
    assert scheduled.program == default.program
    assert scheduled.ledger.spent == default.ledger.spent
    assert _tree(scheduled) == _tree(default)


def test_a_schedule_gives_each_depth_its_own_plans_and_none_past_its_end():
    first, second = plans.D1B[0], plans.D1B[2]
    cols, target = _sum_of_two()
    solved = questions.solve(problems.Problem.whole(cols, target), P.CURRENT, 80000,
                             flat_share=0.5, schedule=[(first,), (second,)])
    depth = _depths(solved)
    names = {1: {e.name for e in first.entries}, 2: {e.name for e in second.entries}}
    for n in solved.nodes[1:]:
        assert n.problem.operator in names[depth[n.problem.parent]]
    assert all(not n.children for n in solved.nodes if depth[n.index] >= 3)
    for r in solved.records:
        assert r.plan == (first.name if depth[r.parent] == 1 else second.name)
    assert any(depth[n.index] == 3 for n in solved.nodes)


def test_a_plan_keeps_its_derivation_when_the_experiment_changes():
    """D3 invents the experiment; the derivation and the rebuild stay."""
    mine = plans.with_experiment(plans.D1B[2], ("item", ("leaves",), ("const", 0)))
    assert mine.entries == plans.D1B[2].entries and mine.rebuild == plans.D1B[2].rebuild
    assert plans.with_experiment(plans.D1B[0], plans.D1B[0].experiment) == plans.D1B[0]
    cols, target = _sum_of_two()
    solved = questions.solve(problems.Problem.whole(cols, target), P.CURRENT, 60000,
                             plans=(mine,), flat_share=0.5)
    posed = [r for r in solved.records if r.plan == mine.name]
    assert posed and posed[0].experiment == mine.experiment
    assert {n.problem.operator for n in solved.nodes[1:]} <= {e.name for e in mine.entries}


# -- candidates(s) (D3-prep-0) -------------------------------------------------

LEAVES = (get("x"), get("y"), const(0))
CONDITIONS = (cmp(LESS, get("x"), const(0)), cmp(EQUAL, get("x"), get("y")))


def _of_the_vocabulary(p):
    if p in LEAVES or p in CONDITIONS:
        return True
    if p[0] in ("add", "sub"):
        return _of_the_vocabulary(p[1]) and _of_the_vocabulary(p[2])
    return p[0] == "if" and p[1] in CONDITIONS and _of_the_vocabulary(p[2]) \
        and _of_the_vocabulary(p[3])


def _counted(most):
    """How many programs the grammar has of each size, counted, not listed."""
    count = {}
    for n in range(1, most + 1):
        total = sum(1 for p in LEAVES + CONDITIONS if size(p) == n)
        total += 2 * sum(count.get(i, 0) * count.get(n - 1 - i, 0) for i in range(1, n - 1))
        total += sum(count.get(i, 0) * count.get(n - 1 - size(k) - i, 0)
                     for k in CONDITIONS for i in range(1, n - 1 - size(k)))
        count[n] = total
    return {n: t for n, t in count.items() if t}


def _swap(e, old, new):
    if e == old:
        return new
    return tuple(_swap(x, old, new) for x in e) if type(e) is tuple else e


def test_candidates_are_the_programs_of_the_vocabulary_by_size():
    listed = list(questions._Candidates(LEAVES, CONDITIONS, 8))
    sizes = [size(p) for p in listed]
    assert len(set(listed)) == len(listed)
    assert sizes == sorted(sizes) and max(sizes) == 8
    assert dict(Counter(sizes)) == _counted(8)
    assert all(_of_the_vocabulary(p) for p in listed)
    assert tuple(listed[:5]) == LEAVES + CONDITIONS
    lazy = questions._Candidates(LEAVES, CONDITIONS, 8)
    assert [lazy[i] for i in (0, 7, len(listed) - 1)] == [listed[0], listed[7], listed[-1]]


def test_candidates_of_size_one_are_the_leaves_in_their_order():
    """C with candidates(1) in place of the leaves is C, field for field."""
    swapped = questions.Plan(
        plans.C_MASK.name,
        _swap(plans.C_MASK.experiment, ("leaves",), ("candidates", ("const", 1))),
        tuple(questions.Entry(x.name, _swap(x.question, ("leaves",), ("candidates", ("const", 1))),
                              x.budget) for x in plans.C_MASK.entries),
        plans.C_MASK.rebuild)
    assert swapped.experiment != plans.C_MASK.experiment
    cols, target = _sum_of_two()
    runs = [questions.solve(problems.Problem.whole(cols, target), P.CURRENT, 60000,
                            plans=(plan,), flat_share=0.5) for plan in (plans.C_MASK, swapped)]
    record = lambda r: (r.parent, r.plan, r.observation, r.derived, r.cost,  # noqa: E731
                        r.candidate, r.kept, r.stopped)
    assert runs[0].records and runs[0].program == runs[1].program
    assert runs[0].ledger.spent == runs[1].ledger.spent
    assert _tree(runs[0]) == _tree(runs[1])
    assert [record(r) for r in runs[0].records] == [record(r) for r in runs[1].records]


def test_candidates_are_built_only_as_far_as_they_are_read():
    first = questions.Plan(
        "первые пять",
        ("argmax", ("take", ("const", 5), ("candidates", ("const", 9))),
         ("lambda", "x", ("count", ("eq", ("eval", ("var", "x")), ("target",))))),
        (), ("?", "X"))
    cols, target = _sum_of_two()
    solved = questions.solve(problems.Problem.whole(cols, target), P.CURRENT, 60000,
                             plans=(first,), flat_share=0.5)
    posed = [r for r in solved.records if r.plan == first.name]
    assert len(posed) == 1 and not posed[0].stopped
    assert posed[0].cost == {a: (5 if a == problems.BUILD else 0) for a in problems.ARTICLES}


def test_argmin_and_argmax_keep_the_first_best():
    cols, target = _sum_of_two()
    scope = questions._Scope(None, None, cols, target, None, [], [], None, [])
    items = ("list", ("const", 3), ("const", 1), ("const", 1), ("const", 5), ("const", 5))
    score = ("lambda", "v", ("minus", ("var", "v"), ("const", 0)))
    ones = ("lambda", "v", ("const", 0))
    assert scope.value(("argmin", items, score), {}, False) == 1
    assert scope.value(("argmax", items, score), {}, False) == 5
    tagged = ("list", ("list", ("const", "a"), ("const", 0)), ("list", ("const", "b"), ("const", 0)))
    assert scope.value(("argmin", tagged, ones), {}, False)[0] == "a"
    assert scope.value(("argmax", tagged, ones), {}, False)[0] == "a"
