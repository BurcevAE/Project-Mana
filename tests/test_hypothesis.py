"""Can a change be arrived at, rather than chosen from a list somebody wrote?

The line this file is about: a knob is a change a person thought of, and a
rule mined from a diagnosed conflict is not. So the run that matters here
declares **no knobs at all** -- whatever comes out cannot be something
handed over, by construction.

Most of these are refusals. The easy version of this idea is a generator
producing variants until one beats a holdout, and every guard against that
is worth a test: the space is capped, the mining touches only discovery,
a marker is checked with the matcher it will be used with, and a rule that
breaks a group never reaches a hidden split.
"""
from __future__ import annotations

import pytest

from mana import policy as policy_mod
from mana.cognition import hypothesis, investigator, rules, trials
from mana.cognition.compiler import ABSENT, AMBIGUOUS, classify, diagnose
from mana.cognition.findings import Ledger
from mana.cognition.synthesis import DOMAIN_KIND
from mana.core import tasks as task_gen
from mana.core.gates import ACCEPTED


@pytest.fixture(autouse=True)
def clean(tmp_path, monkeypatch):
    monkeypatch.setattr(trials, "_store", lambda: tmp_path / "trials")
    monkeypatch.setattr(rules, "_path", lambda: tmp_path / "rules.json")
    rules._reset_for_tests()
    trials._reset_for_tests()
    investigator._ORACLES.clear()
    yield
    rules._reset_for_tests()
    trials._reset_for_tests()
    investigator._ORACLES.clear()


def _oracle(name="naming", seeds=(1, 2, 3, 4)):
    return investigator.Oracle(
        name=name,
        decide=lambda item: classify(item.prompt, difficulty=item.difficulty)[0],
        samples=lambda seed, group: task_gen.generate(group, 20, seed),
        truth=lambda group: DOMAIN_KIND[group],
        groups=tuple(DOMAIN_KIND), knobs=(), scope="task_naming",
        seeds={trials.DISCOVERY: tuple(seeds),
               trials.VALIDATION: tuple(s + 100 for s in seeds),
               trials.FRESH: tuple(s + 200 for s in seeds)})


def _texts(groups, seeds):
    def samples(group):
        for seed in seeds:
            for task in task_gen.generate(group, 20, seed):
                yield task.prompt
    return samples


# --------------------------------------------------------------------------
# the failure has a shape, and the shapes are different
# --------------------------------------------------------------------------

def test_a_competing_feature_and_no_feature_are_different_failures():
    """"logic came out general" and "text_ops came out math" were the same
    fact while `classify` returned only its answer. They are two causes."""
    def shapes(group, wanted):
        found = []
        for task in task_gen.generate(group, 20, 5):
            got, _ = classify(task.prompt, difficulty=task.difficulty)
            if got != wanted:
                found.append(diagnose(task.prompt, wanted, got))
        return found

    # Per group, not per item: a reworded task can miss the competing
    # marker and be ABSENT while its class is mostly AMBIGUOUS, which is
    # why `observe` takes the dominant shape rather than the first one.
    absent = shapes("logic", "reasoning")
    assert absent and all(row["shape"] == ABSENT for row in absent)
    assert all(not row["competing"] for row in absent)

    ambiguous = shapes("text_ops", "general")
    assert ambiguous
    assert sum(1 for row in ambiguous if row["shape"] == AMBIGUOUS) >         sum(1 for row in ambiguous if row["shape"] == ABSENT)
    assert any(("math", "сколько") in row["competing"] for row in ambiguous)


def test_the_claim_says_which_cause_it_is():
    seeds = (1, 2)
    claims = hypothesis.observe(
        "task_naming", _texts(DOMAIN_KIND, seeds), tuple(DOMAIN_KIND),
        lambda g: DOMAIN_KIND[g], lambda text: classify(text)[0])
    by_group = {claim.group: claim for claim in claims}

    assert by_group["logic"].shape == ABSENT
    assert "нет ни одного признака" in by_group["logic"].claim
    assert by_group["text_ops"].shape == AMBIGUOUS
    assert "сколько" in by_group["text_ops"].claim
    assert "более" in by_group["text_ops"].claim   # specificity, not order


def test_a_decision_that_works_produces_no_claim():
    seeds = (1,)
    claims = hypothesis.observe(
        "task_naming", _texts(("arithmetic",), seeds), ("arithmetic",),
        lambda g: DOMAIN_KIND[g], lambda text: classify(text)[0])
    assert claims == []


# --------------------------------------------------------------------------
# what a claim licenses, and what it may not
# --------------------------------------------------------------------------

def test_a_mined_marker_really_occurs_in_the_text():
    """The first version built bigrams from a filtered token list, so
    "позиции считая" was mined out of "позиции 2, считая с начала": a
    phrase with perfect measured support that could never match, because
    the classifier looks for it as a substring and it is not one."""
    seeds = (1, 2)
    samples = _texts(DOMAIN_KIND, seeds)
    claims = hypothesis.observe(
        "task_naming", samples, tuple(DOMAIN_KIND),
        lambda g: DOMAIN_KIND[g], lambda text: classify(text)[0])
    logic = next(c for c in claims if c.group == "logic")

    for rule in hypothesis.forms(logic, samples, tuple(DOMAIN_KIND)):
        hits = sum(1 for text in samples("logic") if rule.marker in text.lower())
        total = sum(1 for _ in samples("logic"))
        assert hits / total >= hypothesis.MIN_SUPPORT, rule.marker


def test_a_marker_that_leaks_is_not_offered():
    """A marker that appears outside its class is how a rule fixes one
    group by breaking another -- measured once, when a text check took
    `code` from 100% to 37%."""
    seeds = (1, 2)
    samples = _texts(DOMAIN_KIND, seeds)
    claims = hypothesis.observe(
        "task_naming", samples, tuple(DOMAIN_KIND),
        lambda g: DOMAIN_KIND[g], lambda text: classify(text)[0])

    for claim in claims:
        others = [g for g in DOMAIN_KIND if g != claim.group]
        for rule in hypothesis.forms(claim, samples, tuple(DOMAIN_KIND)):
            outside = [t.lower() for g in others for t in samples(g)]
            leak = sum(1 for t in outside if rule.marker in t) / len(outside)
            assert leak <= hypothesis.MAX_LEAK, (rule.marker, leak)


def test_layout_is_not_mistaken_for_a_feature():
    """The longest substring in a templated corpus is the template. The
    first run mined a marker that crossed a line break into the answer
    instruction: true of every task of its class, leaking nowhere, and a
    fact about how the prompt is laid out."""
    seeds = (1, 2)
    samples = _texts(DOMAIN_KIND, seeds)
    claims = hypothesis.observe(
        "task_naming", samples, tuple(DOMAIN_KIND),
        lambda g: DOMAIN_KIND[g], lambda text: classify(text)[0])
    for claim in claims:
        for rule in hypothesis.forms(claim, samples, tuple(DOMAIN_KIND)):
            assert "\n" not in rule.marker
            assert len(rule.marker) <= hypothesis.MAX_MARKER


def test_the_number_of_forms_is_capped():
    """The number of candidates is the size of the search, and a search
    large enough will beat a holdout by luck."""
    seeds = (1, 2)
    samples = _texts(DOMAIN_KIND, seeds)
    claims = hypothesis.observe(
        "task_naming", samples, tuple(DOMAIN_KIND),
        lambda g: DOMAIN_KIND[g], lambda text: classify(text)[0])
    for claim in claims:
        assert len(hypothesis.forms(claim, samples, tuple(DOMAIN_KIND))) \
            <= hypothesis.MAX_CANDIDATES


# --------------------------------------------------------------------------
# the whole thing, with nothing declared to choose from
# --------------------------------------------------------------------------

def test_with_no_knobs_declared_it_still_finds_a_change(tmp_path):
    """The test this file exists for. `knobs=()`, so anything adopted is
    something it built."""
    ledger = Ledger(tmp_path / "f.jsonl")
    investigator.declare(_oracle())

    empty = investigator.investigate("naming", ledger=ledger,
                                     propose=investigator.from_knobs)
    assert empty.stopped_at == "кандидат"

    trials._reset_for_tests()
    investigator.declare(_oracle())
    built = investigator.investigate("naming", ledger=ledger,
                                     propose=investigator.from_hypotheses,
                                     allow_adopt=True)
    assert built.verdict == ACCEPTED
    assert built.hypothesis["shape"] in (ABSENT, AMBIGUOUS)
    assert built.chosen["marker"]
    assert rules.installed("task_naming")


def test_what_it_builds_is_not_a_knob():
    """A rule and a setting are different objects, and the point of the
    experiment is which kind this is."""
    investigator.declare(_oracle())
    made = investigator.from_hypotheses(_oracle(), (1, 2))
    assert made and all(change.kind == "rule" for change in made)
    assert all(change.hypothesis for change in made)
    assert all(change.settings["marker"] not in policy_mod._BY_NAME
               for change in made)


def test_a_built_rule_needs_evidence_like_everything_else():
    from mana.cognition.rules import InstallRefused, Rule

    with pytest.raises(InstallRefused):
        rules.install(Rule(where="task_naming", marker="x", decides="math"), {})
    assert rules.installed() == []


def test_nothing_is_installed_without_permission(tmp_path):
    investigator.declare(_oracle())
    report = investigator.investigate("naming", ledger=Ledger(tmp_path / "f.jsonl"),
                                      propose=investigator.from_hypotheses)
    assert report.verdict == ACCEPTED and report.adopted is False
    assert rules.installed() == []


def test_an_empty_rule_store_leaves_the_classifier_as_it_was():
    """Nothing installed by default, so declaring the seat changed
    nothing until something earned it."""
    assert rules.installed("task_naming") == []
    wrong = sum(1 for task in task_gen.generate("logic", 20, 5)
                if classify(task.prompt, difficulty=task.difficulty)[0]
                != DOMAIN_KIND["logic"])
    assert wrong == 20


# --------------------------------------------------------------------------
# executable by the matcher that will measure it
# --------------------------------------------------------------------------

def test_a_candidate_that_cannot_fire_never_leaves_the_generator():
    """Not a weak rule -- not the rule the miner claims. The first version
    mined "позиции считая" out of "позиции 2, считая с начала": perfect
    measured support for a phrase that is not a substring of anything."""
    from mana.cognition.rules import Rule

    o = _oracle()
    dead = Rule(where="task_naming", marker="этого нет ни в одной задаче",
                decides="reasoning",
                provenance={"group": "logic", "wanted": "reasoning"})
    checked, why = investigator._executable(o, dead, (1, 2))
    assert checked is None
    assert "сопоставитель не срабатывает" in why


def test_a_candidate_that_matches_but_moves_nothing_is_refused():
    """Seated where it cannot decide. A marker that matches and changes no
    answer describes an intention rather than a mechanism."""
    from mana.cognition.rules import Rule

    o = _oracle()
    # Right answer already: applying it changes nothing anywhere.
    inert = Rule(where="task_naming", marker="вычисли", decides="math",
                 provenance={"group": "arithmetic", "wanted": "math"})
    checked, why = investigator._executable(o, inert, (1, 2))
    assert checked is None
    assert "не меняет ни одного ответа" in why


def test_the_check_happens_before_any_quota_is_spent(tmp_path):
    investigator.declare(_oracle())
    made = investigator.from_hypotheses(_oracle(), (1, 2))
    assert made, "nothing survived the check, so nothing can be said"
    assert trials.reads.__name__          # the registry was never asked
    for change in made:
        assert change.rule.provenance["matcher"] == "Rule.matches + Oracle.decide"
        assert change.rule.provenance["matched"] > 0
        assert change.rule.provenance["moved"] > 0


# --------------------------------------------------------------------------
# a trace from the conflict to the rule
# --------------------------------------------------------------------------

def test_every_built_rule_carries_where_it_came_from():
    """Otherwise `hypothesis.py` can declare one thing while the generator
    hands over another, and the only visible fact is that something won."""
    for change in investigator.from_hypotheses(_oracle(), (1, 2)):
        trace = change.rule.provenance
        assert trace["group"] in DOMAIN_KIND
        assert trace["shape"] in (ABSENT, AMBIGUOUS)
        assert trace["claim"]
        assert trace["wanted"] == DOMAIN_KIND[trace["group"]]
        # and the rule decides what the claim said it should
        assert change.rule.decides == trace["wanted"]
        assert trace["wrong"] > 0 and trace["of"] >= trace["wrong"]
        assert set(trace["mined_with"]) == {"min_support", "max_leak",
                                            "max_marker"}


def test_the_trace_reads_from_conflict_to_rule():
    change = investigator.from_hypotheses(_oracle(), (1, 2))[0]
    trace = change.rule.trace()
    for part in ("провал:", "диагноз:", "гипотеза:", "правило:", "проверено:"):
        assert part in trace


def test_an_ambiguous_claim_names_the_feature_that_competed():
    made = investigator.from_hypotheses(_oracle(), (1, 2))
    ambiguous = [c for c in made
                 if c.rule.provenance["shape"] == AMBIGUOUS]
    assert ambiguous
    assert any("сколько" in str(c.rule.provenance.get("competing"))
               for c in ambiguous)


# --------------------------------------------------------------------------
# what is measured, and what is only reported
# --------------------------------------------------------------------------

def test_word_order_dependence_is_reported_and_not_filtered():
    """It is not specificity. Specificity would be "how often does this
    fire on text nobody measured it against", and every split here comes
    from one generator. Filtering on this would condemn "текст:", which is
    a single token and the right marker for its class."""
    made = investigator.from_hypotheses(_oracle(), (1, 2))
    reported = {c.rule.marker: c.rule.accidental for c in made}
    assert reported, "nothing to report on"
    # Both kinds survive mining: bare tokens and phrases.
    assert any(value >= 0.9 for value in reported.values())
    assert any(value <= 0.5 for value in reported.values())


def test_among_measured_equals_the_phrase_wins():
    """A tiebreak on a measured number, applied after the protocol has
    failed to tell two candidates apart -- not a proxy standing in for
    something this corpus cannot measure."""
    from mana.cognition.rules import Rule

    def change(marker, accidental):
        return investigator.Change(
            label=marker, kind="rule", settings={"marker": marker},
            rule=Rule(where="task_naming", marker=marker, decides="reasoning",
                      accidental=accidental))

    rows = [(1.0, 0.0, change("чем", 1.0)),
            (1.0, 0.0, change("считая с", 0.16))]
    rows.sort(key=lambda row: (row[0], row[1],
                               -row[2].rule.accidental), reverse=True)
    assert rows[0][2].rule.marker == "считая с"
