"""
tests/test_research.py — the cycle that runs itself, and the two claims
that make it more than a script.

Claim 1: failures have structure the self-model cannot see, and that
structure is computable from history without a single brain call.

Claim 2: measuring and experimenting compete for one budget on one
scale -- total interval width -- so the cycle spends where the most gets
resolved rather than always doing the same thing. Examining the
vocabulary is deliberately NOT on that scale; several tests below are
about why putting it there broke the arbitration.

Everything else here checks that an interrupted cycle is findable and
that the cycle cannot quietly widen its own representation.
"""
from __future__ import annotations

import pytest

from mana.cognition import research
from mana.cognition.curriculum import EASY, HARD, MEDIUM
from mana.cognition.research import (EXPERIMENT, MEASURE, ActivityOption,
                                     ResearchCycle, choose, cluster_failures,
                                     options_for, representation_findings)
from mana.cognition.self_model import Observation, SelfModel


def observation(task_id, correct, domain="arithmetic", difficulty=0.5, reason="wrong"):
    return Observation(task_id, domain, difficulty, correct,
                       reason="" if correct else reason, calls=1)


def corpus(rows):
    """rows: (task_id, text, correct) -> (observations, task_texts)."""
    return ([observation(tid, ok) for tid, _t, ok in rows],
            {tid: text for tid, text, _ok in rows})


def always(correct):
    def runner(task):
        return correct, "brain", 1
    return runner


def model_with(domain, difficulty, correct, n):
    model = SelfModel()
    for i in range(n):
        model.record(observation(f"{domain}{i}", correct, domain, difficulty))
    return model


# ---------------------------------------------------------------------------
# clustering: what the failures have in common
# ---------------------------------------------------------------------------

def test_a_structure_shared_by_failures_is_found():
    """The point of the module: not "hard arithmetic is 0.2" but "every
    failure was nested".

    Same numbers, same operators, brackets the only difference -- so
    nesting depth is the one field that can explain the split, and the
    test is not satisfied by a field that merely happened to correlate.
    """
    rows = [(f"flat{i}", f"Вычисли {i} + 1 * 2", True) for i in range(10)]
    rows += [(f"deep{i}", f"Вычисли ({i} + 1) * 2", False) for i in range(10)]
    observations, texts = corpus(rows)
    clusters = cluster_failures(observations, texts)
    assert clusters
    top = clusters[0]
    assert top.field_name == "nesting_depth"
    assert top.failures == 10 and top.successes == 0
    assert top.purity == 1.0


def test_the_commonest_value_is_not_reported_just_for_being_common():
    """Raw counts would send every cycle chasing whatever the generator
    produces most; the measure asks whether failures are concentrated."""
    rows = [(f"t{i}", f"Вычисли {i} + 1", i % 2 == 0) for i in range(40)]
    observations, texts = corpus(rows)
    assert cluster_failures(observations, texts) == []


def test_a_pure_cluster_is_still_found_when_most_things_fail():
    """Lift was the first measure here and it broke exactly here: at a
    58% base failure rate no group can reach 2x, so a perfectly pure
    cluster scored below threshold. A system worth researching is one
    that fails often, so the measure has to survive that."""
    rows = [(f"flat{i}", f"Вычисли {i} + 1 * 2", i % 2 == 0) for i in range(30)]
    rows += [(f"deep{i}", f"Вычисли ({i} + 1) * 2", False) for i in range(6)]
    observations, texts = corpus(rows)
    clusters = cluster_failures(observations, texts)
    assert clusters, "a pure failure group must be visible at any base rate"
    assert clusters[0].purity == 1.0
    assert clusters[0].lift < 2.0, "and the old threshold would have hidden it"


def test_a_handful_of_failures_is_not_a_pattern():
    rows = [(f"ok{i}", f"Вычисли {i} + 1", True) for i in range(20)]
    rows += [("bad", "Вычисли (1 + 1)", False)]
    observations, texts = corpus(rows)
    assert cluster_failures(observations, texts) == []


def test_clusters_are_ranked_by_concentration_not_by_size():
    """A big cluster that is half successes explains less than a small
    pure one."""
    rows = [(f"flat{i}", f"Вычисли {i} + 1 * 2", i % 3 == 0) for i in range(30)]
    rows += [(f"deep{i}", f"Вычисли ({i} + 1) * 2", False) for i in range(6)]
    observations, texts = corpus(rows)
    clusters = cluster_failures(observations, texts)
    assert clusters
    assert clusters[0].excess >= clusters[-1].excess


def test_a_group_holding_everything_explains_nothing():
    """With no outside to compare against, every corpus would report its
    own failure rate as a discovery."""
    observations, texts = corpus([(f"t{i}", f"Вычисли {i} + 1", False)
                                  for i in range(20)])
    assert cluster_failures(observations, texts) == []


def test_tasks_with_no_recorded_text_are_skipped():
    observations = [observation(f"t{i}", False) for i in range(10)]
    assert cluster_failures(observations, {}) == []


def test_clustering_consults_no_model():
    """A clustering judged by a model would put its opinion inside the
    definition of MANA's own weaknesses."""
    import inspect
    source = inspect.getsource(cluster_failures).lower()
    for forbidden in (".ask(", "ask_many", "brain", "llm", "prompt", "_call("):
        assert forbidden not in source


def test_a_cluster_says_what_it_found_in_words():
    rows = [(f"flat{i}", f"Вычисли {i} + 1 * 2", True) for i in range(10)]
    rows += [(f"deep{i}", f"Вычисли ({i} + 1) * 2", False) for i in range(10)]
    observations, texts = corpus(rows)
    text = cluster_failures(observations, texts)[0].describe()
    assert "отказов" in text and "чаще" in text


# ---------------------------------------------------------------------------
# arbitration: what to spend the next call on
# ---------------------------------------------------------------------------

def test_resolution_per_call_beats_raw_resolution():
    """A cheap measurement can outrank an expensive experiment, which is
    the whole reason the two share a scale."""
    cheap = ActivityOption(MEASURE, "измерить", expected_resolution=0.3, estimated_calls=12)
    dear = ActivityOption(EXPERIMENT, "проверить", expected_resolution=0.5, estimated_calls=180)
    assert choose([cheap, dear], budget_left=1000) is cheap


def test_nothing_unaffordable_is_started():
    """An experiment the budget cuts off halfway spends the calls and
    produces no verdict."""
    dear = ActivityOption(EXPERIMENT, "проверить", expected_resolution=9.0, estimated_calls=500)
    assert choose([dear], budget_left=100) is None


def test_an_unaffordable_option_does_not_block_an_affordable_one():
    cheap = ActivityOption(MEASURE, "измерить", expected_resolution=0.1, estimated_calls=10)
    dear = ActivityOption(EXPERIMENT, "проверить", expected_resolution=9.0, estimated_calls=500)
    assert choose([dear, cheap], budget_left=100) is cheap


def test_a_cycle_starting_from_nothing_has_something_to_do():
    """An untouched slice produces no gap, because a gap needs
    observations to exist at all -- so the first version of this stopped
    on its first move with an empty model. Untouched is not absent from
    the space of work; it is the widest interval in it."""
    options = options_for(SelfModel(), {})
    assert options
    assert all(o.activity == MEASURE for o in options)
    assert options[0].expected_resolution == 1.0


def test_an_untouched_slice_stops_being_offered_once_it_is_measured():
    model = SelfModel()
    for i in range(12):
        model.record(observation(f"t{i}", i % 2 == 0, "logic", 0.8))
    described = " ".join(o.description for o in options_for(model, {}))
    assert "logic/hard: ни разу" not in described


def test_an_unmeasured_capability_produces_a_measurement_not_an_experiment():
    """Experimenting on numbers too soft to conclude from wastes the
    budget that would have made them firm."""
    model = model_with("logic", 0.8, True, 3)
    measures = [o for o in options_for(model, {})
                if o.activity == MEASURE and "logic/hard" in o.description]
    assert measures, "the soft slice must be offered as something to measure"


def test_a_measured_weakness_produces_an_experiment():
    model = model_with("arithmetic", 0.9, False, 30)
    activities = {o.activity for o in options_for(model, {})}
    assert EXPERIMENT in activities


def test_the_vocabulary_never_competes_for_a_step():
    """Its payoff is in collision pairs, not interval width, so putting
    it on the arbitration scale compared two different quantities. Being
    free, it then won every round: a live run spent three of five steps
    recommending fields, resolved nothing, and tripped the
    diminishing-returns guard with 94% of its budget unspent."""
    model = SelfModel()
    texts = {}
    for i in range(24):
        model.record(observation(f"t{i}", i % 2 == 0))
        texts[f"t{i}"] = f"Вычисли {i} + 1"
    assert {o.activity for o in options_for(model, texts)} <= set(research.ACTIVITIES)
    assert research.ACTIVITIES == (MEASURE, EXPERIMENT)


def test_the_vocabulary_is_examined_when_it_stops_explaining():
    """And the baseline vocabulary stops explaining almost immediately.

    task_view is ("task", "category", "difficulty"), and all three are
    read off the task text -- so every arithmetic task gets the same
    description and any mixed outcome inside it is a collision. That is
    not a flaw in this test; it is the finding Level 3 exists for.
    """
    model = SelfModel()
    texts = {}
    for i in range(24):
        model.record(observation(f"t{i}", i % 2 == 0))
        texts[f"t{i}"] = f"Вычисли {i} + 1"
    findings = representation_findings(model, texts)
    assert findings
    assert findings[0]["separates_pairs"] > 0


def test_a_field_already_recommended_is_not_recommended_again():
    """Repeating a recommendation adds nothing to the report."""
    model = SelfModel()
    texts = {}
    for i in range(24):
        model.record(observation(f"t{i}", i % 2 == 0))
        texts[f"t{i}"] = f"Вычисли {i} + 1"
    name = representation_findings(model, texts)[0]["field_name"]
    again = representation_findings(model, texts, already_proposed={name})
    assert all(f["field_name"] != name for f in again)


def test_a_cycle_does_not_spend_its_budget_on_vocabulary(isolated_config):
    """The regression for the livelock, at the level it happened: every
    step the cycle takes must buy interval width."""
    model = SelfModel()
    texts = {}
    for i in range(24):
        model.record(observation(f"t{i}", i % 2 == 0))
        texts[f"t{i}"] = f"Вычисли {i} + 1"
    cycle = ResearchCycle(model, task_texts=texts, budget_calls=400, max_steps=8)
    report = cycle.run(always(True))
    assert all(s["activity"] in research.ACTIVITIES for s in report["history"])
    assert report["calls_used"] > 0


def test_the_vocabulary_is_still_reported_without_costing_a_step(isolated_config):
    """Free means recorded on every step, not dropped."""
    model = SelfModel()
    texts = {}
    for i in range(24):
        model.record(observation(f"t{i}", i % 2 == 0))
        texts[f"t{i}"] = f"Вычисли {i} + 1"
    cycle = ResearchCycle(model, task_texts=texts, budget_calls=200, max_steps=3)
    report = cycle.run(always(True))
    assert report["representation_findings"]
    names = [f["field_name"] for f in report["representation_findings"]]
    assert len(names) == len(set(names))


def test_a_vocabulary_that_explains_the_outcomes_yields_no_finding():
    """No contradicting pair means nothing to explain, whatever the
    representation is."""
    model = SelfModel()
    texts = {}
    for i in range(24):
        model.record(observation(f"t{i}", True))
        texts[f"t{i}"] = f"Вычисли {i} + 1"
    assert representation_findings(model, texts) == []


def test_examining_the_vocabulary_needs_no_brain_call():
    """Insufficiency and its proposals are computed from history. That
    is why it is free -- and, once it was free, why it could not stay on
    the arbitration scale."""
    import inspect
    import re
    # The docstring is prose *about* brain calls; the code is what counts.
    code = re.sub(r'"""3?.*?"""', "", inspect.getsource(representation_findings),
                  flags=re.S).lower()
    for forbidden in (".ask(", "brain", "llm", "prompt", "_call("):
        assert forbidden not in code, f"{forbidden} in the executed code"


# ---------------------------------------------------------------------------
# the cycle
# ---------------------------------------------------------------------------

def test_a_cycle_runs_without_being_told_each_step(isolated_config):
    model = model_with("logic", 0.8, True, 3)
    cycle = ResearchCycle(model, budget_calls=120, max_steps=4)
    report = cycle.run(always(True))
    assert report["steps"] >= 1
    assert report["calls_used"] > 0
    assert report["stop_reason"]


def test_the_cycle_stops_when_the_budget_runs_out(isolated_config):
    model = model_with("logic", 0.8, True, 3)
    cycle = ResearchCycle(model, budget_calls=24, max_steps=99)
    report = cycle.run(always(True))
    assert report["calls_used"] <= 24 + 12
    assert "бюджет" in report["stop_reason"] or "предел" in report["stop_reason"]


def test_a_newly_measured_slice_counts_as_progress_not_as_loss(isolated_config):
    """An unmeasured slice sits in the total at width 1.0, because
    "nothing known" is the widest interval there is."""
    cycle = ResearchCycle(SelfModel())
    empty = cycle._uncertainty()
    cycle.model.record(observation("t", True, "logic", 0.8))
    for i in range(11):
        cycle.model.record(observation(f"t{i}", True, "logic", 0.8))
    assert cycle._uncertainty() < empty


def test_a_cycle_with_nothing_affordable_left_says_so(isolated_config):
    """A run that ends without naming a stop condition cannot be told
    apart from a run that crashed."""
    cycle = ResearchCycle(SelfModel(), budget_calls=1, max_steps=4)
    report = cycle.run(always(True))
    assert report["steps"] == 0
    assert "нечего делать" in report["stop_reason"]


def test_a_cycle_with_an_empty_model_still_runs(isolated_config):
    """The regression for the run that did nothing: a system that knows
    nothing about itself has the most to find out, not the least."""
    cycle = ResearchCycle(SelfModel(), budget_calls=60, max_steps=2)
    report = cycle.run(always(True))
    assert report["steps"] >= 1
    assert report["total_resolution"] > 0


def test_the_cycle_stops_when_nothing_is_being_resolved(isolated_config):
    """Three steps that changed no interval means the cycle is buying
    nothing, whatever the budget still allows."""
    cycle = ResearchCycle(SelfModel(), budget_calls=999, max_steps=9)
    for _ in range(3):
        cycle.steps.append(research.CycleStep("s", MEASURE, "d", 12, 0.0, "ok"))
    assert "убывающая отдача" in cycle.stop_reason()


def test_a_step_is_valued_on_uncertainty_removed_not_on_the_score(isolated_config):
    """A step that confirms a capability is poor has resolved something;
    one that changes nothing has not.

    Also the regression for the accounting bug: the lesson here measures
    a band nothing had measured before, and summing only the *measured*
    slices made that look like uncertainty going up.
    """
    model = model_with("logic", 0.8, True, 3)
    cycle = ResearchCycle(model, budget_calls=200, max_steps=3)
    first = cycle.step(always(False))
    assert first is not None
    assert first.resolution > 0.0


def test_every_step_leaves_a_finished_transaction(isolated_config):
    from mana.core import transaction
    model = model_with("logic", 0.8, True, 3)
    ResearchCycle(model, budget_calls=60, max_steps=2).run(always(True))
    assert transaction.unfinished() == []
    kinds = {t.kind for t in transaction.read_journal()}
    assert "research" in kinds


def test_the_cycle_cannot_adopt_a_field_by_itself(isolated_config):
    """Widening its own representation would change the terms of its own
    measurement; the proposal goes no further than a recommendation."""
    model = SelfModel()
    texts = {}
    for i in range(24):
        model.record(observation(f"t{i}", i % 2 == 0))
        texts[f"t{i}"] = f"Вычисли {i} + 1"
    cycle = ResearchCycle(model, task_texts=texts, budget_calls=200, max_steps=1)
    from mana.cognition.genome import CognitiveGenome
    before = CognitiveGenome().representations["task_view"].fields
    report = cycle.run(always(True))
    assert report["representation_findings"], "it must still have noticed"
    assert CognitiveGenome().representations["task_view"].fields == before


def test_the_cycle_re_reads_its_failures_as_they_accumulate(isolated_config):
    """Analysing once at the start finds nothing, because at the start
    nothing has run. A live cycle reported no failure structure at all
    for exactly that reason."""
    model = SelfModel()
    cycle = ResearchCycle(model, budget_calls=200, max_steps=2)

    def failing_on_nested(task):
        return "(" not in task.prompt, "brain", 1

    cycle.run(failing_on_nested)
    assert cycle.model.observations, "the cycle must have produced evidence"
    assert cycle.clusters == cluster_failures(cycle.model.observations,
                                              cycle.task_texts), \
        "the reported clusters must describe the end of the run"


def test_the_cycle_records_the_prompts_it_practised_on(isolated_config):
    """Relying on the caller's runner to record them made the cycle's own
    analysis a function of how someone else wired it up."""
    cycle = ResearchCycle(SelfModel(), budget_calls=60, max_steps=1)
    cycle.run(always(True))
    assert cycle.task_texts
    assert set(cycle.task_texts) >= {o.task_id for o in cycle.model.observations}


def test_the_cycle_analyses_before_it_acts(isolated_config):
    """Clusters are what a hypothesis gets written against, so they have
    to exist before the first decision."""
    model = SelfModel()
    texts = {}
    for i in range(10):
        model.record(observation(f"flat{i}", True))
        texts[f"flat{i}"] = f"Вычисли {i} + 1 * 2"
        model.record(observation(f"deep{i}", False))
        texts[f"deep{i}"] = f"Вычисли ({i} + 1) * 2"
    cycle = ResearchCycle(model, task_texts=texts, budget_calls=60, max_steps=1)
    report = cycle.run(always(True))
    assert report["failure_clusters"]
    assert report["failure_clusters"][0]["field_name"] == "nesting_depth"


def test_an_experiment_needs_a_runner_before_it_will_be_run(isolated_config):
    """No trial runner means no experiment -- and the step says so
    instead of reporting a result nothing produced."""
    model = model_with("arithmetic", 0.9, False, 30)
    cycle = ResearchCycle(model, budget_calls=400, max_steps=1)
    step = cycle.step(always(True))
    if step and step.activity == EXPERIMENT:
        assert "нет исполнителя" in step.outcome


def test_the_report_can_be_read_backwards(isolated_config, tmp_path):
    model = model_with("logic", 0.8, True, 3)
    cycle = ResearchCycle(model, budget_calls=60, max_steps=2)
    report = cycle.run(always(True))
    assert report["history"]
    assert set(report["by_activity"]) <= set(research.ACTIVITIES)
    cycle.save(tmp_path / "research.json")
    assert (tmp_path / "research.json").exists()


def test_the_cycle_is_not_a_memory_cleanup():
    """The brief warns against turning this into housekeeping; nothing
    here compacts, prunes or summarises anything."""
    import inspect
    source = inspect.getsource(research)
    for forbidden in ("def compact", "def prune", "def summarise", "def cleanup"):
        assert forbidden not in source

# ---------------------------------------------------------------------------
# the whole chain, without a human between the steps
# ---------------------------------------------------------------------------

def test_weakness_to_capability_without_anything_directing_it(isolated_config):
    """Weakness → hypothesis → experiment → discovery → independent
    confirmation → a capability the compiler then chooses.

    Every earlier phase is one link. This is the only test that runs the
    whole chain, and the assertion that matters is the last one: after
    all of it, the system compiles a hard arithmetic task differently
    than it did before.
    """
    from mana.cognition.compiler import Capabilities, compile_program
    from mana.cognition.genome import CognitiveGenome
    from mana.cognition.programs import Budget
    from mana.core import tasks as core_tasks

    # A system that already knows its own shape. Starting from an empty
    # model the cycle spends its first fifteen steps measuring, which is
    # the right order -- ground before theory -- but means "find and fix
    # a problem" only begins once there is a measured problem to find.
    model = SelfModel()
    for domain in core_tasks.DOMAINS:
        for band, mid in (("easy", 0.2), ("medium", 0.5), ("hard", 0.8)):
            weak = domain == "arithmetic" and band == "hard"
            for i in range(24):
                ok = False if weak else i % 4 != 0
                model.record(Observation(
                    f"{domain}{band}{i}", domain, mid, correct=ok,
                    reason="" if ok else "wrong", calls=1))

    def trial_runner(steps, task):
        """Longer chains do better here -- a mechanism the cycle has to
        find rather than be told."""
        return len(steps) > 3, len(steps)

    def task_source(domain, n):
        return core_tasks.generate(domain, n, seed=4242,
                                   difficulty_range=(0.65, 1.01))

    def hidden_fn(steps):
        """The same mechanism, measured on tasks the cycle cannot see."""
        return 0.75 if len(steps) > 3 else 0.45

    def counterexample_fn(hypothesis):
        return 4, 0                 # four probes, the effect held

    base = CognitiveGenome()
    cycle = ResearchCycle(model, budget_calls=4000, max_steps=3, genome=base,
                          hidden_fn=hidden_fn, counterexample_fn=counterexample_fn)
    report = cycle.run(always(False), trial_runner, task_source)

    assert report["adopted_capabilities"], (
        f"nothing persisted; steps were {[s['outcome'] for s in report['history']]}")
    name = report["adopted_capabilities"][0]
    assert report["genome"] != base.signature(), "the genome must have changed"

    task = "Вычисли: (91767 - 690) * 86 + 8"
    caps = Capabilities(brains=2, has_memory=True, has_web=True, has_sandbox=True)
    chosen = compile_program(task, cycle.synthesizer.genome, caps,
                             Budget(calls=12), difficulty=0.72)
    before = compile_program(task, base, caps, Budget(calls=12), difficulty=0.72)
    assert chosen is not None and chosen.template == name
    assert before is None or before.template != name


def test_a_discovery_cannot_be_adopted_on_its_own_evidence(isolated_config):
    """The confirming run is not politeness: `genome.apply` matches the
    verdict's claim id against the proposal, and the discovery's verdict
    carries the experiment's id."""
    import inspect
    source = inspect.getsource(ResearchCycle._maybe_synthesize)
    assert "run_experiment" in source, "a second measurement must be run"
    assert "discovery.verdict" not in source, "the first verdict must not be reused"


def test_a_capability_waits_when_the_budget_cannot_confirm_it(isolated_config):
    """A confirmation cut off halfway spends the calls and settles
    nothing."""
    from mana.core import tasks as core_tasks

    model = SelfModel()
    for i in range(24):
        model.record(Observation(f"a{i}", "arithmetic", 0.8, False,
                                 reason="wrong", calls=1))

    def trial_runner(steps, task):
        return len(steps) > 3, len(steps)

    def task_source(domain, n):
        return core_tasks.generate(domain, n, seed=99,
                                   difficulty_range=(0.65, 1.01))

    cycle = ResearchCycle(model, budget_calls=420, max_steps=2)
    report = cycle.run(always(False), trial_runner, task_source)
    waiting = [s for s in report["history"] if "ждёт подтверждения" in s["outcome"]]
    if waiting:
        assert not report["adopted_capabilities"], \
            "nothing may be installed on an unaffordable confirmation"


def test_a_gate_with_no_evidence_is_not_reported_as_a_refutation(isolated_config):
    """Without a hidden-set scorer every experiment came back "REFUTED:
    failed hidden_confirms, counterexamples" -- a false negative that
    also made a supported discovery impossible to ever produce."""
    from mana.core import tasks as core_tasks

    model = SelfModel()
    for domain in core_tasks.DOMAINS:
        for band, mid in (("easy", 0.2), ("medium", 0.5), ("hard", 0.8)):
            weak = domain == "arithmetic" and band == "hard"
            for i in range(24):
                ok = False if weak else i % 4 != 0
                model.record(Observation(f"{domain}{band}{i}", domain, mid,
                                         correct=ok, reason="" if ok else "wrong",
                                         calls=1))

    cycle = ResearchCycle(model, budget_calls=4000, max_steps=1)
    report = cycle.run(always(False), lambda steps, task: (len(steps) > 3, len(steps)),
                       lambda d, n: core_tasks.generate(d, n, seed=7,
                                                        difficulty_range=(0.65, 1.01)))
    experiments = [s for s in report["history"] if s["activity"] == EXPERIMENT]
    assert experiments
    assert "не удалось судить" in experiments[0]["outcome"]
    assert "REFUTED" not in experiments[0]["outcome"]


def test_nothing_is_installed_when_the_gates_cannot_be_measured(isolated_config):
    """A confirmation that cannot clear the gates the discovery could not
    clear spends the budget and refuses the capability every time."""
    from mana.core import tasks as core_tasks

    model = SelfModel()
    for domain in core_tasks.DOMAINS:
        for band, mid in (("easy", 0.2), ("medium", 0.5), ("hard", 0.8)):
            weak = domain == "arithmetic" and band == "hard"
            for i in range(24):
                ok = False if weak else i % 4 != 0
                model.record(Observation(f"{domain}{band}{i}", domain, mid,
                                         correct=ok, reason="" if ok else "wrong",
                                         calls=1))

    cycle = ResearchCycle(model, budget_calls=4000, max_steps=2)
    report = cycle.run(always(False), lambda steps, task: (len(steps) > 3, len(steps)),
                       lambda d, n: core_tasks.generate(d, n, seed=8,
                                                        difficulty_range=(0.65, 1.01)))
    assert report["adopted_capabilities"] == []
