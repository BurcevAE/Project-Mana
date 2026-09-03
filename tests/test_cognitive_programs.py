"""
tests/test_cognitive_programs.py — compiling and running a program of
thought.

The properties worth protecting:

  1. the compiler produces *different* programs for the same task, or
     nothing can be compared and nothing discovered;
  2. a composed operator actually executes as its chain, or composition is
     a naming exercise;
  3. a run cannot outspend its budget, and a budget stop is not scored as
     an inability to answer.
"""
from __future__ import annotations

import pytest

from mana.cognition import compiler, genome as g, runtime
from mana.cognition.compiler import Capabilities
from mana.cognition.ir import ANSWER, CONTEXT, CRITIQUE, DRAFT, PLAN, TASK
from mana.cognition.programs import Budget, CognitiveProgram, ProgramState
from mana.cognition.runtime import Runtime, StepOutcome


FULL = Capabilities(brains=2, has_sandbox=True, has_web=True)
NO_BRAIN = Capabilities(brains=0, has_sandbox=True, has_web=True)
NO_SANDBOX = Capabilities(brains=2, has_sandbox=False, has_web=True)


def fake_impls(**overrides):
    """Implementations that cost nothing and always work, unless a test
    says otherwise. Lets the runtime be tested for control flow rather
    than for whether a model happened to answer."""
    def make(op_id, produces):
        def impl(state):
            return StepOutcome(values={t: f"{op_id}:{state.task[:12]}" for t in produces},
                               calls=1 if produces else 0, brain=f"brain-{op_id.lower()}")
        return impl
    base = {
        "OBSERVE": make("OBSERVE", []),
        "RETRIEVE": make("RETRIEVE", [CONTEXT]),
        "GENERATE": make("GENERATE", [DRAFT]),
        "CRITIQUE": make("CRITIQUE", [CRITIQUE]),
        "REPAIR": make("REPAIR", [DRAFT]),
        "VERIFY": make("VERIFY", ["evidence"]),
        "DECOMPOSE": make("DECOMPOSE", [PLAN]),
        "SYNTHESIZE": make("SYNTHESIZE", [ANSWER]),
        "ABSTRACT": make("ABSTRACT", ["abstraction"]),
        "PREDICT": make("PREDICT", ["prediction"]),
        "COUNTEREXAMPLE": make("COUNTEREXAMPLE", [CRITIQUE]),
        "COMPARE": make("COMPARE", [CRITIQUE]),
        "ANSWER": make("ANSWER", [ANSWER]),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# compiling
# ---------------------------------------------------------------------------

def test_the_compiler_offers_several_programs_for_one_task():
    """A compiler that returns THE program is a router, and a router
    cannot be searched."""
    candidates = compiler.compile_candidates("Вычисли 17 * 23", g.CognitiveGenome(), FULL)
    assert len(candidates) >= 2
    assert len({c.program.signature() for c in candidates}) == len(candidates)


def test_alternatives_are_structurally_different_not_just_renamed():
    """Two templates that expand to the same chain are one alternative;
    comparing them would spend a budget measuring a thing against itself."""
    alts = compiler.compile_alternatives("Объясни, зачем нужен контроль версий",
                                         g.CognitiveGenome(), FULL, count=3)
    assert len({p.signature() for p in alts}) == len(alts)


def test_a_program_whose_steps_cannot_run_here_is_not_offered():
    """VERIFY without a sandbox is a missing check, not a weaker one --
    and the program would be scored for its absence."""
    with_sandbox = compiler.compile_candidates("Вычисли 12 * 5", g.CognitiveGenome(), FULL)
    without = compiler.compile_candidates("Вычисли 12 * 5", g.CognitiveGenome(), NO_SANDBOX)
    assert any("VERIFY" in c.program.steps for c in with_sandbox)
    assert not any("VERIFY" in c.program.steps for c in without)


def test_nothing_is_offered_when_nothing_can_run():
    """An empty result is information; a silently substituted trivial
    program would be measured as if it had been chosen."""
    assert compiler.compile_candidates("любая задача", g.CognitiveGenome(), NO_BRAIN) == []


def test_a_chain_that_would_be_cut_off_by_the_budget_is_not_offered():
    tight = Budget(calls=1)
    candidates = compiler.compile_candidates("Объясни trade-off двух подходов",
                                             g.CognitiveGenome(), FULL, budget=tight)
    assert all(c.estimated_calls <= 1 for c in candidates)


def test_task_kind_moves_the_ranking():
    genome = g.CognitiveGenome()
    math_top = compiler.compile_program("Вычисли 144 / 12", genome, FULL)
    reasoning_top = compiler.compile_program(
        "Почему полезно отделять память агента от весов модели?", genome, FULL)
    assert math_top is not None and reasoning_top is not None
    assert math_top.template != reasoning_top.template


def test_measured_evidence_shifts_the_score_only_where_it_exists():
    """A compiler that leans on measured performance from the first run
    locks in whatever the first few samples said."""
    from mana.cognition.ir import OperatorEvidence
    genome = g.CognitiveGenome()
    before = compiler.compile_candidates("Вычисли 5 + 5", genome, FULL)
    ops = dict(genome.operators)
    ops["GENERATE"] = ops["GENERATE"].with_evidence(
        OperatorEvidence(runs=200, successes=20))
    measured = g.CognitiveGenome(operators=ops)
    after = compiler.compile_candidates("Вычисли 5 + 5", measured, FULL)
    assert after[0].score < before[0].score
    assert any("measured success" in r for c in after for r in c.reasons)


def test_a_grown_genome_widens_the_compiler_without_touching_it():
    """The point of putting the vocabulary in data."""
    parent = g.CognitiveGenome()
    grown = g.propose(parent, "compose_operators", steps=("GENERATE", "CRITIQUE", "REPAIR"),
                      op_id="COUNTERFACTUAL_REFINEMENT").candidate
    grown = g.propose(grown, "create_program_template", name="counterfactual",
                      steps=("OBSERVE", "COUNTERFACTUAL_REFINEMENT", "ANSWER"),
                      applicability=("reasoning",)).candidate
    picked = compiler.compile_candidates("Почему это работает?", grown, FULL)
    assert any(c.program.template == "counterfactual" for c in picked)


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

def test_a_straight_chain_runs_and_produces_an_answer():
    genome = g.CognitiveGenome()
    program = CognitiveProgram.build(("OBSERVE", "GENERATE", "ANSWER"))
    result = Runtime(genome.operators, fake_impls()).run(program, "задача")
    assert result.ok is True
    assert [s.op_id for s in result.trace] == ["OBSERVE", "GENERATE", "ANSWER"]
    assert result.calls_used == 2


def test_a_composed_operator_executes_as_its_chain():
    """Otherwise composition is a naming exercise: the search discovers an
    operator, the runtime has no implementation, and the measurement
    scores a step that never ran."""
    genome = g.propose(g.CognitiveGenome(), "compose_operators",
                       steps=("GENERATE", "CRITIQUE", "REPAIR"),
                       op_id="COUNTERFACTUAL_REFINEMENT").candidate
    program = CognitiveProgram.build(("OBSERVE", "COUNTERFACTUAL_REFINEMENT", "ANSWER"))
    result = Runtime(genome.operators, fake_impls()).run(program, "задача")
    executed = [s.op_id for s in result.trace]
    assert executed == ["OBSERVE", "GENERATE", "CRITIQUE", "REPAIR", "ANSWER"]


def test_a_missing_implementation_is_refused_before_any_budget_is_spent():
    """A program that dies at step four has already paid for three."""
    genome = g.CognitiveGenome()
    program = CognitiveProgram.build(("OBSERVE", "GENERATE", "ANSWER"))
    impls = fake_impls()
    impls.pop("GENERATE")
    result = Runtime(genome.operators, impls).run(program, "задача")
    assert result.ok is False
    assert result.calls_used == 0
    assert "GENERATE" in result.stop_reason


def test_a_failing_step_does_not_collapse_the_run():
    """'Which step failed' is the first question about a bad answer."""
    def broken(state):
        raise RuntimeError("brain died")

    genome = g.CognitiveGenome()
    program = CognitiveProgram.build(("OBSERVE", "GENERATE", "CRITIQUE", "ANSWER"))
    result = Runtime(genome.operators, fake_impls(CRITIQUE=broken)).run(program, "задача")
    failed = [s for s in result.trace if not s.ok]
    assert [s.op_id for s in failed] == ["CRITIQUE"]
    assert "brain died" in failed[0].error
    assert result.ok is True, "the draft survived; the answer should too"


def test_a_step_whose_input_never_arrived_is_skipped_and_recorded():
    def barren(state):
        return StepOutcome(values={}, calls=1, brain="b")

    genome = g.CognitiveGenome()
    program = CognitiveProgram.build(("OBSERVE", "GENERATE", "REPAIR", "ANSWER"))
    result = Runtime(genome.operators, fake_impls(GENERATE=barren)).run(program, "задача")
    skipped = [s for s in result.trace if not s.ok]
    assert [s.op_id for s in skipped] == ["REPAIR", "ANSWER"] or "REPAIR" in [s.op_id for s in skipped]
    assert any("needs" in s.error for s in skipped)


def test_the_budget_stops_the_chain_and_says_so():
    genome = g.CognitiveGenome()
    program = CognitiveProgram.build(
        ("OBSERVE", "RETRIEVE", "GENERATE", "CRITIQUE", "REPAIR", "ANSWER"))
    result = Runtime(genome.operators, fake_impls()).run(program, "задача", Budget(calls=2))
    assert result.stopped_early is True
    assert "call budget" in result.stop_reason
    assert result.calls_used <= 3


def test_a_budget_stop_with_a_usable_draft_is_not_a_failure():
    """Folding the two together makes every budget cut look like an
    inability to answer."""
    genome = g.CognitiveGenome()
    program = CognitiveProgram.build(("OBSERVE", "GENERATE", "CRITIQUE", "REPAIR", "ANSWER"))
    result = Runtime(genome.operators, fake_impls()).run(program, "задача", Budget(calls=2))
    assert result.stopped_early is True
    assert result.ok is True
    assert result.answer.startswith("GENERATE:")


def test_the_trace_names_which_brain_did_which_step():
    """An answer assembled from several models is undebuggable without it."""
    genome = g.CognitiveGenome()
    program = CognitiveProgram.build(("OBSERVE", "GENERATE", "CRITIQUE", "ANSWER"))
    result = Runtime(genome.operators, fake_impls()).run(program, "задача")
    assert "brain-generate" in result.brains_used
    assert all(s.brain or not s.ok or s.op_id in {"OBSERVE"} for s in result.trace)


def test_history_keeps_every_revision_of_a_draft():
    """Aggregating would erase exactly what a critique loop is for."""
    state = ProgramState(task="t")
    state.put(DRAFT, "first")
    state.put(DRAFT, "second")
    assert state.history[DRAFT] == ["first", "second"]
    assert state.get(DRAFT) == "second"


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_a_program_referring_to_operators_a_genome_lacks_is_invalid():
    program = CognitiveProgram.build(("OBSERVE", "TELEPATHY", "ANSWER"))
    assert "unknown operators" in program.validate(g.CognitiveGenome().operators)[0]


def test_an_untypeable_program_is_invalid():
    program = CognitiveProgram.build(("REPAIR", "GENERATE"))
    assert program.validate(g.CognitiveGenome().operators)


def test_two_programs_with_the_same_chain_share_a_signature():
    a = CognitiveProgram.build(("OBSERVE", "GENERATE", "ANSWER"))
    b = CognitiveProgram.build(("OBSERVE", "GENERATE", "ANSWER"))
    assert a.program_id != b.program_id
    assert a.signature() == b.signature()


def test_expansion_survives_a_genome_with_a_cyclic_composite():
    """Validation forbids it; an interpreter should still not loop forever
    on a genome that got there another way."""
    from mana.cognition.ir import CognitiveOperator
    from mana.cognition.ir import DRAFT as D, TASK as T
    looping = CognitiveOperator("LOOP", (T,), (D,), implementation="composite",
                                components=("LOOP",))
    ops = dict(g.CognitiveGenome().operators)
    ops["LOOP"] = looping
    program = CognitiveProgram.build(("LOOP",))
    assert len(program.expand(ops)) < 20
