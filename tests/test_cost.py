"""
tests/test_cost.py — cost in units that mean something, and the two
traps that make a cost model lie.

Everything counted "calls" before this. One call to a 120B remote model
and one call to a local 7B were the same number, which made "cheap
computation first" unmeasurable and every claim of saving anything
rhetoric.
"""
from __future__ import annotations

import pytest

from mana.core import cost as c
from mana.core.cost import CostVector, efficiency, one_call
from mana.core.gates import Claim, Evidence, PairedOutcome, judge


# ---------------------------------------------------------------------------
# trap 1: unmeasured is not zero
# ---------------------------------------------------------------------------

def test_a_call_that_reported_no_tokens_is_unmeasured_not_free():
    """A remote API that does not report token use has an unknown token
    cost. Treating it as zero makes the most expensive substrate look
    free -- exactly backwards."""
    v = one_call("remote_llm", 0.9)
    assert v.tokens == 0
    assert v.unmeasured_token_calls == 1
    assert v.tokens_complete is False


def test_a_call_that_reported_zero_tokens_is_measured():
    """Zero and unknown are different facts."""
    v = one_call("algorithmic", 0.001, tokens_in=0, tokens_out=0)
    assert v.unmeasured_token_calls == 0
    assert v.tokens_complete is True


def test_an_aggregate_says_how_much_of_it_was_unmeasured():
    mixed = one_call("remote_llm", 1.0) + one_call("remote_llm", 1.0, tokens_in=50,
                                                   tokens_out=10)
    assert mixed.tokens == 60
    assert mixed.unmeasured_token_calls == 1
    assert mixed.tokens_complete is False
    assert "без учёта" in mixed.describe()


def test_a_ratio_for_an_unmeasured_unit_is_none_not_zero():
    """0.0 would read as "infinitely inefficient" for exactly the
    substrates whose cost we failed to record."""
    ratios = efficiency(0.5, one_call("remote_llm", 1.0))
    assert ratios["per_1k_tokens"] is None
    assert ratios["per_mb"] is None
    assert ratios["per_second"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# trap 2: one number hides which unit bought the gain
# ---------------------------------------------------------------------------

def test_there_is_no_single_efficiency_score():
    """Collapsing seconds, tokens and megabytes into one number needs
    weights nobody has measured; an invented weighting reads as a
    measurement while being a preference."""
    assert not hasattr(c, "efficiency_score")
    assert isinstance(efficiency(1.0, one_call("remote_llm", 1.0)), dict)


def test_the_unit_change_is_what_makes_two_substrates_distinguishable():
    """The whole point of the phase, as an assertion: under the old unit
    these two were identical."""
    llm = one_call("remote_llm", 0.9, tokens_in=120, tokens_out=40)
    alg = one_call("algorithmic", 0.0004, tokens_in=0, tokens_out=0)
    same_gain = 0.5
    assert efficiency(same_gain, llm)["per_call"] == efficiency(same_gain, alg)["per_call"]
    assert (efficiency(same_gain, alg)["per_second"] >
            efficiency(same_gain, llm)["per_second"] * 100)


# ---------------------------------------------------------------------------
# arithmetic that would report a machine that never existed
# ---------------------------------------------------------------------------

def test_peak_memory_of_two_sequential_things_is_the_larger_peak():
    a = CostVector(calls=1, rss_bytes=100)
    b = CostVector(calls=1, rss_bytes=250)
    assert (a + b).rss_bytes == 250


def test_memory_stays_unknown_when_neither_side_measured_it():
    assert (CostVector(calls=1) + CostVector(calls=1)).rss_bytes is None


def test_the_substrate_mix_is_carried_through_an_aggregate():
    mixed = one_call("algorithmic", 0.001) + one_call("remote_llm", 1.0) \
        + one_call("remote_llm", 1.0)
    assert mixed.by_substrate == {"algorithmic": 1, "remote_llm": 2}


# ---------------------------------------------------------------------------
# the core still does not judge on cost
# ---------------------------------------------------------------------------

def _outcomes(n, base_ok, cand_ok):
    return [PairedOutcome(task_id=f"t{i}", domain="d",
                          baseline_correct=(i / n) < base_ok,
                          candidate_correct=(i / n) < cand_ok) for i in range(n)]


def test_no_gate_reads_the_cost():
    """Cost is recorded, never judged. A gate that started reading it
    would make acceptance depend on how expensive the proof was, which is
    not a property of whether the claim is true."""
    claim = Claim(claim_id="c1", kind="program", description="test")
    common = dict(paired_dev=_outcomes(40, 0.4, 0.8),
                  baseline_hidden=0.40, candidate_hidden=0.72,
                  counterexamples_sought=4, counterexamples_found=0)
    cheap = judge(claim, Evidence(cost=CostVector(calls=1, wall_seconds=0.001), **common))
    dear = judge(claim, Evidence(cost=CostVector(calls=99999, wall_seconds=9e6,
                                                 tokens_in=10 ** 8), **common))
    assert cheap.accepted == dear.accepted
    assert cheap.failed_gates == dear.failed_gates


def test_the_cost_is_still_recorded_in_the_measurements():
    claim = Claim(claim_id="c1", kind="program", description="test")
    verdict = judge(claim, Evidence(paired_dev=_outcomes(40, 0.4, 0.8),
                                    cost=CostVector(calls=7, wall_seconds=1.25)))
    assert verdict.measurements["cost"]["calls"] == 7


def test_no_gate_name_mentions_cost():
    """A defence against the fix drifting back: if a gate is ever named
    after cost, acceptance has started to depend on price."""
    gates = Claim(claim_id="c", kind="program", description="",
                  asserts_transfer=True).required_gates()
    assert not any("cost" in name or "cheap" in name for name in gates)


# ---------------------------------------------------------------------------
# the pool measures what it spent
# ---------------------------------------------------------------------------

def test_the_pool_totals_its_own_spending(isolated_config):
    from mana.brains import BrainPool, BrainSpec
    isolated_config.enable_llm = True
    pool = BrainPool(isolated_config, transport=lambda **kw: "ok")
    pool.brains = {"local": BrainSpec("local", "ollama", "m", local=True)}
    pool.health = {"local": __import__("mana.brains", fromlist=["BrainHealth"]).BrainHealth()}
    for _ in range(3):
        pool.ask_brain("local", "задача")
    total = pool.total_cost()
    assert total.calls == 3
    assert total.by_substrate == {"local_llm": 3}
    assert total.wall_seconds >= 0.0


def test_a_failed_call_still_costs(isolated_config):
    """Reporting a failure as free makes an unreliable brain look cheap,
    though its failures are paid for twice -- once here and again by the
    failover that follows."""
    from mana.brains import BrainHealth, BrainPool, BrainSpec

    def broken(**kw):
        raise RuntimeError("nope")

    isolated_config.enable_llm = True
    pool = BrainPool(isolated_config, transport=broken)
    pool.brains = {"local": BrainSpec("local", "ollama", "m", local=True)}
    pool.health = {"local": BrainHealth()}
    result = pool.ask_brain("local", "задача")
    assert result["ok"] is False
    assert result["cost"]["calls"] == 1
    assert pool.total_cost().calls == 1


# ---------------------------------------------------------------------------
# the activation counter
# ---------------------------------------------------------------------------

def test_watching_reports_only_what_happened_inside_it():
    from mana.core import instrument
    instrument.record_read("before")
    with instrument.watching() as used:
        instrument.record_read("inside")
        instrument.record_read("inside")
    assert used.get("inside") == 2
    assert "before" not in used


def test_the_counter_never_raises():
    """Instrumentation that can break the thing it measures is worse
    than no instrumentation."""
    from mana.core import instrument
    instrument.record_read(None)          # type: ignore[arg-type]
    instrument.record_read("ok")
    assert instrument.reads("ok") >= 1


def test_exercised_is_a_question_about_one_name():
    from mana.core import instrument
    assert instrument.exercised({"gap.cost": 3}, "gap.cost") is True
    assert instrument.exercised({"gap.cost": 0}, "gap.cost") is False
    assert instrument.exercised({}, "gap.cost") is False
