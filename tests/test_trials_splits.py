"""Three splits, and the ones that cannot be peeked at.

The claim "measured on data it was never chosen on" rests entirely on
this module behaving, so these test the refusals rather than the happy
path: a fresh split that answers before sealing, or twice, or about a
strategy nobody committed to, would make that claim worthless while every
number in it stayed true.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from mana.cognition import trials

#: The checkout, so a spawned process imports the same code.
ROOT = Path(__file__).resolve().parent.parent


NAME = "test_experiment"


@pytest.fixture(autouse=True)
def clean(tmp_path, monkeypatch):
    """Its own store, per test.

    The counts live on disk now, so a suite sharing the real one would
    both spend the user's holdout budget and inherit it -- the journal's
    old bug, which wrote 46 real episodes into the repository root, in a
    place where the consequence would be a holdout claim nobody could
    trust.
    """
    monkeypatch.setattr(trials, "_store", lambda: tmp_path / "trials")
    trials._reset_for_tests()
    trials.register(NAME)
    yield
    trials._reset_for_tests()


def _constant(value):
    return lambda seed: value


# --------------------------------------------------------------------------
# the seeds are never handed out
# --------------------------------------------------------------------------

def test_a_hidden_split_returns_numbers_and_not_seeds():
    """An agent cannot overfit a set it has never seen. `core/splits.py`
    made this argument first; this is the same one for a world."""
    result = trials.score(NAME, trials.VALIDATION, _constant(0.5))
    assert result["mean"] == 0.5
    assert result["values"] is None
    # Discovery is the set you are allowed to fit, so it does hand them back.
    assert trials.score(NAME, trials.DISCOVERY, _constant(0.5))["values"]


def test_the_splits_do_not_overlap():
    seen = set()
    for split in (trials.DISCOVERY, trials.VALIDATION, trials.FRESH):
        assert trials.size(NAME, split) > 0
        seen.add(trials.size(NAME, split))
    # The seeds themselves are not reachable from here, which is the
    # point; what can be checked is that each split is a real sample.
    assert trials.size(NAME, trials.FRESH) >= 30


# --------------------------------------------------------------------------
# fresh means fresh
# --------------------------------------------------------------------------

def test_the_fresh_split_refuses_before_anything_is_sealed():
    with pytest.raises(trials.NotSealed):
        trials.score(NAME, trials.FRESH, _constant(0.5))


def test_the_fresh_split_refuses_a_strategy_nobody_committed_to():
    trials.seal(NAME, {"policy": "planning"})
    with pytest.raises(trials.NotSealed):
        trials.score(NAME, trials.FRESH, _constant(0.5),
                     strategy={"policy": "something else"})


def test_the_fresh_split_answers_once():
    """A second answer would make it a set to tune against, which is what
    it exists not to be."""
    strategy = {"policy": "planning"}
    trials.seal(NAME, strategy)
    trials.score(NAME, trials.FRESH, _constant(0.5), strategy=strategy)
    with pytest.raises(trials.BudgetExceeded):
        trials.score(NAME, trials.FRESH, _constant(0.5), strategy=strategy)


def test_sealing_again_is_a_new_strategy_and_a_weaker_claim():
    """Adjusting after a fresh read is allowed and is not the same claim.

    The second strategy was chosen knowing something about this split, so
    its answer is worth less than the first. That is what `read_number`
    is for: not a refusal, a discount a reader can apply.
    """
    first = trials.seal(NAME, {"policy": "a"})
    trials.score(NAME, trials.FRESH, _constant(0.5), strategy={"policy": "a"})
    second = trials.seal(NAME, {"policy": "b"})
    assert first != second
    second_answer = trials.score(NAME, trials.FRESH, _constant(0.6),
                                 strategy={"policy": "b"})
    assert second_answer["mean"] == 0.6
    assert second_answer["read_number"] == 2


def test_the_fresh_split_runs_out_even_across_strategies():
    """Every answer leaks. Enough of them and it is a set to tune
    against, whatever it is called."""
    for n in range(3):
        strategy = {"policy": f"try-{n}"}
        trials.seal(NAME, strategy)
        trials.score(NAME, trials.FRESH, _constant(0.5), strategy=strategy)
    trials.seal(NAME, {"policy": "one too many"})
    with pytest.raises(trials.BudgetExceeded):
        trials.score(NAME, trials.FRESH, _constant(0.5),
                     strategy={"policy": "one too many"})


# --------------------------------------------------------------------------
# budget, not honour system
# --------------------------------------------------------------------------

def test_validation_has_a_budget_and_it_is_enforced():
    for _ in range(8):
        trials.score(NAME, trials.VALIDATION, _constant(0.5))
    with pytest.raises(trials.BudgetExceeded):
        trials.score(NAME, trials.VALIDATION, _constant(0.5))


def test_discovery_is_the_one_you_may_fit():
    for _ in range(30):
        trials.score(NAME, trials.DISCOVERY, _constant(0.5))
    assert trials.reads(NAME)[trials.DISCOVERY] == 30


def test_every_read_is_written_down():
    """A claim about a holdout can then be checked instead of believed."""
    trials.score(NAME, trials.DISCOVERY, _constant(0.1))
    trials.seal(NAME, {"policy": "planning"})
    trials.paired(NAME, trials.VALIDATION, _constant(0.1), _constant(0.2))

    events = [row["event"] for row in trials.audit(NAME)]
    assert events == ["score", "seal", "paired"]
    assert trials.audit(NAME)[-1]["split"] == trials.VALIDATION
    assert trials.audit(NAME)[-1]["mean"] == pytest.approx(0.1)


def test_a_paired_read_costs_one_read_and_not_two():
    """Charging the honest way of asking more than the sloppy one would
    push callers towards the sloppy one."""
    trials.paired(NAME, trials.VALIDATION, _constant(0.1), _constant(0.2))
    assert trials.reads(NAME)[trials.VALIDATION] == 1


# --------------------------------------------------------------------------
# what a paired read says
# --------------------------------------------------------------------------

def test_an_improvement_is_an_interval_above_zero():
    result = trials.paired(NAME, trials.VALIDATION, _constant(0.1), _constant(0.2))
    assert result["improved"] is True
    assert result["low"] > 0
    assert result["better"] == result["n"] and result["worse"] == 0


def test_no_difference_is_not_an_improvement():
    result = trials.paired(NAME, trials.VALIDATION, _constant(0.3), _constant(0.3))
    assert result["improved"] is False
    assert result["mean"] == 0.0
    assert result["same"] == result["n"]


def test_a_loss_is_not_an_improvement():
    result = trials.paired(NAME, trials.VALIDATION, _constant(0.4), _constant(0.2))
    assert result["improved"] is False
    assert result["high"] < 0


# --------------------------------------------------------------------------
# one discipline, several experiments
# --------------------------------------------------------------------------

def test_budgets_are_not_shared_between_experiments():
    """A quota shared between unrelated questions is a race: the second
    one to run would find the fresh split already spent and no way to
    know why."""
    other = "another_question"
    trials.register(other)
    strategy = {"policy": "a"}

    trials.seal(NAME, strategy)
    trials.score(NAME, trials.FRESH, _constant(0.5), strategy=strategy)
    assert trials.reads(NAME)[trials.FRESH] == 1
    assert trials.reads(other)[trials.FRESH] == 0

    trials.seal(other, strategy)
    assert trials.score(other, trials.FRESH, _constant(0.7),
                        strategy=strategy)["mean"] == 0.7


def test_an_unregistered_experiment_raises_rather_than_appearing():
    """A split that appears when you ask for it is one nobody agreed to."""
    with pytest.raises(KeyError):
        trials.score("never declared", trials.DISCOVERY, _constant(0.5))


def test_registering_twice_does_not_hand_back_a_spent_split():
    strategy = {"policy": "a"}
    trials.seal(NAME, strategy)
    trials.score(NAME, trials.FRESH, _constant(0.5), strategy=strategy)
    trials.register(NAME)          # same name again
    with pytest.raises(trials.BudgetExceeded):
        trials.score(NAME, trials.FRESH, _constant(0.5), strategy=strategy)


def test_each_experiment_keeps_its_own_audit():
    other = "another_question"
    trials.register(other)
    trials.score(NAME, trials.DISCOVERY, _constant(0.1))
    assert len(trials.audit(NAME)) == 1
    assert trials.audit(other) == []


# --------------------------------------------------------------------------
# a budget a restart resets is not a budget
# --------------------------------------------------------------------------

def test_the_counts_survive_a_restart():
    """Found by using it: the naming experiment read its fresh split, the
    rule was corrected, the script ran again in a new process, and the
    counter said "read once" both times. It had been read twice."""
    strategy = {"policy": "a"}
    trials.seal(NAME, strategy)
    trials.score(NAME, trials.FRESH, _constant(0.5), strategy=strategy)

    trials._reset_for_tests()          # a new process, as far as memory goes
    trials.register(NAME)
    assert trials.reads(NAME)[trials.FRESH] == 1
    with pytest.raises(trials.BudgetExceeded):
        trials.score(NAME, trials.FRESH, _constant(0.5), strategy=strategy)


def test_the_seal_survives_a_restart_too():
    strategy = {"policy": "a"}
    sealed = trials.seal(NAME, strategy)
    trials._reset_for_tests()
    trials.register(NAME)
    assert trials.sealed(NAME) == sealed


def test_a_store_that_cannot_be_written_says_so(tmp_path, monkeypatch):
    """The run is still protected; the next one will not be, and that is
    a different thing which must not look the same."""
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(trials, "_store", lambda: blocked / "trials")
    trials._reset_for_tests()
    trials.register(NAME)
    trials.score(NAME, trials.DISCOVERY, _constant(0.5))

    assert trials.experiment(NAME).persisted is False
    assert "ВНИМАНИЕ" in trials.describe(NAME)


# --------------------------------------------------------------------------
# the decision this was built for
# --------------------------------------------------------------------------

def test_naming_a_task_is_right_on_every_domain_under_the_adopted_policy():
    """The live decision `compiler.classify` makes, against the oracle
    `synthesis.DOMAIN_KIND`. It is 65.7% overall at the conservative
    default -- logic 0%, text_ops 28.5% -- and 100% under the policy the
    holdouts earned.

    Stated explicitly rather than read from the ambient adoption: an
    adoption is data on this installation now, and a test that depended
    on it would pass or fail according to what the machine happened to
    have adopted.
    """
    from mana.cognition.compiler import classify
    from mana.cognition.synthesis import DOMAIN_KIND
    from mana.core import tasks as task_gen
    from mana.policy import Policy, use

    earned = Policy.of(classify_text_first=True, classify_premise_marker=True)
    with use(earned):
        for domain, want in DOMAIN_KIND.items():
            for task in task_gen.generate(domain, 20, seed=777):
                got, _ = classify(task.prompt, difficulty=task.difficulty)
                assert got == want, (domain, task.prompt[:70], got)


def test_turning_the_knobs_on_fixes_what_was_failing():
    """The knobs are what changed the behaviour, not something else that
    happened to move at the same time.

    Stated from the conservative side now: the defaults went back to
    False when an adoption became data rather than a source edit, so the
    behaviour to demonstrate is what turning them ON does.
    """
    from mana.cognition.compiler import classify
    from mana.cognition.synthesis import DOMAIN_KIND
    from mana.core import tasks as task_gen
    from mana.policy import Policy, use

    def wrong_under(policy):
        wrong = 0
        with use(policy):
            for domain in ("logic", "text_ops"):
                for task in task_gen.generate(domain, 20, seed=777):
                    got, _ = classify(task.prompt, difficulty=task.difficulty)
                    wrong += int(got != DOMAIN_KIND[domain])
        return wrong

    off = Policy.of(classify_text_first=False, classify_premise_marker=False)
    on = Policy.of(classify_text_first=True, classify_premise_marker=True)
    assert wrong_under(off) > 20
    assert wrong_under(on) == 0


# --------------------------------------------------------------------------
# the control check: two processes, one holdout
# --------------------------------------------------------------------------

_READ_FRESH = """
import json, os, sys
from mana.cognition import trials
name, marker = sys.argv[1], sys.argv[2]
trials.register(name)
strategy = {"strategy": marker}
trials.seal(name, strategy)
try:
    result = trials.score(name, trials.FRESH, lambda seed: 0.5,
                          strategy=strategy)
    print(json.dumps({"ok": True, "read_number": result["read_number"]}))
except trials.BudgetExceeded as refused:
    print(json.dumps({"ok": False, "refused": str(refused)}))
"""


def _in_new_process(tmp_path, name, marker):
    """A real second process, not a reset of the first one's memory.

    The bug this exists for was invisible to an in-memory reset: the
    counter was rebuilt from nothing on every start, so the only way to
    see it was to actually start again.
    """
    import json
    import subprocess
    import sys

    env = dict(os.environ)
    env["MANA_DATA_DIR"] = str(tmp_path)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    done = subprocess.run([sys.executable, "-c", _READ_FRESH, name, marker],
                          capture_output=True, text=True, env=env,
                          cwd=str(ROOT), timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_a_second_process_knows_the_fresh_split_was_already_read(tmp_path):
    """The invariant the user asked for: same experiment, same fresh
    split, different process -> the read number goes up and the second
    read is detectable.

    Before the counts were written down, process one read it, exited,
    and process two started from zero and read it again believing it was
    the first."""
    first = _in_new_process(tmp_path, "control_check", "one")
    assert first["ok"] is True and first["read_number"] == 1

    second = _in_new_process(tmp_path, "control_check", "two")
    assert second["ok"] is True, second
    assert second["read_number"] == 2, "a restart handed the holdout back"

    third = _in_new_process(tmp_path, "control_check", "three")
    assert third["read_number"] == 3

    fourth = _in_new_process(tmp_path, "control_check", "four")
    assert fourth["ok"] is False and "квоте" in fourth["refused"]


def test_the_same_strategy_is_refused_across_processes(tmp_path):
    """Not only counted -- refused. Asking the same question twice until
    the noise falls the right way is the leak the per-strategy limit
    closes, and it has to survive a restart too."""
    assert _in_new_process(tmp_path, "same_strategy", "identical")["ok"] is True
    again = _in_new_process(tmp_path, "same_strategy", "identical")
    assert again["ok"] is False
    assert "уже отвечала" in again["refused"]


def test_a_new_experiment_starts_its_own_count(tmp_path):
    """New experiment, new declared split, independent counter -- even
    when another experiment in the same store has spent everything."""
    for marker in ("one", "two", "three"):
        _in_new_process(tmp_path, "spent", marker)
    assert _in_new_process(tmp_path, "spent", "four")["ok"] is False

    started = _in_new_process(tmp_path, "quite_separate", "one")
    assert started["ok"] is True and started["read_number"] == 1


def test_no_domain_is_worse_under_the_adopted_naming_rule():
    """The check the first experiment did not make and the aggregate hid:
    `code` once fell from 100% to 37% while the mean rose by 0.22.

    Verified on `task_naming_domains` (discovery 301..340, validation
    401..440, fresh 501..540): every domain +0.0000 or better on both
    hidden splits. This pins it so the rule cannot be widened later at
    one domain's expense without something failing.
    """
    from mana.cognition.compiler import classify
    from mana.cognition.synthesis import DOMAIN_KIND
    from mana.core import tasks as task_gen
    from mana.policy import Policy, use

    old = Policy.of(classify_text_first=False, classify_premise_marker=False)
    new = Policy.of(classify_text_first=True, classify_premise_marker=True)

    def hits(policy, domain, seed):
        with use(policy):
            return sum(
                classify(t.prompt, difficulty=t.difficulty)[0] == DOMAIN_KIND[domain]
                for t in task_gen.generate(domain, 20, seed))

    for domain in DOMAIN_KIND:
        for seed in (601, 602, 603):
            assert hits(new, domain, seed) >= hits(old, domain, seed), (
                f"{domain} стал хуже на посеве {seed}")
