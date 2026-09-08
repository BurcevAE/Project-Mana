"""Can the protocol be walked without a person in the middle?

Not "is the classifier better" -- that was settled and is not what this
tests. What is tested here is the loop: given only an oracle, a region of
change, split sizes and quotas, does it find the failure, pick a
candidate, take it through validation and a sealed fresh split, apply a
criterion nobody moved, and come back with a verdict.

The refusals matter more than the happy path, so most of these are about
what it must not do: adopt without permission, adopt without evidence,
accept a candidate that breaks a group, or accept one that only wins where
it was chosen.
"""
from __future__ import annotations

import pytest

from mana import policy as policy_mod
from mana.cognition import investigator, trials
from mana.cognition.findings import Ledger
from mana.core.gates import ACCEPTED, REJECTED
from mana.policy import Policy


@pytest.fixture(autouse=True)
def clean(tmp_path, monkeypatch):
    monkeypatch.setattr(trials, "_store", lambda: tmp_path / "trials")
    trials._reset_for_tests()
    investigator._ORACLES.clear()
    yield
    trials._reset_for_tests()
    investigator._ORACLES.clear()


def _oracle(name="probe", knobs=("classify_text_first",),
            broken=("logic",), fixed_by=("classify_text_first",)):
    """A decision that gets one group wrong until a knob is turned on.

    Synthetic on purpose: the point is the protocol, and a real domain
    would make a failure here look like a fact about that domain.
    """
    groups = ("logic", "arithmetic")

    def decide(item):
        group, _ = item
        if group in broken:
            on = all(policy_mod.get(name) for name in fixed_by)
            return "right" if on else "wrong"
        return "right"

    return investigator.Oracle(
        name=name, decide=decide,
        samples=lambda seed, group: [(group, n) for n in range(10)],
        truth=lambda group: "right", groups=groups, knobs=knobs,
        seeds={trials.DISCOVERY: tuple(range(1, 11)),
               trials.VALIDATION: tuple(range(101, 111)),
               trials.FRESH: tuple(range(201, 211))})


# --------------------------------------------------------------------------
# it walks the whole thing
# --------------------------------------------------------------------------

def test_it_finds_the_failure_without_being_told_which(tmp_path):
    investigator.declare(_oracle())
    report = investigator.investigate("probe", ledger=Ledger(tmp_path / "f.jsonl"))
    assert set(report.failing) == {"logic"}
    assert report.failing["logic"] == 0.0


def test_it_reaches_a_verdict_through_every_stage(tmp_path):
    investigator.declare(_oracle())
    report = investigator.investigate("probe", ledger=Ledger(tmp_path / "f.jsonl"))
    assert report.verdict == ACCEPTED
    assert report.validation["improved"] and report.fresh["improved"]
    assert report.fresh["worst_group"] >= 0
    assert report.chosen == {"classify_text_first": True}


def test_the_fresh_split_is_sealed_before_it_is_read(tmp_path):
    investigator.declare(_oracle())
    investigator.investigate("probe", ledger=Ledger(tmp_path / "f.jsonl"))
    events = [row["event"] for row in trials.audit("probe")]
    assert events.index("seal") < len(events) - 1
    assert events[-1] == "paired"
    assert trials.sealed("probe")


def test_a_clean_decision_is_left_alone(tmp_path):
    investigator.declare(_oracle(broken=()))
    report = investigator.investigate("probe", ledger=Ledger(tmp_path / "f.jsonl"))
    assert report.stopped_at == "провал"
    assert "исправный конец" in report.why
    assert trials.reads("probe")[trials.VALIDATION] == 0


# --------------------------------------------------------------------------
# what it must not do
# --------------------------------------------------------------------------

def test_it_does_not_adopt_unless_it_is_allowed_to(tmp_path):
    """Searching and measuring change nothing by themselves. Putting the
    result into force is the step that alters the program for the person
    using it, and it is theirs to permit."""
    investigator.declare(_oracle())
    report = investigator.investigate("probe", ledger=Ledger(tmp_path / "f.jsonl"))
    assert report.verdict == ACCEPTED
    assert report.adopted is False
    assert policy_mod.active() is policy_mod.BASELINE
    assert "не разрешено" in report.why


def test_when_allowed_it_adopts_with_the_evidence_attached(tmp_path):
    investigator.declare(_oracle())
    report = investigator.investigate("probe", ledger=Ledger(tmp_path / "f.jsonl"),
                                      allow_adopt=True)
    assert report.adopted is True
    assert policy_mod.active().get("classify_text_first") is True

    evidence = policy_mod.adoption_evidence()
    assert evidence["experiment"] == "probe"
    assert evidence["verdict"] == ACCEPTED
    assert evidence["fresh"]["n"] == 10
    assert policy_mod.revert() and policy_mod.active() is policy_mod.BASELINE


def test_an_adoption_without_evidence_is_refused():
    """Not politeness: an overlay that can be written on an argument is
    the same thing as a default that can be edited on one."""
    with pytest.raises(policy_mod.AdoptionRefused):
        policy_mod.adopt(Policy.of(classify_text_first=True), {})
    with pytest.raises(policy_mod.AdoptionRefused):
        policy_mod.adopt(Policy.of(classify_text_first=True),
                         {"experiment": "probe"})
    assert policy_mod.active() is policy_mod.BASELINE


def test_a_candidate_that_breaks_a_group_never_reaches_a_holdout(tmp_path):
    """The no-regression rule is applied on the discovery half too, so a
    candidate that could not pass is not sent to spend a quota."""
    groups = ("logic", "arithmetic")

    def decide(item):
        group, _ = item
        on = policy_mod.get("classify_text_first")
        if group == "logic":
            return "right" if on else "wrong"
        return "wrong" if on else "right"      # fixes one, breaks the other

    investigator.declare(investigator.Oracle(
        name="tradeoff", decide=decide,
        samples=lambda seed, group: [(group, n) for n in range(10)],
        truth=lambda group: "right", groups=groups,
        knobs=("classify_text_first",),
        seeds={trials.DISCOVERY: tuple(range(1, 11)),
               trials.VALIDATION: tuple(range(101, 111)),
               trials.FRESH: tuple(range(201, 211))}))

    report = investigator.investigate("tradeoff",
                                      ledger=Ledger(tmp_path / "f.jsonl"),
                                      allow_adopt=True)
    assert report.stopped_at == "кандидат"
    assert report.adopted is False
    assert trials.reads("tradeoff")[trials.VALIDATION] == 0
    assert trials.reads("tradeoff")[trials.FRESH] == 0


def test_a_candidate_that_only_wins_where_it_was_chosen_is_rejected(tmp_path):
    """The discovery seeds are 1..10 and the hidden ones are not. A
    change that helps only on the first is what the split exists to
    catch."""
    groups = ("logic", "arithmetic")

    def decide(item):
        group, _ = item
        on = policy_mod.get("classify_text_first")
        if group != "logic":
            return "right"
        # Right only on the seeds it was chosen on.
        return "right" if (on and item[1] < 10 and _seed[0] < 100) else "wrong"

    _seed = [0]

    def samples(seed, group):
        _seed[0] = seed
        return [(group, n) for n in range(10)]

    investigator.declare(investigator.Oracle(
        name="fitted", decide=decide, samples=samples,
        truth=lambda group: "right", groups=groups,
        knobs=("classify_text_first",),
        seeds={trials.DISCOVERY: tuple(range(1, 11)),
               trials.VALIDATION: tuple(range(101, 111)),
               trials.FRESH: tuple(range(201, 211))}))

    report = investigator.investigate("fitted",
                                      ledger=Ledger(tmp_path / "f.jsonl"),
                                      allow_adopt=True)
    assert report.verdict == REJECTED
    assert report.stopped_at == "валидация"
    assert report.adopted is False
    assert trials.reads("fitted")[trials.FRESH] == 0


def test_the_acceptance_rule_is_written_down_and_not_chosen(tmp_path):
    """An investigator that could pick its own success criterion would
    find success."""
    import inspect

    source = inspect.getsource(investigator)
    assert len(investigator.ACCEPTANCE) == 3
    # Nothing in the loop may set the threshold or the acceptance clauses.
    assert "FAILING_BELOW =" in source
    assert source.count("FAILING_BELOW =") == 1
    for forbidden in ("FAILING_BELOW +", "FAILING_BELOW -", "ACCEPTANCE ="):
        assert source.count(forbidden) <= 1


def test_every_result_is_written_to_the_ledger(tmp_path):
    ledger = Ledger(tmp_path / "f.jsonl")
    investigator.declare(_oracle())
    report = investigator.investigate("probe", ledger=ledger)
    assert report.finding_id
    assert [f.finding_id for f in ledger.findings()] == [report.finding_id]
    assert ledger.findings()[0].verdict == ACCEPTED
