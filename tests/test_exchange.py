"""
tests/test_exchange.py — what may pass between two installations.

The rule under test is one sentence: hypotheses cross, verdicts do not.
Most of what follows is about the second half, because that is the half a
future change could quietly break -- adding a convenience that adopts an
imported result would look helpful and would turn a system of proof into
a system of rumour.
"""
from __future__ import annotations

import json

import pytest

from mana.cognition import exchange, genome


def _genome():
    return genome.CognitiveGenome(
        operators=dict(genome.baseline_representations()) and {},
        representations=genome.baseline_representations(),
        program_templates=genome.baseline_templates(),
        learning_rules=genome.baseline_learning_rules())


# ------------------------------------------------------------- the allowlist

def test_identifier_parameters_pass():
    exchange.check_shareable({"name": "tokenize", "domain": "arithmetic",
                              "depth": 3, "enabled": True, "ratio": 0.5})
    exchange.check_shareable({"chain": ["tokenize", "count"]})


@pytest.mark.parametrize("params,what", [
    ({"task": "сколько продали гаек в марте"}, "текст задачи"),
    ({"field": "Номенклатура.ХитПродаж 2024"}, "имя с пробелом"),
    ({"note": "клиент ООО Ромашка"}, "название контрагента"),
])
def test_free_text_never_crosses(params, what):
    """The whole privacy rule in one check.

    Anything with a space in it is not an identifier, and a task someone
    typed or a business object name is exactly what must not leave.
    """
    with pytest.raises(exchange.ExchangeError):
        exchange.check_shareable(params)


def test_a_nested_structure_is_refused_rather_than_flattened():
    """Refused, not filtered: silently dropping a field would produce a
    hypothesis meaning something other than what was proposed, and the
    receiver would measure the wrong thing without either side noticing."""
    with pytest.raises(exchange.ExchangeError):
        exchange.check_shareable({"spec": {"nested": "thing"}})


def test_an_unknown_mutation_is_refused_at_construction():
    with pytest.raises(exchange.ExchangeError):
        exchange.Hypothesis(mutation="выдуманная_мутация")


# ------------------------------------------------------------------ identity

def test_the_id_comes_from_the_content_not_from_a_counter():
    """Two instances that independently think of the same change must
    produce the same id, or every rediscovery looks like a new idea and
    replication cannot be counted at all."""
    one = exchange.Hypothesis("compose_operators", {"a": "tokenize", "b": "count"})
    two = exchange.Hypothesis("compose_operators", {"b": "count", "a": "tokenize"})
    assert one.hypothesis_id == two.hypothesis_id      # key order is irrelevant
    other = exchange.Hypothesis("compose_operators", {"a": "tokenize", "b": "sum"})
    assert other.hypothesis_id != one.hypothesis_id


def test_a_record_edited_in_transit_is_refused(tmp_path):
    """The id is checkable against the content, so tampering shows."""
    path = tmp_path / "bundle.json"
    exchange.export_bundle(path, [exchange.Hypothesis(
        "compose_operators", {"a": "tokenize", "b": "count"})])
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["hypotheses"][0]["params"]["b"] = "sum"        # changed, id left alone
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    result = exchange.import_bundle(path)
    assert result.hypotheses == []
    assert "не сходится" in result.refused[0]["reason"]


# ------------------------------------------------------------------- bundles

def test_a_bundle_round_trips(tmp_path):
    path = tmp_path / "b.json"
    hypotheses = [exchange.Hypothesis("compose_operators",
                                      {"a": "tokenize", "b": "count"},
                                      note="стоит попробовать")]
    reports = [exchange.Report("abc123", "3f2a91bb", "REJECTED", trials=40,
                               discordant=9, p_value=0.31, holdout="v1+3f2a91bb")]
    exchange.export_bundle(path, hypotheses, reports)

    back = exchange.import_bundle(path)
    assert len(back.hypotheses) == 1
    assert back.hypotheses[0].params == {"a": "tokenize", "b": "count"}
    assert back.reports[0].verdict == "REJECTED"
    assert back.reports[0].holdout == "v1+3f2a91bb"


def test_one_bad_record_does_not_discard_the_bundle(tmp_path):
    path = tmp_path / "b.json"
    exchange.export_bundle(path, [exchange.Hypothesis(
        "compose_operators", {"a": "tokenize", "b": "count"})])
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["hypotheses"].append({"mutation": "нет_такой", "params": {}})
    raw["reports"] = [{"hypothesis_id": "x", "instance": "y", "verdict": "МОЖЕТ"}]
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    result = exchange.import_bundle(path)
    assert len(result.hypotheses) == 1
    assert len(result.refused) == 2
    assert {r["kind"] for r in result.refused} == {"hypothesis", "report"}


def test_already_known_hypotheses_are_skipped_not_refused(tmp_path):
    path = tmp_path / "b.json"
    hypothesis = exchange.Hypothesis("compose_operators",
                                     {"a": "tokenize", "b": "count"})
    exchange.export_bundle(path, [hypothesis])
    result = exchange.import_bundle(path, known={hypothesis.hypothesis_id})
    assert result.hypotheses == []
    assert result.refused == []          # not new is not the same as bad


def test_a_newer_format_is_refused_rather_than_guessed(tmp_path):
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"format": exchange.FORMAT,
                                "format_version": exchange.FORMAT_VERSION + 5}),
                    encoding="utf-8")
    with pytest.raises(exchange.ExchangeError):
        exchange.import_bundle(path)


def test_a_foreign_file_is_refused(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"format": "что-то другое"}), encoding="utf-8")
    with pytest.raises(exchange.ExchangeError):
        exchange.import_bundle(path)


def test_the_exported_bundle_carries_a_fingerprint_not_an_id(tmp_path, monkeypatch):
    from mana.core import identity
    monkeypatch.setenv(identity.INSTANCE_ENV, "рабочая-станция-ООО-Ромашка")
    identity.reset_cache()
    path = tmp_path / "b.json"
    exchange.export_bundle(path, [])
    blob = path.read_text(encoding="utf-8")
    assert "Ромашка" not in blob
    assert identity.fingerprint() in blob


# -------------------------------------------------------------- replication

def test_replication_counts_environments_not_reports():
    """An instance that ran the same experiment ten times has ten pieces
    of evidence and still one environment. Letting it vote ten times
    would turn persistence into agreement."""
    reports = [exchange.Report("h1", "aaaa", "ACCEPTED", created=1),
               exchange.Report("h1", "aaaa", "ACCEPTED", created=2),
               exchange.Report("h1", "aaaa", "ACCEPTED", created=3),
               exchange.Report("h1", "bbbb", "REJECTED", created=1)]
    summary = exchange.replication(reports)["h1"]
    assert summary["instances"] == 2
    assert summary["accepted"] == 1
    assert summary["rejected"] == 1


def test_a_later_report_supersedes_the_same_instance_earlier_one():
    reports = [exchange.Report("h1", "aaaa", "NOT_EVALUATED", trials=10, created=1),
               exchange.Report("h1", "aaaa", "ACCEPTED", trials=60, created=2)]
    summary = exchange.replication(reports)["h1"]
    assert summary["instances"] == 1
    assert summary["accepted"] == 1
    assert summary["not_evaluated"] == 0


def test_not_evaluated_is_not_counted_as_disagreement():
    """"Nobody had the power to test it" and "it does not work" are
    different facts, and the three-state verdict exists to keep them
    apart. `ruled` is what a reader should divide by."""
    reports = [exchange.Report("h1", "a", "ACCEPTED"),
               exchange.Report("h1", "b", "NOT_EVALUATED"),
               exchange.Report("h1", "c", "NOT_EVALUATED")]
    summary = exchange.replication(reports)["h1"]
    assert summary["instances"] == 3
    assert summary["ruled"] == 1
    assert summary["not_evaluated"] == 2


def test_distinct_holdouts_are_visible():
    """Since phase 25 each installation draws its own hidden set. Three
    verdicts over three different sets is a stronger claim than three
    over one, and the count says which happened."""
    reports = [exchange.Report("h1", "a", "ACCEPTED", holdout="v1+aaaa"),
               exchange.Report("h1", "b", "ACCEPTED", holdout="v1+bbbb")]
    assert exchange.replication(reports)["h1"]["distinct_holdouts"] == 2


# ------------------------------------------------------- nothing is adopted

def test_the_module_never_adopts_anything():
    """The rule the whole design rests on, read off the source.

    A convenience that adopted an imported result would look helpful and
    would destroy the only reason a verdict here means anything. Checked
    structurally, because a reviewer will not notice it being added.
    """
    import ast
    import inspect

    forbidden = {"apply", "adopt", "judge", "accept"}
    tree = ast.parse(inspect.getsource(exchange))
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            name = getattr(target, "id", None) or getattr(target, "attr", None)
            if name:
                called.add(name)
    assert not (called & forbidden), called & forbidden


def test_an_imported_hypothesis_is_rebuilt_against_the_local_genome():
    """The mutation travels; the candidate does not.

    Applying a foreign genome would import somebody's entire cognitive
    state along with the one change being tested, and the change is the
    only part anyone proposed.
    """
    local = genome.CognitiveGenome(
        representations=genome.baseline_representations(),
        program_templates=genome.baseline_templates(),
        learning_rules=genome.baseline_learning_rules())
    hypothesis = exchange.Hypothesis(
        "create_program_template",
        {"name": "foreign_template", "steps": ["OBSERVE", "GENERATE", "ANSWER"]})
    proposal = exchange.to_local_proposal(hypothesis, local)
    assert proposal.candidate.parent_id == local.genome_id
    assert "локальным доказательствам" in proposal.rationale


def test_a_mutation_that_does_not_apply_here_raises():
    """A real answer rather than a wasted experiment: "compose these two
    operators" is meaningless in a genome holding neither."""
    local = genome.CognitiveGenome(
        representations=genome.baseline_representations(),
        program_templates=genome.baseline_templates(),
        learning_rules=genome.baseline_learning_rules())
    hypothesis = exchange.Hypothesis("compose_operators",
                                     {"first": "nope", "second": "alsonope"})
    with pytest.raises(Exception):
        exchange.to_local_proposal(hypothesis, local)


def test_non_latin_identifiers_do_not_cross_and_that_is_deliberate():
    """A recorded decision, not an accident of the regular expression.

    MANA's own structural names are Latin throughout -- domains, nodes,
    operators. A name written in the user's own language is far more
    likely to be a business object out of their configuration
    ("Номенклатура") than a piece of MANA's vocabulary, and a privacy
    boundary should be narrower than strictly necessary rather than
    wider. If this ever blocks something legitimate it surfaces as a
    refusal with the reason attached, which is the failure worth having.
    """
    with pytest.raises(exchange.ExchangeError):
        exchange.check_shareable({"name": "Номенклатура"})
    exchange.check_shareable({"name": "nomenclature"})


# ------------------------------------------------------------------- queue

def test_an_annotative_parameter_is_dropped_and_a_behavioural_one_is_not(tmp_path):
    """The distinction the queue rests on.

    `create_program_template` carries a `description` written for a person
    -- "синтезировано из открытия ..." -- which free-text vetting rightly
    refuses and which changes nothing about what the template DOES.
    Dropping it is safe. Dropping `steps` would produce a hypothesis
    meaning something other than what was proposed.
    """
    queue = exchange.Queue(tmp_path / "q.json")
    record = queue.record_hypothesis("create_program_template", {
        "name": "probe_then_answer",
        "steps": ["OBSERVE", "RETRIEVE", "ANSWER"],
        "applicability": ["arithmetic"],
        "description": "Синтезировано из открытия d42; доказано на arithmetic."})
    assert record["shareable"] is True
    assert "description" not in record["params"]
    assert record["params"]["steps"] == ["OBSERVE", "RETRIEVE", "ANSWER"]


def test_a_hypothesis_that_cannot_be_shared_is_kept_with_the_reason(tmp_path):
    """It is still something to try here. Losing local work to a
    federation concern would be the wrong trade, and `export` filters, so
    it cannot leave by accident."""
    queue = exchange.Queue(tmp_path / "q.json")
    record = queue.record_hypothesis(
        "create_brain", {"brain_id": "мозг для 1С", "substrate": "algorithmic"})
    assert record["shareable"] is False
    assert record["refusal"]
    assert queue.stats()["not_shareable"] == 1
    assert queue.hypotheses(shareable_only=True) == []


def test_the_same_hypothesis_twice_is_recorded_once(tmp_path):
    """A cycle that rediscovers the same change on three runs contributes
    it once -- otherwise a persistent instance would look like agreement."""
    queue = exchange.Queue(tmp_path / "q.json")
    params = {"name": "t", "steps": ["OBSERVE", "ANSWER"]}
    first = queue.record_hypothesis("create_program_template", params)
    second = queue.record_hypothesis("create_program_template", dict(params))
    assert second.get("known") is True
    assert first["hypothesis_id"] == second["hypothesis_id"]
    assert queue.stats()["hypotheses"] == 1


def test_a_truncated_queue_file_does_not_lose_the_next_run(tmp_path):
    """Written through a temporary file, so a cycle dying mid-write cannot
    leave a half-file the next run reads as empty."""
    path = tmp_path / "q.json"
    path.write_text('{"hypotheses": [{"mutat', encoding="utf-8")
    queue = exchange.Queue(path)
    assert queue.stats()["hypotheses"] == 0        # unreadable, not fatal
    queue.record_hypothesis("create_program_template",
                            {"name": "t", "steps": ["OBSERVE", "ANSWER"]})
    assert queue.stats()["hypotheses"] == 1


def test_absorbing_an_import_marks_what_came_from_elsewhere(tmp_path):
    queue = exchange.Queue(tmp_path / "q.json")
    bundle = tmp_path / "b.json"
    exchange.export_bundle(bundle, [exchange.Hypothesis(
        "create_program_template", {"name": "t", "steps": ["OBSERVE", "ANSWER"]})])
    added = queue.absorb(exchange.import_bundle(bundle))
    assert added["added"] == 1
    stored = json.loads((tmp_path / "q.json").read_text(encoding="utf-8"))
    assert stored["hypotheses"][0]["foreign"] is True


# ------------------------------------------------- the cycle writes into it

class _FakeProposal:
    name = "probe_then_answer"
    steps = ("OBSERVE", "RETRIEVE", "ANSWER")
    applicability = ("arithmetic",)


def _cycle(tmp_path):
    from mana.cognition.research import ResearchCycle
    from mana.cognition.self_model import SelfModel
    return ResearchCycle(SelfModel(), task_texts={},
                         exchange_path=tmp_path / "q.json")


def test_the_cycle_records_its_hypothesis_before_confirming_it(tmp_path):
    """Recorded whether or not it is confirmed here.

    A queue holding only confirmed changes would share exactly the things
    everyone else can already derive; the value of a hypothesis is that
    somebody else might have the evidence this installation lacks.
    """
    cycle = _cycle(tmp_path)
    hypothesis_id = cycle._record_hypothesis(_FakeProposal())
    assert hypothesis_id
    stored = cycle.exchange.hypotheses()
    assert len(stored) == 1
    assert stored[0].params["steps"] == ["OBSERVE", "RETRIEVE", "ANSWER"]


@pytest.mark.parametrize("verdict", ["ACCEPTED", "REJECTED", "NOT_EVALUATED"])
def test_all_three_verdicts_are_recorded(tmp_path, verdict):
    """NOT_EVALUATED as carefully as the other two: "nobody here had the
    evidence" is a different fact from "it does not work", and collapsing
    them would read an unmeasured gate as a refutation."""
    cycle = _cycle(tmp_path)
    hypothesis_id = cycle._record_hypothesis(_FakeProposal())
    cycle._record_verdict(hypothesis_id, verdict, trials=30)
    reports = cycle.exchange.reports()
    assert [r.verdict for r in reports] == [verdict]
    assert reports[0].holdout.startswith("v1+")   # its own salted set


def test_a_cycle_without_a_queue_records_nothing_and_does_not_crash(tmp_path):
    """Optional on purpose: a cycle run for measurement should not
    accumulate a queue nobody asked for."""
    from mana.cognition.research import ResearchCycle
    from mana.cognition.self_model import SelfModel
    cycle = ResearchCycle(SelfModel(), task_texts={})
    assert cycle.exchange is None
    assert cycle._record_hypothesis(_FakeProposal()) == ""
    cycle._record_verdict("", "ACCEPTED")          # must not raise


def test_bookkeeping_failure_does_not_take_down_the_run(tmp_path, monkeypatch):
    """The federation is a side concern. A cycle spending real brain calls
    must not die because a queue file could not be written."""
    cycle = _cycle(tmp_path)

    def explode(*a, **k):
        raise OSError("диск переполнен")

    monkeypatch.setattr(cycle.exchange, "record_hypothesis", explode)
    assert cycle._record_hypothesis(_FakeProposal()) == ""


# --------------------------------------------------------------- consensus

def test_disagreement_is_the_finding_and_it_is_reported_as_one():
    """Accepted somewhere, rejected somewhere else.

    The most valuable outcome the bridge can produce: it says the change
    is CONDITIONAL, and the condition is a fact about environments that no
    single instance could have seen. `replication` counts votes; this says
    what the count means, because the interesting category is easy to miss
    in a table of numbers.
    """
    reports = [exchange.Report("h1", "a", "ACCEPTED"),
               exchange.Report("h1", "b", "REJECTED")]
    groups = exchange.consensus(reports)
    assert [e["hypothesis_id"] for e in groups["divergent"]] == ["h1"]
    assert groups["confirmed"] == []


def test_two_agreeing_instances_are_not_yet_agreement():
    """Two is not a pattern; it is two. Divergence is flagged from two
    because one contradiction is enough to know a claim is conditional --
    the asymmetry is deliberate."""
    reports = [exchange.Report("h1", "a", "ACCEPTED"),
               exchange.Report("h1", "b", "ACCEPTED")]
    groups = exchange.consensus(reports)
    assert groups["confirmed"] == []
    assert [e["hypothesis_id"] for e in groups["undecided"]] == ["h1"]


def test_enough_agreeing_instances_confirm():
    reports = [exchange.Report("h1", who, "ACCEPTED") for who in "abc"]
    assert [e["hypothesis_id"] for e in exchange.consensus(reports)["confirmed"]] == ["h1"]


def test_a_unanimous_refutation_is_kept_not_discarded():
    """Worth as much as a confirmation, and the result human research
    systematically loses: nobody publishes what did not work."""
    reports = [exchange.Report("h1", who, "REJECTED") for who in "abc"]
    groups = exchange.consensus(reports)
    assert [e["hypothesis_id"] for e in groups["refuted"]] == ["h1"]


def test_untestable_everywhere_is_undecided_not_refuted():
    """"Nobody could test it" is not "it does not work", and folding the
    two together would publish an unmeasured gate as a refutation."""
    reports = [exchange.Report("h1", who, "NOT_EVALUATED") for who in "abcde"]
    groups = exchange.consensus(reports)
    assert [e["hypothesis_id"] for e in groups["undecided"]] == ["h1"]
    assert groups["refuted"] == []


# ------------------------------------------------------------------ signing

@pytest.fixture
def signer(monkeypatch):
    from mana.core import identity
    monkeypatch.setenv(identity.INSTANCE_ENV, "test-signer")
    identity.reset_cache()
    return identity


def test_a_report_is_signed_and_verifies(signer):
    report = exchange.Report("h1", signer.fingerprint(), "ACCEPTED",
                             trials=40).sign()
    assert report.signed_by_someone
    assert report.verified


def test_changing_any_field_breaks_the_signature(signer):
    from dataclasses import replace
    report = exchange.Report("h1", signer.fingerprint(), "ACCEPTED",
                             trials=40).sign()
    assert not replace(report, verdict="REJECTED").verified
    assert not replace(report, trials=999).verified
    assert not replace(report, hypothesis_id="h2").verified


def test_claiming_someone_elses_fingerprint_fails(signer):
    """Both halves are checked. Verifying only the signature would let
    somebody sign with their own key while claiming another instance's
    fingerprint -- which is the impersonation signing exists to stop."""
    from dataclasses import replace
    report = exchange.Report("h1", signer.fingerprint(), "ACCEPTED").sign()
    assert not replace(report, instance="deadbeef").verified


def test_a_tampered_report_is_refused_on_import(tmp_path, signer):
    path = tmp_path / "b.json"
    exchange.export_bundle(path, [], [exchange.Report(
        "h1", signer.fingerprint(), "ACCEPTED", trials=40).sign()])
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["reports"][0]["verdict"] = "REJECTED"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    result = exchange.import_bundle(path)
    assert result.reports == []
    assert "подпись не сходится" in result.refused[0]["reason"]


def test_an_unsigned_report_is_kept_but_not_verified(tmp_path):
    """Absent is not the same as wrong. A record from an older version is
    merely unverified; discarding it would lose history to a format
    change."""
    path = tmp_path / "b.json"
    path.write_text(json.dumps({
        "format": exchange.FORMAT, "format_version": exchange.FORMAT_VERSION,
        "hypotheses": [],
        "reports": [{"hypothesis_id": "h1", "instance": "old", "verdict": "ACCEPTED"}],
    }, ensure_ascii=False), encoding="utf-8")
    result = exchange.import_bundle(path)
    assert len(result.reports) == 1
    assert result.reports[0].verified is False


def test_replication_can_be_asked_to_count_only_verified(signer):
    """The Sybil count this closes: before signing, `instance` was a
    self-declared string, so one installation could invent twelve of them
    and manufacture "confirmed on twelve"."""
    real = exchange.Report("h1", signer.fingerprint(), "ACCEPTED").sign()
    invented = [exchange.Report("h1", f"fake{i:04d}", "ACCEPTED")
                for i in range(11)]
    everything = [real] + invented

    assert exchange.replication(everything)["h1"]["instances"] == 12
    verified = exchange.replication(everything, verified_only=True)
    assert verified["h1"]["instances"] == 1


def test_signing_does_not_stop_sybil_and_the_docstring_says_so():
    """Recorded because overstating it would be worse than not having it.

    Signatures stop impersonation. One person can still generate twelve
    key pairs; what limits that here is deciding whose keys count, not
    cryptography.
    """
    import inspect
    source = inspect.getsource(exchange.Report)
    assert "Sybil" in source
    assert "does NOT" in source or "not stop" in source


def test_the_queue_signs_what_it_records(tmp_path, signer):
    queue = exchange.Queue(tmp_path / "q.json")
    record = queue.record_hypothesis(
        "create_program_template", {"name": "t", "steps": ["OBSERVE", "ANSWER"]})
    queue.record_report(record["hypothesis_id"], "ACCEPTED", trials=30)
    stored = queue.reports()
    assert len(stored) == 1
    assert stored[0].verified is True
