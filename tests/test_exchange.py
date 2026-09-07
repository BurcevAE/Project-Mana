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
