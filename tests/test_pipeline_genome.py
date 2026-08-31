"""
tests/test_pipeline_genome.py — PipelineSpec/PipelineFactory genome
validity. Pure computation, no LLM/web/exec needed; safe to run anywhere.
"""
from __future__ import annotations

from mana.config import RandomManager
from mana.pipeline import PipelineFactory, PipelineSpec


def _validate_graph(cfg, spec: PipelineSpec, ctx: str = "") -> None:
    g = spec.graph_nodes
    assert g[-1] == "EVALUATE", f"{ctx}: EVALUATE must be last: {g}"
    assert g.count("EVALUATE") == 1, f"{ctx}: EVALUATE must appear exactly once: {g}"
    assert "LLM" in g, f"{ctx}: LLM must be present: {g}"
    assert len(g) == len(set(g)), f"{ctx}: no duplicate nodes allowed: {g}"
    assert len(g) <= cfg.graph_max_nodes, f"{ctx}: graph too long: {g}"
    allowed = {"MEMORY", "WEB", "LLM", "CRITIC", "REPAIR", "SYNTHESIS", "EVALUATE", "EXECUTE"}
    assert set(g) <= allowed, f"{ctx}: unknown node in graph: {g}"


def test_random_pipeline_always_valid(isolated_config):
    rm = RandomManager(7)
    for i in range(300):
        spec = PipelineFactory.random(rm, isolated_config)
        _validate_graph(isolated_config, spec, f"random draw {i}")


def test_random_pipeline_reaches_beyond_seven_fixed_graphs(isolated_config):
    """The whole point of the structural generator: reachable-graph count
    should be well beyond the 7 hand-picked tuples the old code used."""
    rm = RandomManager(11)
    seen = {PipelineFactory.random(rm, isolated_config).graph_nodes for _ in range(400)}
    assert len(seen) > 30, f"expected much more diversity than 7 fixed graphs, got {len(seen)}"


def test_sequential_mutation_always_stays_valid(isolated_config):
    rm = RandomManager(3)
    spec = PipelineFactory.random(rm, isolated_config)
    for i in range(500):
        spec = PipelineFactory.mutate(spec, rm, isolated_config, rate=0.5, max_changes=2)
        _validate_graph(isolated_config, spec, f"mutation step {i}")


def test_crossover_always_valid(isolated_config):
    rm = RandomManager(5)
    for i in range(150):
        a = PipelineFactory.random(rm, isolated_config)
        b = PipelineFactory.random(rm, isolated_config)
        c = PipelineFactory.crossover(a, b, rm, isolated_config)
        _validate_graph(isolated_config, c, f"crossover {i}")


def test_normalize_repairs_evaluate_before_llm(isolated_config):
    """Regression test for a real bug found & fixed during development:
    EVALUATE occurring before LLM used to abort the adaptive loop
    immediately (see ExecutionMixin._adaptive_answer_v41)."""
    bad = PipelineSpec(graph_nodes=("EVALUATE", "LLM", "CRITIC"))
    bad.normalize(isolated_config)
    _validate_graph(isolated_config, bad, "EVALUATE-first adversarial input")


def test_normalize_keeps_execute_node(isolated_config):
    """Regression test: EXECUTE used to be missing from allowed_nodes, so
    normalize() silently stripped it out of any evolved genome."""
    spec = PipelineSpec(graph_nodes=("LLM", "EXECUTE", "EVALUATE"))
    spec.normalize(isolated_config)
    assert "EXECUTE" in spec.graph_nodes


def test_normalize_respects_length_cap_without_losing_evaluate(isolated_config):
    isolated_config.graph_max_nodes = 4
    spec = PipelineSpec(graph_nodes=("MEMORY", "WEB", "CRITIC", "REPAIR", "SYNTHESIS", "EXECUTE", "LLM", "EVALUATE"))
    spec.normalize(isolated_config)
    assert spec.graph_nodes[-1] == "EVALUATE"
    assert len(spec.graph_nodes) <= 4
