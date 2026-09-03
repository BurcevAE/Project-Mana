"""
tests/test_versioning.py — version numbers must have exactly one source.

Before mana/version.py existed, the release number was hardcoded in at
least five places and had already drifted into THREE simultaneous values:
"5.4.0" in mana/__init__, "5.4" in CoreMixin and MemoryManager, "5.3.1" in
the CLI description and the interactive banner. Reports, logs and the
banner disagreed with each other, so no one could say which build they
were looking at.

These tests fail if a hardcoded version reappears anywhere.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from mana.version import (PRODUCT_VERSION, VERSIONED_MODULES,
                          component_versions, format_version_report, version_report)

REPO = pathlib.Path(__file__).resolve().parent.parent


def _source_files():
    return list((REPO / "mana").glob("*.py")) + list((REPO / "mana" / "agent_parts").glob("*.py"))


def test_product_version_is_declared_once():
    """Pinned deliberately: the product version must be bumped as part of
    a change, not drift. Update this line in the same commit that changes
    PRODUCT_VERSION -- the failure is the reminder."""
    assert PRODUCT_VERSION == "5.19.0"


def test_every_listed_module_declares_a_version():
    """Every module must carry a parseable version. The value is NOT
    pinned to 1.0 any more -- modules version independently and are bumped
    when they change (knowledge 1.1 and context 1.1 in 5.7.9, for
    example). What is enforced is that a version exists and is well
    formed, so a module added without one shows up here."""
    import re
    versions = component_versions()
    assert set(versions) == set(VERSIONED_MODULES)
    for name, ver in versions.items():
        assert re.fullmatch(r"\d+\.\d+(\.\d+)?", ver), (
            f"{name} reports {ver!r}, which is not a version number")


def test_modules_changed_in_this_release_were_bumped():
    """Guards the discipline itself: these modules gained new behaviour in
    5.7.9, so they must no longer report the 1.0 baseline. If a future
    change touches them again without a bump, this is where it shows."""
    versions = component_versions()
    for name in ("knowledge", "agent_parts.context", "config", "graph_memory", "intent",
                 "agent_parts.routing", "llm", "tools", "pipeline", "cli",
                 "agent_parts.core", "agent_parts.execution", "brains",
                 "agent_parts.evolution", "verifier",
                 "code_evolution"):
        assert versions[name] != "1.0", (
            f"{name} changed in this release but still reports 1.0 -- bump it")


def test_new_modules_are_registered_for_reporting():
    """A module that exists but is missing from VERSIONED_MODULES would
    silently vanish from --version output."""
    assert "intent" in VERSIONED_MODULES


def test_package_version_matches_product_version():
    import mana
    assert mana.__version__ == PRODUCT_VERSION


def test_agent_and_memory_report_the_same_version(isolated_agent):
    assert isolated_agent.VERSION == PRODUCT_VERSION
    assert isolated_agent.persistent_memory.VERSION == PRODUCT_VERSION


def test_no_stale_hardcoded_versions_remain():
    """The specific drift that motivated this module.

    Comments and docstrings are excluded: mana/version.py documents the
    old values on purpose, and forbidding that would push the explanation
    out of the code. Only executable lines are checked.
    """
    import ast
    stale = re.compile(r"^5\.(3\.1|4(\.0)?)$")
    offenders = []
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    docstrings.add(doc)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in docstrings:
                    continue
                if stale.match(node.value.strip()):
                    offenders.append(f"{path.relative_to(REPO)}:{node.lineno}: {node.value!r}")
    assert not offenders, "hardcoded version strings found:\n" + "\n".join(offenders)


def test_no_module_hardcodes_the_product_version_itself():
    """Modules must import PRODUCT_VERSION, not repeat the literal."""
    literal = f'"{PRODUCT_VERSION}"'
    offenders = []
    for path in _source_files():
        if path.name == "version.py":
            continue              # the one place it is allowed to live
        if literal in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, f"product version literal duplicated in: {offenders}"


def test_version_report_shapes():
    report = version_report()
    assert report["product"] == PRODUCT_VERSION
    assert isinstance(report["components"], dict) and report["components"]
    text = format_version_report()
    assert PRODUCT_VERSION in text and "agent_parts.core" in text


def test_component_versions_never_raise_on_a_bad_module(monkeypatch):
    """Version reporting must not be the thing that breaks a run."""
    import importlib

    from mana import version as version_mod

    def boom(name):
        raise ImportError("simulated")

    monkeypatch.setattr(importlib, "import_module", boom)
    result = version_mod.component_versions()
    assert all("unavailable" in v for v in result.values())
