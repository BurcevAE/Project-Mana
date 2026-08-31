"""
tests/conftest.py — shared fixtures for the MANA pytest suite.

Every test that constructs a Config or ManaAgent must use the
`isolated_config`/`isolated_agent` fixtures below, never Config()/ManaAgent()
directly with default paths -- those defaults point at mana_memory/ etc.
relative to the CURRENT WORKING DIRECTORY, which would pollute (or worse,
be polluted by) whatever real MANA state exists wherever `pytest` happens
to be invoked from. Everything here is redirected into pytest's per-test
tmp_path, so the suite is safe to run repeatedly, in parallel, and against
a real MANA installation without touching its actual data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the package importable when tests are run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mana.config import Config  # noqa: E402


@pytest.fixture
def isolated_config(tmp_path: Path) -> Config:
    """A Config with every filesystem path redirected under tmp_path, LLM
    and web disabled by default (tests that need them override explicitly),
    and hardware auto-adaptation left on (detect_hardware() is read-only
    and safe everywhere)."""
    root = tmp_path / "mana_state"
    cfg = Config(
        enable_llm=False,
        enable_web=False,
        knowledge_db_path=str(root / "knowledge.pkl"),
        state_file=str(root / "state.pkl"),
        history_file=str(root / "history.json"),
        cache_file=str(root / "cache.pkl"),
        experience_db_path=str(root / "experience.sqlite3"),
        evolution_report_file=str(root / "evolution_reports.json"),
        memory_root=str(root / "memory"),
        memory_db_path=str(root / "memory" / "memory.sqlite3"),
        local_exec_workdir=str(root / "exec_sandbox"),
    )
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def isolated_agent(isolated_config: Config):
    """A fully constructed ManaAgent on isolated storage, LLM/web off."""
    from mana import ManaAgent
    agent = ManaAgent(isolated_config)
    yield agent
    # NOTE: pytest's tmp_path cleanup is lazy (keeps recent dirs, prunes
    # older ones later), so an unclosed sqlite3 handle here didn't fail
    # tests outright on Windows -- but it's exactly the kind of thing that
    # bit scripts/run_diagnostics.py's benchmark (which deletes its temp
    # dir immediately). Closing explicitly avoids relying on that timing.
    try:
        agent.persistent_memory.close()
    except Exception:
        pass
    try:
        agent.experience.close()
    except Exception:
        pass


@pytest.fixture
def isolated_agent_exec_enabled(isolated_config: Config):
    """Same as isolated_agent, but with the local code-exec sandbox turned
    on -- needed for run_code / verify_answer / code_evolution tests."""
    isolated_config.local_exec_enabled = True
    from mana import ManaAgent
    agent = ManaAgent(isolated_config)
    yield agent
    try:
        agent.persistent_memory.close()
    except Exception:
        pass
    try:
        agent.experience.close()
    except Exception:
        pass
