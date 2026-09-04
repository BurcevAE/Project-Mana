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


#: Every environment variable a brain reads its key from. Kept as an
#: explicit list rather than "anything ending in _API_KEY", so a new
#: provider added without a line here shows up as a test that behaves
#: differently on a developer machine than in CI -- which is exactly the
#: failure this exists to stop, and better caught than silently absorbed.
_PROVIDER_KEY_VARS = (
    "GROQ_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY", "DEEPSEEK_API_KEY",
    "MISTRAL_API_KEY", "GITHUB_TOKEN", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
    "CLOUDFLARE_API_TOKEN", "MANA_CLOUDFLARE_ACCOUNT_ID",
)


@pytest.fixture(autouse=True)
def _no_ambient_api_keys(monkeypatch, request):
    """Hide provider keys from every test that has not asked for them.

    Found by a live run on a real machine: with GROQ_API_KEY exported,
    four tests that assert offline behaviour started talking to cloud
    APIs and failed. Under a bare `pytest` they passed. A suite whose
    result depends on whether a shell happened to export a key cannot
    tell a regression from an environment difference -- and this project's whole
    argument rests on measurements that do not move with ambient state.

    `enable_llm=False` does NOT cover this: it means "no local backend",
    and a remote brain with a key stays usable on purpose, so that a
    machine with a free key and no Ollama still works. The two readings
    of that name are what let the assumption through.

    Tests that genuinely need a live provider mark themselves `llm`;
    they keep the real environment.
    """
    if request.node.get_closest_marker("llm"):
        return
    for name in _PROVIDER_KEY_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _restore_search_weights():
    """Undo any change a test makes to the tuneable search weights.

    Since phase 16 an accepted meta-change writes into the module dicts
    the search actually reads -- which is the point, and which makes them
    global mutable state that leaks from one test into the next. The same
    class of problem as ambient API keys: a suite whose result depends on
    what ran before it cannot tell a regression from an ordering.
    """
    from mana.cognition import experiments, gaps, novelty
    tables = (gaps.PRIORITY_WEIGHTS, experiments.VALUE_WEIGHTS,
              novelty.CHANNEL_WEIGHTS)
    saved = [dict(t) for t in tables]
    yield
    for table, original in zip(tables, saved):
        table.clear()
        table.update(original)


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
