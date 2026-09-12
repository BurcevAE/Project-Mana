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


#: One directory for the whole session, and a counter for the names
#: inside it. `tmp_path_factory.mktemp` scans its parent for the next free
#: number on every call, which is fine once and quadratic when two
#: autouse fixtures do it for every test in a suite of eighteen hundred.
_ISOLATION_COUNTER = [0]


def _next_id() -> int:
    _ISOLATION_COUNTER[0] += 1
    return _ISOLATION_COUNTER[0]


def _isolation_dir(tmp_path_factory):
    root = getattr(_isolation_dir, "_root", None)
    if root is None:
        root = tmp_path_factory.mktemp("isolation", numbered=False)
        _isolation_dir._root = root
    return root


@pytest.fixture(autouse=True)
def _no_ambient_adoption(tmp_path_factory, monkeypatch):
    """No test reads or writes the machine's adopted policy.

    An adoption is now data on disk rather than an edit to a default, so
    a suite that read it would pass or fail depending on what this
    installation had adopted -- and one that wrote it would change the
    user's program from a test run. The same lesson as the findings
    ledger, which once had 46 real episodes written into the repository
    root before its path answered to Config.
    """
    from mana import policy as policy_mod

    from mana.cognition import rules as rules_mod

    # One directory and a name, not a directory per test: `mktemp` scans
    # its parent for the next free number, so with a fixture on every test
    # the scan grows with the suite. Same fix as the chess fixtures below,
    # and the same reason the whole suite crawled while every file alone
    # was fast.
    root = _isolation_dir(tmp_path_factory)
    mark = _next_id()
    monkeypatch.setattr(policy_mod, "_overlay_path",
                        lambda: root / f"adopted-{mark}.json")
    monkeypatch.setattr(policy_mod, "_adopted_cache", None, raising=False)
    # Rules MANA wrote are the same kind of state as a setting it adopted,
    # and a test reading them would pass or fail by what this machine had
    # installed. Found that way: a knob test came back clean because a
    # rule installed by an earlier run was still deciding.
    monkeypatch.setattr(rules_mod, "_path", lambda: root / f"rules-{mark}.json")
    rules_mod._reset_for_tests()
    yield
    monkeypatch.setattr(policy_mod, "_adopted_cache", None, raising=False)
    rules_mod._reset_for_tests()


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
        journal_path=str(root / "journal" / "episodes.jsonl"),
        findings_path=str(root / "findings" / "findings.jsonl"),
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


@pytest.fixture(autouse=True)
def _no_ambient_game_record(tmp_path_factory, monkeypatch):
    """Point the chess record at a temporary file for every test.

    The same reason the policy overlay and the installed rules are
    isolated: a test that forgets writes to the machine's real state. It
    was not hypothetical -- a full re-judge rewrote the real record during
    a suite run and could not afterwards be attributed to a test, which
    is worse than a wrong write, because a wrong write can at least be
    found.
    """
    from mana.cognition import chess_bench, chess_bot

    # Not the test's own `tmp_path`: a fixture that adds a folder there
    # changes what every test listing that directory sees, and one of
    # them counts the files it wrote. And not a fresh `mktemp` either --
    # that scans its parent for the next free number, so the scan grows
    # with the number of tests and cost eight seconds of setup each by
    # the end of a suite.
    # File names, not directories. Creating a folder for every test made
    # nearly four thousand of them in one place, and NTFS degrades as that
    # grows -- which is why the files run fast one at a time and the whole
    # suite crawled. Nothing is created unless a test actually writes.
    root = _isolation_dir(tmp_path_factory)
    mark = _next_id()
    monkeypatch.setattr(chess_bot, "games_path",
                        lambda: root / f"games-{mark}.jsonl")
    monkeypatch.setattr(chess_bench, "state_path",
                        lambda: root / f"bench-{mark}.json")


@pytest.fixture(autouse=True)
def _no_ambient_findings(tmp_path_factory, monkeypatch):
    """Point the findings ledger at a temporary file for every test.

    The game record was isolated after a rewrite nobody could attribute;
    this is the same hole in the neighbouring file. A `Ledger()` with no
    path defaults to the machine's real ledger, so a test that forgets
    both writes to it and reads all of it -- which is also how one test
    came to take four hundred and fifty-nine seconds.
    """
    from mana.cognition import findings

    root = _isolation_dir(tmp_path_factory)
    mark = _next_id()
    monkeypatch.setattr(findings, "ledger_path",
                        lambda: root / f"findings-{mark}.jsonl")


@pytest.fixture(autouse=True)
def _no_ambient_transactions(tmp_path_factory, monkeypatch):
    """No test writes the machine's self-modification journal.

    `core/transaction.py` writes under the data root, which during tests
    was the repository itself: a brain-retirement test left three lines in
    `experiments/transactions.jsonl` on every run. Every adoption now opens
    a transaction, so every test that adopts would have done the same.
    """
    from mana.core import transaction

    root = _isolation_dir(tmp_path_factory)
    mark = _next_id()
    monkeypatch.setattr(transaction, "journal_path",
                        lambda: root / f"transactions-{mark}.jsonl")


@pytest.fixture(autouse=True)
def _no_ambient_player_version(tmp_path_factory, monkeypatch):
    """No test reads or writes the machine's adopted player.

    An adoption is data on disk, so a suite that read it would pass or
    fail by what this installation had adopted, and one that wrote it
    would change the user's player from a test run. The same lesson as
    the policy overlay, the installed rules, the game record and the
    findings ledger -- four times now.
    """
    from mana.cognition import chess_version

    root = _isolation_dir(tmp_path_factory)
    mark = _next_id()
    monkeypatch.setattr(chess_version, "path",
                        lambda: root / f"player-{mark}.json")
    chess_version._reset_for_tests()
    yield
    chess_version._reset_for_tests()
