# MANA test suite

Closes checklist item #1 ("no persisted test suite") -- everything here
was, until now, one-off scripts run in a chat session and thrown away.

## Quick start

```bash
pip install pytest
python -m pytest tests/ -m "not llm and not web" -v
```

This runs everything that needs no LLM, no network, and no microphone --
41 tests, pure computation + local SQLite + local sandboxed code exec.
Safe to run anywhere, repeatedly, in CI.

## What needs your real environment

| Flag | Requires | File |
|---|---|---|
| `MANA_TEST_LLM=1` | a reachable LLM backend (Ollama by default) | `test_llm_dependent.py` |
| `MANA_TEST_WEB=1` | real network access | `test_web_dependent.py` |
| `MANA_TEST_LLM_CODE_EVOLUTION=1` (needs `MANA_TEST_LLM=1` too) | LLM + writes to real `mana/agent_parts/` files, **always rolled back** | one test in `test_llm_dependent.py` |

```bash
MANA_TEST_LLM=1 python -m pytest tests/ -v
MANA_TEST_LLM=1 MANA_TEST_WEB=1 python -m pytest tests/ -v
```

Voice has no automated test -- see `scripts/voice_manual_test.py`, it needs
a human to actually speak and listen.

## The easy way: run everything at once

```bash
python scripts/run_diagnostics.py
```

Runs the offline suite + a hardware profile of your machine + an
at-scale memory benchmark, writes `mana_diagnostic_report.json` and
`mana_diagnostic_report.txt` next to this repo. Send both back.

## Isolation guarantee

Every test uses the `isolated_config`/`isolated_agent` fixtures in
`conftest.py`, which redirect all storage (SQLite DBs, knowledge base,
cache, exec sandbox) under pytest's per-test `tmp_path`. None of them
touch your real `mana_memory/` or any other state from normal use --
**except** the one `code_evolution`-related test explicitly called out
above, because `code_evolution.py`'s history/backup paths are hardcoded
relative to the installed package, not redirectable via Config (a known
limitation, see that module's docstring). That test rolls back
unconditionally in a `finally` block.
