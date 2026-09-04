"""
tests/test_desktop_session.py — the window's side of the app.

Two properties matter here and neither is about the UI looking right:

  * the session must never let a caller wedge the agent (it is not
    re-entrant: one SQLite connection, one pipeline, one session id);
  * the provenance row must report what actually happened, including the
    unflattering cases -- a fallback answer, a critic that was not
    independent, brains that disagreed. A status line that only ever shows
    good news is worse than no status line, because it teaches the user to
    trust it.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from mana_desktop.session import AgentSession


class _StubAgent:
    """Stands in for ManaAgent. The session must not know the difference --
    it only ever calls the public API."""

    def __init__(self, answer="ответ", trace=None, delay=0.0, raises=None):
        self.answer = answer
        self.trace = trace or {}
        self.delay = delay
        self.raises = raises
        self.calls = 0
        self.session_id = "default"

    def solve_task(self, task):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.raises:
            raise self.raises
        return {"answer": self.answer, "trace": dict(self.trace), "fallback": False}


def make_session(agent):
    session = AgentSession.__new__(AgentSession)
    session._config = None
    session._agent = agent
    session._agent_error = None
    session._lock = threading.RLock()
    session._busy = threading.Event()
    session._cancel = threading.Event()
    session._current_id = 0
    session._ready = threading.Event()
    session._ready.set()
    return session


# ---------------------------------------------------------------------------
# asking
# ---------------------------------------------------------------------------

def test_answer_carries_provenance_not_just_text():
    session = make_session(_StubAgent(answer="391", trace={
        "brain": "groq", "brain_attempts": ["gemini", "groq"],
        "brain_strategy": "single", "verification_kind": "arithmetic",
        "verification_used": True, "critic_brain": "gemini",
        "critic_independent": True}))
    result = session.ask("17*23?")
    p = result["provenance"]
    assert result["answer"] == "391"
    assert p["brain"] == "groq"
    assert p["attempts"] == ["gemini", "groq"]
    assert p["verified"] is True
    assert p["critic_independent"] is True


def test_provenance_reports_the_unflattering_cases_too():
    """Disagreement and a non-independent critic are exactly the facts a
    user needs and the ones a status line is tempted to hide."""
    session = make_session(_StubAgent(trace={
        "brain": "ollama", "critic_brain": "ollama", "critic_independent": False,
        "consensus": {"agreement": 0.21, "disagreement": True, "brains": ["ollama", "groq"]}}))
    p = session.ask("вопрос")["provenance"]
    assert p["disagreement"] is True
    assert p["agreement"] == pytest.approx(0.21)
    assert p["critic_independent"] is False


def test_a_second_question_is_refused_while_one_is_running():
    """The agent is not re-entrant. Two concurrent solve_task calls share a
    SQLite connection and one pipeline object -- the failure would be
    corrupted state, not an exception."""
    agent = _StubAgent(delay=0.4)
    session = make_session(agent)
    results = []
    t = threading.Thread(target=lambda: results.append(session.ask("первый")))
    t.start()
    time.sleep(0.1)
    second = session.ask("второй")
    t.join()
    assert second["ok"] is False
    assert "предыдущ" in second["error"]
    assert agent.calls == 1


def test_an_exception_in_the_agent_becomes_a_reportable_result():
    """Every caller is an HTTP handler whose job is to show the failure,
    not to propagate it."""
    session = make_session(_StubAgent(raises=RuntimeError("модель упала")))
    result = session.ask("вопрос")
    assert result["ok"] is False
    assert "модель упала" in result["error"]
    assert session.state()["busy"] is False, "busy must clear even on failure"


def test_cancel_says_plainly_that_the_work_was_not_stopped():
    """solve_task has no interruption point. Naming this honestly is the
    difference between a Stop button that lies and one that does not."""
    agent = _StubAgent(delay=0.4)
    session = make_session(agent)
    holder = {}
    t = threading.Thread(target=lambda: holder.update(session.ask("вопрос")))
    t.start()
    time.sleep(0.1)
    cancelled = session.cancel()
    t.join()
    assert cancelled["ok"] is True
    assert "продолжает выполняться" in cancelled["note"]
    assert holder["cancelled"] is True
    assert agent.calls == 1, "the call still ran -- only its result was dropped"


def test_cancel_with_nothing_running_is_not_an_error_state():
    session = make_session(_StubAgent())
    assert session.cancel()["ok"] is False


def test_asking_before_the_agent_is_built_is_refused_cleanly():
    session = make_session(_StubAgent())
    session._ready.clear()
    result = session.ask("рано")
    assert result["ok"] is False
    assert "запускается" in result["error"]


def test_a_failed_startup_is_reported_not_swallowed():
    session = make_session(None)
    session._agent_error = "ImportError: numpy"
    assert session.ask("вопрос")["error"] == "ImportError: numpy"
    assert session.state()["ready"] is False


# ---------------------------------------------------------------------------
# the server surface
# ---------------------------------------------------------------------------

def test_json_encoder_survives_objects_the_agent_puts_in_traces():
    """Traces carry dataclasses, tuples and numpy scalars. A serialiser
    that raises here would turn a good answer into a 500."""
    from dataclasses import dataclass
    from mana_desktop.server import _json_safe

    @dataclass
    class Meta:
        brain: str
        latency: float

    payload = {"meta": Meta("groq", 1.5), "attempts": ("a", "b"), "set": {1},
               "obj": object(), "nested": {"none": None, "ok": True}}
    encoded = json.dumps(_json_safe(payload), ensure_ascii=False)
    assert '"brain": "groq"' in encoded
    assert '"attempts": ["a", "b"]' in encoded


def test_saving_an_empty_env_var_name_is_rejected():
    from mana_desktop.server import save_api_key
    assert save_api_key("", "секрет")["ok"] is False


def test_a_saved_key_reaches_the_environment_for_this_process(monkeypatch):
    """BrainPool reads api_key_env when it is constructed, so a key that
    only lands in the credential store would not exist for the pool that
    is about to be built."""
    import os
    import sys
    import types

    stub = types.SimpleNamespace(
        set_password=lambda *a: None,
        get_password=lambda *a: None,
        delete_password=lambda *a: None,
    )
    monkeypatch.setitem(sys.modules, "keyring", stub)
    monkeypatch.delenv("TEST_BRAIN_KEY", raising=False)

    from mana_desktop.server import save_api_key
    result = save_api_key("TEST_BRAIN_KEY", "sk-test")
    assert result["ok"] is True
    assert os.environ["TEST_BRAIN_KEY"] == "sk-test"


def test_key_storage_failure_is_reported_rather_than_raised(monkeypatch):
    import sys
    import types

    def boom(*_a, **_k):
        raise RuntimeError("credential store locked")

    monkeypatch.setitem(sys.modules, "keyring", types.SimpleNamespace(set_password=boom))
    from mana_desktop.server import save_api_key
    result = save_api_key("SOME_KEY", "value")
    assert result["ok"] is False
    assert "credential store locked" in result["error"]


# ---------------------------------------------------------------------------
# the window as an instrument
# ---------------------------------------------------------------------------

def test_an_answer_reports_what_it_cost_not_the_running_total(isolated_config):
    """A running total says nothing about the question just asked, and
    the substrate mix is the interesting part: an answer from an
    algorithmic brain cost no tokens at all, which a call count hides."""
    from mana.core.cost import CostVector
    from mana_desktop.session import AgentSession

    class FakePool:
        def __init__(self):
            self.total = CostVector()

        def total_cost(self):
            return self.total

    session = AgentSession.__new__(AgentSession)
    session._cost_mark = None
    pool = FakePool()
    session._pool = lambda: pool

    pool.total = CostVector(calls=10, tokens_in=500, by_substrate={"remote_llm": 10})
    first = session._cost_since_mark()
    assert first["calls"] == 10

    pool.total = CostVector(calls=13, tokens_in=500,
                            by_substrate={"remote_llm": 10, "algorithmic": 3})
    second = session._cost_since_mark()
    assert second["calls"] == 3, "the second answer must not be charged for the first"
    assert second["tokens_in"] == 0
    assert second["by_substrate"] == {"algorithmic": 3}


def test_a_cycle_does_not_charge_its_cost_to_the_next_answer(isolated_config):
    """Otherwise someone asks one question after a cycle and sees a
    hundred calls under it, spent on something else."""
    import inspect
    from mana_desktop.session import AgentSession
    source = inspect.getsource(AgentSession._run_cycle)
    assert "self._cost_mark = spent" in source


def test_stopping_a_cycle_promises_only_what_it_can_do(isolated_config):
    """A step is a batch of brain calls with no interruption point, so
    the current one finishes and is paid for. Promising otherwise would
    be the lie `cancel` already refuses to tell."""
    import inspect
    from mana_desktop.session import AgentSession
    doc = inspect.getdoc(AgentSession.stop_cycle) or ""
    assert "after the step it is on" in doc


def test_the_genome_view_reads_rather_than_builds(isolated_config, tmp_path):
    """A window that constructed a fresh genome to display would show a
    baseline and call it the system's state."""
    from mana_desktop.session import AgentSession
    session = AgentSession.__new__(AgentSession)
    view = session.genome(str(tmp_path / "absent.json"))
    assert view["present"] is False
    assert "нет сохранённого" in view["note"]
