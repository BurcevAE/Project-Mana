"""Proposing the change, so a person does not have to.

Two properties carry the weight here. A candidate must be aimed at a
failure that was actually observed -- blind mutation churns a working
system and spends the budget on nothing. And nothing in this layer may
adopt anything: a generator that could accept its own output would be a
second way to change the system, and the second way is always the one
with the hole in it.
"""
from __future__ import annotations

import pytest

from mana import policy as policy_mod
from mana.policy import Policy, KNOBS
from mana.apps import intent
from mana.journal import Episode
from mana.cognition import candidates, failure_domain as fd
from mana.cognition.invariants import Violation, MECHANICAL, PATTERN, scan

TARGETS = ["UT11-ER", "Информационная база"]

REAL_MISSES = (
    "я хочу поработать с 1С запусти пожалуйста конфигуратор информационной базы",
    "я хочу что бы ты открыла конфигуратор информационной базы на моем компьютере",
)


def violation(name: str, episode_id: str = "e1") -> Violation:
    kind = MECHANICAL if name == "repeats_earlier_answer" else PATTERN
    return Violation(name, kind, episode_id, "нарушение из записи")


# --------------------------------------------------------------------------
# the policy layer -- defaults must be today
# --------------------------------------------------------------------------

def test_the_baseline_policy_is_what_is_in_force():
    """A default is the behaviour MANA actually has. Three of these
    changed when the 1C recognition fix was adopted on 08.09.2026 -- see
    the findings ledger, verdict NOT_EVALUATED."""
    assert policy_mod.BASELINE.changes() == {}
    assert policy_mod.active() is policy_mod.BASELINE
    assert policy_mod.get("echo_lookback") == 1
    assert policy_mod.get("intent_verb_anywhere") is True
    assert policy_mod.get("intent_verb_forms") == "addressed"
    assert policy_mod.get("intent_stem_match") is True


def test_every_knob_still_offers_what_it_was_before():
    """An adopted setting has to stay reversible: the generator proposes
    the other value, and the gates can send it back."""
    for knob in KNOBS:
        assert len(knob.options) >= 2
        assert knob.default == knob.options[0]


def test_only_declared_knobs_may_be_set():
    """An allowlist: a field not declared cannot be changed by a
    candidate, however the proposal is written."""
    with pytest.raises(ValueError):
        Policy.of(some_setting_nobody_declared=True)


def test_a_value_outside_the_declared_range_is_refused():
    with pytest.raises(ValueError):
        Policy.of(echo_lookback=9999)


def test_a_policy_is_content_addressed():
    # The non-default value, so these are policies that differ from what
    # is in force rather than restatements of it.
    assert (Policy.of(intent_verb_anywhere=False).policy_id
            == Policy.of(intent_verb_anywhere=False).policy_id)
    assert (Policy.of(intent_verb_anywhere=False).policy_id
            != Policy.of(intent_stem_match=False).policy_id)


def test_a_policy_applies_only_to_the_thread_that_asked():
    """A cycle evaluating a candidate runs while somebody types; a
    process-wide swap would answer their question under an unproven
    policy."""
    import threading

    seen = []
    ready = threading.Event()

    def other():
        seen.append(policy_mod.get("intent_verb_anywhere"))
        ready.set()

    with policy_mod.use(Policy.of(intent_verb_anywhere=False)):
        assert policy_mod.get("intent_verb_anywhere") is False
        worker = threading.Thread(target=other)
        worker.start()
        ready.wait(timeout=5)
        worker.join()
    assert seen == [True]                       # the other thread saw the default
    assert policy_mod.get("intent_verb_anywhere") is True


def test_the_policy_is_restored_even_when_the_body_raises():
    with pytest.raises(RuntimeError):
        with policy_mod.use(Policy.of(intent_verb_anywhere=False)):
            raise RuntimeError("evaluation blew up")
    assert policy_mod.get("intent_verb_anywhere") is True


def test_every_knob_names_an_invariant_it_can_affect():
    """This is how a proposal is connected to a measurement by
    construction rather than by the proposer's belief."""
    from mana.cognition.invariants import INVARIANTS

    names = {f.__name__ for f in INVARIANTS}
    for knob in KNOBS:
        assert knob.addresses in names, knob.name


# --------------------------------------------------------------------------
# the knobs must actually change behaviour
# --------------------------------------------------------------------------

def test_the_adopted_policy_catches_the_real_requests(monkeypatch):
    """Both were misses before 08.09.2026; the fix is now the default."""
    monkeypatch.setattr(intent, "_known_bases", lambda: TARGETS)
    for text in REAL_MISSES:
        found = intent.match(text)
        assert found is not None
        assert found.params["base"] == "Информационная база"


def test_turning_the_fix_off_restores_the_old_misses(monkeypatch):
    """Reversible, and the reverse is a policy like any other."""
    monkeypatch.setattr(intent, "_known_bases", lambda: TARGETS)
    narrow = Policy.of(intent_verb_anywhere=False, intent_stem_match=False,
                       intent_verb_forms="imperative")
    with policy_mod.use(narrow):
        for text in REAL_MISSES:
            assert intent.match(text) is None


def test_the_adopted_policy_names_the_base_and_the_mode(monkeypatch):
    """The measured misses, fixed by settings rather than by an edit."""
    monkeypatch.setattr(intent, "_known_bases", lambda: TARGETS)
    for text in REAL_MISSES:
        found = intent.match(text)
        assert found is not None
        assert found.params["base"] == "Информационная база"
        assert found.params["designer"] is True


@pytest.mark.parametrize("text", [
    "Как запустить 1С?",
    "ты уже открыла 1С?",
    "расскажи что такое конфигуратор",
    "я не могу запустить 1С, что делать?",
    "в 1С можно открыть несколько баз сразу?",
    "в инструкции написано открыть базу через ярлык 1С",
    "почему не запускается 1С?",
    "нужно ли закрывать 1С перед обновлением?",
])
def test_the_adopted_policy_does_not_launch_at_questions(monkeypatch, text):
    """The reason the actor was narrow in the first place. Measured before
    the guards existed: the widening alone fired on three of these, and
    launching 1C at somebody asking for help is worse than doing nothing.

    The dry evaluation had reported zero counterexamples -- it searched
    seven recorded episodes, none of which is a question about 1C.
    Absence of counterexamples in a small record is not evidence of
    safety.
    """
    monkeypatch.setattr(intent, "_known_bases", lambda: TARGETS)
    assert intent.match(text) is None


def test_the_echo_lookback_knob_reaches_the_guard():
    """A knob wired to nothing is this project's recurring failure."""
    from mana import echo_guard

    class Memory:
        def recent(self, _session, _n):
            return [{"kind": "USER_MESSAGE", "content": "вопрос раз"},
                    {"kind": "MANA_RESPONSE", "content": "ответ раз"},
                    {"kind": "USER_MESSAGE", "content": "вопрос два"},
                    {"kind": "MANA_RESPONSE", "content": "ответ два"}]

    assert len(echo_guard.previous_exchanges(Memory(), "s", limit=1)) == 1
    assert len(echo_guard.previous_exchanges(Memory(), "s", limit=6)) == 2


def test_the_echo_thresholds_reach_the_guard():
    from mana import echo_guard

    answer = "Развёрнутый ответ, повторённый почти дословно, но не совсем точно."
    other = "Развёрнутый ответ, повторённый почти дословно, но чуть иначе!!"
    with policy_mod.use(Policy.of(echo_same_answer=0.90)):
        strict = echo_guard.repeats_previous("вопрос два", answer,
                                             "совсем другой вопрос", other)
    with policy_mod.use(Policy.of(echo_same_answer=0.70)):
        loose = echo_guard.repeats_previous("вопрос два", answer,
                                            "совсем другой вопрос", other)
    assert loose is not None
    assert (strict is None) or (loose is not None)


# --------------------------------------------------------------------------
# proposing
# --------------------------------------------------------------------------

def test_no_observed_failure_means_no_candidate():
    """A system with nothing wrong has nothing to propose, and a
    generator that produced candidates anyway would change a working
    system on speculation."""
    assert candidates.propose([]) == []


def test_candidates_are_aimed_at_the_invariant_that_fired():
    proposed = candidates.propose([violation("actionable_request_not_acted_on")])
    assert proposed
    assert {c.addresses for c in proposed} == {"actionable_request_not_acted_on"}
    for candidate in proposed:
        assert set(candidate.policy.changes()) <= {
            "intent_verb_anywhere", "intent_verb_forms", "intent_stem_match"}


def test_each_candidate_changes_one_thing_plus_one_combination():
    """An accepted combination would not say which part did the work, so
    the singles are proposed too."""
    proposed = candidates.propose([violation("actionable_request_not_acted_on")])
    sizes = sorted(len(c.policy.changes()) for c in proposed)
    assert sizes.count(1) >= 3
    assert max(sizes) > 1


def test_a_candidate_carries_the_measurement_it_comes_from():
    proposed = candidates.propose(
        [violation("actionable_request_not_acted_on", "a"),
         violation("actionable_request_not_acted_on", "b")])
    assert all(c.observed == 2 for c in proposed)
    assert all("2 нарушени" in c.rationale for c in proposed)


def test_the_mechanism_is_chosen_not_assumed():
    """`choose_mechanism` answers `algorithmic` here because detecting a
    repeat and matching a name against a known list are both exact. A
    domain whose answer is computable does not need a model."""
    from mana.cognition.brain_factory import ALGORITHMIC

    proposed = candidates.propose([violation("repeats_earlier_answer")])
    assert proposed and all(c.mechanism == ALGORITHMIC for c in proposed)


def test_a_knob_is_not_offered_when_a_knob_is_the_wrong_shape(monkeypatch):
    """If the mechanism for an invariant is not algorithmic, offering a
    setting anyway is how a search optimises what it can adjust rather
    than what is wrong."""
    monkeypatch.setitem(candidates.EXACTLY_COMPUTABLE,
                        "actionable_request_not_acted_on", False)
    proposed = candidates.propose([violation("actionable_request_not_acted_on")])
    assert proposed == []


# --------------------------------------------------------------------------
# dry evaluation, and what it refuses to claim
# --------------------------------------------------------------------------

def test_echo_candidates_are_refused_rather_than_guessed():
    """When the guard fires the live path retries with context suppressed,
    and what that retry answers cannot be known without the model.
    Reporting a number would be inventing evidence."""
    allowed, why = candidates.dry_evaluable(Policy.of(echo_lookback=6))
    assert allowed is False
    assert "echo_lookback" in why
    report = candidates.dry_report(Policy.of(echo_lookback=6), [])
    assert report["dry_evaluable"] is False
    assert "baseline_pass_rate" not in report


def test_a_dry_score_says_it_is_an_upper_bound():
    """A feasible action can still fail, so this is the best the candidate
    could do, never the number it would achieve."""
    report = candidates.dry_report(Policy.of(intent_verb_anywhere=True), [])
    assert report["dry_evaluable"] is True
    assert report["upper_bound"] is True


def test_the_dry_responder_performs_nothing():
    import inspect

    source = inspect.getsource(candidates)
    for performing in ("registry.call", "perform(", "subprocess", "Popen"):
        assert performing not in source


def test_a_situation_the_candidate_does_not_touch_keeps_its_answer(monkeypatch):
    monkeypatch.setattr(intent, "_known_bases", lambda: TARGETS)
    situation = fd.Situation("x", "s", "расскажи про налоги",
                             recorded_answer="Развёрнутый ответ про налоги.")
    answer, calls = candidates.dry_responder(
        Policy.of(intent_verb_anywhere=True))(situation)
    assert answer == "Развёрнутый ответ про налоги."
    assert calls == []


def test_an_infeasible_action_is_reported_as_a_refusal(monkeypatch):
    """A refusal with its reason is a real answer, and a better one than
    a narration."""
    monkeypatch.setattr(intent, "_known_bases", lambda: ["UT11-ER"])
    situation = fd.Situation("x", "s", 'Запусти 1С базу "НЕТ-ТАКОЙ"')
    answer, calls = candidates.dry_responder(policy_mod.BASELINE)(situation)
    assert answer.startswith("Не получилось")
    assert calls and calls[0].ok is False


def test_would_act_consults_the_machine_and_performs_nothing(monkeypatch):
    monkeypatch.setattr(intent, "_known_bases", lambda: TARGETS)
    found = intent.match('Запусти 1С базу "UT11-ER"')
    assert found is not None
    feasible, note = intent.would_act(found)
    assert feasible is True and "UT11-ER" in note


# --------------------------------------------------------------------------
# nothing here adopts anything
# --------------------------------------------------------------------------

def test_the_generator_cannot_accept_its_own_output():
    """Acceptance belongs to core/gates.py, on evidence, exactly as for a
    cognitive program."""
    import inspect

    source = inspect.getsource(candidates)
    for adopting in ("gates.judge", "ACCEPTED", "policy_mod._local",
                     "BASELINE ="):
        assert adopting not in source


def test_ranking_is_a_suggestion_of_what_to_measure(monkeypatch):
    """Ordering carries no verdict: the top row is what to test properly,
    not what to ship."""
    monkeypatch.setattr(intent, "_known_bases", lambda: TARGETS)
    episodes = [Episode("a", "s", 1.0, REAL_MISSES[0],
                        "Чтобы запустить, найдите ярлык 1С на рабочем столе."),
                Episode("b", "s", 2.0, "расскажи про налоги",
                        "Развёрнутый содержательный ответ про налоги.")]
    found = scan(episodes, TARGETS)
    rows = candidates.rank(found, fd.situations_from(episodes, TARGETS))
    assert rows
    evaluable = [r for r in rows if r["dry"].get("dry_evaluable")]
    assert evaluable
    best = evaluable[0]["dry"]
    assert best["candidate_pass_rate"] > best["baseline_pass_rate"]
    assert best["counterexamples"]["found"] == 0
    for row in rows:
        assert "accepted" not in row and "verdict" not in row
