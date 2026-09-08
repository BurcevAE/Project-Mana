"""Does a law change anything, and does the right kind of law change it?

The chain the roadmap asks for -- закон найден → следующий план →
поведение изменилось -- with the gate in the middle. A PROPOSED law is
licensed to point at the next experiment and no further; a SUPPORTED one
may change a decision; a REFUTED one changes nothing and is kept.

Every law here is built by hand with the evidence that earns its status,
because the point is to test the gate at each status rather than to test
whichever status today's book happens to hold.
"""
from __future__ import annotations

import pytest

from mana.cognition import acting
from mana.cognition.laws import (CognitiveLaw, Condition, LawBook, PROPOSED,
                                 REFUTED, SUPPORTED, VALIDATED)


def _law(book: LawBook, *, domain: str = "world_model",
         axis: str = "steps", was: str = "1000", now: str = "2500",
         effect: float = 0.4, trials: int = 40,
         hidden: bool = True, transfer: str = "",
         counterexample: bool = False) -> CognitiveLaw:
    law = book.propose(condition=Condition(domain=domain),
                       intervention=(f"{axis}: {was} -> {now}",),
                       claimed_effect=f"{axis} с {was} на {now} меняет исход",
                       discovered_in=domain,
                       # Which way the claim points, the way `lawgiver`
                       # sets it. Without it a law claiming harm is
                       # refuted by the evidence it was proposed on.
                       direction=1 if effect >= 0 else -1)
    law.record_evidence(effect=effect, trials=trials,
                        hidden_confirmed=hidden or None,
                        transfer_domain=transfer,
                        transfer_ok=True if transfer else None,
                        counterexamples_sought=1 if counterexample else 0,
                        counterexamples_found=1 if counterexample else 0,
                        experiment_id="e1")
    return law


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

def test_a_supported_law_changes_the_decision():
    book = LawBook()
    law = _law(book)
    assert law.status == SUPPORTED

    chosen = acting.setting_for("steps", 1000, "world_model", book)
    assert chosen.changed is True
    assert chosen.value == 2500
    assert chosen.status == SUPPORTED
    assert chosen.law_id == law.law_id


def test_a_proposed_law_changes_nothing():
    """It may point at the next experiment. That licence stops here."""
    book = LawBook()
    law = _law(book, hidden=False)
    assert law.status == PROPOSED

    chosen = acting.setting_for("steps", 1000, "world_model", book)
    assert chosen.changed is False
    assert chosen.value == 1000
    assert chosen.law_id == ""


def test_a_refuted_law_changes_nothing_and_is_kept():
    book = LawBook()
    law = _law(book, counterexample=True)
    assert law.status == REFUTED

    assert acting.setting_for("steps", 1000, "world_model", book).changed is False
    assert law in book.all()


def test_a_validated_law_changes_the_decision_too():
    book = LawBook()
    law = _law(book, transfer="other_world")
    law.record_evidence(counterexamples_sought=20, counterexamples_found=0,
                        experiment_id="hunt")
    assert law.status == VALIDATED
    assert acting.setting_for("steps", 1000, "world_model", book).value == 2500


def test_demotion_takes_the_behaviour_back():
    """A law that stops holding stops being acted on, without anybody
    deciding to check. Promotion and demotion are the same code path."""
    book = LawBook()
    law = _law(book)
    assert acting.setting_for("steps", 1000, "world_model", book).changed

    law.record_evidence(counterexamples_sought=5, counterexamples_found=1,
                        experiment_id="hunt")
    assert law.status == REFUTED
    assert not acting.setting_for("steps", 1000, "world_model", book).changed


# --------------------------------------------------------------------------
# scope
# --------------------------------------------------------------------------

def test_a_law_from_another_domain_says_nothing_here():
    """A claim measured on chess says nothing about how long to explore a
    world. Letting it speak is how a law gets applied where it was never
    measured."""
    book = LawBook()
    _law(book, domain="chess")
    assert acting.setting_for("steps", 1000, "world_model", book).changed is False


def test_a_law_about_another_axis_says_nothing_about_this_one():
    book = LawBook()
    _law(book, axis="episode_steps")
    assert acting.setting_for("steps", 1000, "world_model", book).changed is False


def test_an_empty_domain_matches_nothing_rather_than_everything():
    book = LawBook()
    _law(book, domain="")
    assert acting.setting_for("steps", 1000, "", book).changed is False
    assert acting.setting_for("steps", 1000, "world_model", book).changed is False


def test_no_book_is_the_default_and_not_an_error():
    chosen = acting.setting_for("steps", 1000, "world_model", None)
    assert chosen.value == 1000 and chosen.law_id == ""
    assert "по умолчанию" in chosen.describe()


# --------------------------------------------------------------------------
# what the law actually recommends
# --------------------------------------------------------------------------

def test_a_law_that_lowers_the_outcome_recommends_staying_put():
    """"Beyond here, more of this buys nothing" is the more useful half:
    it is a saving, and it is what diminishing returns produce."""
    book = LawBook()
    _law(book, was="1000", now="2500", effect=-0.3)
    chosen = acting.setting_for("steps", 2500, "world_model", book)
    assert chosen.value == 1000
    assert chosen.changed is True
    assert "опускает" in chosen.why


def test_the_value_comes_back_in_the_type_the_caller_uses():
    book = LawBook()
    _law(book, axis="episode_steps", was="25", now="50")
    chosen = acting.setting_for("episode_steps", 25, "world_model", book)
    assert chosen.value == 50 and isinstance(chosen.value, int)


def test_a_law_agreeing_with_the_default_is_not_a_change():
    book = LawBook()
    _law(book, was="400", now="1000")
    chosen = acting.setting_for("steps", 1000, "world_model", book)
    assert chosen.value == 1000
    assert chosen.changed is False
    # But it is recorded as having spoken, so a reader can tell "no law
    # applies" from "a law applies and agrees".
    assert chosen.law_id


# --------------------------------------------------------------------------
# what today's real book actually does
# --------------------------------------------------------------------------

def test_the_real_book_changes_nothing_yet_and_that_is_the_honest_state():
    """The only law on this machine is PROPOSED, because these
    experiments had no hidden holdout. So the path exists, is gated, and
    the gate is shut -- which is different from being connected to
    nothing, and is worth asserting rather than hoping."""
    from mana.cognition import lawgiver

    book = lawgiver.load_book()
    for law in book.all():
        assert law.status in (PROPOSED, REFUTED, SUPPORTED, VALIDATED)
    standing = acting.standing_laws(book, "world_model")
    assert all(law.status in acting.STANDING for law in standing)


def test_applicable_and_standing_answer_different_questions():
    """`applicable` is "does this claim cover this situation" and admits
    PROPOSED. Merging the two would let a claim with no hidden
    confirmation decide something."""
    book = LawBook()
    _law(book, hidden=False)
    assert book.applicable("world_model", 0.5)
    assert acting.standing_laws(book, "world_model") == []


# --------------------------------------------------------------------------
# where it was measured
# --------------------------------------------------------------------------

def test_a_law_measured_elsewhere_does_not_speak_here():
    """The world series found a plateau at episode length 25 and at no
    other. A law that dropped where it was measured would claim the whole
    domain on evidence from one corner of it."""
    book = LawBook()
    law = _law(book)
    law.scope = {"episode_steps": 25}

    here = acting.setting_for("steps", 1000, "world_model", book,
                              conditions={"episode_steps": 25})
    elsewhere = acting.setting_for("steps", 1000, "world_model", book,
                                   conditions={"episode_steps": 50})
    assert here.changed is True
    assert elsewhere.changed is False


def test_a_caller_that_says_nothing_is_not_out_of_scope():
    """Refusing there would make every law unusable by anyone who had not
    enumerated the whole world."""
    book = LawBook()
    law = _law(book)
    law.scope = {"episode_steps": 25}
    assert acting.setting_for("steps", 1000, "world_model", book).changed is True


def test_the_scope_survives_the_book_file(tmp_path):
    from mana.cognition import lawgiver

    book = LawBook()
    law = _law(book)
    law.scope = {"episode_steps": 25}
    path = tmp_path / "laws.json"
    assert lawgiver.save_book(book, path)
    restored = lawgiver.load_book(path)
    assert restored.all()[0].scope == {"episode_steps": 25}
