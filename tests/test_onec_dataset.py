"""
tests/test_onec_dataset.py — the loop that manufactures verified examples.

Nothing here touches 1C. The teacher and the executor are both passed in
or patched, because what needs testing is the bookkeeping and the
read-only rule -- and the one infobase on this machine holds real trade
records.

The read-only tests carry the most weight. Generated text goes into a live
database, and this check is the only thing between a model's output and
that database.
"""
from __future__ import annotations

import pytest

from mana.apps import onec, onec_dataset as dataset


# ------------------------------------------------------------- read-only

def test_a_plain_select_passes():
    dataset.check_read_only("ВЫБРАТЬ Наименование ИЗ Справочник.Номенклатура")
    dataset.check_read_only("SELECT 1")


def test_a_batch_with_a_temporary_table_passes():
    """This is how a real 1C query batch cleans up after itself, so
    refusing it would refuse most non-trivial correct queries."""
    dataset.check_read_only(
        "ВЫБРАТЬ 1 ПОМЕСТИТЬ ВТ; ВЫБРАТЬ * ИЗ ВТ; УНИЧТОЖИТЬ ВТ")


@pytest.mark.parametrize("text,why", [
    ("", "пустой"),
    ("ВЫБРАТЬ 1; УДАЛИТЬ ИЗ Справочник.Номенклатура", "запись во втором операторе"),
    ('ВЫПОЛНИТЬ("что угодно")', "исполнение"),
    ("ОБНОВИТЬ Справочник.Номенклатура", "запись"),
])
def test_anything_that_is_not_a_read_is_refused(text, why):
    with pytest.raises(dataset.NotReadOnly):
        dataset.check_read_only(text)


def test_a_write_hidden_behind_a_comment_is_still_refused():
    """The case a naive "does it start with ВЫБРАТЬ" check would pass.

    Comments are stripped before the statements are read, so the leading
    /* ВЫБРАТЬ */ buys nothing.
    """
    with pytest.raises(dataset.NotReadOnly):
        dataset.check_read_only("/* ВЫБРАТЬ */ ОБНОВИТЬ Справочник.Номенклатура")
    with pytest.raises(dataset.NotReadOnly):
        dataset.check_read_only("// ВЫБРАТЬ\nУДАЛИТЬ ИЗ т")


def test_the_rule_is_an_allowlist_not_a_denylist():
    """An operator nobody thought of is refused, which is the whole point.

    The project already carries a debt for the opposite construction -- an
    AST denylist in the sandbox that `f = eval` walks through.
    """
    with pytest.raises(dataset.NotReadOnly):
        dataset.check_read_only("СОВЕРШЕННО_НОВЫЙ_ОПЕРАТОР х")


def test_not_read_only_is_an_onec_error():
    """So a caller catching OneCError does not miss the refusal."""
    assert issubclass(dataset.NotReadOnly, onec.OneCError)


# ------------------------------------------------------------- what is kept

def test_a_query_that_runs_but_returns_nothing_is_not_useful():
    """`ГДЕ Ложь` executes perfectly and proves nothing.

    Kept in the bank, because the distinction between "wrong query" and
    "right query, no data" is exactly what a success rate would hide.
    """
    empty = dataset.Attempt(task="t", query="ВЫБРАТЬ 1", ok=True, row_count=0)
    full = dataset.Attempt(task="t", query="ВЫБРАТЬ 1", ok=True, row_count=3)
    assert empty.useful is False
    assert full.useful is True


def test_errors_are_reduced_to_a_kind_with_no_identifiers():
    kind = dataset._error_kind(
        "Поле не найдено \"Номенклатура.ХитПродаж2024\"")
    assert kind == "field_not_found"
    assert "Номенклатура" not in kind


def test_the_shareable_form_carries_no_business_data():
    """The rule that makes a federation possible later: what leaves this
    machine names no object, no table and no question."""
    attempt = dataset.Attempt(
        task="сколько продали гаек в марте",
        query="ВЫБРАТЬ СУММА(Количество) ИЗ РегистрНакопления.ПродажиОбороты",
        ok=False, error='Поле не найдено "ПродажиОбороты.Гайка"',
        columns=["Количество", "Номенклатура"], row_count=0, teacher="groq")
    shared = dataset.Attempt.shared(attempt)
    blob = repr(shared)
    for secret in ("гаек", "Номенклатура", "ПродажиОбороты", "Количество",
                   "сколько продали"):
        assert secret not in blob, secret
    assert shared["error_kind"] == "field_not_found"
    assert shared["ok"] is False


# ------------------------------------------------------------------ the bank

def test_the_bank_is_append_only_and_reads_back(tmp_path):
    bank = dataset.LessonBank(str(tmp_path / "bank.jsonl"))
    bank.add(dataset.Attempt(task="a", query="ВЫБРАТЬ 1", ok=True, row_count=2))
    bank.add(dataset.Attempt(task="b", query="ВЫБРАТЬ 2", ok=False,
                             error="Синтаксическая ошибка"))
    back = bank.all()
    assert [a.task for a in back] == ["a", "b"]
    assert back[0].useful is True


def test_stats_separate_executed_from_useful(tmp_path):
    bank = dataset.LessonBank(str(tmp_path / "bank.jsonl"))
    bank.add(dataset.Attempt(task="a", query="q", ok=True, row_count=5))
    bank.add(dataset.Attempt(task="b", query="q", ok=True, row_count=0))
    bank.add(dataset.Attempt(task="c", query="q", ok=False,
                             error="Поле не найдено X"))
    stats = bank.stats()
    assert stats["attempts"] == 3
    assert stats["executed"] == 2
    assert stats["useful"] == 1
    assert stats["errors"] == {"field_not_found": 1}


def test_the_bank_does_not_decide_when_training_is_worth_it(tmp_path):
    """It reports the count; brain_factory.choose_mechanism decides.

    Two places deciding the same thing is how they come to disagree.
    """
    from mana.cognition.brain_factory import MIN_ML_EXAMPLES
    bank = dataset.LessonBank(str(tmp_path / "bank.jsonl"))
    bank.add(dataset.Attempt(task="a", query="q", ok=True, row_count=1))
    stats = bank.stats()
    assert stats["ready_for_ml"] is False
    assert stats["needed_for_ml"] == MIN_ML_EXAMPLES - 1


def test_an_empty_bank_reports_zeroes_rather_than_failing(tmp_path):
    assert dataset.LessonBank(str(tmp_path / "none.jsonl")).stats()["attempts"] == 0


# ------------------------------------------------------------------ the loop

def test_the_loop_records_a_teacher_that_wrote_a_write(tmp_path, monkeypatch):
    """A teacher producing a non-read is itself a measurement.

    Recorded, never executed: the point of the refusal is that nothing
    reaches the database.
    """
    executed = []
    monkeypatch.setattr(dataset.onec, "query",
                        lambda *a, **k: executed.append(a) or {})
    bank = dataset.LessonBank(str(tmp_path / "bank.jsonl"))
    report = dataset.run_loop(
        ["удали всё"],
        write_query=lambda task, meta: "УДАЛИТЬ ИЗ Справочник.Номенклатура",
        bank=bank, metadata={})
    assert report["refused_not_read_only"] == 1
    assert executed == []                     # nothing reached 1C
    assert bank.all()[0].ok is False


def test_the_loop_records_a_teacher_that_failed_without_stopping(
        tmp_path, monkeypatch):
    monkeypatch.setattr(dataset.onec, "query",
                        lambda *a, **k: {"columns": ["A"], "count": 1})

    def flaky(task, meta):
        if task == "плохая":
            raise RuntimeError("квота исчерпана")
        return "ВЫБРАТЬ 1"

    bank = dataset.LessonBank(str(tmp_path / "bank.jsonl"))
    report = dataset.run_loop(["хорошая", "плохая", "ещё"], flaky, bank,
                              metadata={})
    assert report["tasks"] == 3
    assert report["useful"] == 2
    assert "квота" in bank.all()[1].error


def test_a_failing_query_is_a_result_not_an_error(tmp_path, monkeypatch):
    """probe never raises for a bad query -- that is the thing being
    measured, and raising would stop the run that is collecting it."""
    def refuse(*a, **k):
        raise onec.OneCError("Поле не найдено \"Х\"")

    monkeypatch.setattr(dataset.onec, "query", refuse)
    attempt = dataset.probe("задача", "ВЫБРАТЬ Х ИЗ Справочник.Номенклатура")
    assert attempt.ok is False
    assert dataset._error_kind(attempt.error) == "field_not_found"


def test_metadata_is_fetched_once_and_handed_to_every_call(
        tmp_path, monkeypatch):
    """It is the thing a model cannot guess, and re-fetching it per task
    would spend a COM round trip on an answer that did not change."""
    monkeypatch.setattr(dataset.onec, "query",
                        lambda *a, **k: {"columns": ["A"], "count": 1})
    seen = []
    bank = dataset.LessonBank(str(tmp_path / "bank.jsonl"))
    dataset.run_loop(["a", "b", "c"],
                     write_query=lambda task, meta: seen.append(meta) or "ВЫБРАТЬ 1",
                     bank=bank, metadata={"Справочники": ["Номенклатура"]})
    assert len(seen) == 3
    assert all(m == {"Справочники": ["Номенклатура"]} for m in seen)
