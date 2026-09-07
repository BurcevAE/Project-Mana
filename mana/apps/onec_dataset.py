"""
mana.apps.onec_dataset — manufacturing verified 1C examples.

The point of this module is an unusual one: it does not answer questions.
It spends the expensive, rate-limited external models on producing
training data whose correctness has been **checked by execution**, and it
keeps only what survived.

Why that is worth more than the answers
----------------------------------------
An external model with a daily quota is a scarce resource. Spent on user
questions it produces answers nobody verified; spent here it produces a
dataset that is *better than the teacher*, because every attempt that
failed to execute is thrown away along with the reasoning that produced
it. A few hundred requests a day compound into thousands of verified
examples a month, and unlike answers, examples accumulate.

1C is unusually well suited to this. A query either runs or it does not,
and running it is cheap. That is an oracle -- and the whole "spend time
instead of parameters" trade only works where there is one.

Read-only, by allowlist rather than denylist
---------------------------------------------
The text is required to consist of ВЫБРАТЬ statements (plus УНИЧТОЖИТЬ of
a temporary table, which is how a 1C query batch cleans up after itself).
That is a positive property of the text, not the absence of a list of bad
words -- and this project already carries a documented debt for a sandbox
policy built the other way round, where `f = eval` walks through an AST
denylist.

The 1C `Запрос` object cannot write; that is true and it is not the
reason this check exists. Generated text goes into a live trade database,
and "the API probably cannot do damage" is not a sentence to bet someone
else's accounting records on.

What is stored, and what is deliberately not
---------------------------------------------
A record keeps the query and the SHAPE of what came back -- column names,
row count, timing. It does not keep the rows. Business data has no reason
to be in a training corpus, and the shape is what says whether the query
was right.

That split is also what makes the bank shareable later: `Attempt.shared()`
returns the part that carries no configuration names and no data, which is
the only part that could ever leave this machine.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import onec

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Statements a read-only batch may consist of. An allowlist: anything not
#: named here is refused, including anything nobody thought of.
ALLOWED_STATEMENTS = ("ВЫБРАТЬ", "SELECT", "УНИЧТОЖИТЬ", "DROP")

#: Rows fetched while probing. The question is whether the query is right,
#: and that is answered by the first few rows; pulling a ledger to find out
#: costs the server real work for nothing.
PROBE_ROW_LIMIT = 20

_COMMENT = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)


class NotReadOnly(onec.OneCError):
    """The generated text is not a plain read. It is not executed."""


def strip_comments(text: str) -> str:
    return _COMMENT.sub(" ", _BLOCK_COMMENT.sub(" ", text or ""))


def check_read_only(text: str) -> None:
    """Refuse anything that is not a batch of reads.

    Raises rather than returning a flag: a caller that forgets to look at
    a boolean still gets stopped, and this is the one place in the module
    where forgetting would matter.
    """
    body = strip_comments(text).strip()
    if not body:
        raise NotReadOnly("пустой запрос")
    for piece in body.split(";"):
        statement = piece.strip()
        if not statement:
            continue
        head = statement.split(None, 1)[0].upper()
        if head not in ALLOWED_STATEMENTS:
            raise NotReadOnly(
                f"оператор {head!r} не разрешён; допускаются только "
                f"{', '.join(ALLOWED_STATEMENTS)} — это разрешительный "
                f"список, а не запретительный")


# ------------------------------------------------------------------- records


@dataclass
class Attempt:
    """One question, one generated query, and what executing it proved."""
    task: str                       # the question in natural language
    query: str                      # what the teacher wrote
    ok: bool                        # did it execute
    error: str = ""                 # 1C's message, when it did not
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    seconds: float = 0.0
    teacher: str = ""               # which brain produced it
    created: float = field(default_factory=time.time)

    @property
    def useful(self) -> bool:
        """Executed AND returned something.

        A query that runs and returns nothing is not evidence of a correct
        query: `ГДЕ Ложь` runs perfectly. Kept in the bank because the
        failure is informative, excluded from the trainable set.
        """
        return self.ok and self.row_count > 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def shared(self) -> Dict[str, Any]:
        """The part that could leave this machine, and nothing else.

        No task text, no query, no column names: all three carry names
        from a proprietary configuration, and the task text is whatever a
        person typed. What survives is the shape of the outcome, which is
        what another instance could learn from without learning anything
        about this business.
        """
        return {
            "ok": self.ok,
            "returned_rows": self.row_count > 0,
            "error_kind": _error_kind(self.error),
            "seconds": round(self.seconds, 3),
            "teacher": self.teacher,
        }


def _error_kind(message: str) -> str:
    """A 1C error reduced to its class, with all identifiers dropped.

    "Поле не найдено Номенклатура.ХитПродаж" names an object in someone's
    configuration. "field_not_found" says the same thing about the mistake
    and nothing about the business.
    """
    if not message:
        return ""
    lowered = message.lower()
    for needle, kind in (
            ("поле не найдено", "field_not_found"),
            ("не найден", "not_found"),
            ("синтаксическая ошибка", "syntax"),
            ("ожидается", "syntax"),
            ("таблица не найдена", "table_not_found"),
            ("недостаточно прав", "permissions"),
            ("тип не может", "type_mismatch"),
    ):
        if needle in lowered:
            return kind
    return "other"


# ---------------------------------------------------------------- the bank


class LessonBank:
    """Append-only store of attempts, on disk as JSON lines.

    Append-only on purpose: a corpus that can be edited in place is a
    corpus whose history nobody can reconstruct, and the failures are half
    of what makes this one worth having.
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, attempt: Attempt) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(attempt.as_dict(), ensure_ascii=False) + "\n")

    def all(self) -> List[Attempt]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(Attempt(**json.loads(line)))
        return out

    def stats(self) -> Dict[str, Any]:
        """What the bank knows about itself.

        `useful` is reported separately from `ok` because the difference is
        the interesting number: queries that execute and return nothing are
        the failure mode a success rate would hide.
        """
        attempts = self.all()
        if not attempts:
            return {"attempts": 0, "executed": 0, "useful": 0,
                    "errors": {}, "ready_for_ml": False}
        errors: Dict[str, int] = {}
        for attempt in attempts:
            if not attempt.ok:
                kind = _error_kind(attempt.error)
                errors[kind] = errors.get(kind, 0) + 1
        useful = sum(1 for a in attempts if a.useful)
        from ..cognition.brain_factory import MIN_ML_EXAMPLES
        return {
            "attempts": len(attempts),
            "executed": sum(1 for a in attempts if a.ok),
            "useful": useful,
            "errors": dict(sorted(errors.items(), key=lambda kv: -kv[1])),
            # The bank does not decide when training is worth it --
            # brain_factory.choose_mechanism does, from this number.
            "ready_for_ml": useful >= MIN_ML_EXAMPLES,
            "needed_for_ml": max(0, MIN_ML_EXAMPLES - useful),
        }

    def shareable(self) -> List[Dict[str, Any]]:
        return [a.shared() for a in self.all()]


# ------------------------------------------------------------------- the loop


def probe(task: str, query: str, teacher: str = "",
          limit: int = PROBE_ROW_LIMIT) -> Attempt:
    """Execute one generated query read-only and record what happened.

    Never raises for a bad query: a query that fails is the outcome being
    measured, not an error in measuring. It raises only when the text is
    not a read at all, because that is a refusal to run rather than a
    result.
    """
    check_read_only(query)

    started = time.perf_counter()
    try:
        result = onec.query(query, limit=limit)
    except onec.OneCError as exc:
        return Attempt(task=task, query=query, ok=False, error=str(exc),
                       seconds=time.perf_counter() - started, teacher=teacher)
    return Attempt(
        task=task, query=query, ok=True,
        columns=list(result.get("columns") or []),
        row_count=int(result.get("count") or 0),
        truncated=bool(result.get("truncated")),
        seconds=time.perf_counter() - started, teacher=teacher)


def run_loop(tasks: Sequence[str], write_query: Callable[[str, Dict[str, Any]], str],
             bank: LessonBank, metadata: Optional[Dict[str, Any]] = None,
             on_attempt: Optional[Callable[[Attempt], None]] = None
             ) -> Dict[str, Any]:
    """Generate, execute, record -- for a batch of questions.

    `write_query` is handed the task and the infobase metadata and returns
    a query. It is passed in rather than built here so the loop does not
    care whether the teacher is a remote model, a local one, or a person:
    what the loop contributes is the execution and the bookkeeping.

    The metadata is fetched once and given to every call. It is the thing
    a model cannot guess -- which catalogs and documents this particular
    configuration has -- and handing it over is what separates a query
    that names real objects from one that invents plausible ones.
    """
    if metadata is None:
        onec.connect()
        metadata = onec.metadata()

    produced: List[Attempt] = []
    refused = 0
    for task in tasks:
        try:
            query = write_query(task, metadata)
        except Exception as exc:                    # a teacher that failed
            attempt = Attempt(task=task, query="", ok=False,
                              error=f"учитель не ответил: {exc}")
            bank.add(attempt)
            produced.append(attempt)
            continue
        try:
            attempt = probe(task, query)
        except NotReadOnly as exc:
            # Recorded, not executed. A teacher that writes a non-read is
            # itself a measurement worth keeping.
            attempt = Attempt(task=task, query=query, ok=False,
                              error=f"не только чтение: {exc}")
            refused += 1
        bank.add(attempt)
        produced.append(attempt)
        if on_attempt is not None:
            on_attempt(attempt)

    return {
        "tasks": len(tasks),
        "executed": sum(1 for a in produced if a.ok),
        "useful": sum(1 for a in produced if a.useful),
        "refused_not_read_only": refused,
        "bank": bank.stats(),
    }
