"""
mana.policy — the answering decisions a candidate is allowed to change.

Why a separate module
---------------------
`cognition/candidates.py` proposes changes and `core/gates.py` judges
them, but a proposal that reaches nothing is theatre. This is the surface
those proposals act on: a small, explicit set of decision points that the
answering path really consults.

Explicit is the whole design. The knobs are declared here, by name, with
a default that reproduces today's behaviour exactly and a note saying
which invariant each one bears on. A candidate generator that could
change anything would be a second way to modify the system, and the
second way is always the one with the hole in it -- the same reason
`brain_factory` may assemble candidates and may not adopt them.

Defaults are today
------------------
Every default below is the value the code had before this module existed,
so importing it changes no behaviour. A knob only matters once a policy
carrying a different value is put into effect, which happens during
evaluation and, if the gates accept it, afterwards.

Per thread, deliberately
------------------------
`use()` overrides the policy on the calling thread only. A research cycle
evaluating a candidate can run while somebody is typing; a process-wide
swap would silently answer their question under an unproven policy. The
same reasoning as the journal's per-thread episode.
"""
from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Iterator, Sequence, Tuple

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"


@dataclass(frozen=True)
class Knob:
    """One decision a candidate may change, and what it bears on."""
    name: str
    default: Any
    options: Tuple[Any, ...]
    addresses: str              # the invariant this knob can affect
    why: str

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "default": self.default,
                "options": list(self.options), "addresses": self.addresses,
                "why": self.why}


#: The search space. An allowlist: a field not declared here cannot be
#: changed by a candidate, however the proposal is written.
#:
#: Each `options` tuple begins with the current default, so a generator
#: enumerating them always has the do-nothing candidate available and a
#: comparison against it is a comparison against today.
KNOBS: Tuple[Knob, ...] = (
    Knob("echo_lookback", 1, (1, 3, 6, 12), "repeats_earlier_answer",
         "Сколько прошлых обменов сравнивать. Сегодня — один: "
         "`previous_exchange` останавливается на первом найденном ответе, "
         "поэтому повтор старше одного хода невидим. Замерено: ответ, "
         "идентичный данному тремя ходами раньше, набрал 1.00 против того "
         "хода и 0.018 против того, который guard смотрит."),
    Knob("echo_same_answer", 0.80, (0.80, 0.70, 0.90), "repeats_earlier_answer",
         "Насколько похожими должны быть два ответа, чтобы считаться одним."),
    Knob("echo_different_question", 0.75, (0.75, 0.60, 0.85),
         "repeats_earlier_answer",
         "Насколько разными должны быть вопросы, чтобы повтор был подозрителен. "
         "Слишком низко — MANA откажется быть последовательной, что хуже дефекта."),
    Knob("intent_verb_anywhere", False, (False, True),
         "actionable_request_not_acted_on",
         "Считать ли повелительный глагол в середине сообщения командой. "
         "Сегодня требуется начало строки, поэтому «я хочу поработать с 1С "
         "запусти пожалуйста конфигуратор» не распознаётся вовсе."),
    Knob("intent_verb_forms", "imperative", ("imperative", "addressed"),
         "actionable_request_not_acted_on",
         "Какие формы глагола считать просьбой. Сегодня только повелительные "
         "(«запусти», «открой»). Замерено: «я хочу что бы ты открыла "
         "конфигуратор» не распознаётся — сослагательного «открыла» в списке "
         "нет. «addressed» добавляет обращённые к MANA формы, и это заметно "
         "рискованнее: «ты уже открыла отчёт?» — вопрос, а не команда. "
         "Поэтому решают ворота, а не мнение."),
    Knob("intent_stem_match", False, (False, True),
         "actionable_request_not_acted_on",
         "Сопоставлять ли имена баз по основам слов. Сегодня — подстрокой, "
         "поэтому «информационной базы» не находит «Информационная база»: "
         "по-русски никто не пишет имя в именительном падеже."),
)

_BY_NAME = {knob.name: knob for knob in KNOBS}


@dataclass(frozen=True)
class Policy:
    """An assignment to the knobs above.

    Frozen and content-addressed: two policies with the same settings are
    the same policy, and a recorded verdict names the settings it judged
    rather than a serial number that means nothing later.
    """
    settings: Tuple[Tuple[str, Any], ...] = ()

    @classmethod
    def of(cls, **settings: Any) -> "Policy":
        unknown = sorted(set(settings) - set(_BY_NAME))
        if unknown:
            raise ValueError(f"неизвестные настройки политики: {unknown}")
        for name, value in settings.items():
            allowed = _BY_NAME[name].options
            if value not in allowed:
                raise ValueError(
                    f"{name}={value!r} вне объявленного диапазона {allowed}")
        return cls(tuple(sorted(settings.items())))

    def get(self, name: str) -> Any:
        knob = _BY_NAME.get(name)
        if knob is None:
            raise KeyError(name)
        for key, value in self.settings:
            if key == name:
                return value
        return knob.default

    def as_dict(self) -> Dict[str, Any]:
        return {knob.name: self.get(knob.name) for knob in KNOBS}

    def changes(self) -> Dict[str, Any]:
        """Only what differs from today. What a report should print."""
        return {name: value for name, value in self.settings
                if value != _BY_NAME[name].default}

    @property
    def policy_id(self) -> str:
        body = json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.blake2b(body.encode("utf-8"), digest_size=6).hexdigest()

    def describe(self) -> str:
        changed = self.changes()
        if not changed:
            return "как сейчас (без изменений)"
        return ", ".join(f"{k}={v}" for k, v in sorted(changed.items()))


#: The policy in force when nothing overrides it: every knob at its
#: default, i.e. exactly the behaviour before this module existed.
BASELINE = Policy()

_local = threading.local()


def active() -> Policy:
    """The policy this thread must answer under."""
    return getattr(_local, "policy", None) or BASELINE


@contextmanager
def use(policy: Policy) -> Iterator[Policy]:
    """Answer under `policy` on this thread, then put it back.

    Restored in a `finally`: an evaluation that raises must not leave an
    unproven policy in force for the next question on this thread.
    """
    previous = getattr(_local, "policy", None)
    _local.policy = policy
    try:
        yield policy
    finally:
        _local.policy = previous


def get(name: str) -> Any:
    """One setting from the policy in force. What callers use."""
    return active().get(name)


def knobs_for(invariant: str) -> Tuple[Knob, ...]:
    """The decision points that could affect this invariant.

    A generator asks this instead of guessing, so a candidate is always
    aimed at a failure that was actually observed.
    """
    return tuple(k for k in KNOBS if k.addresses == invariant)
