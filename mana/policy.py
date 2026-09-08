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

Defaults are what is in force
------------------------------
A default here is the behaviour MANA actually has. Three of them changed
on 08.09.2026, when the 1C recognition fix was adopted: the loop proposed
it, a dry run scored it, and it was turned on **under stated uncertainty**
rather than on a verdict -- the gates want thirty paired trials of live
evidence and there were seven recorded turns.

That adoption is in the findings ledger with verdict NOT_EVALUATED, so
the record says what it rests on. When live evidence accumulates the
gates judge it properly, and the reverse setting is a candidate the
generator will offer, because every knob's options include what it was
before.

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
__version__ = "1.2"


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
    Knob("echo_lookback", 6, (6, 1, 3, 12), "repeats_earlier_answer",
         "Сколько прошлых обменов сравнивать. Было один, и повтор старше "
         "одного хода оставался невидим. Принято 08.09.2026 по замеру на "
         "живой сессии: глубина 1 ловила 0 повторов из 12 ходов, 3 — два, "
         "6 — все четыре, 12 — те же четыре. Ложных срабатываний ни на "
         "одной глубине не было. Цена ложного — один лишний вызов: при "
         "срабатывании ответ переспрашивается без вспомненной памяти, и "
         "если повтор остаётся, возвращается исходный."),
    Knob("echo_same_answer", 0.80, (0.80, 0.70, 0.90), "repeats_earlier_answer",
         "Насколько похожими должны быть два ответа, чтобы считаться одним."),
    Knob("echo_different_question", 0.75, (0.75, 0.60, 0.85),
         "repeats_earlier_answer",
         "Насколько разными должны быть вопросы, чтобы повтор был подозрителен. "
         "Слишком низко — MANA откажется быть последовательной, что хуже дефекта."),
    Knob("intent_verb_anywhere", True, (True, False),
         "actionable_request_not_acted_on",
         "Считать ли повелительный глагол в середине сообщения командой. "
         "Принято 08.09.2026: до этого требовалось начало строки, и «я хочу "
         "поработать с 1С запусти пожалуйста конфигуратор» не распознавалось "
         "вовсе. Включено вместе с ограждениями в apps/intent.py — без них "
         "давало 3 ложных запуска из 12 вопросов про 1С."),
    Knob("intent_verb_forms", "addressed", ("addressed", "imperative"),
         "actionable_request_not_acted_on",
         "Какие формы глагола считать просьбой. «addressed» добавляет "
         "обращённые к MANA формы к повелительным: «я хочу что бы ты открыла "
         "конфигуратор» без них не распознавалось. Риск назван в замере — "
         "«ты уже открыла 1С?» вопрос, а не команда, — и снят требованием "
         "маркера просьбы («хочу», «прошу», «можешь») перед такой формой."),
    Knob("intent_stem_match", True, (True, False),
         "actionable_request_not_acted_on",
         "Сопоставлять ли имена баз по основам слов. До принятия — "
         "подстрокой, поэтому «информационной базы» не находило "
         "«Информационная база»: по-русски никто не пишет имя базы в "
         "именительном падеже."),
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
