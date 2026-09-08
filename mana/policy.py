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
import time
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import (Any, Dict, Iterator, Optional, Sequence,
                    Tuple)

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
    Knob("classify_text_first", False, (False, True), "task_naming",
         "Смотреть ли сначала, О ЧЁМ задача, и лишь потом, что она просит. "
         "«Сколько раз буква «и» встречается в тексте» считалось "
         "арифметикой, потому что в списке math есть «сколько»: слово "
         "делает свою работу, а не хватало того, что вопрос о тексте "
         "остаётся о тексте, что бы в нём ни считали. Принято 08.09.2026 "
         "по трём выборкам: открытие 0.654 → 1.000, валидация +0.3535 "
         "[+0.3473, +0.3592], свежая +0.3465 [+0.3400, +0.3532], лучше "
         "40 из 40 на обеих скрытых, ни одного домена не сломано."),
    Knob("classify_premise_marker", False, (False, True), "task_naming",
         "Считать ли перечень посылок признаком рассуждения. Задача вида "
         "«Известно: … Кто стоит на позиции 2?» не содержит ни одного "
         "маркера из списка — там слова объяснения («почему», «объясни»), "
         "а не вывода из данных. Принято тем же замером, что и "
         "classify_text_first: поодиночке ни одна из двух настроек "
         "полного разбора не даёт."),
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
_adopted_lock = threading.RLock()
_adopted_cache: Optional[Dict[str, Any]] = None


def _overlay_path() -> Path:
    """Beside the journal and the ledger, not inside the package.

    An adoption is a fact about this installation. Kept in the package it
    would be lost on reinstall and shared by a repository, and a change
    one machine earned would arrive on another with no evidence.
    """
    from .paths import resolve_data_path
    return Path(resolve_data_path("policy/adopted.json"))


class AdoptionRefused(RuntimeError):
    """An adoption without evidence behind it."""


#: What an adoption must name before it may be written. Not politeness:
#: an overlay that can be written on an argument is the same thing as a
#: default that can be edited on one.
REQUIRED_EVIDENCE = ("experiment", "fresh", "verdict")


def _load_adopted() -> Dict[str, Any]:
    global _adopted_cache
    with _adopted_lock:
        if _adopted_cache is not None:
            return _adopted_cache
        try:
            _adopted_cache = json.loads(
                _overlay_path().read_text(encoding="utf-8"))
        except Exception:
            _adopted_cache = {}
        return _adopted_cache


def adopted() -> Policy:
    """What evidence has earned, as a policy. BASELINE when nothing has."""
    settings = (_load_adopted().get("settings") or {})
    usable = {name: value for name, value in settings.items()
              if name in _BY_NAME}
    if not usable:
        return BASELINE
    try:
        overlay = Policy.of(**usable)
        # An overlay that changes nothing is the baseline. Returning the
        # same object keeps "nothing is adopted" a single identity rather
        # than a fresh equal-looking one every call.
        return overlay if overlay.changes() else BASELINE
    except Exception:
        # A stored setting the knobs no longer allow. Ignored rather than
        # raised: an old overlay must not stop the program starting, and
        # falling back to the conservative behaviour is the safe way to
        # be wrong.
        return BASELINE


def adoption_evidence() -> Dict[str, Any]:
    """What was measured, and where. Empty when nothing is adopted."""
    return dict(_load_adopted().get("evidence") or {})


def adopt(policy: Policy, evidence: Dict[str, Any]) -> Policy:
    """Put this policy in force, with the measurement that earned it.

    Refuses without evidence naming the experiment, the fresh-split
    result and the verdict. What is stored is the settings and the
    numbers, so a person can read why their program behaves as it does
    and take it off again.
    """
    missing = [key for key in REQUIRED_EVIDENCE if not (evidence or {}).get(key)]
    if missing:
        raise AdoptionRefused(
            "принятие без свидетельства: не названо " + ", ".join(missing))
    global _adopted_cache
    payload = {"settings": policy.as_dict(), "changes": policy.changes(),
               "evidence": dict(evidence), "at": time.time()}
    with _adopted_lock:
        path = _overlay_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        _adopted_cache = payload
    return policy


def revert() -> bool:
    """Take the adoption off. The record of it stays in the ledger."""
    global _adopted_cache
    with _adopted_lock:
        _adopted_cache = {}
        try:
            _overlay_path().unlink()
            return True
        except Exception:
            return False


def active() -> Policy:
    """The policy this thread must answer under.

    Most specific first: an experiment's `use()` beats an adoption, which
    beats the conservative default. That order is what lets a candidate be
    measured against current behaviour whatever current behaviour is.
    """
    return getattr(_local, "policy", None) or adopted()


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
