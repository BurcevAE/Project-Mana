"""
mana.cognition.cycle — the whole loop in one place, and what stops it.

    реальный опыт → ошибка → исследование → изменение
                  → повторная ситуация → улучшенный результат

Every stage of that already existed and each was reached by a different
command. A person read `--findings`, then `--propose`, then decided, then
would have had to assemble the paired comparison by hand and remember to
write the result into the ledger. A loop that only a person can walk is
not a loop; it is a set of parts that happen to fit.

So this runs the six stages against a record and reports each one, and the
last thing it reports is what stopped the cycle from being proven. That
last line is the point of the module. There will nearly always be one, and
a loop that could not say which stage it stalled at would be indistinguish-
able from a loop that quietly did nothing.

Nothing here decides anything
------------------------------
The verdict comes from `core/gates.py` through `failure_domain.judge_change`
-- the same gates, the same sample floor, the same three states as every
other claim in the project. A cycle runner with its own acceptance rule
would be a second way to change the system, which is the loophole the
gates exist to close.

The dry evaluation is an upper bound
-------------------------------------
Candidates are replayed with `candidates.dry_responder`, which assumes a
feasible action succeeds. So a positive result here is a ceiling, it is
labelled one, and the gate is fed it knowing that: what the cycle can
prove is "this change would not help even at its best", and what it cannot
prove alone is that a change that looks good here helps in live use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..journal import Episode
from ..policy import Policy
from . import candidates as candidates_mod
from . import failure_domain as fd
from . import invariants as inv
from . import lessons as lessons_mod
from .findings import Finding, Ledger, measurement_of
from ..core.gates import ACCEPTED, MIN_PAIRED_TRIALS, NOT_EVALUATED, REJECTED

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

EXPERIENCE = "опыт"
FAILURE = "ошибка"
INVESTIGATION = "исследование"
CHANGE = "изменение"
REPLAY = "повторная ситуация"
RESULT = "результат"

STAGES = (EXPERIENCE, FAILURE, INVESTIGATION, CHANGE, REPLAY, RESULT)


@dataclass(frozen=True)
class Stage:
    """One step of the loop: whether it produced anything, and what."""
    name: str
    reached: bool
    summary: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        mark = "+" if self.reached else "-"
        return f"  [{mark}] {self.name}: {self.summary}"

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "reached": self.reached,
                "summary": self.summary, "detail": dict(self.detail)}


@dataclass(frozen=True)
class Cycle:
    """One pass of the loop over one record."""
    stages: Tuple[Stage, ...] = ()
    verdict: str = ""
    #: What stopped it short of a proven improvement. Empty only when
    #: the gates accepted, which on a real record is rare and is
    #: supposed to be.
    missing: str = ""
    finding_id: str = ""
    #: How the change did on the half it was never chosen on. None when
    #: there was no holdout to ask.
    hidden_margin: Optional[float] = None
    hidden_size: int = 0

    @property
    def loop_closed(self) -> bool:
        """Did experience reach a verdict without a person in the middle?

        A fact about machinery. It can be true on four turns, and its
        being true says nothing about whether MANA got better -- which is
        why it is reported apart from the next one.
        """
        return all(stage.reached for stage in self.stages)

    @property
    def improvement_proven(self) -> bool:
        """Did a change chosen on one half do better on the other?

        A fact about the world, and the only one that supports the
        sentence "MANA improved". No number of gates passed on the half a
        candidate was chosen on can establish it: that half is where the
        choosing happened, and a change that wins there may only have
        been fitted to it.
        """
        return (self.verdict == ACCEPTED and self.loop_closed
                and self.hidden_size > 0
                and (self.hidden_margin or 0.0) > 0.0)

    @property
    def proven(self) -> bool:
        return self.improvement_proven

    def stage(self, name: str) -> Optional[Stage]:
        for found in self.stages:
            if found.name == name:
                return found
        return None

    def describe(self) -> str:
        lines = ["цикл: реальный опыт → ошибка → исследование → изменение "
                 "→ повторная ситуация → улучшенный результат"]
        lines.extend(stage.describe() for stage in self.stages)
        lines.append("")
        lines.append(f"вердикт: {self.verdict or '—'}")
        # Two facts, never one. The first is about the machinery and can
        # be true on four turns; the second is about the world.
        lines.append(f"контур замкнут:      "
                     f"{'да' if self.loop_closed else 'нет'}")
        lines.append(f"улучшение доказано:  "
                     f"{'да' if self.improvement_proven else 'нет'}"
                     + (f" (на скрытой половине {self.hidden_margin:+.2f} "
                        f"по {self.hidden_size} ситуациям)"
                        if self.hidden_margin is not None else
                        " (скрытой половины не было)"))
        if self.missing:
            lines.append(f"не хватает: {self.missing}")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {"stages": [s.as_dict() for s in self.stages],
                "verdict": self.verdict, "missing": self.missing,
                "loop_closed": self.loop_closed,
                "improvement_proven": self.improvement_proven,
                "hidden_margin": self.hidden_margin,
                "hidden_size": self.hidden_size,
                "finding_id": self.finding_id}


def _already_run(row: Dict[str, Any]) -> bool:
    """Has this exact experiment already been run on this evidence?

    Not "was it rejected". A NOT_EVALUATED result is not a rejection, and
    re-running it against the same record produces the same "could not
    tell" -- so the verdict is the wrong thing to ask. What matters is
    that the approach and the conditions are the ones already on record,
    which is what `already_tried` calls an exact match.

    Found by a test: keying on REJECTED let an unevaluated candidate come
    round again on every pass.
    """
    prior = row.get("already_tried") or {}
    if prior.get("match") == "exact":
        return True
    return (prior.get("verdict") == REJECTED
            and not prior.get("conditions_moved"))


def _stop(stages: List[Stage], name: str, summary: str,
          missing: str, detail: Optional[Dict[str, Any]] = None) -> Cycle:
    stages.append(Stage(name=name, reached=False, summary=summary,
                        detail=dict(detail or {})))
    return Cycle(stages=tuple(stages), verdict="", missing=missing)


def run(episodes: Sequence[Episode],
        current: Optional[Policy] = None,
        ledger: Optional[Ledger] = None,
        targets: Optional[Sequence[str]] = None,
        record: bool = True) -> Cycle:
    """Walk the loop once over a record, and say where it stopped.

    `record` writes the result to the ledger, which is the point of
    running it: a cycle whose outcome is not written down has to be
    re-run to be remembered, and re-running is how a rejected change gets
    proposed again next week.
    """
    from .. import policy as policy_mod

    current = current or policy_mod.active()
    ledger = ledger if ledger is not None else Ledger()
    stages: List[Stage] = []

    # ---------- 1. real experience ----------
    if not episodes:
        return _stop(stages, EXPERIENCE, "журнал пуст",
                     "записанных ходов нет — цикл начинается с работы, "
                     "а не с кода")
    sessions = len({e.session for e in episodes})
    stages.append(Stage(EXPERIENCE, True,
                        f"{len(episodes)} ходов в {sessions} сессиях",
                        {"episodes": len(episodes), "sessions": sessions}))

    # ---------- 2. failure ----------
    if targets is None:
        targets = inv.known_targets()
    violations = inv.scan(episodes, targets)
    if not violations:
        return _stop(stages, FAILURE, "нарушений в записи нет",
                     "нечего чинить — это исправный конец, а не тупик")
    by_invariant: Dict[str, int] = {}
    for violation in violations:
        by_invariant[violation.invariant] = by_invariant.get(violation.invariant, 0) + 1
    stages.append(Stage(FAILURE, True,
                        f"{len(violations)} нарушений: "
                        + ", ".join(f"{k}×{v}" for k, v in sorted(by_invariant.items())),
                        {"by_invariant": by_invariant}))

    # ---------- 3. investigation ----------
    learned = lessons_mod.read(ledger, [v.invariant for v in violations])
    domain = fd.build(episodes, targets=targets)
    situations = list(domain.train)
    plan = candidates_mod.plan(violations, current, learned)
    ranked = candidates_mod.rank(violations, situations, current, ledger, learned)
    measurable = [row for row in ranked
                  if (row.get("dry") or {}).get("dry_evaluable")]
    # A candidate already measured and rejected under conditions that
    # still hold is not the next experiment. `rank` sinks it rather than
    # hiding it, which is right for a reader and wrong for a loop: the
    # loop would pick the least bad of the settled ones and re-run it for
    # ever. Measured on the real record, which did exactly that.
    settled = [row for row in measurable if _already_run(row)]
    usable = [row for row in measurable if not _already_run(row)]
    if not usable:
        if settled:
            why = (f"измеримого не осталось: все {len(settled)} кандидатов "
                   f"уже проведены при этих же условиях, и на той же записи "
                   f"дадут тот же ответ. Сдвинуть это может только новый "
                   f"опыт — с ним меняются условия — или механизм, которого "
                   f"нет в пространстве настроек")
        else:
            why = "; ".join(r["why"] for r in plan.refusals) or (
                "все кандидаты не оцениваются всухую — их эффект не увидеть "
                "без запуска модели")
        return _stop(stages, INVESTIGATION, "нечего измерить", why,
                     {"refusals": [dict(r) for r in plan.refusals],
                      "candidates": len(ranked), "settled": len(settled)})
    chosen = usable[0]
    stages.append(Stage(
        INVESTIGATION, True,
        f"кандидатов {len(ranked)}, измеримо всухую {len(usable)}; "
        f"выбран {chosen['changes']} (ценность {chosen.get('value', 0):+.3f})",
        {"chosen": chosen["changes"], "candidates": len(ranked),
         "refusals": [dict(r) for r in plan.refusals]}))

    # ---------- 4. the change ----------
    candidate_policy = Policy.of(**{k: v for k, v in chosen["changes"].items()})
    stages.append(Stage(CHANGE, True,
                        f"{current.describe()} → {candidate_policy.describe()}",
                        {"from": current.changes(), "to": chosen["changes"]}))

    # ---------- 5. the same situations again ----------
    # Both arms answered by the same machinery, the baseline being the
    # policy in force rather than the recorded answers. Scoring against
    # the record would compare every candidate with a version of MANA
    # that no longer exists, and after an adoption they all look good for
    # free -- measured once already, in `dry_report`.
    baseline = fd.evaluate(situations, candidates_mod.dry_responder(current),
                           targets)
    after = fd.evaluate(situations,
                        candidates_mod.dry_responder(candidate_policy), targets)
    # The half the candidate was not chosen on. `build` has been splitting
    # it out on every pass since this module was written and nothing read
    # it, so every run reported "hidden: not measured" -- the holdout
    # existed and was consulted by nobody.
    hidden_before = fd.evaluate(list(domain.hidden),
                                candidates_mod.dry_responder(current), targets)
    hidden_after = fd.evaluate(list(domain.hidden),
                               candidates_mod.dry_responder(candidate_policy),
                               targets)
    fixed = [b.situation_id for b, c in zip(baseline, after)
             if not b.passed and c.passed]
    broke = [b.situation_id for b, c in zip(baseline, after)
             if b.passed and not c.passed]
    hidden_margin = (fd.pass_rate(hidden_after) - fd.pass_rate(hidden_before)
                     if domain.hidden else None)
    stages.append(Stage(
        REPLAY, True,
        f"переиграно {len(situations)}: стало проходить {len(fixed)}, "
        f"сломано {len(broke)}"
        + (f"; на скрытой половине ({len(domain.hidden)}) "
           f"{hidden_margin:+.2f}" if hidden_margin is not None else
           "; скрытой половины нет"),
        {"fixed": fixed, "broke": broke,
         "baseline_pass_rate": round(fd.pass_rate(baseline), 4),
         "candidate_pass_rate": round(fd.pass_rate(after), 4),
         "hidden": len(domain.hidden),
         "hidden_baseline": round(fd.pass_rate(hidden_before), 4)
                            if domain.hidden else None,
         "hidden_candidate": round(fd.pass_rate(hidden_after), 4)
                             if domain.hidden else None,
         "hidden_margin": round(hidden_margin, 4)
                          if hidden_margin is not None else None}))

    # ---------- 6. the verdict, from the gates ----------
    judged = fd.judge_change(
        description=f"политика {chosen['changes']} против «{chosen['addresses']}»",
        domain=domain, baseline=baseline, candidate=after,
        hidden_baseline=hidden_before, hidden_candidate=hidden_after)
    verdict = judged["verdict"]
    status = verdict.get("status") or (ACCEPTED if verdict["accepted"] else REJECTED)
    stages.append(Stage(RESULT, True,
                        f"{status}: {verdict['reason']}",
                        {"verdict": verdict}))

    # ---------- 7. write it down ----------
    finding_id = ""
    if record:
        measured = verdict.get("measurements") or {}
        trials = int(measured.get("paired_trials") or len(situations))
        margin = float(measured.get("dev_margin") or 0.0)
        finding = Finding(
            question=candidates_mod.question_for(chosen["addresses"]),
            approach=dict(chosen["changes"]),
            verdict=status if status in (ACCEPTED, REJECTED, NOT_EVALUATED)
                    else NOT_EVALUATED,
            measurement=measurement_of(trials=trials,
                                       interval=(margin, margin), null=0.0),
            # Exactly the conditions `candidates.prior_for` probes with.
            # Recording anything else means the loop cannot find its own
            # results next time round, and re-runs what it has just run --
            # which is what happened on the first six passes over the real
            # record before this line was fixed.
            conditions={"observed_failures": chosen["observed_failures"]},
            note=(f"сухой прогон по записанным ходам ({domain.identity}): "
                  f"верхняя граница, а не живой результат. "
                  + verdict["reason"]))
        if ledger.record(finding):
            finding_id = finding.finding_id

    missing = _missing(verdict, len(situations))
    return Cycle(stages=tuple(stages), verdict=status, missing=missing,
                 finding_id=finding_id, hidden_margin=hidden_margin,
                 hidden_size=len(domain.hidden))


def _missing(verdict: Dict[str, Any], trials: int) -> str:
    """What stands between this run and a proven improvement.

    Named per failed gate rather than summarised, because "not proven"
    covers "we have no evidence" and "we have evidence that it does not
    work", and those call for opposite next moves.
    """
    failed = list(verdict.get("failed_gates") or ())
    if not failed:
        return ""
    reasons: List[str] = []
    if "sample_size" in failed:
        reasons.append(f"записанных ходов {trials} против {MIN_PAIRED_TRIALS} — "
                       f"нужно ещё {max(0, MIN_PAIRED_TRIALS - trials)}, "
                       f"и они появляются от работы, а не от кода")
    if "dev_improvement" in failed:
        reasons.append("выигрыша на записи нет — менять нечего")
    if "significance" in failed:
        reasons.append("различие не отделимо от случайности")
    if "hidden_confirmation" in failed:
        reasons.append("на скрытой выборке не подтверждено")
    if "no_domain_regression" in failed:
        reasons.append("что-то сломалось в другом домене")
    if "counterexamples" in failed:
        reasons.append("нашлись контрпримеры")
    for name in failed:
        if name not in ("sample_size", "dev_improvement", "significance",
                        "hidden_confirmation", "no_domain_regression",
                        "counterexamples"):
            reasons.append(f"не пройден гейт «{name}»")
    return "; ".join(reasons)
