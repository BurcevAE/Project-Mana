"""
mana.cognition.investigator — the protocol, walked without a person in it.

What a person still supplies, and what they no longer do
---------------------------------------------------------
The infrastructure, and only that:

    an oracle          a live decision and the free, exact answer to it
    a region           which knobs a change may touch
    split sizes        how much data each role gets
    quotas             how often a hidden split may be asked

What a person no longer supplies:

    which failure to attack
    what the cause is
    which candidate to try
    which candidate to seal
    what counts as success

The last of those is the one that matters. `ACCEPTANCE` is written down
here, once, and nothing in the loop may argue with it: the candidate must
beat the baseline on a split it was not chosen on, the effect must be
reproduced on a split that did not exist during the choosing, and no group
may get worse on either. An investigator that could pick its own success
criterion would find success.

Why the strategy space is a person's to declare
------------------------------------------------
A knob is a change somebody thought of and wrote down. MANA searching
`policy.KNOBS` is MANA choosing among changes it did not invent, and that
is the honest description of what this does. What it removes from the
person is everything after that choice -- and everything after that choice
is where this project has repeatedly gone wrong.

Adoption is data
-----------------
A passing verdict writes `policy.adopt`, which refuses without evidence
naming the experiment, the fresh result and the verdict. A failing one
writes a REJECTED finding and changes nothing. Either way the ledger holds
what happened, so the next run finds it there instead of rediscovering it.

`allow_adopt` is off unless a caller says otherwise. Searching and
measuring change nothing by themselves; putting the result into force is
the one step that alters how the program behaves for the person using it,
and it is theirs to permit.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .. import policy as policy_mod
from ..policy import Policy
from . import trials
from .findings import Finding, Ledger, measurement_of
from ..core.gates import ACCEPTED, REJECTED

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: A group whose accuracy is below this is a failure worth investigating.
#: Infrastructure, not a judgement the loop may revise: a threshold the
#: searcher could move is a searcher that declares its own success.
FAILING_BELOW = 0.95

#: What acceptance means here, written once. Every clause has to hold.
ACCEPTANCE = (
    "интервал парной разницы целиком выше нуля на валидации",
    "то же на свежей выборке, запечатанной до чтения",
    "ни одна группа не стала хуже ни на одной скрытой выборке",
)


@dataclass(frozen=True)
class Oracle:
    """A live decision with a free exact answer, and where it may change.

    Everything here is a person's declaration. The investigator reads it
    and never writes it: a loop that could widen its own region of change
    would be choosing what it is allowed to become.
    """
    name: str
    #: What the program actually decides, under whatever policy is in force.
    decide: Callable[[Any], str]
    #: Items to decide about: (seed, group) -> an iterable of items.
    samples: Callable[[int, str], Iterable[Any]]
    #: The right answer for a group. Free and exact, or this is not an oracle.
    truth: Callable[[str], str]
    #: The groups a regression must be checked on, separately.
    groups: Tuple[str, ...]
    #: The knobs a candidate may touch. The declared region of change.
    knobs: Tuple[str, ...]
    seeds: Optional[Dict[str, Tuple[int, ...]]] = None

    def score(self, policy: Policy, seed: int) -> Dict[str, float]:
        """Per-group accuracy under one policy on one seed."""
        out: Dict[str, float] = {}
        with policy_mod.use(policy):
            for group in self.groups:
                want = self.truth(group)
                items = list(self.samples(seed, group))
                if not items:
                    out[group] = 0.0
                    continue
                right = sum(1 for item in items if self.decide(item) == want)
                out[group] = right / len(items)
        return out

    def overall(self, policy: Policy, seed: int) -> float:
        by_group = self.score(policy, seed)
        return statistics.mean(by_group.values()) if by_group else 0.0


#: Oracles a person has declared. Explicit, like every other allowlist
#: here: a decision becomes investigable when somebody says it is.
_ORACLES: Dict[str, Oracle] = {}


def declare(oracle: Oracle) -> Oracle:
    _ORACLES[oracle.name] = oracle
    return oracle


def known() -> List[str]:
    return sorted(_ORACLES)


def oracle(name: str) -> Oracle:
    found = _ORACLES.get(name)
    if found is None:
        raise KeyError(f"оракул «{name}» не объявлен")
    return found


@dataclass
class Investigation:
    """One pass of the protocol over one decision, and where it stopped."""
    oracle: str
    failing: Dict[str, float] = field(default_factory=dict)
    candidates_tried: int = 0
    chosen: Dict[str, Any] = field(default_factory=dict)
    discovery: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    fresh: Dict[str, Any] = field(default_factory=dict)
    verdict: str = ""
    adopted: bool = False
    stopped_at: str = ""
    why: str = ""
    finding_id: str = ""

    def describe(self) -> str:
        lines = [f"расследование «{self.oracle}»"]
        if self.failing:
            lines.append("  провалы: " + ", ".join(
                f"{group} {share:.0%}" for group, share in
                sorted(self.failing.items(), key=lambda kv: kv[1])))
        else:
            lines.append("  провалов нет")
        if self.chosen:
            lines.append(f"  кандидатов рассмотрено: {self.candidates_tried}; "
                         f"выбран {self.chosen}")
        for name, result in (("валидация", self.validation),
                             ("свежая", self.fresh)):
            if result:
                lines.append(
                    f"  {name}: {result['mean']:+.4f} "
                    f"[{result['low']:+.4f}, {result['high']:+.4f}], "
                    f"худшая группа {result['worst_group']:+.4f}")
        lines.append(f"  вердикт: {self.verdict or '—'}"
                     + ("  ПРИНЯТО" if self.adopted else ""))
        if self.stopped_at:
            lines.append(f"  остановлено на «{self.stopped_at}»: {self.why}")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {"oracle": self.oracle, "failing": dict(self.failing),
                "candidates_tried": self.candidates_tried,
                "chosen": dict(self.chosen), "discovery": dict(self.discovery),
                "validation": dict(self.validation), "fresh": dict(self.fresh),
                "verdict": self.verdict, "adopted": self.adopted,
                "stopped_at": self.stopped_at, "why": self.why,
                "finding_id": self.finding_id}


def _candidate_policies(found: Oracle, current: Policy) -> List[Policy]:
    """Every setting in the declared region, one at a time and all at once.

    Coordinate-wise plus the combination, the same shape
    `cognition/candidates.py` uses for an invariant: enumerating the full
    product is a budget nobody has, and an accepted combination would not
    say which part of it did the work -- so the singles run too.
    """
    settings = current.as_dict()
    out: List[Policy] = []
    combined: Dict[str, Any] = {}
    for name in found.knobs:
        knob = policy_mod._BY_NAME.get(name)
        if knob is None:
            continue
        for option in knob.options:
            if option == settings.get(name):
                continue
            out.append(Policy.of(**dict(settings, **{name: option})))
            combined.setdefault(name, option)
    if len(combined) > 1:
        out.append(Policy.of(**dict(settings, **combined)))
    return out


def _paired_with_groups(found: Oracle, experiment: str, split: str,
                        base: Policy, candidate: Policy,
                        strategy: Optional[Dict[str, Any]] = None
                        ) -> Dict[str, Any]:
    """One read of a split: the paired difference and the per-group margins.

    Both come from the same pass. Asking twice would spend two reads on
    one question, and the no-regression check is not a second question --
    it is the same measurement, read by group. `core/gates.py` reads a
    hidden set by domain for exactly this.
    """
    collected: Dict[str, Dict[str, List[float]]] = {
        "base": {group: [] for group in found.groups},
        "candidate": {group: [] for group in found.groups}}

    def arm(policy: Policy, side: str):
        def run(seed: int) -> float:
            by_group = found.score(policy, seed)
            for group, value in by_group.items():
                collected[side][group].append(value)
            return statistics.mean(by_group.values()) if by_group else 0.0
        return run

    result = trials.paired(experiment, split, arm(base, "base"),
                           arm(candidate, "candidate"), strategy=strategy)
    margins = {
        group: (statistics.mean(collected["candidate"][group])
                - statistics.mean(collected["base"][group]))
        for group in found.groups}
    result["by_group"] = margins
    result["worst_group"] = min(margins.values()) if margins else 0.0
    return result


def investigate(name: str, ledger: Optional[Ledger] = None,
                allow_adopt: bool = False) -> Investigation:
    """Walk the whole protocol on one declared decision.

    Stops at the first stage that gives an answer, and says which. Every
    stop is a real answer: no failure, no candidate that helps, a
    candidate that does not survive a holdout, a group that got worse.
    """
    found = oracle(name)
    ledger = ledger if ledger is not None else Ledger()
    report = Investigation(oracle=name)
    experiment = trials.register(name, seeds=found.seeds)
    current = policy_mod.active()

    # ---------- 1. is anything failing, and where ----------
    base_discovery = trials.score(
        name, trials.DISCOVERY, lambda seed: found.overall(current, seed))
    by_group = _mean_by_group(found, current, experiment.seeds[trials.DISCOVERY])
    report.failing = {group: share for group, share in by_group.items()
                      if share < FAILING_BELOW}
    report.discovery = {"baseline": base_discovery["mean"],
                        "by_group": by_group}
    if not report.failing:
        report.stopped_at = "провал"
        report.why = (f"каждая группа выше {FAILING_BELOW:.0%} — расследовать "
                      f"нечего, и это исправный конец")
        return report

    # ---------- 2. what in the declared region helps ----------
    options = _candidate_policies(found, current)
    report.candidates_tried = len(options)
    scored: List[Tuple[float, float, Policy]] = []
    for option in options:
        marks = _mean_by_group(found, option, experiment.seeds[trials.DISCOVERY])
        overall = statistics.mean(marks.values()) if marks else 0.0
        worst = min((marks[g] - by_group[g] for g in found.groups), default=0.0)
        scored.append((overall, worst, option))
    # Better overall AND nothing worse: the same rule the holdouts will
    # apply, used here so a candidate that could not pass is not sent.
    viable = [row for row in scored
              if row[0] > base_discovery["mean"] and row[1] >= 0.0]
    if not viable:
        report.stopped_at = "кандидат"
        report.why = ("в объявленной области нет настройки, которая "
                      "улучшает и никого не ломает")
        return report
    viable.sort(key=lambda row: (row[0], row[1]), reverse=True)
    best_overall, best_worst, chosen = viable[0]
    report.chosen = chosen.changes()
    report.discovery.update({"candidate": best_overall,
                             "worst_group": best_worst})

    # ---------- 3. a split it was not chosen on ----------
    report.validation = _paired_with_groups(
        found, name, trials.VALIDATION, current, chosen)
    if not report.validation["improved"] or report.validation["worst_group"] < 0:
        report.verdict = REJECTED
        report.stopped_at = "валидация"
        report.why = _why_not(report.validation)
        report.finding_id = _record(ledger, found, report, chosen)
        return report

    # ---------- 4. a split that did not exist during the choosing ----------
    strategy = dict(chosen.changes(), oracle=name)
    trials.seal(name, strategy)
    report.fresh = _paired_with_groups(found, name, trials.FRESH, current,
                                       chosen, strategy=strategy)
    if not report.fresh["improved"] or report.fresh["worst_group"] < 0:
        report.verdict = REJECTED
        report.stopped_at = "свежая"
        report.why = _why_not(report.fresh)
        report.finding_id = _record(ledger, found, report, chosen)
        return report

    # ---------- 5. the verdict, and only then the change ----------
    report.verdict = ACCEPTED
    report.finding_id = _record(ledger, found, report, chosen)
    if allow_adopt:
        policy_mod.adopt(chosen, {
            "experiment": name,
            "fresh": {"mean": round(report.fresh["mean"], 4),
                      "interval": [round(report.fresh["low"], 4),
                                   round(report.fresh["high"], 4)],
                      "worst_group": round(report.fresh["worst_group"], 4),
                      "n": report.fresh["n"]},
            "verdict": ACCEPTED, "acceptance": list(ACCEPTANCE),
            "finding_id": report.finding_id})
        report.adopted = True
    else:
        report.why = ("принятие не разрешено вызывающим: изменение "
                      "измерено и не введено в силу")
    return report


def _mean_by_group(found: Oracle, policy: Policy,
                   seeds: Sequence[int]) -> Dict[str, float]:
    totals: Dict[str, List[float]] = {group: [] for group in found.groups}
    for seed in seeds:
        for group, value in found.score(policy, seed).items():
            totals[group].append(value)
    return {group: statistics.mean(values) if values else 0.0
            for group, values in totals.items()}


def _why_not(result: Dict[str, Any]) -> str:
    if result["worst_group"] < 0:
        worst = min(result["by_group"], key=lambda g: result["by_group"][g])
        return (f"группа «{worst}» стала хуже на "
                f"{result['by_group'][worst]:+.4f}")
    return (f"интервал {result['low']:+.4f}..{result['high']:+.4f} "
            f"накрывает ноль")


def _record(ledger: Ledger, found: Oracle, report: Investigation,
            chosen: Policy) -> str:
    result = report.fresh or report.validation
    finding = Finding(
        question=f"Какая настройка в объявленной области чинит «{found.name}»?",
        approach=dict(chosen.changes(), domain=found.name),
        verdict=report.verdict or REJECTED,
        measurement=measurement_of(
            trials=int(result.get("n") or 0),
            interval=(round(result.get("low", 0.0), 4),
                      round(result.get("high", 0.0), 4)),
            null=0.0),
        conditions={"oracle": found.name, "stopped_at": report.stopped_at,
                    "split": result.get("split", ""),
                    "groups": list(found.groups)},
        note=("найдено и проведено без человека: провал обнаружен по "
              "оракулу, кандидаты перебраны в объявленной области, "
              "критерий приёмки задан заранее. " + (report.why or "")),
    )
    return finding.finding_id if ledger.record(finding) else ""
