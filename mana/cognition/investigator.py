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

from contextlib import nullcontext

from .. import policy as policy_mod
from ..policy import Policy
from . import rules as rules_mod
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
class Change:
    """Something that can be put in force for the length of a measurement.

    Two kinds so far: a policy setting somebody declared, and a rule MANA
    constructed from a diagnosis. They have the same obligations, so they
    take the same path -- a second path beside this one is how the second
    kind would end up held to an easier bar.
    """
    label: str
    kind: str                        # policy | rule | baseline
    settings: Dict[str, Any] = field(default_factory=dict)
    policy: Optional[Policy] = None
    rule: Optional[Any] = None
    #: The claim this came from, when it came from one. Empty for a knob.
    hypothesis: Dict[str, Any] = field(default_factory=dict)

    def apply(self):
        """In force on this thread, for this measurement only."""
        if self.kind == "policy" and self.policy is not None:
            return policy_mod.use(self.policy)
        if self.kind == "rule" and self.rule is not None:
            return rules_mod.Proposed(self.rule)
        # The baseline is current behaviour, not a reconstruction of it.
        return nullcontext()

    def as_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "kind": self.kind,
                "settings": dict(self.settings),
                "hypothesis": dict(self.hypothesis)}


#: What the candidate is measured against: whatever the program does now.
BASELINE = Change(label="как сейчас", kind="baseline")


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
    #: Where a constructed rule belongs, in the decision's own vocabulary.
    #: Separate from `name` because the decision has an identity that does
    #: not change when a second experiment looks at it: a rule found while
    #: investigating `task_naming_built` is still a rule about task
    #: naming, and the classifier asks for that scope and no other.
    scope: str = ""

    @property
    def rule_scope(self) -> str:
        return self.scope or self.name

    def score(self, change: Change, seed: int) -> Dict[str, float]:
        """Per-group accuracy under one change on one seed."""
        out: Dict[str, float] = {}
        with change.apply():
            for group in self.groups:
                want = self.truth(group)
                items = list(self.samples(seed, group))
                if not items:
                    out[group] = 0.0
                    continue
                right = sum(1 for item in items if self.decide(item) == want)
                out[group] = right / len(items)
        return out

    def overall(self, change: Change, seed: int) -> float:
        by_group = self.score(change, seed)
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
    #: The claim the chosen change came from, when it came from one.
    hypothesis: Dict[str, Any] = field(default_factory=dict)
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
        if self.hypothesis:
            lines.append(f"  гипотеза: {self.hypothesis.get('claim', '')}")
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
                "chosen": dict(self.chosen),
                "hypothesis": dict(self.hypothesis),
                "discovery": dict(self.discovery),
                "validation": dict(self.validation), "fresh": dict(self.fresh),
                "verdict": self.verdict, "adopted": self.adopted,
                "stopped_at": self.stopped_at, "why": self.why,
                "finding_id": self.finding_id}


def from_knobs(found: Oracle, seeds: Sequence[int]) -> List[Change]:
    """Every setting in the declared region, one at a time and all at once.

    Coordinate-wise plus the combination, the same shape
    `cognition/candidates.py` uses for an invariant: enumerating the full
    product is a budget nobody has, and an accepted combination would not
    say which part of it did the work -- so the singles run too.
    """
    current = policy_mod.active()
    settings = current.as_dict()
    out: List[Change] = []
    combined: Dict[str, Any] = {}
    for name in found.knobs:
        knob = policy_mod._BY_NAME.get(name)
        if knob is None:
            continue
        for option in knob.options:
            if option == settings.get(name):
                continue
            made = Policy.of(**dict(settings, **{name: option}))
            out.append(Change(label=made.describe(), kind="policy",
                              settings=made.changes(), policy=made))
            combined.setdefault(name, option)
    if len(combined) > 1:
        made = Policy.of(**dict(settings, **combined))
        out.append(Change(label=made.describe(), kind="policy",
                          settings=made.changes(), policy=made))
    return out


def from_hypotheses(found: Oracle, seeds: Sequence[int]) -> List[Change]:
    """Watch it fail, say why, and build what the claim licenses.

    The other side of the line: a knob is a change somebody wrote down,
    and this is a change constructed from a diagnosed conflict. The
    claims come from `cognition/hypothesis.py`, which has no model in it
    and a deliberately narrow space -- a search wide enough will beat a
    holdout by luck, so the number of forms per claim is capped there.
    """
    from . import hypothesis as hypothesis_mod

    def samples(group):
        for seed in seeds:
            for item in found.samples(seed, group):
                yield item

    def as_text(item) -> str:
        return getattr(item, "prompt", item) if not isinstance(item, str) else item

    def texts(group):
        return (as_text(item) for item in samples(group))

    claims = hypothesis_mod.observe(
        found.rule_scope, texts, found.groups, found.truth,
        lambda text: found.decide(_Textish(text)))
    out: List[Change] = []
    for claim in claims:
        for rule in hypothesis_mod.forms(claim, texts, found.groups):
            checked, note = _executable(found, rule, seeds)
            if checked is None:
                # Not a weak candidate -- not the rule the miner claims.
                # Discarded here rather than measured, because a
                # candidate that cannot be applied should not cost a
                # discovery pass.
                continue
            out.append(Change(
                label=checked.describe(), kind="rule",
                settings={"marker": checked.marker, "decides": checked.decides},
                rule=checked, hypothesis=claim.as_dict()))
    return out


#: How many seeds the executability check looks at. Two is enough to see
#: whether a marker fires at all, and the point is to spend nothing.
EXECUTABILITY_SEEDS = 2


def _executable(found: Oracle, rule: Any, seeds: Sequence[int]):
    """Does this fire through the real matcher, and does it move a decision?

    Returns the rule with its check recorded, or None and why. Two
    different failures: a marker that never matches (the representation
    the miner used is not the one the matcher uses) and a marker that
    matches but changes no answer (it is seated where it cannot matter).
    """
    from dataclasses import replace

    group = (rule.provenance or {}).get("group")
    wanted = (rule.provenance or {}).get("wanted")
    if not group or not wanted:
        return None, "у кандидата нет происхождения — нечем проверить"

    matched = moved = looked = 0
    candidate = Change(label=rule.describe(), kind="rule",
                       settings={"marker": rule.marker}, rule=rule)
    for seed in list(seeds)[:EXECUTABILITY_SEEDS]:
        for item in found.samples(seed, group):
            looked += 1
            text = getattr(item, "prompt", item)
            if rule.matches(str(text).lower()):
                matched += 1
            before = found.decide(item)
            with candidate.apply():
                after = found.decide(item)
            if before != after and after == wanted:
                moved += 1
    if not looked:
        return None, "не на чем проверить"
    if not matched:
        return None, (f"сопоставитель не срабатывает ни разу из {looked}: "
                      f"представление, которым добывали, — не то, которым "
                      f"сопоставляют")
    if not moved:
        return None, (f"совпадает {matched} раз из {looked} и не меняет ни "
                      f"одного ответа — стоит там, где не решает")
    return replace(rule, provenance=dict(
        rule.provenance,
        matcher="Rule.matches + Oracle.decide",
        checked_on=f"{looked} задач с {EXECUTABILITY_SEEDS} посевов открытия",
        matched=matched, moved=moved)), ""


class _Textish(str):
    """A string that also answers to `.prompt`.

    The oracle's `decide` takes whatever `samples` yields, and the
    hypothesis layer works in text. Rather than make every oracle carry a
    text extractor, the text is handed back in a shape both accept.
    """

    @property
    def prompt(self) -> str:
        return str(self)

    @property
    def difficulty(self) -> Optional[float]:
        return None


def _paired_with_groups(found: Oracle, experiment: str, split: str,
                        base: Change, candidate: Change,
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

    def arm(change: Change, side: str):
        def run(seed: int) -> float:
            by_group = found.score(change, seed)
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
                allow_adopt: bool = False,
                propose: Optional[Callable[[Oracle, Sequence[int]],
                                           List[Change]]] = None
                ) -> Investigation:
    """Walk the whole protocol on one declared decision.

    `propose` decides where candidates come from: `from_knobs` searches
    the region a person declared, `from_hypotheses` diagnoses the failure
    and builds what its claim licenses. Neither gets a say in what counts
    as success.

    Stops at the first stage that gives an answer, and says which. Every
    stop is a real answer: no failure, no candidate that helps, a
    candidate that does not survive a holdout, a group that got worse.
    """
    found = oracle(name)
    ledger = ledger if ledger is not None else Ledger()
    propose = propose or from_knobs
    report = Investigation(oracle=name)
    experiment = trials.register(name, seeds=found.seeds)
    current = BASELINE

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

    # ---------- 2. what helps, from wherever candidates come from ----------
    options = propose(found, experiment.seeds[trials.DISCOVERY])
    report.candidates_tried = len(options)
    scored: List[Tuple[float, float, Change]] = []
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
    # Measured first: overall, then the worst group. Among candidates the
    # protocol cannot tell apart, prefer the one that depends least on a
    # bare token -- `accidental` is the share of matches that survive
    # shuffling the words, and a phrase is strictly harder to hit by
    # accident than a single word in any text, not only in this corpus.
    # A tiebreak among measured equals, on a measured number; not a proxy
    # standing in for specificity, which this corpus cannot measure.
    def _order(row):
        overall, worst, change = row
        rule = getattr(change, "rule", None)
        accidental = getattr(rule, "accidental", 0.0) if rule else 0.0
        return (overall, worst, -accidental)

    viable.sort(key=_order, reverse=True)
    best_overall, best_worst, chosen = viable[0]
    report.chosen = dict(chosen.settings)
    report.hypothesis = dict(chosen.hypothesis)
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
    strategy = dict(chosen.settings, oracle=name, kind=chosen.kind)
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
        evidence = {
            "experiment": name,
            "fresh": {"mean": round(report.fresh["mean"], 4),
                      "interval": [round(report.fresh["low"], 4),
                                   round(report.fresh["high"], 4)],
                      "worst_group": round(report.fresh["worst_group"], 4),
                      "n": report.fresh["n"]},
            "verdict": ACCEPTED, "acceptance": list(ACCEPTANCE),
            "finding_id": report.finding_id}
        if chosen.hypothesis:
            evidence["hypothesis"] = dict(chosen.hypothesis)
        if chosen.kind == "rule":
            # A rule MANA wrote about its own behaviour. Same bar as a
            # setting, and `install` refuses without the evidence too.
            rules_mod.install(chosen.rule, evidence)
        else:
            policy_mod.adopt(chosen.policy, evidence)
        report.adopted = True
    else:
        report.why = ("принятие не разрешено вызывающим: изменение "
                      "измерено и не введено в силу")
    return report


def _mean_by_group(found: Oracle, change: Change,
                   seeds: Sequence[int]) -> Dict[str, float]:
    totals: Dict[str, List[float]] = {group: [] for group in found.groups}
    for seed in seeds:
        for group, value in found.score(change, seed).items():
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
            chosen: Change) -> str:
    result = report.fresh or report.validation
    asked = ("Какое построенное правило чинит «{}»?" if chosen.kind == "rule"
             else "Какая настройка в объявленной области чинит «{}»?")
    finding = Finding(
        question=asked.format(found.name),
        approach=dict(chosen.settings, domain=found.name, kind=chosen.kind),
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
