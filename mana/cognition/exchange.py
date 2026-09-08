"""
mana.cognition.exchange — what may pass between two installations.

The rule this module exists to enforce
---------------------------------------
**Hypotheses cross. Verdicts do not.**

MANA's acceptance is worth something because every accepted claim was
measured *here*: a paired experiment, McNemar, a minimum absolute margin,
against a hidden set this installation drew for itself. Importing another
instance's verdict throws all of that away. Its brains differ, its
hardware differs, its task distribution differs, and since phase 25 its
hidden set differs too. "Accepted there" is not evidence here, and a
knowledge store that pools verdicts turns a system of proof into a system
of rumour.

So the bridge carries *what to try*. Each instance re-derives the
candidate against its own genome and rules on it with its own gates.

Three things follow, and none of them is obvious
-------------------------------------------------
**Poisoning solves itself.** If only hypotheses cross, a malicious or
merely broken instance cannot corrupt anything: a bad hypothesis fails a
local gate like any other. No trust is required. Signatures would be for
spam, not for correctness -- a far weaker requirement than federated
systems usually carry.

**The expensive half is what gets shared.** A research cycle spends most
of its budget deciding *what* to measure; the measuring is comparatively
cheap. Sharing hypotheses multiplies the costly part and leaves the
trustworthy part local.

**External validity becomes reachable.** One instance can only know "true
here". Twelve can know "accepted in nine, and the three exceptions all
lack a local model" -- which is the difference between a result and a
finding. Negative results matter most here: REJECTED and NOT_EVALUATED
are first-class verdicts in this project, and a store of "tried, did not
work" is the thing human research systematically loses.

What is allowed to be in a hypothesis
--------------------------------------
A mutation name and its parameters, and nothing else. Parameters must be
JSON scalars, or lists of them, and every string must look like an
identifier -- letters, digits, `_`, `.`, `-`, no spaces. That is an
allowlist over shape: an operator name passes, a domain name passes, a
task someone typed does not, and neither does a business object name with
a space in it.

`rationale` is deliberately NOT exported. It is generated locally from
local observations and is the one field likely to quote them. A receiving
instance gains little from it anyway -- it is about to derive its own
reason from its own evidence. `note` exists instead: optional, and meant
to be written by a person who knows what they are publishing.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .genome import MUTATIONS, CognitiveGenome, MutationProposal, propose

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Bundles say what they are, so a future change can refuse an old file
#: instead of misreading it.
FORMAT = "mana.exchange"
FORMAT_VERSION = 1

#: The shape a parameter string may have. An allowlist: anything that is
#: not plainly an identifier is refused, including anything nobody
#: anticipated. A denylist here would be the same mistake this project
#: already documents in its sandbox policy.
IDENTIFIER = re.compile(r"^[A-Za-z0-9_.\-:+]{1,120}$")

#: Verdicts a report may carry. Mirrors the gates; anything else is a
#: record from a version that means something we do not know.
VERDICTS = ("ACCEPTED", "REJECTED", "NOT_EVALUATED")


class ExchangeError(ValueError):
    """A record cannot be exported or imported. Never raised for a record
    that is merely unwelcome -- only for one that is malformed."""


# ------------------------------------------------------------------ vetting


def check_shareable(params: Dict[str, Any]) -> None:
    """Refuse parameters that could carry local data off the machine.

    Raises rather than filtering. Silently dropping a field would produce
    a hypothesis that means something other than what was proposed, and a
    receiving instance would measure the wrong thing without either side
    noticing.
    """
    if not isinstance(params, dict):
        raise ExchangeError("параметры должны быть словарём")
    for key, value in sorted(params.items()):
        if not IDENTIFIER.match(str(key)):
            raise ExchangeError(f"имя параметра {key!r} не похоже на идентификатор")
        for item in (value if isinstance(value, (list, tuple)) else [value]):
            if isinstance(item, bool) or isinstance(item, (int, float)) or item is None:
                continue
            if isinstance(item, str):
                if not IDENTIFIER.match(item):
                    raise ExchangeError(
                        f"значение {key}={item!r} не похоже на идентификатор: "
                        f"через мост не проходит свободный текст, потому что "
                        f"в нём уезжают чужие данные")
                continue
            raise ExchangeError(
                f"{key}: тип {type(item).__name__} не передаётся; "
                f"допустимы числа, строки-идентификаторы, списки из них")


#: Parameters that document a change rather than determine it.
#:
#: `check_shareable` refuses free text, and it is right to: a task
#: somebody typed must not leave. But `create_program_template` carries a
#: `description` written for a human -- "синтезировано из открытия ..." --
#: and dropping it changes nothing about what the template DOES. The
#: distinction is behavioural versus annotative, and it has to be explicit:
#: silently dropping a behavioural parameter would produce a hypothesis
#: that means something other than what was proposed, and the receiving
#: instance would measure the wrong thing without either side noticing.
ANNOTATIVE = frozenset({"description", "rationale", "reason", "note", "comment"})


def try_hypothesis(mutation: str, params: Dict[str, Any], note: str = ""
                   ) -> Tuple[Optional["Hypothesis"], str]:
    """Build a shareable hypothesis, or say why it is not one.

    Returns a reason instead of raising because the caller is a research
    cycle mid-run: a proposal that cannot be shared is still a proposal
    worth pursuing locally, and an exception here would stop the work to
    report on a side concern.
    """
    behavioural = {k: v for k, v in (params or {}).items()
                   if k not in ANNOTATIVE}
    try:
        return Hypothesis(mutation=mutation, params=behavioural, note=note), ""
    except ExchangeError as exc:
        return None, str(exc)


# ------------------------------------------------------------------ records


@dataclass(frozen=True)
class Hypothesis:
    """Something to try, addressed to no genome in particular."""
    mutation: str
    params: Dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        if self.mutation not in MUTATIONS:
            raise ExchangeError(f"неизвестная мутация {self.mutation!r}")
        check_shareable(self.params)

    @property
    def hypothesis_id(self) -> str:
        """Derived from the content, not assigned.

        Two instances that independently think of the same change produce
        the same id, which is what makes deduplication and replication
        counting work at all. A random id would make every rediscovery
        look like a new idea.
        """
        canonical = json.dumps({"mutation": self.mutation, "params": self.params},
                               sort_keys=True, ensure_ascii=False,
                               separators=(",", ":"))
        return hashlib.blake2b(canonical.encode("utf-8"),
                               digest_size=8).hexdigest()

    def as_dict(self) -> Dict[str, Any]:
        return {"hypothesis_id": self.hypothesis_id, "mutation": self.mutation,
                "params": self.params, "note": self.note}


@dataclass(frozen=True)
class Report:
    """One instance's verdict on one hypothesis.

    Carried across the bridge, and never imported as truth. It says how a
    change fared somewhere else, which is worth knowing when deciding
    what to measure next and worth nothing as evidence here.
    """
    hypothesis_id: str
    instance: str                     # fingerprint, not the id itself
    verdict: str
    trials: int = 0
    discordant: int = 0
    p_value: float = 1.0
    margin: float = 0.0
    #: The holdout identity, e.g. "v1+3f2a91bb". Present precisely so a
    #: reader can see this was a DIFFERENT hidden set from their own.
    holdout: str = ""
    created: float = field(default_factory=time.time)
    #: Ed25519 public key and signature over `payload()`.
    #:
    #: `instance` used to be a self-declared string, so one installation
    #: could invent twelve fingerprints and manufacture "confirmed on
    #: twelve instances" -- a Sybil attack on the replication count, which
    #: was the one real hole the federation had. A signed report provably
    #: comes from whoever it names.
    #:
    #: What this does NOT stop is one person generating twelve key pairs.
    #: Cryptocurrencies answer that with proof of work, which is expensive
    #: because they must agree among parties who cannot check a claim.
    #: MANA checks locally, so it never needs agreement -- what limits
    #: Sybil here is deciding whose keys count, the way known_hosts does.
    public_key: str = ""
    signature: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ExchangeError(f"неизвестный вердикт {self.verdict!r}")

    def payload(self) -> Dict[str, Any]:
        """Exactly what the signature covers: the claim, without itself."""
        return {"hypothesis_id": self.hypothesis_id, "instance": self.instance,
                "verdict": self.verdict, "trials": self.trials,
                "discordant": self.discordant, "p_value": self.p_value,
                "margin": self.margin, "holdout": self.holdout,
                "created": self.created}

    def sign(self) -> "Report":
        """A copy signed with this installation's key."""
        from ..core.identity import signed
        key, signature = signed(self.payload())
        return replace(self, public_key=key, signature=signature)

    @property
    def signed_by_someone(self) -> bool:
        return bool(self.public_key and self.signature)

    @property
    def verified(self) -> bool:
        """Signature valid AND the fingerprint really is that key's.

        Both halves are required. Checking only the signature would let
        somebody sign with their own key while claiming another
        instance's fingerprint, which is the impersonation this exists to
        stop.
        """
        if not self.signed_by_someone:
            return False
        from ..core.identity import fingerprint, signature_holds
        if self.instance and self.instance != fingerprint(self.public_key):
            return False
        return signature_holds(self.payload(), self.public_key, self.signature)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.payload(), public_key=self.public_key,
                    signature=self.signature)


# ------------------------------------------------------------------ bundles


def export_bundle(path: Any, hypotheses: Sequence[Hypothesis] = (),
                  reports: Sequence[Report] = ()) -> Dict[str, Any]:
    """Write a bundle. Every record is vetted again on the way out.

    Vetted twice on purpose: `Hypothesis` checks at construction, and a
    record could have been built before a rule tightened. The file is the
    thing that leaves, so the file is where the last check belongs.
    """
    from ..core.identity import fingerprint

    for hypothesis in hypotheses:
        check_shareable(hypothesis.params)

    bundle = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "instance": fingerprint(),
        "created": time.time(),
        "hypotheses": [h.as_dict() for h in hypotheses],
        "reports": [r.as_dict() for r in reports],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(bundle, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return {"path": str(target), "hypotheses": len(hypotheses),
            "reports": len(reports), "instance": bundle["instance"]}


@dataclass
class ImportResult:
    """What arrived, and what was turned away with the reason.

    Refusals are returned rather than raised: one malformed record in a
    bundle of two hundred should not discard the other hundred and
    ninety-nine, and a silent drop would leave nobody able to find out
    why a hypothesis never showed up.
    """
    hypotheses: List[Hypothesis] = field(default_factory=list)
    reports: List[Report] = field(default_factory=list)
    refused: List[Dict[str, str]] = field(default_factory=list)
    instance: str = ""

    def summary(self) -> Dict[str, Any]:
        return {"instance": self.instance, "hypotheses": len(self.hypotheses),
                "reports": len(self.reports), "refused": len(self.refused)}


def import_bundle(path: Any, known: Iterable[str] = ()) -> ImportResult:
    """Read a bundle, vetting every record as untrusted input.

    It is untrusted: it came from another machine. Nothing here adopts
    anything -- see the module note. What comes back is a list of things
    to *try*, and the local gates decide the rest.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("format") != FORMAT:
        raise ExchangeError(f"не пакет обмена MANA: format={raw.get('format')!r}")
    if int(raw.get("format_version", 0)) > FORMAT_VERSION:
        raise ExchangeError(
            f"пакет версии {raw.get('format_version')} новее, чем понимает "
            f"эта MANA ({FORMAT_VERSION}); обновитесь, а не угадывайте")

    result = ImportResult(instance=str(raw.get("instance", "")))
    seen = set(known)

    for record in raw.get("hypotheses") or []:
        try:
            hypothesis = Hypothesis(
                mutation=str(record.get("mutation", "")),
                params=dict(record.get("params") or {}),
                note=str(record.get("note", ""))[:500])
        except (ExchangeError, TypeError, ValueError) as exc:
            result.refused.append({"kind": "hypothesis", "reason": str(exc)})
            continue
        # A content-derived id that disagrees with the claimed one means
        # the record was edited in transit, deliberately or not.
        claimed = str(record.get("hypothesis_id", ""))
        if claimed and claimed != hypothesis.hypothesis_id:
            result.refused.append(
                {"kind": "hypothesis",
                 "reason": f"идентификатор не сходится с содержимым: "
                           f"заявлен {claimed}, вычислен {hypothesis.hypothesis_id}"})
            continue
        if hypothesis.hypothesis_id in seen:
            continue                       # already known; not a refusal
        seen.add(hypothesis.hypothesis_id)
        result.hypotheses.append(hypothesis)

    for record in raw.get("reports") or []:
        try:
            report = Report(
                hypothesis_id=str(record.get("hypothesis_id", "")),
                instance=str(record.get("instance", "")),
                verdict=str(record.get("verdict", "")),
                trials=int(record.get("trials", 0)),
                discordant=int(record.get("discordant", 0)),
                p_value=float(record.get("p_value", 1.0)),
                margin=float(record.get("margin", 0.0)),
                holdout=str(record.get("holdout", "")),
                created=float(record.get("created", time.time())),
                public_key=str(record.get("public_key", "")),
                signature=str(record.get("signature", "")))
        except (ExchangeError, TypeError, ValueError) as exc:
            result.refused.append({"kind": "report", "reason": str(exc)})
            continue
        # A signature that is present and wrong means the record was
        # altered, or someone signed a claim they are not entitled to.
        # Either way it is refused; an ABSENT signature is merely old,
        # and is kept unverified rather than thrown away.
        if report.signed_by_someone and not report.verified:
            result.refused.append(
                {"kind": "report",
                 "reason": f"подпись не сходится для {report.instance or '?'}: "
                           f"запись изменена или подписана чужим ключом"})
            continue
        result.reports.append(report)

    return result


# -------------------------------------------------------------- replication


def replication(reports: Sequence[Report], verified_only: bool = False
                ) -> Dict[str, Dict[str, Any]]:
    """How each hypothesis fared, per hypothesis, across instances.

    Counted over DISTINCT instances rather than over reports: an instance
    that ran the same experiment ten times has ten pieces of evidence and
    still one environment, and letting it vote ten times would turn
    persistence into agreement.

    This is the only thing here a single installation cannot compute for
    itself, and it is the point of the whole exercise.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for report in reports:
        if verified_only and not report.verified:
            continue
        entry = out.setdefault(report.hypothesis_id, {
            "instances": {}, "holdouts": set()})
        # Latest report from an instance wins: a re-run with more trials
        # supersedes the earlier, weaker one from the same environment.
        previous = entry["instances"].get(report.instance)
        if previous is None or report.created >= previous.created:
            entry["instances"][report.instance] = report
        if report.holdout:
            entry["holdouts"].add(report.holdout)

    summary: Dict[str, Dict[str, Any]] = {}
    for hypothesis_id, entry in out.items():
        verdicts = [r.verdict for r in entry["instances"].values()]
        accepted = verdicts.count("ACCEPTED")
        summary[hypothesis_id] = {
            "instances": len(verdicts),
            "accepted": accepted,
            "rejected": verdicts.count("REJECTED"),
            "not_evaluated": verdicts.count("NOT_EVALUATED"),
            # Reported as a fraction of instances that actually RULED.
            # Counting NOT_EVALUATED as disagreement would read "nobody
            # had the power to test it" as "it does not work".
            "ruled": accepted + verdicts.count("REJECTED"),
            "distinct_holdouts": len(entry["holdouts"]),
        }
    return summary


#: How many ruling instances it takes before agreement is worth calling
#: agreement. Two is not a pattern; it is two.
MIN_FOR_CONSENSUS = 3


def consensus(reports: Sequence[Report], verified_only: bool = False
              ) -> Dict[str, List[Dict[str, Any]]]:
    """Sort hypotheses into what the federation actually learned.

    `replication` counts votes. This says what the counts mean, and it
    exists because the interesting category is easy to miss in a table:

      divergent  -- accepted somewhere and rejected somewhere else. The
                    most valuable outcome there is. It says the change is
                    conditional, and the condition is a fact about
                    environments that no single instance could have seen.
      confirmed  -- ruled by enough instances, all accepting.
      refuted    -- ruled by enough instances, all rejecting. Worth as
                    much as `confirmed` and normally thrown away: this is
                    the result human research systematically loses.
      undecided  -- too few rulings to say anything, usually because
                    NOT_EVALUATED dominates. Reported rather than folded
                    into `refuted`, because "nobody could test it" is not
                    "it does not work".

    Divergence is flagged from two instances, while agreement needs three.
    That asymmetry is deliberate: one contradiction is enough to know a
    claim is conditional, whereas two instances agreeing is a coincidence
    with a small sample.
    """
    out: Dict[str, List[Dict[str, Any]]] = {
        "divergent": [], "confirmed": [], "refuted": [], "undecided": []}
    for hypothesis_id, row in sorted(
            replication(reports, verified_only=verified_only).items()):
        entry = dict(row, hypothesis_id=hypothesis_id)
        if row["accepted"] and row["rejected"]:
            out["divergent"].append(entry)
        elif row["ruled"] < MIN_FOR_CONSENSUS:
            out["undecided"].append(entry)
        elif row["rejected"] == 0:
            out["confirmed"].append(entry)
        else:
            out["refuted"].append(entry)
    return out


# ------------------------------------------------------- back into the local


def to_local_proposal(hypothesis: Hypothesis, parent: CognitiveGenome,
                      note: str = "") -> MutationProposal:
    """Re-derive the candidate against THIS installation's genome.

    The mutation travels; the candidate does not. Applying a foreign
    genome would import someone else's whole cognitive state along with
    the one change being tested -- and the change is the only part
    anybody proposed.

    Raises if the mutation does not apply here, which is a real answer:
    "compose these two operators" is meaningless in a genome that has
    neither, and that is worth learning before spending trials on it.
    """
    rationale = note or hypothesis.note or (
        f"гипотеза извне ({hypothesis.hypothesis_id}); "
        f"принимается только по локальным доказательствам")
    return propose(parent, hypothesis.mutation, rationale=rationale,
                   **hypothesis.params)


# ----------------------------------------------------------------- the queue


class Queue:
    """Everything this installation has to offer, and what it heard back.

    Kept as one file rather than two so that a hypothesis and the reports
    about it cannot drift apart across a backup or a copy.

    A hypothesis that could not be made shareable is stored anyway, marked,
    with the reason. It is still something to try here, and dropping it
    would lose local work to a federation concern; `export` filters, so
    nothing unshareable can leave by accident.
    """

    def __init__(self, path: Any) -> None:
        self.path = Path(path)

    def _load(self) -> Dict[str, Any]:
        if not self.path.is_file():
            return {"hypotheses": [], "reports": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"hypotheses": [], "reports": []}
        data.setdefault("hypotheses", [])
        data.setdefault("reports", [])
        return data

    def _save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Written through a temporary file: a cycle that dies mid-write
        # would otherwise leave a truncated queue, and the next run would
        # read it as empty and quietly start over.
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        temporary.replace(self.path)

    def record_hypothesis(self, mutation: str, params: Dict[str, Any],
                          note: str = "") -> Dict[str, Any]:
        """Add one, unless it is already here.

        Deduplicated by the content-derived id, so a cycle that rediscovers
        the same change on three separate runs contributes it once.
        """
        hypothesis, refusal = try_hypothesis(mutation, params, note)
        data = self._load()
        if hypothesis is None:
            record = {"hypothesis_id": "", "mutation": mutation,
                      "shareable": False, "refusal": refusal}
            data["hypotheses"].append(record)
            self._save(data)
            return record

        existing = {h.get("hypothesis_id") for h in data["hypotheses"]}
        if hypothesis.hypothesis_id in existing:
            return {"hypothesis_id": hypothesis.hypothesis_id, "known": True}
        record = dict(hypothesis.as_dict(), shareable=True)
        data["hypotheses"].append(record)
        self._save(data)
        return record

    def record_report(self, hypothesis_id: str, verdict: str, **fields: Any
                      ) -> Dict[str, Any]:
        """Add this installation's own verdict on a hypothesis."""
        from ..core.identity import fingerprint
        report = Report(hypothesis_id=hypothesis_id, instance=fingerprint(),
                        verdict=verdict, **fields)
        try:
            report = report.sign()
        except Exception:
            # An unsigned report is still this installation's own record.
            # Losing the verdict because a key could not be read would be
            # the worse failure.
            pass
        data = self._load()
        data["reports"].append(report.as_dict())
        self._save(data)
        return report.as_dict()

    def hypotheses(self, shareable_only: bool = False) -> List[Hypothesis]:
        out = []
        for record in self._load()["hypotheses"]:
            if shareable_only and not record.get("shareable", True):
                continue
            try:
                out.append(Hypothesis(mutation=record["mutation"],
                                      params=record.get("params") or {},
                                      note=record.get("note", "")))
            except (ExchangeError, KeyError):
                continue
            except TypeError:
                continue
        return out

    def reports(self) -> List[Report]:
        out = []
        for record in self._load()["reports"]:
            try:
                out.append(Report(**record))
            except (ExchangeError, TypeError, ValueError):
                continue
        return out

    def known_ids(self) -> set:
        return {h.get("hypothesis_id") for h in self._load()["hypotheses"]
                if h.get("hypothesis_id")}

    def stats(self) -> Dict[str, Any]:
        data = self._load()
        shareable = [h for h in data["hypotheses"] if h.get("shareable", True)]
        return {"hypotheses": len(data["hypotheses"]),
                "shareable": len(shareable),
                "not_shareable": len(data["hypotheses"]) - len(shareable),
                "reports": len(data["reports"])}

    def absorb(self, result: "ImportResult") -> Dict[str, Any]:
        """Take in what an import produced. Hypotheses only become known --
        nothing here adopts anything."""
        data = self._load()
        existing = {h.get("hypothesis_id") for h in data["hypotheses"]}
        added = 0
        for hypothesis in result.hypotheses:
            if hypothesis.hypothesis_id in existing:
                continue
            data["hypotheses"].append(dict(hypothesis.as_dict(), shareable=True,
                                           foreign=True))
            existing.add(hypothesis.hypothesis_id)
            added += 1
        for report in result.reports:
            data["reports"].append(report.as_dict())
        self._save(data)
        return {"added": added, "reports": len(result.reports)}


# ---------------------------------------------------------- the command line


def default_queue_path() -> Path:
    from ..paths import data_root
    return Path(data_root()) / "exchange_queue.json"


def command_line(argv: Sequence[str]) -> int:
    """`export FILE`, `import FILE`, or nothing for a summary.

    Lives here rather than in a script because the packaged application
    needs it too: an installed MANA had no way to export or import at
    all, so every installation that was not also a source checkout could
    not join the federation. A second copy of this in a script is how the
    two would come to disagree.
    """
    from ..core import identity

    queue = Queue(default_queue_path())
    action = (argv[0] if argv else "show").lower()
    target = argv[1] if len(argv) > 1 else ""

    if action == "export":
        if not target:
            print("нужно имя файла: --exchange export пакет.json")
            return 2
        # shareable_only: a hypothesis that could not be vetted stays
        # here, and must not leave by accident.
        written = export_bundle(target, queue.hypotheses(shareable_only=True),
                                queue.reports())
        print(f"записано: {written['path']}")
        print(f"  гипотез: {written['hypotheses']}, отчётов: {written['reports']}")
        print(f"  от экземпляра: {written['instance']}")
        return 0

    if action == "import":
        if not target:
            print("нужно имя файла: --exchange import пакет.json")
            return 2
        try:
            result = import_bundle(target, known=queue.known_ids())
        except ExchangeError as exc:
            print(f"пакет не принят: {exc}")
            return 1
        queue.absorb(result)
        print(f"от экземпляра {result.instance or '(не указан)'}:")
        print(f"  новых гипотез: {len(result.hypotheses)}")
        print(f"  отчётов:       {len(result.reports)}")
        if result.refused:
            print(f"  отклонено:     {len(result.refused)}")
            for refusal in result.refused[:5]:
                print(f"    [{refusal['kind']}] {refusal['reason'][:90]}")
        print()
        print("Вердикты из пакета НЕ приняты как истина — они лишь показывают,")
        print("что у кого получилось. Принять гипотезу здесь может только")
        print("локальный эксперимент на локальной скрытой выборке.")
        return 0

    if action not in ("show", ""):
        print(f"неизвестная команда {action!r}; есть export, import, show")
        return 2

    stats = queue.stats()
    from ..core import splits
    print(f"экземпляр:       {identity.fingerprint()}")
    print(f"скрытая выборка: {splits.HOLDOUT_V1.identity}")
    print(f"очередь:         {queue.path}")
    print()
    print(f"гипотез: {stats['hypotheses']} "
          f"(передаваемых {stats['shareable']}, "
          f"непередаваемых {stats['not_shareable']}), "
          f"отчётов: {stats['reports']}")

    reports = queue.reports()
    if not reports:
        print()
        print("воспроизведений нет: отчётов ни от кого не поступало")
        return 0

    verified = sum(1 for r in reports if r.verified)
    print(f"из них с проверенной подписью: {verified}")
    print()
    print("что где воспроизвелось:")
    print(f"  {'гипотеза':18s} {'сред':>5} {'судили':>7} {'принято':>8} "
          f"{'откл':>6} {'без силы':>9} {'выборок':>8}")
    for hypothesis_id, row in sorted(replication(reports).items()):
        print(f"  {hypothesis_id:18s} {row['instances']:5d} {row['ruled']:7d} "
              f"{row['accepted']:8d} {row['rejected']:6d} "
              f"{row['not_evaluated']:9d} {row['distinct_holdouts']:8d}")

    groups = consensus(reports)
    print()
    print("что из этого следует:")
    for label, key, note in (
            ("РАСХОЖДЕНИЯ", "divergent",
             "принято на одних, отклонено на других — изменение условно"),
            ("подтверждено", "confirmed",
             f"судили не меньше {MIN_FOR_CONSENSUS}, все приняли"),
            ("опровергнуто", "refuted",
             "все отклонили — результат, который обычно теряют"),
            ("не решено", "undecided",
             "рулений мало, чтобы говорить о чём-то")):
        ids = [e["hypothesis_id"] for e in groups[key]]
        print(f"  {label:14s} {len(ids):3d}  {note}")
        for hypothesis_id in ids[:5]:
            print(f"      {hypothesis_id}")
    return 0
