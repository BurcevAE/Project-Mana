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
from dataclasses import dataclass, field
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

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ExchangeError(f"неизвестный вердикт {self.verdict!r}")

    def as_dict(self) -> Dict[str, Any]:
        return {"hypothesis_id": self.hypothesis_id, "instance": self.instance,
                "verdict": self.verdict, "trials": self.trials,
                "discordant": self.discordant, "p_value": self.p_value,
                "margin": self.margin, "holdout": self.holdout,
                "created": self.created}


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
            result.reports.append(Report(
                hypothesis_id=str(record.get("hypothesis_id", "")),
                instance=str(record.get("instance", "")),
                verdict=str(record.get("verdict", "")),
                trials=int(record.get("trials", 0)),
                discordant=int(record.get("discordant", 0)),
                p_value=float(record.get("p_value", 1.0)),
                margin=float(record.get("margin", 0.0)),
                holdout=str(record.get("holdout", "")),
                created=float(record.get("created", time.time()))))
        except (ExchangeError, TypeError, ValueError) as exc:
            result.refused.append({"kind": "report", "reason": str(exc)})

    return result


# -------------------------------------------------------------- replication


def replication(reports: Sequence[Report]) -> Dict[str, Dict[str, Any]]:
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
