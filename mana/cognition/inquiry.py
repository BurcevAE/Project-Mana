"""
mana.cognition.inquiry — asking the world a question on purpose.

What was missing
-----------------
MANA learned from what arrived. A world that produced no events produced
no statistics, no findings and no hypotheses, and the agent stopped
developing there. Two places already did better -- `ResearchCycle`, which
generates tasks to narrow its own capability intervals, and
`world/explore`, which walks to the state that would settle a
precondition -- but each was bound to its own subject. This module is the
contract both are instances of:

    Detector      knowledge -> questions that are still open
    Question      what is unknown, why it is open, the competing answers
    HypothesisSet the answers, each able to predict what a probe would show
    ProbeProvider the world: which probes exist, what they cost, acting
    choose        the probe with the most expected bits per unit of cost
    update        Bayes over the hypotheses, from what was observed
    settle        when a question is answered, and what the answer is

Three rules carry the design
-----------------------------
**A hypothesis must predict an observation.** "The side with more of X
lost" is a finding, not an explanation. An explanation is admitted only as
a function from a probe to a distribution over outcomes, because that is
the only thing that lets two explanations be told apart by acting. Every
set also carries OTHER -- "none of mine explains this" -- which predicts
every outcome equally.

**One unit.** A probe is worth its expected reduction in the entropy of
the hypothesis set, in bits, per unit of cost. Everything that competes
for an action is priced in bits.

**A question closes on its own terms.** The first version borrowed the
lab's MIN_EXPERIMENT_VALUE as a floor on bits per action. It was a floor
in another module's units, and under a noisy world it fired while answers
were one or two presses away (measured: leaders at 0.91-0.95, best probes
at 0.037-0.048 against 0.05). Now:

    answered   the leading CLASS holds CONFIDENCE of the belief. A class is
               the hypotheses no available probe can tell apart: a button
               on a switch that is locked on and a button that always
               works are one answer in that world, not an open question.
    deferred   what is still needed to reach CONFIDENCE, at the best rate
               any probe offers, will not fit in the budget left. What is
               needed is counted between classes: belief spread inside a
               class is bits no probe can deliver, and owing them made the
               loop give up with three quarters of its budget unspent
               whenever closing waited for anything else (measured, 1.8).

Falsification before closing
-----------------------------
The first version closed a question as soon as its leader reached 0.95
among the hypotheses it had. Given a button that needs two switches at
once, it tested the leader "works when s1 is on" in three of the four
combinations of s1 and s5, never in the one where s1 on and s5 off would
have refuted it, and named the wrong rule in 43 devices of 50. Choosing
by information only separates hypotheses that exist.

So a leader about to be accepted is challenged: `pairwise_rivals` adds
every rule that agrees with it except in one combination of two
conditions, each half as likely a priori as the leader it challenges, and
the weights are recomputed from the whole history. Rivals the history
already refutes vanish; the rest make exactly the untested combinations
informative, and ordinary choice by bits goes there. Only a leader that
survives is accepted. This is strength-2 falsification: it catches any
two-way interaction with an unmodelled condition -- and, stated plainly,
not every three-way one.

What did not work: a probability that the model is wrong
--------------------------------------------------------
1.2-1.11 carried a belief that the hypothesis family misses the truth
(misfit) and closed when it was low. Against the device it was wrong in
kind, not only in size: it took one check to catch a wrong model with
0.83-0.89 where a check does so with 0.32-0.37, and the claim-level
variant took 0.10-0.18. Stated at the measured values (1.11) the belief
closed better than any fixed number of checks -- but the values came from
an oracle. From MANA's own record they are not identifiable (1.12
diagnosis): the record depends on what the closing rule chose to check,
and a right simple model runs out of checks after one or two, so "not a
miss" and "a miss not caught yet" look alike; checking every question to
exhaustion did not change that. The direction is closed. Kept from it:
only checks ever showed a miss (none of 49 by distinguishing
observations), and the errors came from closing, not from choosing.

The claims economy
------------------
A model commits to claims. The task says what a wrong answer costs and how
much it relies on each configuration. A probe is worth taking when the
claims it would test for the first time -- weighted by how much of the
task rests on them -- times the cost of an error exceed what the probe
costs; the question stays open while such a probe exists. No P(model
wrong) and no hidden prior: one exchange rate, set by the task, so the same
knowledge is checked deeply for a task that relies on it and hardly at all
for one that does not. What it can catch is bounded by what the claims can
say: "in this pattern the outcome does not depend on d" cannot state an
interaction of two conditions the model ignores.

Graded (1.13), measured and set aside: a claim seen at one of its
configurations was kept open by the share of its task weight still
unseen. With nothing to say how likely the unseen rest is to fail, every
unseen configuration kept nearly its full stake, and the claims meeting
at one configuration added up. On device A, where every model is right,
it took 125 actions at an error cost of 2 and 210 at 16 -- against 65 and
74 binary, and 149 for walking every configuration. It caught more wrong
models at a middle price (D: 4 wrong instead of 16 at cost 12) only by
observing nearly everything. The switch stays, off, for the A/B.

Boundary of applicability (1.14): an answer does not say how likely it
is to be right; it says where it is backed. Every configuration observed
is inside -- the outcome there is known and a living leader agrees with
it. Beyond them only independent observations reach, as far as an
inclusion rule allows: `observed` (no further), `claims` (a configuration
whose every claim an independent observation bore on) or `pattern` (all
of a pattern seen independently once). A probe is worth the share of the
task it would bring inside, times the cost of an error; each
configuration counts once, so a question never spends more than an error
would cost. The answer closes VERIFIED_FOR_TASK when the task lies
inside, CONDITIONAL otherwise -- with the part it relies on unverified.

Measured on A, C and D, three tasks, fresh devices too: with `claims` no
answer was wrong inside its declared boundary -- wrong answers on D
closed CONDITIONAL, about half their task unverified; with `pattern` up to
12 of 50 answers on D closed VERIFIED_FOR_TASK and wrong; `observed` is
honest by construction and verifies a task only by walking it (189-200
actions, more than enumeration's 149). The honesty of `claims` is a
property of these worlds, not a guarantee: a claim speaks for one ignored
condition at a time, and an interaction between two ignored conditions can
hide inside a claim that was seen once. World E was built for exactly
that (b0 works when a is on unless e and f are both on), and `claims`
broke there in 9 runs of 400 -- 4 devices of 100 -- every one of them a
model that ignored both e and f, wrong at a=e=f=1, whose claims there had
each been confirmed by a different observation. `pattern` broke in up to
13 answers of 50 per price on the same world.

What this is not
-----------------
Not a planner: it never serves a user's turn. Not a gate: a hypothesis at
0.99 is well supported, not ACCEPTED. Not evolution: it changes what MANA
knows, not what MANA is. And not yet wired into anything -- this is the
vertical slice that has to earn a migration.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.14"

# Why a question is open.
NOT_ENOUGH_DATA = "NOT_ENOUGH_DATA"   # nothing observed about it yet
UNCERTAIN = "UNCERTAIN"               # observed, the answers still compete
UNTESTED = "UNTESTED"                 # a leader is being challenged before acceptance
UNEXPLAINED = "UNEXPLAINED"           # "none of my hypotheses" leads
CONTRADICTION = "CONTRADICTION"       # something happened that no hypothesis allows

# How a question was settled.
ANSWERED = "ANSWERED"

# Why the loop stopped.
NO_QUESTIONS = "NO_QUESTIONS"
NOTHING_WORTH_ASKING = "NOTHING_WORTH_ASKING"
BUDGET = "BUDGET"

# What a probe may do to the world. A field of the probe, never a decision
# of the chooser: on a person's machine only the first is allowed without
# asking.
READ_ONLY = "read_only"
REVERSIBLE = "reversible"
IRREVERSIBLE = "irreversible"

OTHER = "не объясняется ни одной из гипотез"
ALWAYS = "всегда"
NEVER = "никогда"

#: A question is answered when its leading class holds this much belief.
CONFIDENCE = 0.95

#: A rival starts half as likely as the leader it challenges. A stated
#: prior, like the numbers in `probes.py`: a rule that is the leader plus
#: one exception is a more complicated rule, and complication is paid for.
RIVAL_PRIOR = 0.5

#: Two reasons to act. `distinguish`: which known explanation is right.
#: `model_check`: is the set of known explanations wrong altogether.
DISTINGUISH = "distinguish"
MODEL_CHECK = "model_check"
MISFIT = "модель не объясняет мир"

#: Prior belief that the hypothesis family does not contain the truth. The
#: same number OTHER already carried -- not a new knob.
MISFIT_PRIOR = 0.05

#: If the family is wrong, the chance that any one condition which differs
#: from everything already tested is where it shows. 0.5 is the ignorant
#: choice: no reason to think an untested condition more likely to matter
#: than not. Nothing here knows which conditions exist in the true rule or
#: how many.
DEVIATION = 0.5

#: What makes a model-check probe able to show the model wrong.
#:   GEOMETRIC  how far it is from anything already observed
#:   CLAIMS     how many of the leading explanation's own commitments it
#:              would test for the first time
#: One function, used both to price a check and to update `misfit` from
#: what it showed -- otherwise the price would not be the information the
#: update actually takes.
GEOMETRIC = "geometric"
CLAIMS = "claims"

#: Stated for dynamic closing, measured on the device (see the 1.10
#: diagnosis) and to be learned rather than stated:
#:   CHECK_STRENGTH  the chance that one observation at a configuration not
#:                   seen before shows a model wrong when its family misses
#:                   the truth -- measured 0.32-0.37 per check, nearly the
#:                   same for the 1st, 2nd, 3rd and 4th check of one leader
#:   FAMILY_MISS     the share of questions whose family misses the truth
#:                   -- one button in five on the D device
#: The misfit belief put the first at 0.83-0.89 and the second at 0.05.
CHECK_STRENGTH = 1.0 / 3.0
FAMILY_MISS = 0.2
#: Neither is identifiable from MANA's own record (see "What did not work"
#: above). Both stay only so the A/B against the claims economy can be rerun.

#: Claims economy: a model whose claim a check falsified may give way once
#: to an explanation that fits everything; a second falsified check means
#: the family does not explain the world. The same one revision the misfit
#: rule allowed -- a stated policy, not an estimate.
REVISIONS_AFTER_REFUTATION = 1

#: Boundary of applicability: how far an answer is taken to hold beyond
#: what was observed -- see `HypothesisSet._inside` -- and how it stands
#: for the task at hand.
BOUNDARY_OBSERVED = "observed"
BOUNDARY_CLAIMS = "claims"
BOUNDARY_PATTERN = "pattern"
VERIFIED_FOR_TASK = "VERIFIED_FOR_TASK"
CONDITIONAL = "CONDITIONAL"

#: Below this a hypothesis is treated as refuted and skipped.
ALIVE = 1e-12

#: A set never grows past this many hypotheses; a leader that cannot be
#: challenged for lack of room is accepted, and the settlement says so.
MAX_HYPOTHESES = 600


def _entropy(weights: Sequence[float]) -> float:
    return -sum(w * math.log2(w) for w in weights if w > 0.0)


@dataclass(eq=False)
class Hypothesis:
    """One possible answer, and what it says a probe would show."""
    name: str
    predict: Callable[[Dict[str, Any]], Dict[Any, float]]
    weight: float = 1.0
    prior: float = 0.0


class HypothesisSet:
    """Competing answers to one question, and how much each is believed.

    Keeps the history of what was observed, because a hypothesis added
    later -- a rival -- has to be weighed against everything seen, not
    only against what comes after it.
    """

    def __init__(self, hypotheses: Sequence[Hypothesis]) -> None:
        self.hypotheses = list(hypotheses)
        total = sum(h.weight for h in self.hypotheses)
        if total <= 0:
            raise ValueError("a hypothesis set needs positive weight")
        for h in self.hypotheses:
            h.prior = h.weight / total
            h.weight = h.prior
        self.history: List[Tuple[Dict[str, Any], Any]] = []
        self.observations = 0
        self.contradicted = 0
        self.settled: Optional[Tuple[str, Tuple[str, ...]]] = None
        self.challenged: set = set()
        self.capped = False
        #: Belief that the family misses the truth; None while model checks
        #: are off. `revise_from` is the leader a failed check refuted.
        self.misfit: Optional[float] = None
        self.revise_from: Optional[Hypothesis] = None
        self.vulnerability = GEOMETRIC
        #: The question's probe space, kept so an explanation's relevant
        #: conditions can be read off its predictions.
        self.space: Optional[List[Dict[str, Any]]] = None
        self._relevant_cache: Dict[int, Tuple[str, ...]] = {}
        self._claims_cache: Optional[Tuple[int, int, set]] = None
        #: Which leading explanation `misfit` is currently the belief about.
        self._misfit_model: Optional[int] = None
        #: What each observation risked when it was made, kept beside the
        #: history: (P(outcome) by the family then, was the family
        #: unanimous then, who led then). None where no check was running.
        self.evidence: List[Optional[Tuple[float, bool, Optional[int]]]] = []
        self._pending: Optional[Tuple[float, bool, Optional[int]]] = None
        self.last_replay: Tuple[int, int] = (0, 0)
        self._coverage_cache: Optional[Tuple[int, set]] = None
        #: Experimental: misfit moved by the obligations an observation put
        #: at risk, not by how many it touched. Off = the 1.7 behaviour.
        self.claim_level = False
        self._all_claims_cache: Dict[int, set] = {}
        #: Every observation that moved misfit, for diagnosis.
        self.update_log: List[Dict[str, Any]] = []
        #: Experimental: checks a leader must survive before it is accepted,
        #: counted per explanation -- a revised model starts from none.
        self.checks_to_close = 0
        self._survived: Dict[int, int] = {}
        #: Experimental: close when one more check is not worth its cost.
        #: The cost of a wrong answer, in the units of probe cost; None = off.
        self.error_cost: Optional[float] = None
        self.next_check_cost: Optional[int] = None
        self._confirmed: Dict[int, int] = {}
        #: Claims economy: what rides on this question in the task at hand;
        #: None = off. `claims_open` -- some allowed probe is still worth it.
        self.stakes: Optional["Stakes"] = None
        self.claims_open = False
        #: The explanations are told apart and only checking is left.
        self.check_phase = False
        self.refuted_by_check = 0
        self._last_refuted: Optional[Hypothesis] = None
        self._importance: Dict[int, Dict[tuple, float]] = {}
        #: Graded claims: a claim is tested over a share of its weight.
        self.claims_graded = False
        self._mass: Dict[int, Dict[tuple, float]] = {}
        self._tested_cache: Optional[Tuple[Tuple[int, int], Dict[tuple, float]]] = None
        #: Boundary of applicability: the inclusion rule (None = off), and
        #: how the answer stood for the task when it closed.
        self.boundary: Optional[str] = None
        self.applicability: Optional[Dict[str, Any]] = None
        self._inside_cache: Optional[Tuple[tuple, set]] = None
        self._predicted: Dict[Tuple[int, Any], Dict[Any, float]] = {}
        self._signature: Dict[int, tuple] = {}

    # ---------- predictions ----------

    def _dist(self, h: Hypothesis, params: Dict[str, Any]) -> Dict[Any, float]:
        """What h predicts for these params, cached by the probe's `key`."""
        key = params.get("key")
        if key is None:
            return h.predict(params)
        slot = (id(h), key)
        dist = self._predicted.get(slot)
        if dist is None:
            dist = self._predicted[slot] = h.predict(params)
        return dist

    def live(self) -> List[Hypothesis]:
        return [h for h in self.hypotheses if h.weight > ALIVE]

    def get(self, name: str) -> Optional[Hypothesis]:
        return next((h for h in self.hypotheses if h.name == name), None)

    # ---------- belief ----------

    def entropy(self) -> float:
        return _entropy([h.weight for h in self.live()])

    def expected_gain(self, params: Dict[str, Any]) -> float:
        """Bits this probe is expected to remove: H now minus H after."""
        live = self.live()
        dists = [self._dist(h, params) for h in live]
        predictive: Dict[Any, float] = {}
        for h, dist in zip(live, dists):
            for outcome, p in dist.items():
                predictive[outcome] = predictive.get(outcome, 0.0) + h.weight * p
        after = 0.0
        for outcome, p_outcome in predictive.items():
            if p_outcome <= 0.0:
                continue
            after += p_outcome * _entropy([h.weight * d.get(outcome, 0.0) / p_outcome
                                           for h, d in zip(live, dists)])
        return max(0.0, _entropy([h.weight for h in live]) - after)

    def update(self, params: Dict[str, Any], outcome: Any) -> bool:
        """Fold one observation in. False when no hypothesis allowed it --
        the weights are then left as they were rather than divided by
        zero, and the question is marked as contradicted."""
        self.observations += 1
        live = self.live()
        raw = [h.weight * self._dist(h, params).get(outcome, 0.0) for h in live]
        total = sum(raw)
        if total <= 0.0:
            self.contradicted += 1
            return False
        for h, r in zip(live, raw):
            h.weight = r / total
        self.history.append((params, outcome))
        self.evidence.append(self._pending)
        self._pending = None
        return True

    def add(self, rivals: Sequence[Hypothesis], prior_each: float) -> int:
        """Admit new hypotheses and re-weigh everything from the history."""
        known = {h.name for h in self.hypotheses}
        fresh = [r for r in rivals if r.name not in known]
        room = MAX_HYPOTHESES - len(self.hypotheses)
        if room < len(fresh):
            self.capped = True
            fresh = fresh[:max(0, room)]
        if not fresh:
            return 0
        for r in fresh:
            r.prior = prior_each
        self.hypotheses.extend(fresh)
        total = sum(h.prior for h in self.hypotheses)
        logs = []
        for h in self.hypotheses:
            h.prior /= total
            log_weight = math.log(h.prior) if h.prior > 0 else -math.inf
            for params, outcome in self.history:
                p = self._dist(h, params).get(outcome, 0.0)
                if p <= 0.0:
                    log_weight = -math.inf
                    break
                log_weight += math.log(p)
            logs.append(log_weight)
        top = max(logs)
        raw = [math.exp(l - top) if l > -math.inf else 0.0 for l in logs]
        norm = sum(raw)
        for h, r in zip(self.hypotheses, raw):
            h.weight = r / norm
        return len(fresh)

    # ---------- the model as a whole ----------

    def _family_true(self, params: Dict[str, Any]) -> Optional[float]:
        """P(True) by the named explanations together -- what "my model"
        says, with OTHER left out because it is the absence of one."""
        named = [h for h in self.live() if h.name != OTHER]
        mass = sum(h.weight for h in named)
        if mass <= 0.0:
            return None
        return sum(h.weight * self._dist(h, params).get(True, 0.0) for h in named) / mass

    def novelty(self, params: Dict[str, Any]) -> int:
        """How many conditions separate this probe from the nearest one
        already observed. The only thing the misfit model reads: where the
        evidence is thinnest, a wrong model has had least chance to show."""
        here = params.get("conditions") or {}
        if not self.history:
            return len(here)
        return min(sum(1 for k, v in here.items()
                       if bool((seen.get("conditions") or {}).get(k)) != bool(v))
                   for seen, _ in self.history)

    def _relevant(self, h: Hypothesis) -> Tuple[str, ...]:
        """The conditions h's prediction actually turns on, read off the
        probe space: two probes that differ only in c and get different
        predictions make c relevant. Nothing is assumed about h's form."""
        cached = self._relevant_cache.get(id(h))
        if cached is not None:
            return cached
        space = self.space or []
        names = sorted({k for p in space for k in (p.get("conditions") or {})})
        relevant = []
        for c in names:
            groups: Dict[tuple, set] = {}
            for p in space:
                cond = p["conditions"]
                key = tuple((k, bool(cond[k])) for k in names if k != c)
                groups.setdefault(key, set()).add(tuple(sorted(self._dist(h, p).items())))
            if any(len(g) > 1 for g in groups.values()):
                relevant.append(c)
        self._relevant_cache[id(h)] = tuple(relevant)
        return tuple(relevant)

    @staticmethod
    def _config_key(params: Dict[str, Any]) -> Any:
        key = params.get("key")
        if key is not None:
            return key
        return tuple(sorted((k, bool(v)) for k, v in (params.get("conditions") or {}).items()))

    def observed_coverage(self) -> set:
        """COVERAGE: the configurations the history has observed. A fact about
        the world and the record, about no model: every observation counts,
        whoever it helped choose."""
        if self._coverage_cache is None or self._coverage_cache[0] != len(self.history):
            self._coverage_cache = (len(self.history),
                                    {self._config_key(p) for p, _ in self.history})
        return self._coverage_cache[1]

    def covered(self, params: Dict[str, Any]) -> bool:
        return self._config_key(params) in self.observed_coverage()

    def _vouches(self, record: Optional[Tuple[float, bool, Optional[int]]],
                 params: Dict[str, Any], outcome: Any, lead: Hypothesis) -> bool:
        """May this past observation count for `lead` -- as confirmation,
        and as having tested its commitments? Only if `lead` was committed
        to it before it happened: it led then, or the whole family agreed
        then and `lead` agrees with what the family predicted. An
        observation that split the family is one `lead` was chosen on."""
        if record is None:
            return False
        right, unanimous, led_by = record[:3]
        if led_by == id(lead):
            return True
        return unanimous and abs(self._dist(lead, params).get(outcome, 0.0) - right) < 1e-9

    def _claims_at(self, lead: Hypothesis, params: Dict[str, Any]) -> set:
        """The obligations of `lead` an observation here bears on: in this
        pattern of what it reads, the outcome does not depend on d = v."""
        relevant = self._relevant(lead)
        here = params.get("conditions") or {}
        pattern = tuple(bool(here[c]) for c in relevant)
        return {(pattern, d, bool(v)) for d, v in here.items() if d not in relevant}

    def _all_claims(self, lead: Hypothesis) -> set:
        cached = self._all_claims_cache.get(id(lead))
        if cached is None:
            cached = set()
            for probe in self.space or []:
                cached |= self._claims_at(lead, probe)
            self._all_claims_cache[id(lead)] = cached
        return cached

    def _evidenced(self, lead: Hypothesis) -> set:
        """Obligations of `lead` with independent evidence.

        1.7: every obligation an allowed observation bore on.
        claim level: only those it put at risk and that survived -- as
        recorded when the lead made the prediction, or, for an observation
        the whole family agreed on, everything it tested for the first time.
        """
        if (self._claims_cache is not None and self._claims_cache[0] == id(lead)
                and self._claims_cache[1] == len(self.history)):
            return self._claims_cache[2]
        tested: set = set()
        seen: set = set()
        for (params, outcome), record in zip(self.history, self.evidence):
            key = self._config_key(params)
            repeat = key in seen
            seen.add(key)
            if not self._vouches(record, params, outcome, lead):
                continue
            if not self.claim_level:
                tested |= self._claims_at(lead, params)
            elif self._dist(lead, params).get(outcome, 0.0) > 0.5:
                if record[2] == id(lead):
                    tested |= set(record[3])
                elif not repeat:
                    tested |= self._claims_at(lead, params)
        self._claims_cache = (id(lead), len(self.history), tested)
        return tested

    def _at_risk(self, lead: Hypothesis, params: Dict[str, Any], checked: set) -> set:
        """Of the obligations this observation tests for the first time, the
        ones it actually put at risk. Where living rivals predicted otherwise,
        only obligations about the conditions those rivals read; where the
        whole family agreed, it was a check of the model itself, and every
        one of them was."""
        mine = self._dist(lead, params)
        rivals = [h for h in self.live()
                  if h is not lead and h.name != OTHER and self._dist(h, params) != mine]
        if not rivals:
            return set(checked)
        dims: set = set()
        for h in rivals:
            dims |= set(self._relevant(h))
        dims -= set(self._relevant(lead))
        return {c for c in checked if c[1] in dims}

    def untested_claims(self, params: Dict[str, Any]) -> int:
        """How many of the leading explanation's commitments this probe
        would test for the first time.

        An explanation that reads conditions R commits, for every other
        condition d, to "in this pattern of R the outcome does not depend
        on d". That commitment is tested at (pattern, d=v) once something
        was observed in that pattern with d=v. A probe that tests none of
        them can only confirm; one that tests many is where the model is
        exposed -- whatever the true rule turns out to be.
        """
        named = [h for h in self.live() if h.name != OTHER]
        if not named or not self.space:
            return 0
        # Coverage decides whether the probe risks anything: at a
        # configuration already observed the outcome is known, whoever that
        # observation helped choose. Evidence decides what is still owed:
        # below, only independent observations count as having tested an
        # obligation. The two are kept apart -- an obligation covered by
        # selection data stays owed, and is tested at a configuration not
        # yet seen.
        if self.covered(params):
            return 0
        lead = max(named, key=lambda h: h.weight)
        # Only observations allowed to vouch for the leader test its
        # commitments. The rest still chose between explanations; they
        # leave the commitment open to be tested in advance.
        return len(self._claims_at(lead, params) - self._evidenced(lead))

    def exposure(self, params: Dict[str, Any]) -> int:
        """How much of the model this probe puts at risk, by the chosen measure."""
        if self.vulnerability == CLAIMS:
            return self.untested_claims(params)
        return self.novelty(params)

    def _misfit_likelihoods(self, params: Dict[str, Any], outcome: Any
                            ) -> Optional[Tuple[float, float]]:
        """P(outcome | my model is right), P(outcome | it is wrong).

        If it is wrong, the world departs from the model's prediction here
        with probability q, which grows with how novel the probe is.
        Boolean outcomes, as in `pairwise_rivals`.
        """
        p_true = self._family_true(params)
        if p_true is None:
            return None
        right = p_true if outcome else 1.0 - p_true
        q = 1.0 - (1.0 - DEVIATION) ** self.exposure(params)
        return right, (1.0 - q) * right + q * (1.0 - right)

    def _replay_claim_level(self, lead: Optional[Hypothesis]) -> float:
        """Claim-level replay: each allowed observation moves misfit by the
        share of the lead's still unverified obligations it put at risk."""
        pi = MISFIT_PRIOR
        counted = 0
        if lead is None:
            self.last_replay = (0, len(self.history))
            return pi
        everything = self._all_claims(lead)
        evidenced: set = set()
        seen: set = set()
        for (params, outcome), record in zip(self.history, self.evidence):
            key = self._config_key(params)
            repeat = key in seen
            seen.add(key)
            if not self._vouches(record, params, outcome, lead):
                continue
            if record[2] == id(lead):
                at_risk = set(record[3]) - evidenced
            else:
                at_risk = set() if repeat else self._claims_at(lead, params) - evidenced
            unverified = len(everything) - len(evidenced)
            q = len(at_risk) / unverified if unverified > 0 else 0.0
            right = record[0]
            wrong = (1.0 - q) * right + q * (1.0 - right)
            total = pi * wrong + (1.0 - pi) * right
            if total > 0.0:
                pi = pi * wrong / total
            if self._dist(lead, params).get(outcome, 0.0) > 0.5:
                evidenced |= at_risk
            counted += bool(at_risk)
        self.last_replay = (counted, len(self.history))
        return pi

    def _replay_misfit(self) -> float:
        """The belief that the current model is wrong, from independent
        confirmations only.

        Every observation stays in the history and keeps choosing between
        explanations -- that is compatibility, and all of it counts. But an
        observation vouches for the model only if the model was committed
        to it before it happened:

            it led then             its own prediction, made in advance
            the family was          the commitment every explanation
            unanimous then, and     shared was put at risk, and nothing
            it agrees with that     inside the family was chosen by it

        An observation that split the family is one the model was chosen
        on. Counting it as confirmation is the model choosing itself on the
        data and then being proved by the same data -- the leak the last
        version had, measured at 64 of 65 leader changes.

        Each counted observation weighs what it risked THEN: the family's
        probability of that outcome as recorded at the time, never
        re-derived under the family the observation has since pruned. The
        obligations it tested are this model's, against what had been seen
        before it.
        """
        named = [h for h in self.live() if h.name != OTHER]
        lead = max(named, key=lambda h: h.weight) if named else None
        if self.claim_level:
            return self._replay_claim_level(lead)
        saved = self.history
        pi = MISFIT_PRIOR
        counted = 0
        try:
            for index, ((params, outcome), record) in enumerate(zip(saved, self.evidence)):
                if lead is None or not self._vouches(record, params, outcome, lead):
                    continue                # it chose the model; it cannot vouch for it
                right = record[0]
                self.history = saved[:index]
                self._claims_cache = None
                q = 1.0 - (1.0 - DEVIATION) ** self.exposure(params)
                wrong = (1.0 - q) * right + q * (1.0 - right)
                total = pi * wrong + (1.0 - pi) * right
                if total > 0.0:
                    pi = pi * wrong / total
                    counted += 1
        finally:
            self.history = saved
            self._claims_cache = None
        self.last_replay = (counted, len(saved))
        return pi

    def current_misfit(self) -> Optional[float]:
        """`misfit` for the explanation that leads now.

        Re-derived when the leader changes, so a model inherits no
        confidence earned by the one it replaced. When nothing named is
        alive there is no model to re-derive it for, and the belief stays
        as the last observation left it -- that is the alarm itself.
        """
        if self.misfit is None:
            return None
        named = [h for h in self.live() if h.name != OTHER]
        if not named:
            return self.misfit
        lead = max(named, key=lambda h: h.weight)
        if id(lead) != self._misfit_model:
            self.misfit = self._replay_misfit()
            self._misfit_model = id(lead)
        return self.misfit

    def model_check_gain(self, params: Dict[str, Any]) -> float:
        """Bits this probe is expected to tell about "is my model wrong"."""
        if self.misfit is None:
            return 0.0
        pi = self.current_misfit()
        after = 0.0
        for outcome in (True, False):
            pair = self._misfit_likelihoods(params, outcome)
            if pair is None:
                return 0.0
            right, wrong = pair
            p = pi * wrong + (1.0 - pi) * right
            if p > 0.0:
                post = pi * wrong / p
                after += p * _entropy([post, 1.0 - post])
        return max(0.0, _entropy([pi, 1.0 - pi]) - after)

    def update_misfit(self, params: Dict[str, Any], outcome: Any) -> None:
        """Fold an observation into the belief that the model is wrong.
        Called before `update`, while the model is still the one that made
        the prediction."""
        if self.misfit is None:
            return
        self.current_misfit()
        pair = self._misfit_likelihoods(params, outcome)
        if pair is None:
            return
        right, wrong = pair
        named = [h for h in self.live() if h.name != OTHER]
        lead = max(named, key=lambda h: h.weight) if named else None
        predicted: set = set()
        checked: set = set()
        at_risk: set = set()
        if lead is not None and self.space:
            predicted = self._claims_at(lead, params)
            if not self.covered(params):
                checked = predicted - self._evidenced(lead)
            at_risk = self._at_risk(lead, params, checked)
            if self.claim_level:
                unverified = len(self._all_claims(lead)) - len(self._evidenced(lead))
                q = len(at_risk) / unverified if unverified > 0 else 0.0
                wrong = (1.0 - q) * right + q * (1.0 - right)
        self._pending = (right, not self.discriminates(params),
                         id(lead) if lead is not None else None, frozenset(at_risk))
        pi = self.misfit
        total = pi * wrong + (1.0 - pi) * right
        if total <= 0.0:
            return
        self.misfit = pi * wrong / total
        self.update_log.append({"model": lead.name if lead is not None else "",
                                "config": self._config_key(params),
                                "predicted": len(predicted), "checked": len(checked),
                                "at_risk": len(at_risk), "before": pi, "after": self.misfit})
        if right <= ALIVE or self.misfit >= CONFIDENCE:
            named = [h for h in self.live() if h.name != OTHER]
            if named:
                self.revise_from = max(named, key=lambda h: h.weight)

    def _lead(self) -> Optional[Hypothesis]:
        named = [h for h in self.live() if h.name != OTHER]
        return max(named, key=lambda h: h.weight) if named else None

    def note_check(self, params: Dict[str, Any], outcome: Any) -> None:
        """A model check was made; if the leader predicted what happened, it
        survived one. Called before `update`, while that leader leads."""
        lead = self._lead()
        if lead is not None and self._dist(lead, params).get(outcome, 0.0) > 0.5:
            self._survived[id(lead)] = self._survived.get(id(lead), 0) + 1

    def note_observation(self, params: Dict[str, Any], outcome: Any) -> None:
        """Before `update`: the leader predicted a configuration not seen
        before, and was right -- one chance to be shown wrong, survived.
        Counted for the leader that made the prediction, so a model chosen
        afterwards because it fits the history is not confirmed by it."""
        lead = self._lead()
        if lead is None or self.covered(params):
            return
        if self._dist(lead, params).get(outcome, 0.0) > 0.5:
            self._confirmed[id(lead)] = self._confirmed.get(id(lead), 0) + 1

    def doubt(self) -> float:
        """P(the family misses the truth), for the current leader: the stated
        share, and every advance prediction of its that came true."""
        lead = self._lead()
        n = self._confirmed.get(id(lead), 0) if lead is not None else 0
        odds = FAMILY_MISS / (1.0 - FAMILY_MISS) * (1.0 - CHECK_STRENGTH) ** n
        return odds / (1.0 + odds)

    def check_worth_it(self) -> bool:
        """Is one more check worth what it costs? The error it would catch
        -- doubt x strength x cost of an error -- against the cheapest check
        on offer. Doubt only falls while the model survives, so looking one
        check ahead is enough."""
        if self.error_cost is None or self.misfit is None or self.next_check_cost is None:
            return False
        return self.doubt() * CHECK_STRENGTH * self.error_cost > self.next_check_cost

    def note_claims(self, params: Dict[str, Any], outcome: Any, check: bool = False) -> None:
        """Claims economy, before `update`: what the evidence ledger keeps
        about this observation -- who led, whether the family agreed --
        with no misfit behind it; and whether a check falsified a claim."""
        p = self._family_true(params)
        lead = self._lead()
        if lead is not None and self._dist(lead, params).get(outcome, 0.0) <= 0.5:
            self._last_refuted = lead
            if check:
                self.refuted_by_check += 1
        right = (p if outcome else 1.0 - p) if p is not None else 0.0
        self._pending = (right, not self.discriminates(params),
                         id(lead) if lead is not None else None, frozenset())

    def _claim_importance(self, lead: Hypothesis) -> Dict[tuple, float]:
        """How much of the task rests on each claim of `lead`: the share of
        the task's weight on the configurations the claim speaks for."""
        cached = self._importance.get(id(lead))
        if cached is None:
            raw: Dict[tuple, float] = {}
            total = 0.0
            for q in self.space or []:
                w = max(0.0, float(self.stakes.weight(q))) if self.stakes else 1.0
                total += w
                for c in self._claims_at(lead, q):
                    raw[c] = raw.get(c, 0.0) + w
            cached = {c: v / total for c, v in raw.items()} if total > 0 else {}
            self._importance[id(lead)] = cached
            self._mass[id(lead)] = raw
        return cached

    def _tested_share(self, lead: Hypothesis) -> Dict[tuple, float]:
        """Graded: how much of each claim's task weight has been observed
        independently for `lead` -- each configuration counted once."""
        key = (id(lead), len(self.history))
        if self._tested_cache is not None and self._tested_cache[0] == key:
            return self._tested_cache[1]
        self._claim_importance(lead)
        mass = self._mass[id(lead)]
        done: Dict[tuple, float] = {}
        seen: set = set()
        for (params, outcome), record in zip(self.history, self.evidence):
            k = self._config_key(params)
            if k in seen or not self._vouches(record, params, outcome, lead):
                continue
            seen.add(k)
            w = max(0.0, float(self.stakes.weight(params))) if self.stakes else 1.0
            for c in self._claims_at(lead, params):
                done[c] = done.get(c, 0.0) + w
        share = {c: min(1.0, v / mass[c]) for c, v in done.items() if mass.get(c, 0.0) > 0}
        self._tested_cache = (key, share)
        return share

    def claim_stake(self, params: Dict[str, Any]) -> float:
        """What a probe here protects: the task's reliance on the claims of
        the leader it would test for the first time, times what a wrong
        answer costs. Nothing where the outcome is already known."""
        if self.stakes is None or not self.space or self.covered(params):
            return 0.0
        lead = self._lead()
        if lead is None:
            return 0.0
        if self.boundary is not None:
            return self._boundary_value(lead, params)
        importance = self._claim_importance(lead)
        if self.claims_graded:
            share = self._tested_share(lead)
            return self.stakes.error_cost * sum(
                importance.get(c, 0.0) * (1.0 - share.get(c, 0.0))
                for c in self._claims_at(lead, params))
        untested = self._claims_at(lead, params) - self._evidenced(lead)
        return self.stakes.error_cost * sum(importance.get(c, 0.0) for c in untested)

    def _pattern(self, lead: Hypothesis, params: Dict[str, Any]) -> tuple:
        here = params.get("conditions") or {}
        return tuple(bool(here[c]) for c in self._relevant(lead))

    def _task_weight(self, params: Dict[str, Any]) -> float:
        return max(0.0, float(self.stakes.weight(params))) if self.stakes else 1.0

    def _inside(self, lead: Hypothesis, extra: Optional[Dict[str, Any]] = None) -> set:
        """BOUNDARY: the configurations where `lead`'s answer is backed.

        Every configuration observed is inside: its outcome is known and a
        living leader agrees with it (coverage). Beyond them only
        independent observations -- made while `lead` was committed --
        reach (evidence), as far as the inclusion rule allows:
            observed  no further
            claims    a configuration whose every claim an independent
                      observation bore on, in a pattern seen independently
            pattern   every configuration of a pattern seen independently
        `extra`: as if one more independent observation were made there.
        """
        key = (id(lead), len(self.history), self.boundary)
        if extra is None and self._inside_cache is not None and self._inside_cache[0] == key:
            return self._inside_cache[1]
        inside = set(self.observed_coverage())
        independent = [params for (params, outcome), record in zip(self.history, self.evidence)
                       if self._vouches(record, params, outcome, lead)]
        if extra is not None:
            inside.add(self._config_key(extra))
            independent.append(extra)
        if self.boundary != BOUNDARY_OBSERVED and independent:
            patterns = {self._pattern(lead, q) for q in independent}
            if self.boundary == BOUNDARY_PATTERN:
                inside |= {self._config_key(q) for q in self.space or []
                           if self._pattern(lead, q) in patterns}
            else:
                evidenced: set = set()
                for q in independent:
                    evidenced |= self._claims_at(lead, q)
                inside |= {self._config_key(q) for q in self.space or []
                           if self._pattern(lead, q) in patterns
                           and self._claims_at(lead, q) <= evidenced}
        if extra is None:
            self._inside_cache = (key, inside)
        return inside

    def _boundary_value(self, lead: Hypothesis, params: Dict[str, Any]) -> float:
        """What a probe here would bring inside the boundary -- a share of
        the task's weight -- times what a wrong answer costs. Valued as if it
        confirms; if it refutes, the answer changes, which is worth more."""
        space = self.space or []
        total = sum(self._task_weight(q) for q in space)
        if total <= 0.0:
            return 0.0
        now = self._inside(lead)
        after = self._inside(lead, params)
        gained = sum(self._task_weight(q) for q in space
                     if self._config_key(q) in after and self._config_key(q) not in now)
        return self.stakes.error_cost * gained / total

    def applicability_for(self, lead: Hypothesis) -> Dict[str, Any]:
        """How an answer stands for the task: verified where the task relies
        on it, or conditional -- with the part it relies on unverified."""
        inside = self._inside(lead)
        task = [q for q in self.space or [] if self._task_weight(q) > 0.0]
        total = sum(self._task_weight(q) for q in task)
        out = [q for q in task if self._config_key(q) not in inside]
        share = sum(self._task_weight(q) for q in out) / total if total > 0 else 0.0
        return {"status": VERIFIED_FOR_TASK if not out else CONDITIONAL,
                "rule": self.boundary, "uncovered": [self._config_key(q) for q in out],
                "share": share}

    def checks_owed(self, space: Sequence[Dict[str, Any]]) -> bool:
        """Is the leader still short of its checks, with something left that
        could check it?"""
        if not self.checks_to_close or self.misfit is None:
            return False
        lead = self._lead()
        if lead is None or self._survived.get(id(lead), 0) >= self.checks_to_close:
            return False
        return any(self.exposure(p) > 0 for p in space)

    def leader(self) -> Tuple[str, float]:
        best = max(self.hypotheses, key=lambda h: h.weight)
        return (best.name, best.weight)

    def leading_class(self, space: Sequence[Dict[str, Any]]
                      ) -> Tuple[Tuple[str, ...], float, Hypothesis]:
        """The hypotheses no probe in `space` can tell apart, with the most
        belief between them. The probe space of a question is taken to be
        fixed, which is what lets the signatures be cached.

        With no probes at all nothing is indistinguishable by experiment --
        there is no experiment -- so the class is the leader alone.
        """
        if not space:
            lead = max(self.live(), key=lambda h: h.weight)
            return ((lead.name,), lead.weight, lead)
        best = max(self._classes(space), key=lambda g: sum(h.weight for h in g))
        lead = max(best, key=lambda h: h.weight)
        return (tuple(sorted(h.name for h in best)),
                sum(h.weight for h in best), lead)

    def _classes(self, space: Sequence[Dict[str, Any]]) -> List[List[Hypothesis]]:
        """Live hypotheses grouped by what they predict on every probe in `space`."""
        groups: Dict[tuple, List[Hypothesis]] = {}
        for h in self.live():
            sig = self._signature.get(id(h))
            if sig is None:
                sig = self._signature[id(h)] = tuple(
                    tuple(sorted(self._dist(h, p).items())) for p in space)
            groups.setdefault(sig, []).append(h)
        return list(groups.values())

    def class_entropy(self, space: Sequence[Dict[str, Any]]) -> float:
        """Uncertainty some probe in `space` could still remove: entropy
        between classes, not between hypotheses no probe tells apart."""
        if not space:
            return self.entropy()
        return _entropy([sum(h.weight for h in g) for g in self._classes(space)])

    def discriminates(self, params: Dict[str, Any]) -> bool:
        """Do the explanations still in play disagree about this probe?
        OTHER is left out: confirming what every live explanation already
        predicts tells no explanation apart from another."""
        live = [h for h in self.live() if h.name != OTHER]
        return len({tuple(sorted(self._dist(h, params).items())) for h in live}) > 1


def condition_hypotheses(conditions: Sequence[str], noise: float = 0.0,
                         other: float = 0.05) -> HypothesisSet:
    """Does an action's outcome depend on one observable condition?

    The same family `world/explore` searches for preconditions: for every
    observable condition, "it works when this holds"; plus "always",
    "never", and OTHER. Nothing about any particular world is in it.
    """
    def bern(yes: bool) -> Dict[bool, float]:
        return ({True: 1.0 - noise, False: noise} if yes
                else {True: noise, False: 1.0 - noise})

    named = [Hypothesis(f"зависит от {c}",
                        lambda params, c=c: bern(bool(params["conditions"][c])))
             for c in conditions]
    named.append(Hypothesis(ALWAYS, lambda params: bern(True)))
    named.append(Hypothesis(NEVER, lambda params: bern(False)))
    share = (1.0 - other) / len(named)
    for h in named:
        h.weight = share
    return HypothesisSet(named + [Hypothesis(OTHER, lambda params: {True: 0.5, False: 0.5},
                                             weight=other)])


def pairwise_rivals(conditions: Sequence[str]) -> Callable[[Hypothesis], List[Hypothesis]]:
    """Rules that agree with a leader except in one combination of two
    conditions -- the challenge a boolean leader has to survive.

    Model-free: it does not guess which interaction might exist, it names
    every two-way exception and lets the history and the next probes rule
    them out. Outcomes are taken to be boolean.
    """
    pairs = list(itertools.combinations(conditions, 2))

    def challenge(leader: Hypothesis) -> List[Hypothesis]:
        out = []
        for d1, d2 in pairs:
            for x in (False, True):
                for y in (False, True):
                    def predict(params, base=leader.predict, d1=d1, d2=d2, x=x, y=y):
                        dist = base(params)
                        seen = params["conditions"]
                        if bool(seen[d1]) == x and bool(seen[d2]) == y:
                            return {True: dist.get(False, 0.0), False: dist.get(True, 0.0)}
                        return dist
                    out.append(Hypothesis(
                        f"{leader.name}, кроме {d1}={int(x)} и {d2}={int(y)}", predict))
        return out
    return challenge


def settle(hypotheses: HypothesisSet, space: Sequence[Dict[str, Any]],
           challenge: Optional[Callable[[Hypothesis], List[Hypothesis]]] = None,
           confidence: float = CONFIDENCE) -> Optional[Tuple[str, Tuple[str, ...]]]:
    """Is this question answered? Part of updating knowledge, not of
    choosing actions: the same test whichever policy gathered the data."""
    if hypotheses.settled:
        return hypotheses.settled
    if hypotheses.stakes is not None:
        hypotheses.check_phase = False
        if hypotheses.refuted_by_check > REVISIONS_AFTER_REFUTATION:
            hypotheses.settled = (UNEXPLAINED, (MISFIT,))
            return hypotheses.settled
    if (hypotheses.current_misfit() or 0.0) >= confidence:
        # The model as a whole is judged wrong. One revision around the
        # explanation the check refuted, with the same rivals as any
        # challenge; if anything in it still fits everything seen, the
        # revised model gets its own check. Otherwise that is the answer.
        lost = hypotheses.revise_from
        if challenge is not None and lost is not None and lost.name not in hypotheses.challenged:
            hypotheses.challenged.add(lost.name)
            hypotheses.add(challenge(lost), prior_each=RIVAL_PRIOR * lost.prior)
            hypotheses.revise_from = None
            if any(h.name != OTHER for h in hypotheses.live()):
                hypotheses.misfit = MISFIT_PRIOR
                hypotheses._misfit_model = None
                return None
        hypotheses.settled = (UNEXPLAINED, (MISFIT,))
        return hypotheses.settled
    names, mass, lead = hypotheses.leading_class(space)
    if mass < confidence:
        return None
    if names == (OTHER,):
        lost = hypotheses._last_refuted
        if (hypotheses.stakes is not None and challenge is not None and lost is not None
                and lost.name not in hypotheses.challenged):
            # Claims economy: nothing fits any more. One revision around the
            # explanation the world last refuted, as the misfit rule did,
            # before calling the question unexplained.
            hypotheses.challenged.add(lost.name)
            hypotheses.add(challenge(lost), prior_each=RIVAL_PRIOR * lost.prior)
            hypotheses._last_refuted = None
            return None
        hypotheses.settled = (UNEXPLAINED, names)
        return hypotheses.settled
    if challenge is not None and lead.name not in hypotheses.challenged:
        hypotheses.challenged.add(lead.name)
        hypotheses.add(challenge(lead), prior_each=RIVAL_PRIOR * lead.prior)
        names, mass, lead = hypotheses.leading_class(space)
        if mass < confidence:
            return None                     # the challenge reopened it
        if names == (OTHER,):
            hypotheses.settled = (UNEXPLAINED, names)
            return hypotheses.settled
        if lead.name not in hypotheses.challenged and not hypotheses.capped:
            return None                     # a rival leads now; it is challenged next
    if mass * (1.0 - (hypotheses.current_misfit() or 0.0)) < confidence:
        return None                         # a check of the model itself is still owed
    if hypotheses.checks_owed(space):
        return None                         # it has not yet survived its checks
    if hypotheses.check_worth_it():
        return None                         # one more check is worth its cost
    if hypotheses.stakes is not None:
        hypotheses.check_phase = True       # told apart; only checking is left
    if hypotheses.claims_open:
        return None                         # a claim the task rests on is worth testing
    if hypotheses.boundary is not None:
        hypotheses.applicability = hypotheses.applicability_for(lead)
    hypotheses.settled = (ANSWERED, names)
    return hypotheses.settled


@dataclass
class Stakes:
    """What rides on a question's answer in the task at hand: what a wrong
    answer costs, in the units of probe cost, and how much the task relies
    on each configuration. The same knowledge carries different stakes in
    different tasks."""
    error_cost: float
    weight: Callable[[Dict[str, Any]], float] = field(default=lambda params: 1.0)


@dataclass
class Question:
    """What MANA wants to know. Not a GAP: a gap says something was
    missing while working; a question says what knowledge would help."""
    subject: str
    why_open: str
    hypotheses: HypothesisSet
    source: Dict[str, Any] = field(default_factory=dict)


def unsettled(knowledge: Dict[str, HypothesisSet]) -> List[Question]:
    """A detector: every question not yet settled, and why it is open.

    Derived from the knowledge each time rather than stored: a question
    that outlived its answer would be asked again for nothing.
    """
    out: List[Question] = []
    for subject, hypotheses in knowledge.items():
        if hypotheses.settled:
            continue
        name, _ = hypotheses.leader()
        why = (NOT_ENOUGH_DATA if hypotheses.observations == 0
               else CONTRADICTION if hypotheses.contradicted
               else UNTESTED if hypotheses.challenged
               else UNEXPLAINED if name == OTHER
               else UNCERTAIN)
        out.append(Question(subject=subject, why_open=why, hypotheses=hypotheses,
                            source={"action": subject}))
    return out


@dataclass
class ProbeSpec:
    """An action the world offers, with what it will make hold and its price.
    `params["key"]`, when present, names the probe for caching predictions."""
    action: str
    params: Dict[str, Any]
    cost: int = 1
    safety: str = REVERSIBLE


@dataclass
class Probe:
    question: Question
    spec: ProbeSpec
    gain: float
    estimated_calls: int
    value: float
    motive: str = DISTINGUISH


class ProbeProvider(Protocol):
    def probes_for(self, question: Question) -> List[ProbeSpec]: ...
    def act(self, spec: ProbeSpec) -> Any: ...


@dataclass
class Step:
    subject: str
    why_open: str
    action: str
    cost: int
    gain: float
    value: float
    outcome: Any
    leader: str
    weight: float
    explained: bool
    motive: str = DISTINGUISH
    misfit: Optional[float] = None


@dataclass
class Report:
    steps: List[Step]
    spent: int
    stopped: str
    answers: Dict[str, Tuple[str, Tuple[str, ...]]]
    open_questions: List[Tuple[str, str, str, float]]


def inquire(detect: Callable[[], List[Question]], world: ProbeProvider, budget: int,
            challenge: Optional[Callable[[Hypothesis], List[Hypothesis]]] = None,
            allowed: Sequence[str] = (READ_ONLY, REVERSIBLE),
            confidence: float = CONFIDENCE, model_check: bool = False,
            vulnerability: str = GEOMETRIC, claim_level: bool = False,
            checks_to_close: int = 0, error_cost: Optional[float] = None,
            stakes: Optional[Stakes] = None, graded: bool = False,
            boundary: Optional[str] = None) -> Report:
    """Settle what can be settled, then act where a question gains most per
    unit of cost, until nothing open is worth what it would take."""
    steps: List[Step] = []
    answers: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
    spent = 0
    closing = _entropy([confidence, 1.0 - confidence])
    while True:
        candidates: List[Probe] = []
        still_open = 0
        blocked_by_budget = False
        for question in detect():
            if model_check and question.hypotheses.misfit is None:
                question.hypotheses.misfit = MISFIT_PRIOR
                question.hypotheses.vulnerability = vulnerability
                question.hypotheses.claim_level = claim_level
                question.hypotheses.checks_to_close = checks_to_close
                question.hypotheses.error_cost = error_cost
            if stakes is not None and question.hypotheses.stakes is None:
                question.hypotheses.stakes = stakes
                question.hypotheses.claims_graded = graded
                question.hypotheses.boundary = boundary
            offered = world.probes_for(question)
            if (model_check or stakes is not None) and question.hypotheses.space is None:
                question.hypotheses.space = [spec.params for spec in offered]
            specs = [s for s in offered if s.safety in allowed]
            if stakes is not None:
                # Claims economy: open while some allowed probe protects more
                # of the task than it costs.
                question.hypotheses.claims_open = any(
                    question.hypotheses.claim_stake(s.params) > max(1, int(s.cost))
                    for s in specs)
            if model_check and error_cost is not None:
                # What the next check would cost: the cheapest allowed probe
                # that still puts some of the model at risk. None -- nothing
                # left that could check it.
                question.hypotheses.next_check_cost = min(
                    (max(1, int(s.cost)) for s in specs
                     if question.hypotheses.exposure(s.params) > 0), default=None)
            # What the world could tell apart decides when a question is
            # answered; what MANA is allowed to do decides only what it
            # does next. Settling on the allowed probes alone would call
            # "not permitted to find out" an answer.
            space = [s.params for s in offered]
            done = settle(question.hypotheses, space, challenge, confidence)
            if done:
                answers[question.subject] = done
                continue
            still_open += 1
            priced = []
            for spec in specs:
                # Two motives, one unit: bits about which explanation is
                # right, and bits about whether any of them is.
                gain = question.hypotheses.expected_gain(spec.params)
                motive = DISTINGUISH
                if model_check:
                    check = question.hypotheses.model_check_gain(spec.params)
                    if check > gain:
                        gain, motive = check, MODEL_CHECK
                cost = max(1, int(spec.cost))
                if stakes is not None and question.hypotheses.check_phase:
                    # Claims economy: the explanations are told apart; a
                    # probe is worth the error it could prevent, net of its
                    # cost -- in actions, not bits.
                    stake = question.hypotheses.claim_stake(spec.params)
                    if stake > cost:
                        priced.append(Probe(question, spec, stake, cost,
                                            (stake - cost) / cost, MODEL_CHECK))
                    continue
                priced.append(Probe(question, spec, gain, cost, gain / cost, motive))
            best_rate = max((p.value for p in priced
                             if stakes is None or p.motive == DISTINGUISH), default=0.0)
            checks_here = stakes is not None and any(p.motive == MODEL_CHECK for p in priced)
            if best_rate <= 0.0 and not checks_here:
                continue                    # no allowed probe says anything here
            if best_rate > 0.0:
                needed = max(0.0, question.hypotheses.class_entropy(space) - closing)
                if needed / best_rate > budget - spent:
                    blocked_by_budget = True    # would not finish in what is left
                    continue
            affordable = [p for p in priced
                          if p.gain > 0.0 and p.estimated_calls <= budget - spent]
            if not affordable:
                blocked_by_budget = True
            candidates.extend(affordable)
        if still_open == 0:
            stopped = NO_QUESTIONS
            break
        if not candidates:
            stopped = BUDGET if blocked_by_budget else NOTHING_WORTH_ASKING
            break
        pool = candidates
        if stakes is not None:
            # Bits and actions are different units: tell explanations apart
            # first, check the model when nothing is left to tell.
            pool = [p for p in candidates if p.motive == DISTINGUISH] or candidates
        chosen = max(pool, key=lambda p: p.value)
        outcome = world.act(chosen.spec)
        spent += chosen.estimated_calls
        if stakes is not None:
            chosen.question.hypotheses.note_claims(chosen.spec.params, outcome,
                                                   chosen.motive == MODEL_CHECK)
        if error_cost is not None:
            chosen.question.hypotheses.note_observation(chosen.spec.params, outcome)
        if chosen.motive == MODEL_CHECK:
            chosen.question.hypotheses.note_check(chosen.spec.params, outcome)
        chosen.question.hypotheses.update_misfit(chosen.spec.params, outcome)
        explained = chosen.question.hypotheses.update(chosen.spec.params, outcome)
        name, weight = chosen.question.hypotheses.leader()
        steps.append(Step(subject=chosen.question.subject,
                          why_open=chosen.question.why_open,
                          action=chosen.spec.action, cost=chosen.estimated_calls,
                          gain=round(chosen.gain, 4), value=round(chosen.value, 4),
                          outcome=outcome, leader=name, weight=round(weight, 4),
                          explained=explained, motive=chosen.motive,
                          misfit=chosen.question.hypotheses.misfit))
    remaining = [(q.subject, q.why_open) + q.hypotheses.leader() for q in detect()]
    return Report(steps=steps, spent=spent, stopped=stopped, answers=answers,
                  open_questions=remaining)
