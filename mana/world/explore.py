"""
mana.world.explore — working out the rules of a world by acting in it.

What it may use
----------------
Observations and the outcome of its own attempts. Nothing else. It never
imports the world's true rules, and a test asserts that by reading this
file: a learner that peeks scores perfectly and means nothing, which
would make the whole experiment a demo.

How a precondition is established
----------------------------------
A fact is a candidate precondition of an action when it held in every
success of that action. That alone is weak -- in a world where the power
is usually on, "the power is on" and "it is Tuesday" are equally good
candidates -- so a candidate becomes KNOWN only with a **discriminating
failure**: an attempt that failed while this fact was false and every
other candidate held. That is the one case where the world can be said to
have behaved differently because of this fact and nothing else.

Without such a case the candidate stays BELIEVED. It is written down, it
is used, and it is never called knowledge -- the same line
`cognition/lessons.py` draws between what was tried and what was learned.

A fact that never varies cannot be learned at all
--------------------------------------------------
If nothing in the world ever makes a file stop existing, then "the file
exists" holds in every success of every action, and asserting it as a
precondition of all of them would fill the model with true-but-unearned
claims. Facts observed with only one value are therefore excluded from
candidacy and listed separately as unlearnable-from-here. This is the
same rule `cognition/probes.py` applies to an axis that was never varied,
which is not a coincidence: it is the same fact about evidence.

An effect is measured over opportunities, not over successes
-------------------------------------------------------------
Opening an already-open file succeeds and changes nothing, so counting
"how often did the state change after a success" would rate every effect
as unreliable. The denominator is the number of successes where the fact
was **not already** at the value in question. A share of 1 over enough
opportunities is an effect; anything less is a regularity, recorded with
its share and never promoted to a rule.

A negative is never certain
----------------------------
"I cannot do this" is concluded from many attempts across many different
situations, and it is recorded as BELIEVED whatever the count. Absence of
evidence is not evidence of absence, and this project has already read
"no counterexamples in seven episodes" as safety once.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .schema import (BELIEVED, CAN, CANNOT, Capability, KNOWN, Observation,
                     Rule, Situation, UNKNOWN, WorldModel)

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Successes where a fact was not already at the target value, before its
#: share is worth reading. Below this, one lucky attempt would read as a
#: law and one unlucky one as a coincidence.
MIN_OPPORTUNITIES = 3

#: Attempts, in this many distinct situations, before "I cannot" is said
#: at all. Both are needed: fifty attempts from the same state say only
#: that this state does not work.
MIN_ATTEMPTS_FOR_CANNOT = 20
MIN_SITUATIONS_FOR_CANNOT = 8

#: How long an episode runs before the world is reset. An explorer that
#: cut the power on move two would otherwise spend the rest of the run
#: watching everything fail, and record that as the way the world is.
EPISODE_STEPS = 25

#: Reset after this many attempts in a row have failed, instead of on the
#: clock. Zero keeps the clock.
#:
#: The idea is read out of the record rather than picked off a grid: the
#: schedule decides most of the model's quality -- 0.57 at episode length
#: 10 against 0.91 at 25 on the discovery seeds -- and a clock wastes
#: steps at both ends. It cuts a productive episode short, and it leaves a
#: dead one running: after the power is cut nothing digital works, and
#: every attempt until the next tick is a failure that removes nothing
#: from any intersection.
#:
#: Off by default, and it stays off until a holdout says otherwise. The
#: information policy was argued for just as plausibly and measured worse.
RESET_AFTER_FAILURES = 0

#: What an attempt would be worth, before it is made. The same three
#: numbers `cognition/probes.py` puts on an experimental axis, for the
#: same reason -- this is one fact about evidence, not two.
#:
#: Nothing tried yet: anything that happens is news.
UNTRIED = 1.0
#: Exactly one candidate precondition is false in this situation. A
#: failure confirms that one and a success kills it, so both outcomes
#: conclude something. The only shape where that is true.
DISCRIMINATING = 0.9
#: Several candidates are false. A success would kill all of them at
#: once, which is worth a lot; a failure would explain nothing, which is
#: worth little. Priced between the two.
ELIMINATING = 0.6
#: Never succeeded. Whether this is possible at all is still open, and
#: that is a bigger question than which conditions it needs.
POSSIBILITY = 0.7
#: An effect seen sometimes and not others, with too few opportunities to
#: read the share. Repetition really does buy something here.
UNRESOLVED_REGULARITY = 0.5
#: Every candidate holds; the attempt should work and confirm what is
#: already confirmed.
SETTLED = 0.15

#: Below this, an attempt is not worth making for what it would teach --
#: the same floor `experiments.select` holds everything else to. The
#: explorer falls back to plain coverage rather than stopping, because a
#: world it has not finished touching is not a world it has finished
#: learning.
MIN_ATTEMPT_VALUE = 0.05

#: Choose by what an attempt would teach, or by whose turn it is.
#:
#: Coverage is the default because it measured better, not because it is
#: older. Paired over forty seeds at 400 steps, the length where the two
#: differ most: precondition precision under the information policy came
#: to -0.107 [-0.159, -0.056] against coverage, worse in 24 pairs of 40
#: and better in 6, with twice as many invented preconditions. The
#: interval is entirely below zero, so `findings.classify` puts it in
#: WORSE and it is recorded in the ledger as REJECTED.
#:
#: Kept rather than deleted: "we tried this and it did not hold" is one of
#: the more valuable things a research loop can know, and this is now the
#: baseline any future policy has to beat.
BY_INFORMATION = "information"
BY_COVERAGE = "coverage"
DEFAULT_POLICY = BY_COVERAGE


@dataclass
class Attempt:
    """One thing tried, in one situation, and whether it worked."""
    action: str
    before: Situation
    after: Situation
    succeeded: bool


@dataclass
class Explorer:
    """Acts, watches, and writes down what follows from what.

    Holds every attempt rather than a running summary: a rule that looks
    established after forty attempts can be broken by the forty-first, and
    a summary cannot be re-derived under a stricter rule later.
    """
    attempts: List[Attempt] = field(default_factory=list)
    seen_values: Dict[Tuple[str, str], Set[Any]] = field(
        default_factory=lambda: defaultdict(set))
    domains: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    observations: int = 0
    #: How often each exact (action, situation) pair has been run. An
    #: attempt identical to one already made learns nothing the first one
    #: did not, and without this a greedy policy repeats it for ever.
    repeats: Dict[Tuple[str, Situation], int] = field(
        default_factory=lambda: defaultdict(int))
    #: Derived facts, thrown away whenever the record grows. Recomputed
    #: rather than maintained: a summary that drifts from the attempts it
    #: came from is worse than one that costs a little to rebuild.
    _cache: Dict[str, Any] = field(default_factory=dict)

    # ---------- living in the world ----------

    def note(self, facts: Sequence[Any]) -> None:
        """Take in what the senses report, including which layer it came from."""
        self.observations += 1
        for fact in facts:
            self.seen_values[fact.key].add(fact.value)
            if fact.source:
                self.domains[fact.source].add(fact.entity)

    def record(self, observation: Observation) -> None:
        self.attempts.append(Attempt(
            action=observation.action, before=observation.before,
            after=observation.after, succeeded=observation.succeeded))
        self.repeats[(observation.action, observation.before)] += 1
        for entity, attribute, value in observation.after:
            self.seen_values[(entity, attribute)].add(value)
        self._cache.clear()

    def explore(self, world: Any, steps: int = 400, seed: int = 0,
                episode_steps: int = EPISODE_STEPS,
                policy: str = DEFAULT_POLICY,
                reset_after_failures: int = RESET_AFTER_FAILURES) -> "Explorer":
        """Act in the world, spreading attempts evenly over the actions.

        Even coverage: with a handful of actions the cheap thing to get
        right is that nothing is left untried. It looks like the policy
        with no idea what it is doing, and it beat the one that did --
        see DEFAULT_POLICY above for the number.

        `BY_INFORMATION` prices each attempt by what it would teach in the
        situation the explorer is actually in. It is available, it is
        measured, and it is worse; passing it is how the comparison is
        re-run rather than how the explorer is improved.
        """
        rng = random.Random(seed)
        actions = list(world.actions)
        tried: Dict[str, int] = {name: 0 for name in actions}
        failures_in_a_row = 0
        self.note(world.observe())
        for step in range(steps):
            if reset_after_failures:
                # The world has stopped answering. Resetting on that
                # rather than on a clock keeps a productive episode
                # running and cuts a dead one short.
                if failures_in_a_row >= reset_after_failures:
                    world.reset()
                    failures_in_a_row = 0
                    self.note(world.observe())
            elif episode_steps and step and step % episode_steps == 0:
                world.reset()
                self.note(world.observe())
            if policy == BY_COVERAGE:
                fewest = min(tried.values())
                action = rng.choice([a for a in actions if tried[a] == fewest])
            else:
                action, _, _ = self.choose(actions, world.situation(), rng)
            tried[action] += 1
            outcome = world.act(action)
            failures_in_a_row = 0 if outcome.succeeded else failures_in_a_row + 1
            self.record(outcome)
        self.note(world.observe())
        return self

    # ---------- working it out ----------

    def learnable(self) -> Set[Tuple[str, str]]:
        """Facts seen with more than one value. The rest cannot be learned.

        Not "the rest are not preconditions" -- they may well be. They
        cannot be **established** from this record, which is a different
        statement and the only one the evidence supports.
        """
        cached = self._cache.get("learnable")
        if cached is None:
            cached = {key for key, values in self.seen_values.items()
                      if len(values) > 1}
            self._cache["learnable"] = cached
        return cached

    def _for(self, action: str) -> List[Attempt]:
        by_action = self._cache.get("by_action")
        if by_action is None:
            by_action = defaultdict(list)
            for attempt in self.attempts:
                by_action[attempt.action].append(attempt)
            self._cache["by_action"] = by_action
        return by_action[action]

    # ---------- what an attempt would be worth ----------

    def value_of(self, action: str,
                 situation: Situation) -> Tuple[float, str]:
        """What attempting this here would teach, and why that much.

        Priced against the situation the explorer is actually in, which is
        the whole difference from taking turns: the same action is worth a
        great deal in one state and nothing in another, and a policy that
        cannot see the difference never visits the state that would settle
        a question.
        """
        attempts = self._for(action)
        if not attempts:
            return (UNTRIED, "ещё не пробовали")

        repeated = self.repeats.get((action, situation), 0)
        damping = 1.0 / (1.0 + repeated)

        successes = [a for a in attempts if a.succeeded]
        if not successes:
            # Whether this is possible at all is a bigger question than
            # which conditions it needs, and it stays open until either a
            # success or enough failures in enough different situations.
            return (POSSIBILITY * damping, "ни одной удачи — возможно ли вообще")

        candidates = self._candidates(action)
        missing = [f for f in candidates if f not in situation]
        if len(missing) == 1:
            settled = self._discriminated(action, candidates)
            if missing[0] in settled:
                return (SETTLED * damping,
                        "это условие уже различено отказом")
            return (DISCRIMINATING * damping,
                    f"ровно одно условие ложно: {missing[0][0]}."
                    f"{missing[0][1]} — отказ подтвердит его, удача опровергнет")
        if len(missing) > 1:
            return (ELIMINATING * damping,
                    f"ложных условий {len(missing)}: удача снимет все, "
                    f"отказ не объяснит ничего")

        _, regularities = self._effects(action)
        if regularities:
            return (UNRESOLVED_REGULARITY * damping,
                    "эффект наступает не всегда — доля ещё не устоялась")
        return (SETTLED * damping, "все условия выполнены — подтверждать нечего")

    def choose(self, actions: Sequence[str], situation: Situation,
               rng: random.Random) -> Tuple[str, float, str]:
        """The attempt worth making here, or the least-tried one.

        The fallback is not a failure of the policy. When nothing on offer
        clears the floor, what is left to learn is not in this situation,
        and touching the least-tried action is how the explorer gets out
        of it.
        """
        priced = [(self.value_of(a, situation), a) for a in actions]
        (best, why), action = max(priced, key=lambda row: (row[0][0], row[1]))
        if best >= MIN_ATTEMPT_VALUE:
            top = [a for (v, _), a in priced if v >= best - 1e-9]
            return (rng.choice(top), best, why)
        counts = {a: len(self._for(a)) for a in actions}
        fewest = min(counts.values())
        return (rng.choice([a for a in actions if counts[a] == fewest]),
                best, "ничего не проходит порог — беру наименее испробованное")

    def _candidates(self, action: str) -> Set[Tuple[str, str, Any]]:
        """Facts that held in every success. Weak on their own."""
        learnable = self.learnable()
        successes = [a for a in self._for(action) if a.succeeded]
        if not successes:
            return set()
        common: Optional[Set[Tuple[str, str, Any]]] = None
        for attempt in successes:
            here = {f for f in attempt.before if (f[0], f[1]) in learnable}
            common = here if common is None else (common & here)
        return common or set()

    def _discriminated(self, action: str,
                       candidates: Set[Tuple[str, str, Any]]
                       ) -> Set[Tuple[str, str, Any]]:
        """Candidates a failure has actually singled out.

        The failure must have this fact false and every other candidate
        true. Anything weaker leaves two explanations standing, and
        picking one of them is how a model becomes confident about the
        wrong thing.
        """
        confirmed: Set[Tuple[str, str, Any]] = set()
        failures = [a for a in self._for(action) if not a.succeeded]
        for fact in candidates:
            others = candidates - {fact}
            for attempt in failures:
                if fact in attempt.before:
                    continue
                if others <= attempt.before:
                    confirmed.add(fact)
                    break
        return confirmed

    def _effects(self, action: str
                 ) -> Tuple[Set[Tuple[str, str, Any]],
                            List[Tuple[Tuple[str, str, Any], float]]]:
        """What follows a success, and how reliably.

        Counted over opportunities: a success that could not have changed
        the fact because it already held says nothing about the effect,
        and letting it into the denominator would make every effect look
        unreliable.
        """
        successes = [a for a in self._for(action) if a.succeeded]
        occurrences: Dict[Tuple[str, str, Any], int] = defaultdict(int)
        for attempt in successes:
            for fact in attempt.after - attempt.before:
                occurrences[fact] += 1

        certain: Set[Tuple[str, str, Any]] = set()
        regular: List[Tuple[Tuple[str, str, Any], float]] = []
        for fact, count in sorted(occurrences.items()):
            chances = self._opportunities(action, fact)
            if chances < MIN_OPPORTUNITIES:
                continue
            share = count / chances
            if share >= 1.0:
                certain.add(fact)
            elif share > 0.0:
                regular.append((fact, round(share, 3)))
        return certain, regular

    def _opportunities(self, action: str,
                       fact: Tuple[str, str, Any]) -> int:
        """Successes where this fact was not already at that value."""
        entity, attribute, value = fact
        count = 0
        for attempt in self._for(action):
            if not attempt.succeeded:
                continue
            if (entity, attribute, value) not in attempt.before:
                count += 1
        return count

    def _capability(self, action: str) -> Capability:
        attempts = self._for(action)
        successes = [a for a in attempts if a.succeeded]
        situations = len({a.before for a in attempts})
        if successes:
            # One success settles it. Nothing about the world can make an
            # action that has been performed impossible.
            return Capability(action=action, verdict=CAN, status=KNOWN,
                              attempts=len(attempts), successes=len(successes),
                              distinct_situations=situations,
                              note="выполнено хотя бы раз")
        if (len(attempts) >= MIN_ATTEMPTS_FOR_CANNOT
                and situations >= MIN_SITUATIONS_FOR_CANNOT):
            # Believed, never known: no number of failures proves that a
            # situation allowing it does not exist.
            return Capability(
                action=action, verdict=CANNOT, status=BELIEVED,
                attempts=len(attempts), successes=0,
                distinct_situations=situations,
                note=("ни одной удачи; отсутствие свидетельства — не "
                      "свидетельство отсутствия"))
        return Capability(action=action, verdict=UNKNOWN, status=UNKNOWN,
                          attempts=len(attempts), successes=0,
                          distinct_situations=situations,
                          note="попыток пока мало, чтобы говорить о невозможности")

    def model(self) -> WorldModel:
        """Everything the record supports, marked by how well it does."""
        built = WorldModel(observations=self.observations)
        built.domains = {name: sorted(entities)
                         for name, entities in self.domains.items()}

        for action in sorted({a.action for a in self.attempts}):
            capability = self._capability(action)
            built.capabilities[action] = capability
            if capability.verdict != CAN:
                # Nothing was ever performed, so there is no transition to
                # describe. Saying so beats writing an empty rule that
                # reads like knowledge.
                built.rules[action] = Rule(
                    action=action, status=UNKNOWN,
                    evidence={"attempts": capability.attempts,
                              "successes": 0,
                              "note": "ни одной удачи — переход не наблюдался"})
                continue

            candidates = self._candidates(action)
            confirmed = self._discriminated(action, candidates)
            effects, regularities = self._effects(action)
            status = KNOWN if confirmed and confirmed == candidates else BELIEVED
            if not candidates:
                status = BELIEVED
            built.rules[action] = Rule(
                action=action,
                preconditions=tuple(sorted(candidates)),
                effects=tuple(sorted(effects)),
                regularities=tuple(regularities),
                status=status,
                evidence={"successes": capability.successes,
                          "failures": capability.attempts - capability.successes,
                          "discriminated": sorted(map(list, confirmed)),
                          "assumed": sorted(map(list, candidates - confirmed)),
                          "unlearnable": sorted(
                              f"{e}.{a}" for (e, a) in self.seen_values
                              if (e, a) not in self.learnable())})
        return built
