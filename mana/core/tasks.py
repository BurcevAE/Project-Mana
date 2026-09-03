"""
mana.core.tasks — task generators whose answer is computed, not judged.

Why this replaces BenchmarkSuite
--------------------------------
The whole of MANA's quality signal was 21 hand-written tasks graded by
substring containment: `["391"]`, `["верси"]`, `["памят"]`. That set drives
the fitness of every genome mutation and doubles as the oracle for
patching MANA's own source code. Three consequences, all fatal for
anything above tuning:

  * **Too small to measure with.** A difference of one or two tasks out of
    21 is noise, and the acceptance gates compute z-scores over it.
  * **Gameable by construction.** "верси" is contained in "конверсия"; an
    answer consisting of the union of the required substrings scores well
    on several tasks at once. An optimiser pointed at this metric finds
    that before it finds competence.
  * **Fixed forever.** 21/21 is a ceiling. After it, no further
    improvement is expressible, so open-ended search has nothing to search
    for.

Every task here instead carries a ground truth this module *computed*
while generating it. Grading needs no model, which makes it both
trustworthy and nearly free -- and cost matters: an experiment budget
measured in thousands of LLM calls cannot also spend a call to judge each
answer.

Domains are separate distributions, not paraphrases
---------------------------------------------------
Transfer is the load-bearing claim of the whole project: a cognitive
mechanism that works only where it was found has not been shown to be a
mechanism. That requires domains a strategy cannot bridge by surface
similarity. `arithmetic` and `sequence` share digits but not structure;
`text_ops` and `logic` share neither. A strategy discovered on one and
still winning on another is evidence; the same strategy winning on a
paraphrase of its own training set is not.

What a task is
--------------
Prompts state the required output format explicitly, and an answer that
ignores the format is graded wrong. That is deliberate: following a stated
output contract is part of the capability being measured, and the
alternative -- a lenient extractor -- reintroduces exactly the fuzzy
grading this module exists to remove.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Independent task distributions. Kept small and genuinely different --
#: five weak domains would measure less than four separated ones.
DOMAINS = ("arithmetic", "sequence", "logic", "text_ops", "code")

#: How an answer is compared with the truth. Every one of these is exact;
#: none consults a model.
CHECKERS = ("number", "text", "set", "sequence", "code_tests")


@dataclass(frozen=True)
class Task:
    """One graded question with a computed answer.

    Frozen because a task that can be edited after generation is a task
    that can be edited to match an answer -- and the generator is inside
    the immutable boundary precisely so that cannot happen.
    """
    task_id: str
    domain: str
    prompt: str
    answer: Any
    checker: str
    difficulty: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def public(self) -> Dict[str, Any]:
        """The task as an agent may see it: no answer, ever.

        Used wherever tasks cross out of core -- a training set is handed
        out this way, and the hidden holdout is never handed out at all.
        """
        return {"task_id": self.task_id, "domain": self.domain,
                "prompt": self.prompt, "difficulty": self.difficulty}


# ---------------------------------------------------------------------------
# arithmetic
# ---------------------------------------------------------------------------

def _gen_arithmetic(rng: random.Random, difficulty: float, idx: int) -> Task:
    """Exact integer arithmetic. Difficulty scales operand size and depth.

    Deliberately not "17 * 23": the point of generating is that the answer
    cannot have been memorised from the prompt corpus, and that the space
    is large enough that fitting it means computing rather than recalling.
    """
    if difficulty < 0.3:
        a, b = rng.randint(11, 99), rng.randint(11, 99)
        op = rng.choice(["+", "-", "*"])
        expr = f"{a} {op} {b}"
    elif difficulty < 0.6:
        a, b, c = rng.randint(100, 999), rng.randint(11, 99), rng.randint(2, 19)
        expr = rng.choice([f"{a} * {b} - {c}", f"({a} + {b}) * {c}", f"{a} * {b} + {c} * {b}"])
    else:
        a, b, c, d = (rng.randint(1000, 99999), rng.randint(100, 999),
                      rng.randint(11, 99), rng.randint(2, 9))
        expr = rng.choice([f"({a} - {b}) * {c} + {d}",
                           f"{a} * {c} - {b} * {d}",
                           f"({a} + {b} * {c}) * {d}"])
    answer = eval(expr, {"__builtins__": {}}, {})   # noqa: S307 -- generator-authored, no input
    return Task(
        task_id=f"arith_{idx}",
        domain="arithmetic",
        prompt=f"Вычисли: {expr}\nОтветь одним целым числом, без пояснений.",
        answer=answer, checker="number", difficulty=difficulty,
        metadata={"expression": expr},
    )


# ---------------------------------------------------------------------------
# sequences
# ---------------------------------------------------------------------------

def _gen_sequence(rng: random.Random, difficulty: float, idx: int) -> Task:
    """Continue a generated sequence. The rule is chosen, not guessed, so
    the next term is known exactly rather than being the most plausible
    continuation."""
    n = 6
    if difficulty < 0.3:
        start, step = rng.randint(1, 20), rng.randint(2, 9)
        terms = [start + step * i for i in range(n + 1)]
        rule = f"arithmetic step {step}"
    elif difficulty < 0.6:
        start, ratio = rng.randint(1, 6), rng.randint(2, 4)
        terms = [start * ratio ** i for i in range(n + 1)]
        rule = f"geometric ratio {ratio}"
    else:
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        mul, add = rng.randint(2, 4), rng.randint(1, 15)
        terms = [a, b]
        while len(terms) < n + 1:
            terms.append(terms[-1] * mul - terms[-2] + add)
        rule = f"t[k] = t[k-1]*{mul} - t[k-2] + {add}"
    shown = terms[:n]
    return Task(
        task_id=f"seq_{idx}",
        domain="sequence",
        prompt=("Продолжи последовательность: " + ", ".join(str(t) for t in shown) +
                "\nОтветь одним числом — следующим членом, без пояснений."),
        answer=terms[n], checker="number", difficulty=difficulty,
        metadata={"rule": rule, "shown": shown},
    )


# ---------------------------------------------------------------------------
# logic
# ---------------------------------------------------------------------------

_NAMES = ["Анна", "Борис", "Вера", "Глеб", "Дина", "Егор", "Жанна", "Илья"]


def _gen_logic(rng: random.Random, difficulty: float, idx: int) -> Task:
    """An ordering puzzle with a unique solution.

    The order is drawn first and the constraints are derived from it, so
    the answer is known and consistency is guaranteed. Generating
    constraints and then solving would risk unsolvable or ambiguous
    instances, and an ambiguous task silently punishes a correct answer.
    """
    count = 3 if difficulty < 0.3 else (4 if difficulty < 0.6 else 5)
    people = rng.sample(_NAMES, count)
    order = people[:]                      # position 0 = first
    facts: List[str] = []
    for i in range(len(order) - 1):
        facts.append(f"{order[i]} стоит раньше, чем {order[i + 1]}")
    if difficulty >= 0.6 and len(order) >= 4:
        facts.append(f"{order[0]} стоит раньше, чем {order[-1]}")   # redundant, adds noise
    rng.shuffle(facts)
    asked = rng.randrange(len(order))
    return Task(
        task_id=f"logic_{idx}",
        domain="logic",
        prompt=("Известно:\n" + "\n".join(f"- {f}" for f in facts) +
                f"\n\nКто стоит на позиции {asked + 1} (считая с начала)?"
                "\nОтветь одним именем, без пояснений."),
        answer=order[asked], checker="text", difficulty=difficulty,
        metadata={"order": order},
    )


# ---------------------------------------------------------------------------
# text operations
# ---------------------------------------------------------------------------

def _gen_text_ops(rng: random.Random, difficulty: float, idx: int) -> Task:
    """Deterministic string work. No world knowledge, no reasoning about
    meaning -- which is the point: it shares nothing with the other
    domains, so transfer onto it is not explained by surface similarity."""
    words = ["дом", "река", "стол", "книга", "окно", "поле", "город", "лампа",
             "мост", "сад", "камень", "птица", "море", "лес", "ключ"]
    sample = [rng.choice(words) for _ in range(rng.randint(6, 12))]
    kind = rng.choice(["count_letter", "longest", "unique_sorted"]) if difficulty >= 0.3 else "count_letter"
    text = " ".join(sample)
    if kind == "count_letter":
        letter = rng.choice("аоеирс")
        return Task(
            task_id=f"text_{idx}", domain="text_ops",
            prompt=(f"Текст: {text}\n\nСколько раз буква «{letter}» встречается в этом тексте?"
                    "\nОтветь одним числом, без пояснений."),
            answer=text.count(letter), checker="number", difficulty=difficulty,
            metadata={"kind": kind, "letter": letter})
    if kind == "longest":
        # max() returns the first maximal element, which is exactly the
        # tie-break the prompt promises ("назови первое из них").
        longest = max(sample, key=len)
        return Task(
            task_id=f"text_{idx}", domain="text_ops",
            prompt=(f"Текст: {text}\n\nКакое слово в нём самое длинное?"
                    " Если таких несколько — назови первое из них."
                    "\nОтветь одним словом, без пояснений."),
            answer=longest, checker="text", difficulty=difficulty,
            metadata={"kind": kind})
    unique = sorted(set(sample))
    return Task(
        task_id=f"text_{idx}", domain="text_ops",
        prompt=(f"Текст: {text}\n\nВыпиши все различные слова по алфавиту."
                "\nОтветь списком через запятую, без пояснений."),
        answer=unique, checker="sequence", difficulty=difficulty,
        metadata={"kind": kind})


# ---------------------------------------------------------------------------
# code
# ---------------------------------------------------------------------------

def _gen_code(rng: random.Random, difficulty: float, idx: int) -> Task:
    """Write a small function; graded by running hidden tests.

    The tests travel with the task but never reach the agent -- `public()`
    drops them along with the answer. Grading is the strongest kind
    available here: the code either satisfies the cases or it does not,
    and no model is asked for an opinion.
    """
    specs = [
        ("sum_even", "Напиши функцию sum_even(numbers), возвращающую сумму чётных чисел списка.",
         [("sum_even([1,2,3,4])", 6), ("sum_even([])", 0), ("sum_even([7])", 0),
          ("sum_even([-2,-3,4])", 2)], 0.2),
        ("count_vowels", "Напиши функцию count_vowels(text), считающую гласные a,e,i,o,u в строке.",
         [("count_vowels('hello')", 2), ("count_vowels('')", 0), ("count_vowels('xyz')", 0),
          ("count_vowels('aeiou')", 5)], 0.3),
        ("second_largest", "Напиши функцию second_largest(numbers), возвращающую второе по величине "
                           "различное значение списка, или None, если такого нет.",
         [("second_largest([1,2,3])", 2), ("second_largest([5,5])", None),
          ("second_largest([9,1,9,4])", 4), ("second_largest([])", None)], 0.55),
        ("run_length", "Напиши функцию run_length(text), сжимающую строку в формат 'a3b2' "
                       "(символ и число повторов подряд; одиночный символ тоже с числом 1).",
         [("run_length('aaabb')", "a3b2"), ("run_length('')", ""),
          ("run_length('abc')", "a1b1c1")], 0.7),
        ("balanced", "Напиши функцию balanced(text), возвращающую True, если круглые скобки "
                     "в строке сбалансированы, иначе False.",
         [("balanced('(())')", True), ("balanced('(')", False),
          ("balanced('')", True), ("balanced(')(')", False)], 0.6),
    ]
    eligible = [s for s in specs if abs(s[3] - difficulty) <= 0.35] or specs
    name, prompt, cases, spec_difficulty = rng.choice(eligible)
    return Task(
        task_id=f"code_{idx}", domain="code",
        prompt=prompt + "\nВерни только определение функции на Python, без пояснений и без markdown.",
        answer=cases, checker="code_tests", difficulty=spec_difficulty,
        metadata={"function": name},
    )


_GENERATORS: Dict[str, Callable[[random.Random, float, int], Task]] = {
    "arithmetic": _gen_arithmetic,
    "sequence": _gen_sequence,
    "logic": _gen_logic,
    "text_ops": _gen_text_ops,
    "code": _gen_code,
}


def generate(domain: str, count: int, seed: int,
             difficulty_range: Tuple[float, float] = (0.1, 0.9)) -> List[Task]:
    """Deterministic task set for one domain.

    Determinism is what makes a split reproducible: the same seed always
    yields the same tasks, so a hidden holdout stays the same set across
    runs and across restarts without ever being written to disk where an
    agent could read it.
    """
    if domain not in _GENERATORS:
        raise ValueError(f"unknown domain: {domain!r}")
    rng = random.Random(f"{domain}:{seed}")
    lo, hi = difficulty_range
    tasks: List[Task] = []
    for i in range(count):
        # Sweep difficulty rather than sampling it, so a set of N always
        # covers the range instead of clustering by chance.
        difficulty = lo + (hi - lo) * (i / max(1, count - 1)) if count > 1 else (lo + hi) / 2
        tasks.append(_GENERATORS[domain](rng, round(difficulty, 3), i))
    return tasks


def generate_mixed(count_per_domain: int, seed: int,
                   domains: Optional[Sequence[str]] = None,
                   difficulty_range: Tuple[float, float] = (0.1, 0.9)) -> List[Task]:
    out: List[Task] = []
    for domain in (domains or DOMAINS):
        out.extend(generate(domain, count_per_domain, seed, difficulty_range))
    return out
