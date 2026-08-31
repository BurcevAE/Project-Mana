#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/calibrate_memory_relevance.py — measure the right relevance floor
for YOUR machine and model instead of trusting a guessed default.

Why this exists: `Config.memory_min_relevance_embedding` defaults to 0.45,
which is a reasoned figure, NOT a measured one. It was chosen without
access to a sentence-transformers model -- the environment this code was
written in had none, which is exactly how the first version of the fix
shipped a single threshold that silently filtered nothing on machines
where embeddings were available.

This script prints the actual similarity scores your model produces for
related and unrelated pairs, so you can set the floor from data.

Usage:
    python scripts/calibrate_memory_relevance.py

Read the output like this: you want a threshold ABOVE the highest
"unrelated" score and BELOW the lowest "related" score. If those two
ranges overlap, no single threshold separates them cleanly -- the script
says so rather than pretending otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mana.config import Config          # noqa: E402
from mana.knowledge import KnowledgeBase  # noqa: E402
from mana.optional_deps import HAS_SENTENCE_TRANSFORMERS, HAS_SKLEARN  # noqa: E402

STORED = ("Задача: какие последние новости про ИИ\n"
          "Ответ: РИА Новости 18 июня, Известия — прямая трансляция.")

RELATED = [
    "какие последние новости про ИИ",
    "новости про искусственный интеллект",
    "что там было про ИИ в новостях",
    "расскажи про новости ИИ ещё раз",
]

UNRELATED = [
    "Привет, Мана. Я твой создатель, Алексей.",
    "Мана, ты меня понимаешь?",
    "Как дела?",
    "Который час?",
    "напиши функцию сортировки",
    "сколько будет 17 умножить на 23",
]

#: Meta-remarks ABOUT the conversation. They share vocabulary with the
#: stored entry but are not requests for that information -- the user is
#: complaining, correcting, or asking a question about the exchange
#: itself. Whether similarity alone can separate these from RELATED is
#: the open empirical question; this script measures it instead of
#: assuming an answer.
META = [
    "Разве я просил новости?",
    "Я не спрашивал про новости",
    "Зачем ты дал мне новости?",
    "Хватит про новости",
    "Почему ты опять вспомнил про ИИ?",
]


def main() -> int:
    import tempfile

    print(f"sentence-transformers available: {HAS_SENTENCE_TRANSFORMERS}")
    print(f"sklearn (TF-IDF) available:      {HAS_SKLEARN}")
    print()

    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(knowledge_db_path=str(Path(tmp) / "kb.pkl"))
        # Measure whichever path this machine would actually use.
        kb = KnowledgeBase(cfg)
        kb.add(STORED, source="llm", confidence=0.55, status="unverified")
        entry = kb.entries[0]

        path = "embedding" if entry.embedding is not None else ("tfidf/overlap")
        print(f"scoring path in use: {path}")
        current_floor = (cfg.memory_min_relevance_embedding if path == "embedding"
                         else cfg.memory_min_relevance_lexical)
        print(f"current configured floor: {current_floor}\n")

        def score(query: str) -> float:
            """Compute the same similarity KnowledgeBase.search would, on
            whichever path this machine uses."""
            import re

            import numpy as np

            q = kb._embed(query)
            if q is not None and entry.embedding is not None:
                return float(np.dot(q, entry.embedding))
            if kb.tfidf_vectorizer is not None and kb.tfidf_matrix is not None:
                from mana.optional_deps import cosine_similarity
                sims = cosine_similarity(kb.tfidf_vectorizer.transform([query]),
                                          kb.tfidf_matrix).ravel()
                return float(sims[0])
            qw = set(re.findall(r"\w+", query.lower()))
            ew = set(re.findall(r"\w+", entry.content.lower()))
            return len(qw & ew) / max(1, len(qw | ew))

        print("RELATED (should score HIGH -- these must stay retrievable):")
        related_scores = []
        for q in RELATED:
            s = score(q)
            related_scores.append(s)
            print(f"  {s:6.3f}  {q}")

        print("\nUNRELATED (should score LOW -- these must be filtered out):")
        unrelated_scores = []
        for q in UNRELATED:
            s = score(q)
            unrelated_scores.append(s)
            print(f"  {s:6.3f}  {q}")

        print("\nMETA (about the conversation, NOT requests for the stored info):")
        meta_scores = []
        for q in META:
            s = score(q)
            meta_scores.append(s)
            print(f"  {s:6.3f}  {q}")

        lo_related = min(related_scores)
        hi_unrelated = max(unrelated_scores)
        hi_meta = max(meta_scores)
        print("\n" + "=" * 60)
        print(f"lowest related:    {lo_related:.3f}")
        print(f"highest unrelated: {hi_unrelated:.3f}")
        print(f"highest meta:      {hi_meta:.3f}")

        print("\n--- Question 1: can a threshold reject UNRELATED? ---")
        if lo_related > hi_unrelated:
            suggested = round((lo_related + hi_unrelated) / 2, 2)
            print(f"YES. The groups separate cleanly. Suggested floor: {suggested}")
            print(f"Set Config.memory_min_relevance_"
                  f"{'embedding' if path == 'embedding' else 'lexical'} = {suggested}")
            if not (hi_unrelated < current_floor <= lo_related):
                print(f"WARNING: the current floor ({current_floor}) does NOT sit in "
                      f"the separating range ({hi_unrelated:.3f} .. {lo_related:.3f}) "
                      f"-- it is wrong for this machine.")
            else:
                print(f"The current floor ({current_floor}) is inside the separating "
                      f"range and is therefore sound.")
        else:
            print("NO -- the groups OVERLAP. Relevance filtering alone cannot solve "
                  "this; the retrieval signal itself needs improving. Do not tune the "
                  "number until it 'passes'.")

        print("\n--- Question 2: can a threshold reject META? ---")
        if lo_related > hi_meta:
            print(f"YES -- meta-remarks score below every genuine request "
                  f"({hi_meta:.3f} < {lo_related:.3f}). A threshold between them would "
                  f"work, and separate intent detection may not be needed.")
        else:
            print(f"NO -- meta-remarks reach {hi_meta:.3f}, at or above the weakest "
                  f"genuine request ({lo_related:.3f}). No single threshold separates "
                  f"'asking about X' from 'complaining about X': raising the floor to "
                  f"exclude meta-remarks would also start rejecting real queries.")
            print("      This is the empirical case FOR intent detection -- and the "
                  "reason not to fix it by tuning the threshold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
