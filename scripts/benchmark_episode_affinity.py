#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/benchmark_episode_affinity.py — score episode affinity OFFLINE,
on real logged dialogues, before anything touches _build_context.

Why offline first: wiring segmentation into the live context path and
then discovering it dropped half a conversation is the expensive order to
find out. This measures the two failure modes separately, because they
have very different costs:

    CONTAMINATION -- turns from another episode pulled into context.
                     Produced the observed "матч ЦСКА, который мог
                     состояться 126 дней назад": a number from an
                     arithmetic turn welded onto a football question.

    LOSS          -- turns from the SAME episode left out of context.
                     Never measured before this script, and the more
                     dangerous of the two: contamination confuses an
                     answer, loss removes the ability to answer at all.

    RECALL@k      -- is the correct episode among the top k by affinity?
                     If it is, a clarifying question can recover; if not,
                     no amount of asking helps.

The turns below are transcribed from real sessions, not invented. Episode
labels are the honest reading of those sessions, including one turn that
genuinely belongs to two episodes (marked), because pretending every turn
has one true episode is exactly the assumption this design rejects.

Usage:
    python scripts/benchmark_episode_affinity.py
    python scripts/benchmark_episode_affinity.py --floor 0.08
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mana.episode_affinity import SCORERS, Episode, rank_episodes, score_ngrams  # noqa: E402

#: (text, episodes-it-belongs-to). Transcribed from real sessions.
#: "A" = AI news, "B" = date arithmetic, "C" = football.
LABELLED_TURNS = [
    ("Привет, Мана. Какие есть последние новости про ИИ?", {"A"}),
    ("Но 28 мая это очень давно, ты знаешь какая сейчас дата?", {"A"}),
    ("Сколько прошло дней с 29 мая?", {"B"}),
    # Genuinely spans both: it is arithmetic ABOUT the news freshness.
    ("126 дней, думаешь это самые свежие новости?", {"A", "B"}),
    ("То есть РИА новости не публиковала новости свежее чем 28 мая?", {"A"}),
    ("Скажи как прошел матч ЦСКА Локомотив? какой был счет?", {"C"}),
    ("Как сыграли Зенит и ЦСКА? Какой был счет?", {"C"}),
]

#: Follow-up queries and the episode whose turns SHOULD form their context.
QUERIES = [
    ("Узнай когда был последний матч и с каким счетом завершился", "C"),
    ("что там с новостями", "A"),
    ("а какой был счёт", "C"),
    ("сколько это дней получилось", "B"),
    ("что там про ИИ говорили", "A"),
]


def build_episodes(exclude: str = "") -> list:
    episodes = {}
    for text, labels in LABELLED_TURNS:
        for label in labels:
            episodes.setdefault(label, Episode(label)).add(text)
    return [ep for ep in episodes.values() if ep.episode_id != exclude]


def evaluate(scorer: str, floor: float) -> dict:
    episodes = build_episodes()
    by_id = {ep.episode_id: ep for ep in episodes}

    hits_at_1 = hits_at_2 = ties = 0
    contamination_num = contamination_den = 0
    loss_num = loss_den = 0
    detail = []

    for query, truth in QUERIES:
        ranked = rank_episodes(query, episodes, scorer)
        top_ids = [eid for eid, _ in ranked]
        # A tie at the top is NOT a hit. `words` scored R@1 1.00 on an
        # early run purely because every episode scored 0.0 and the first
        # in list order won -- luck recorded as skill. Discrimination
        # requires the top score to be strictly greater than the next.
        top_score = ranked[0][1]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        decisive = top_score > runner_up and top_score > 0.0
        ties += int(not decisive)
        hits_at_1 += int(decisive and top_ids[0] == truth)
        hits_at_2 += int(truth in top_ids[:2] and top_score > 0.0)

        # Context = turns from the top-ranked episode scoring above `floor`
        # against the query. Both failure modes are measured on that set.
        chosen = by_id[top_ids[0]]
        fn = SCORERS[scorer]
        selected = [t for t in chosen.turns if fn(query, t) >= floor]

        truth_turns = {t for t, labels in LABELLED_TURNS if truth in labels}
        foreign = [t for t in selected if t not in truth_turns]
        missed = [t for t in truth_turns if t not in selected]

        contamination_num += len(foreign)
        contamination_den += max(1, len(selected))
        loss_num += len(missed)
        loss_den += max(1, len(truth_turns))

        detail.append({
            "query": query, "truth": truth,
            "ranked": [(eid, round(score, 3)) for eid, score in ranked],
            "top_correct": top_ids[0] == truth,
            "selected": len(selected), "foreign": len(foreign), "missed": len(missed),
        })

    return {
        "scorer": scorer, "floor": floor,
        "undecided_ties": ties / len(QUERIES),
        "recall_at_1": hits_at_1 / len(QUERIES),
        "recall_at_2": hits_at_2 / len(QUERIES),
        "contamination": contamination_num / max(1, contamination_den),
        "loss": loss_num / max(1, loss_den),
        "detail": detail,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor", type=float, default=0.05)
    args = parser.parse_args()

    print(f"Episode affinity benchmark | {len(LABELLED_TURNS)} labelled turns, "
          f"{len(QUERIES)} queries | floor={args.floor}\n")
    print(f"{'scorer':10} {'R@1':>6} {'R@2':>6} {'ties':>6} {'contamination':>15} {'loss':>8}")
    print("-" * 58)

    results = []
    for scorer in SCORERS:
        r = evaluate(scorer, args.floor)
        results.append(r)
        print(f"{scorer:10} {r['recall_at_1']:6.2f} {r['recall_at_2']:6.2f} "
              f"{r['undecided_ties']:6.2f} {r['contamination']:15.2f} {r['loss']:8.2f}")

    best = max(results, key=lambda r: (r["recall_at_1"], -r["loss"], -r["contamination"]))
    print(f"\nBest by R@1, then loss, then contamination: {best['scorer']}")
    print("\nPer-query detail for the best scorer:")
    for row in best["detail"]:
        mark = "ok  " if row["top_correct"] else "MISS"
        print(f"  [{mark}] truth={row['truth']} ranked={row['ranked']}")
        print(f"         selected={row['selected']} foreign={row['foreign']} missed={row['missed']}"
              f"  | {row['query']}")

    print("\nHow to read this. Contamination confuses an answer; LOSS removes the "
          "ability to answer at all, so a scorer with lower loss is preferable "
          "even at slightly higher contamination. R@1 below ~0.8 means the top "
          "episode is often wrong -- but if R@2 is high, a clarifying question "
          "can still recover, which is the whole point of ranking instead of "
          "deciding. Nothing here justifies wiring this into _build_context yet.")

    out = REPO_ROOT / "mana_affinity_benchmark.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
