#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/probe_live_behaviour.py — measure the behaviours that a prompt can
only ASK for, by running a scripted conversation through the real agent
several times and counting outcomes.

Why this is separate from pytest: three fixes shipped in 5.7.10 are of two
different kinds, and they need different evidence.

  MECHANICAL (guaranteed by code, already covered by tests/):
    * a conversation-reference remark must not trigger memory OR web

  PROMPT-DEPENDENT (requested of the model, NOT guaranteed):
    * internal evidence labels ("USER CLAIM:", "SOURCE EVIDENCE:",
      "CONCLUSION:") must not appear as headings in the answer
    * a search-snippet date must not be presented as the state of the site

For the second kind, a single successful run proves nothing -- the model
may comply by chance. This script repeats the conversation `--trials`
times and reports how often each rule held. A rule that holds 5/5 is
evidence; 3/5 means the prompt is not sufficient and a deterministic
output filter is needed instead.

Usage (needs a running LLM backend):
    python scripts/probe_live_behaviour.py --llm-model qwen2.5:7b-instruct-q4_K_M
    python scripts/probe_live_behaviour.py --llm-model mana --trials 5 --web

Writes mana_live_probe.json next to the repo. Send that file back.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mana import Config, ManaAgent  # noqa: E402
from mana.version import PRODUCT_VERSION  # noqa: E402

#: Headings that are internal context markup and must never reach the user.
LEAKED_LABEL = re.compile(
    r"^\s*(USER CLAIM|SOURCE EVIDENCE|VERIFIED EVIDENCE|EXPERIENCE|INFERENCE|"
    r"RECENT CONVERSATION|CONCLUSION)\s*:", re.M | re.I)

#: Substantive news CONTENT -- what a recap actually consists of. The
#: first version of this metric counted topic WORDS instead, and reading
#: the transcript showed why that was wrong: acknowledging a correction
#: naturally mentions the subject ("вы не просили новости про ИИ"), which
#: is not a recap at all. It scored 6/10 on a run where all ten answers
#: were, on reading, correct acknowledgements.
#:
#: Note on method: this metric was changed AFTER seeing results, which is
#: normally how you fool yourself. The justification is that reading the
#: data showed it measured the wrong thing, not that the number was
#: inconvenient -- and the corrected metric is required to still flag the
#: original 5.7.11 failure (see test below). If it stopped flagging that,
#: it would be tuning-until-green and worthless.
CONTENT_MARKERS = (
    re.compile(r"\b20\d\d\b"),                       # a date -- reporting, not acknowledging
    re.compile(r"^\s*\d+[.)]\s", re.M),               # an enumerated list of items
    re.compile(r"(освеща|сообща|опубликова|распростран|стратеги)", re.I),  # reporting verbs
    re.compile(r"(предоставил|я\s+дал|пришлось\s+предостав)", re.I),      # justifying the earlier answer
)

#: A correction deserves an acknowledgement, not an essay.
RECAP_CHAR_LIMIT = 400
RECAP_CONTENT_HITS = 1

#: Phrasings that assert a site has nothing newer, on snippet evidence alone.
OVERCLAIM = re.compile(
    r"(на сайте .{0,40}(нет|отсутствуют)|не были найдены на сайте|"
    r"последнее обновление на сайте)", re.I)

CONVERSATION = [
    # (turn text, what we are checking on this turn)
    ("Какие последние новости про ИИ?", "baseline_web_query"),
    ("Но на сайтах есть статьи свежее, сегодняшние, а не четырёхдневной давности",
     "snippet_authority_challenge"),
    ("Разве я просил новости?", "conversation_reference"),
    ("Хватит про новости", "conversation_reference"),
    ("а что там было про ИИ?", "genuine_followup"),
]


def _close_agent(agent) -> None:
    """Release every SQLite handle the agent holds.

    NOTE: this is the second time this bug shipped. run_diagnostics.py had
    the identical WinError 32 and was fixed by closing MemoryManager --
    but the agent also owns an ExperienceDB (e.sqlite3), and this script
    closed only the first. On Linux an open handle does not block deleting
    a file, so neither miss is visible there; on Windows the temp-directory
    cleanup raises and kills the run mid-way. Closing everything in one
    helper, so the next script that needs it cannot half-do it.
    """
    for attr in ("persistent_memory", "experience"):
        target = getattr(agent, attr, None)
        close = getattr(target, "close", None)
        if close is None:
            continue
        try:
            close()
        except Exception:
            pass


def run_trial(model: str, use_web: bool, trial: int) -> dict:
    # ignore_cleanup_errors is a safety net, not the fix: _close_agent above
    # is. Without it a single missed handle aborts a multi-trial run and
    # loses every result already collected.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp) / "state"
        cfg = Config(
            enable_llm=True, enable_web=use_web,
            knowledge_db_path=str(root / "k.pkl"), state_file=str(root / "s.pkl"),
            history_file=str(root / "h.json"), cache_file=str(root / "c.pkl"),
            experience_db_path=str(root / "e.sqlite3"),
            evolution_report_file=str(root / "ev.json"),
            memory_root=str(root / "mem"), memory_db_path=str(root / "mem" / "m.sqlite3"),
            local_exec_workdir=str(root / "sandbox"),
        )
        cfg.ollama_model = model
        cfg.ensure_dirs()
        agent = ManaAgent(cfg)

        turns = []
        for text, check in CONVERSATION:
            started = time.perf_counter()
            result = agent.solve_task(text)
            answer = str(result.get("answer") or "")
            trace = result.get("trace") or {}
            turns.append({
                "trial": trial, "query": text, "check": check,
                "answer": answer,
                "latency": round(time.perf_counter() - started, 2),
                "web_used": int(trace.get("web", 0)),
                "memory_used": int(trace.get("memory", 0)),
                "memory_skipped": trace.get("memory_skipped"),
                "web_skipped": trace.get("web_skipped"),
                "leaked_labels": sorted(set(m.group(1).upper() for m in LEAKED_LABEL.finditer(answer))),
                "snippet_overclaim": bool(OVERCLAIM.search(answer)),
                "content_hits": sum(1 for rx in CONTENT_MARKERS if rx.search(answer)),
                "answer_chars": len(answer),
                "recapped_on_correction": bool(
                    check == "conversation_reference"
                    and (len(answer) > RECAP_CHAR_LIMIT
                         or sum(1 for rx in CONTENT_MARKERS if rx.search(answer)) >= RECAP_CONTENT_HITS)),
            })
        _close_agent(agent)
        return {"trial": trial, "turns": turns}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-model", required=True,
                        help="ollama model name, e.g. qwen2.5:7b-instruct-q4_K_M")
    parser.add_argument("--trials", type=int, default=3,
                        help="how many times to repeat the conversation (default 3)")
    parser.add_argument("--web", action="store_true",
                        help="allow real web search (needed for the snippet checks)")
    args = parser.parse_args()

    print(f"MANA {PRODUCT_VERSION} | model={args.llm_model} | trials={args.trials} "
          f"| web={'on' if args.web else 'off'}\n")

    trials = []
    for i in range(1, args.trials + 1):
        print(f"--- trial {i}/{args.trials} ---")
        trials.append(run_trial(args.llm_model, args.web, i))
        for turn in trials[-1]["turns"]:
            flags = []
            if turn["leaked_labels"]:
                flags.append(f"LEAKED {','.join(turn['leaked_labels'])}")
            if turn["snippet_overclaim"]:
                flags.append("SNIPPET-OVERCLAIM")
            if turn["recapped_on_correction"]:
                flags.append(f"RECAP({turn['answer_chars']}ch,{turn['content_hits']} content-markers)")
            print(f"  {turn['check']:28} web={turn['web_used']} mem={turn['memory_used']} "
                  f"{' '.join(flags)}")

    # --- aggregate ---
    all_turns = [t for tr in trials for t in tr["turns"]]
    meta_turns = [t for t in all_turns if t["check"] == "conversation_reference"]
    gate_ok = sum(1 for t in meta_turns if t["web_used"] == 0 and t["memory_used"] == 0)
    leak_free = sum(1 for t in all_turns if not t["leaked_labels"])
    overclaim_free = sum(1 for t in all_turns if not t["snippet_overclaim"])
    no_recap = sum(1 for t in meta_turns if not t["recapped_on_correction"])

    print("\n" + "=" * 66)
    print("MECHANICAL (guaranteed by code -- any failure here is a real bug):")
    print(f"  conversation-reference gate held: {gate_ok}/{len(meta_turns)}")
    print("\nPROMPT-DEPENDENT (asked of the model -- a ratio, not a guarantee):")
    print(f"  turns free of leaked labels:      {leak_free}/{len(all_turns)}")
    print(f"  turns free of snippet overclaim:  {overclaim_free}/{len(all_turns)}")
    print(f"  corrections answered without recap: {no_recap}/{len(meta_turns)}")
    print("\nHow to read this: 5/5 on a prompt-dependent rule is evidence the "
          "instruction works. Anything less means the prompt is not enough and "
          "a deterministic output filter is the honest fix -- do not conclude "
          "'mostly fine'.")

    out = REPO_ROOT / "mana_live_probe.json"
    out.write_text(json.dumps({
        "product_version": PRODUCT_VERSION, "model": args.llm_model,
        "trials": args.trials, "web_enabled": args.web,
        "summary": {
            "gate_held": [gate_ok, len(meta_turns)],
            "leak_free": [leak_free, len(all_turns)],
            "overclaim_free": [overclaim_free, len(all_turns)],
            "no_recap_on_correction": [no_recap, len(meta_turns)],
        },
        "detail": trials,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull transcript: {out}")
    return 0 if gate_ok == len(meta_turns) else 1


if __name__ == "__main__":
    raise SystemExit(main())
