"""
mana.agent_parts.context — ContextMixin: intent detection (recall/memory-write/CLI), evidence ranking and prompt/context construction for a task.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import pickle
import random
import re
import statistics
import sys
import threading
import time
import subprocess
import tempfile
import shutil
import platform
import ast
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import Config, RandomManager
from ..knowledge import KnowledgeBase
from ..web import WebSearcher
from ..llm import LLMClient
from ..pipeline import PipelineSpec, PipelineFactory, BenchmarkTask, BenchmarkSuite
from ..intent import refers_to_previous_turn
from ..experience import ExperienceDB
from ..verifier import LocalVerifier
from ..memory import MemoryManager
from ..optional_deps import fitz, HAS_FITZ, HAS_SKLEARN, LogisticRegression, HAS_TORCH, DEVICE, HAS_WEB, WEB_BACKEND, torch

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.5"


class ContextMixin:
    def _is_recall_request(self, task: str) -> bool:
        t=(task or "").lower().strip()
        patterns=("что ты помнишь", "что мы обсуждали", "помнишь", "предыдущий разговор", "вспомни", "что я говорил", "что я тебе говорил")
        return any(p in t for p in patterns)

    def _is_memory_write_request(self, task: str) -> bool:
        t=(task or "").lower().strip()
        return bool(re.match(r"^(запомни|запиши|сохрани|помни|важно помнить|считай что|держи в памяти)\b", t))

    def _extract_memory_text(self, task: str) -> str:
        t=(task or "").strip()
        return re.sub(r"^(запомни|запиши|сохрани|помни|важно помнить|считай что|держи в памяти)\s*[:,-]?\s*", "", t, flags=re.I).strip()

    def _is_cli_command(self, task: str) -> bool:
        t=(task or "").strip().lower()
        return bool(re.match(r"^(python|py)\s+mana_[^\s]+(?:\.py)?\s+--[a-z0-9_-]+", t))

    def _conversation_recall_context(self, query: str) -> Tuple[str, Dict[str, Any]]:
        pdata=self.persistent_memory.get_session(self.session_id) or {}
        recent=self.persistent_memory.recent_events(self.session_id, limit=self.config.memory_recent_messages) if hasattr(self.persistent_memory, "recent_events") else []
        relevant=self.persistent_memory.semantic_search(query, limit=self.config.memory_retrieval_limit, session_id=self.session_id, cross_session=self.config.memory_cross_session)
        facts=[]
        try:
            with self.persistent_memory.lock:
                rows=self.persistent_memory.con.execute("SELECT * FROM facts WHERE provenance LIKE ? ORDER BY confidence DESC LIMIT 12", (f"%\"session_id\": \"{self.session_id}\"%",)).fetchall()
                facts=[dict(r) for r in rows]
        except Exception:
            facts=[]
        parts=["[RECENT CONVERSATION]"]
        for e in recent[-self.config.memory_recent_messages:]:
            parts.append(f"{e.get('kind','')}: {e.get('content','')}")
        if pdata.get("summary"):
            parts += ["[SESSION SUMMARY]", str(pdata.get("summary"))]
        if pdata.get("working_context"):
            parts += ["[WORKING MEMORY]", str(pdata.get("working_context"))]
        if facts:
            parts.append("[USER/SESSION FACTS]")
            parts.extend([f"{f.get('subject')} {f.get('predicate')}: {f.get('object')} (status={f.get('status','')})" for f in facts])
        if relevant:
            parts.append("[RELEVANT MEMORY]")
            parts.extend([str(x.get("text", "")) for x in relevant[:self.config.memory_retrieval_limit]])
        return "\n".join(parts), {"recent": recent, "relevant": relevant, "facts": facts, "session": pdata}

    def _evidence_profile(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize provenance and assign epistemic class/priority."""
        prov = item.get("provenance", {})
        if isinstance(prov, str):
            try: prov = json.loads(prov)
            except Exception: prov = {"raw": prov}
        prov = prov if isinstance(prov, dict) else {}
        item_type = str(item.get("item_type", ""))
        status = str(prov.get("status", item.get("status", "")))
        source = str(prov.get("source", ""))
        if item_type == "user_claim" or status == "user_claim":
            kind, base = "USER_CLAIM", 0.62
        elif status in {"verified", "experiment_verified", "verified_by_execution"}:
            kind, base = "VERIFIED", 1.00
        elif item_type in {"fact", "knowledge_fact"} and source in {"documentation", "official", "experiment"}:
            kind, base = "SOURCE", 0.88
        elif item_type == "document_chunk" or source in {"web", "document", "official"}:
            kind, base = "SOURCE", 0.82
        elif item_type in {"procedure", "experience", "episode", "decision"}:
            kind, base = "EXPERIENCE", 0.78
        else:
            kind, base = "INFERENCE", 0.64
        item["evidence_kind"] = kind
        item["evidence_priority"] = base
        item["provenance"] = prov
        return item

    def _evidence_intent(self, task: str) -> str:
        t=(task or "").lower()
        if self._is_recall_request(t): return "conversation"
        if any(x in t for x in ["кто я", "что обо мне", "меня", "мой проект", "мы решили", "что мы"]): return "conversation"
        if any(x in t for x in ["как работает", "что такое", "объясни", "документац", "1с", "скд", "язык запросов"]): return "knowledge"
        if any(x in t for x in ["почему", "ошиб", "не работает", "как исправ", "что произошло"]): return "mixed_reasoning"
        return "general"

    def _rank_evidence(self, task: str, items: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        intent=self._evidence_intent(task); ranked=[]
        preferred={
            "conversation":{"USER_CLAIM":1.18,"EXPERIENCE":1.05,"VERIFIED":0.95,"SOURCE":0.72,"INFERENCE":0.60},
            "knowledge":{"VERIFIED":1.25,"SOURCE":1.15,"EXPERIENCE":0.96,"INFERENCE":0.72,"USER_CLAIM":0.58},
            "mixed_reasoning":{"VERIFIED":1.18,"EXPERIENCE":1.10,"SOURCE":1.02,"USER_CLAIM":0.78,"INFERENCE":0.70},
            "general":{"VERIFIED":1.08,"SOURCE":1.00,"EXPERIENCE":0.95,"USER_CLAIM":0.86,"INFERENCE":0.72},
        }[intent]
        for raw in items:
            item=self._evidence_profile(dict(raw)); sim=float(item.get("retrieval_score",0.0));
            score=sim*preferred.get(item["evidence_kind"],1.0)*float(item.get("evidence_priority",0.7))
            item["evidence_score"]=float(score); ranked.append(item)
        ranked.sort(key=lambda x:(x.get("evidence_score",0.0),x.get("updated_at",0.0)),reverse=True)
        return ranked[:max(1,int(limit))]

    def _render_evidence_context(self, task: str) -> Tuple[str, Dict[str, Any]]:
        """Build grounded evidence from persistent memory without exposing embeddings."""
        data=self.persistent_memory.context_for(self.session_id, task, True)
        candidates=[]
        candidates.extend(data.get("semantic",[]))
        candidates.extend(data.get("facts",[]))
        candidates.extend(data.get("relevant",[]))
        candidates.extend(data.get("claims",[]))
        unique={}
        for item in candidates:
            txt=str(item.get("text", item.get("object", ""))).strip()
            if not txt: continue
            key=(str(item.get("item_type",item.get("kind",""))),txt)
            unique[key]=dict(item)
        ranked=self._rank_evidence(task,list(unique.values()),limit=max(8,int(getattr(self.config,"memory_retrieval_limit",8))*2))
        parts=[]
        if data.get("summary"): parts.append("[SESSION SUMMARY]\n"+str(data["summary"]))
        if data.get("active_task"): parts.append("[ACTIVE TASK]\n"+str(data["active_task"]))
        recent=data.get("recent",[])[-self.config.memory_recent_messages:]
        if recent:
            parts.append("[RECENT CONVERSATION]\n"+"\n".join(("USER" if r.get("kind")=="USER_MESSAGE" else "MANA")+": "+str(r.get("content","")) for r in recent))
        grouped={"VERIFIED":[],"SOURCE":[],"EXPERIENCE":[],"USER_CLAIM":[],"INFERENCE":[]}
        for item in ranked: grouped.setdefault(item["evidence_kind"],[]).append(item)
        labels={"VERIFIED":"[VERIFIED EVIDENCE]","SOURCE":"[SOURCE EVIDENCE]","EXPERIENCE":"[EXPERIENCE]","USER_CLAIM":"[USER CLAIM]","INFERENCE":"[INFERENCE]"}
        for kind in ("VERIFIED","SOURCE","EXPERIENCE","USER_CLAIM","INFERENCE"):
            vals=grouped.get(kind,[])
            if not vals: continue
            lines=[]
            for item in vals[:6]:
                txt=str(item.get("text",item.get("object",""))).strip()
                prov=item.get("provenance",{}) if isinstance(item.get("provenance",{}),dict) else {}
                src=prov.get("source","")
                tail=(f" source={src}" if src else "")
                lines.append(f"- {txt} (confidence={float(item.get('confidence',0.5)):.2f}{tail})")
            parts.append(labels[kind]+"\n"+"\n".join(lines))
        text="\n\n".join(parts)
        budget=max(3000,int(getattr(self.config,"memory_context_budget_chars",11000)))
        if len(text)>budget: text=text[:budget]+"\n[CONTEXT BUDGET EXHAUSTED]"
        meta={"intent":self._evidence_intent(task),"evidence_count":len(ranked),"evidence":ranked,"session":data}
        return text,meta

    def _available_tools_line(self) -> str:
        """One line per available tool, for the answer prompt.

        Observed in a live session: asked "ты можешь посмотреть в
        интернете?", MANA answered "могу искать, но не могу делать это в
        реальном времени" -- which is false; it had just run three
        searches. The model was inventing a limitation because nothing
        ever told it what it can do. The registry already knows; it simply
        was not being shown.
        """
        try:
            available = [t for t in self.tools.list_tools() if t.get("available")]
        except Exception:
            return "- (список инструментов недоступен)"
        if not available:
            return "- сейчас внешние инструменты недоступны"
        return "\n".join(f"- {t['name']}: {t['description'].split('.')[0]}."
                          for t in available)

    def _detect_conversation_reference(self, task: str):
        """Is this turn about the previous exchange rather than the world?

        Requires a previous assistant turn to actually exist -- at the
        start of a session nothing can be a reference back, and treating
        an opening message as a correction would skip memory for no
        reason. Failure here degrades to "not a reference", i.e. normal
        memory retrieval, because losing context is worse than searching
        it unnecessarily.
        """
        try:
            has_previous = any(
                r.get("kind") == "MANA_RESPONSE"
                for r in self.persistent_memory.recent(self.session_id, 6))
        except Exception as exc:
            self._vlog(f"conversation-reference check could not read history: {exc}")
            return refers_to_previous_turn("", has_previous_assistant_turn=False)
        return refers_to_previous_turn(task, has_previous_assistant_turn=has_previous)

    def _graph_memory_context(self, task: str) -> Tuple[str, int]:
        """Read side of the graph memory layer: seed from semantic search,
        walk the graph a few hops, return a bounded [GRAPH MEMORY] block.
        Routed through self.tools ('search_graph_memory') like every other
        capability in this method, not a direct self.graph_memory call.
        Failure here must never break answering -- same degrade-gracefully
        contract as every other optional context source in this method."""
        if not getattr(self.config, "graph_memory_context_enabled", True):
            return "", 0
        result = self.tools.call(
            "search_graph_memory", query=task, session_id=self.session_id,
            depth=int(self.config.graph_memory_depth),
            limit=int(self.config.graph_memory_limit),
            seed_limit=int(self.config.graph_memory_seed_limit),
            char_budget=int(self.config.graph_memory_char_budget),
        )
        if not result.output:
            return "", 0
        used = result.meta.get("trace", {}).get("used", [])
        return f"[GRAPH MEMORY]\n{result.output}", len(used)

    def _build_context(self, task: str, spec: PipelineSpec) -> Tuple[str, Dict[str, Any]]:
        chunks = []
        conversation_reference = self._detect_conversation_reference(task)
        trace = {"memory": 0, "persistent_memory": 0, "web": 0, "web_attempted": False, "web_ok": False, "web_error": None,
                 "tool_health": 1.0, "web_latency": 0.0, "graph_memory": 0}
        # v5.4: evidence-grounded resolver ranks memory by epistemic type and task intent.
        try:
            pctx, pdata = self._render_evidence_context(task)
            if pctx:
                chunks.append(pctx)
                trace["persistent_memory"] = int(pdata.get("evidence_count", 0))
                trace["memory_intent"] = pdata.get("intent")
        except Exception as exc:
            self._vlog(f"evidence memory unavailable: {exc}")
            try:
                pctx, pdata = self.persistent_memory.render_prompt_context(self.session_id, task)
                if pctx:
                    chunks.append(pctx); trace["persistent_memory"] = len(pdata.get("recent", [])) + len(pdata.get("relevant", []))
            except Exception as exc2:
                self._vlog(f"persistent memory unavailable: {exc2}")
        # v5.5: graph memory -- distilled, multi-hop-linked context that flat
        # top-k retrieval above can miss (see mana/graph_memory.py). Kept as
        # its own labeled block rather than merged into evidence context so
        # its provenance (graph traversal vs. direct semantic/fact lookup)
        # stays visible to the LLM and to anyone reading the trace.
        try:
            gctx, gcount = self._graph_memory_context(task)
            if gctx:
                chunks.append(gctx)
                trace["graph_memory"] = gcount
        except Exception as exc:
            self._vlog(f"graph memory unavailable: {exc}")
        # v5.7.9: a turn that refers to the PREVIOUS EXCHANGE is not a
        # request for stored knowledge, so long-term memory is not
        # consulted at all. Two rounds of measurement showed no similarity
        # threshold can separate "asking about X" from "complaining that
        # you gave me X" -- because it was never a relevance problem. See
        # mana/intent.py.
        if spec.use_memory and conversation_reference:
            trace["memory"] = 0
            trace["memory_skipped"] = "conversation_reference"
            trace["conversation_reference_kind"] = conversation_reference.kind
        elif spec.use_memory:
            # v5.6: routed through self.tools ('search_knowledge_base') instead
            # of self.memory.search directly -- see the module-level design
            # note in mana/tools.py for why this call site qualifies as a
            # Tool (conditional on spec.use_memory) while, e.g.,
            # persistent_memory.context_for above does not (unconditional
            # infrastructure). The tool returns plain dicts (KnowledgeEntry.
            # to_dict()) rather than KnowledgeEntry objects, hence e["x"]
            # instead of e.x below -- same field names, same values.
            mem = self.tools.call("search_knowledge_base", query=task, top_k=spec.memory_top_k,
                                   min_confidence=spec.min_memory_confidence).output or []
            trace["memory"] = len(mem)
            for e in mem:
                chunks.append(f"[MEMORY confidence={e['confidence']:.2f} quality={e['quality']:.2f}]\n{e['content']}")
        # v5.7.10: the same gate must cover the WEB, not just stored memory.
        # Observed on real hardware: "Разве я просил новости?" still issued a
        # live search ("веб: 3 результатов" in the trace) because the earlier
        # fix only skipped KnowledgeBase. A remark about the previous turn is
        # not a request for current information from the world, so no
        # retrieval of any kind belongs here.
        if conversation_reference:
            trace["web_skipped"] = "conversation_reference"
            # v5.7.12: blocking retrieval was not enough. Measured over 5
            # live trials: the gate held (web=0, memory=0) but the answer
            # still recapped the topic, because RECENT CONVERSATION stays
            # in context -- and it must, since answering a correction
            # requires knowing what is being corrected. The missing piece
            # was never "less context", it was telling the model how to
            # RESPOND to a correction. Asked "Хватит про новости", MANA
            # replied with another paragraph of news.
            chunks.append(
                "[РЕПЛИКА О РАЗГОВОРЕ]\n"
                f"Это не новый вопрос, а замечание о предыдущем ответе "
                f"(тип: {conversation_reference.kind}). Ответь коротко на само замечание: "
                "признай его, уточни, что нужно, или предложи сменить тему. "
                "НЕ пересказывай и не повторяй ту тему снова — именно её повтор и "
                "вызвал замечание. Не оправдывайся и не объясняй, почему ты так ответил."
            )
        elif self._should_use_web(task, spec) and spec.web_results > 0:
            web_result = self.tools.call("web_search", query=task, max_results=spec.web_results)
            rows, ws = web_result.output or [], web_result.meta
            retrieved_at = time.strftime("%Y-%m-%d %H:%M")
            trace.update({"web": len(rows), "web_attempted": ws.get("attempted", False), "web_ok": ws.get("ok", False),
                          "web_error": ws.get("reason"), "tool_health": ws.get("tool_health", 1.0), "web_latency": ws.get("latency", 0.0),
                          "web_retrieved_at": retrieved_at})
            for row in rows:
                # Audit #60: stamp WHEN this was fetched, and state plainly
                # that it is a search-result snippet rather than page
                # content. Both were absent, so the model had no way to tell
                # a two-month-old blurb from today's headline -- and no way
                # to know it had not actually read the page.
                chunks.append(
                    f"[WEB | получено: {retrieved_at} | это сниппет поисковой выдачи, "
                    f"не содержимое страницы]\n"
                    f"{row.get('title','')}\n{row.get('body', row.get('snippet',''))}\n"
                    f"URL: {row.get('href', row.get('url',''))}")
        context = "\n\n".join(chunks)
        if len(context) > spec.max_context_chars:
            context = context[:spec.max_context_chars] + "\n[CONTEXT TRUNCATED]"
        return context, trace

    def _prompt_text(self, strategy: str) -> str:
        return {
            "direct": "Реши задачу по существу. Не добавляй лишнюю информацию.",
            "structured": "Дай структурированный ответ. Не раскрывай скрытые рассуждения.",
            "analytical": "Определи ключевые требования, проверь факты и только затем сформулируй ответ.",
            "verification": "Сначала проверь неоднозначность и возможные ошибки исходных данных, затем ответь.",
            "researcher": "Отделяй известные факты от предположений и критически используй внешний контекст.",
        }[strategy]

    def _compose_prompt(self, task: str, context: str, spec: PipelineSpec, previous: str = "") -> str:
        style = {"direct": "Ответь по существу.", "structured": "Дай структурированный ответ.", "concise": "Будь максимально краток, но точен."}[spec.synthesis_style]
        # Audit #60. The model has no clock: nothing in this prompt used to
        # say what day it is. That is why it could write "новости за
        # последние два дня" above a snippet dated 18 June -- not a
        # reasoning failure but a missing input, and no amount of extra
        # self-assessment can recover a fact that was never supplied.
        prompt = f"""Ты — когнитивный модуль MANA. Реши задачу пользователя.

Сегодняшняя дата: {time.strftime('%Y-%m-%d')} (используй её как точку отсчёта для любых суждений о времени).

Что ты умеешь прямо сейчас (это выполняет агент, не ты сама — результаты уже
в контексте ниже, если инструмент вызывался):
{self._available_tools_line()}
Не заявляй, что не можешь искать в интернете или работать в реальном времени,
если web_search доступен: поиск выполняется на каждом подходящем запросе.
Если данных в контексте нет — скажи, что в выдаче их не нашлось, а не что ты
принципиально не умеешь искать.

Задача:
{task}

Стратегия:
{self._prompt_text(spec.prompt_strategy)}

Правила:
- Не выдумывай факты.
- Внешний контекст не является автоматически истинным.
- Различай уровни доказательности: VERIFIED EVIDENCE, SOURCE EVIDENCE, EXPERIENCE, USER CLAIM и INFERENCE.
- Эти метки — служебная разметка контекста, а НЕ формат ответа. Никогда не выводи их пользователю как заголовки (`USER CLAIM:`, `SOURCE EVIDENCE:`, `CONCLUSION:` и т.п.). Отвечай обычным текстом; разницу в надёжности передавай словами.
- USER CLAIM никогда не превращай в VERIFIED FACT. Говори «ты сообщил», «в памяти сохранено как утверждение» и т.п.
- SOURCE EVIDENCE отражает внешний источник, но тоже не гарантирует истину без проверки.
- VERIFIED EVIDENCE имеет высший приоритет, но не выдумывай детали, которых в evidence нет.
- Для вопросов о памяти используй RECENT CONVERSATION и релевантные evidence, а не общую догадку.
- Если evidence противоречит друг другу, укажи противоречие вместо произвольного выбора.
- Не называй информацию свежей, актуальной или последней, если в контексте нет даты, подтверждающей это относительно сегодняшней даты. Если дата источника неизвестна или заметно старше сегодняшней — скажи об этом прямо.
- Дата в сниппете поисковой выдачи описывает ТОЛЬКО сам сниппет, а не сайт. Отсутствие свежих статей в сниппете не доказывает, что их нет на сайте: страницу ты не открывал. Не утверждай, что на ресурсе нет более новых материалов — говори «в выдаче не видно», и если пользователь сообщает о более свежих данных, не спорь со ссылкой на сниппет.
- Отвечай на языке пользователя. Не переключайся на другой язык посреди ответа.
- Если данных нет, не придумывай их: не сочиняй цифры, прогнозы, счёт или даты,
  которых нет в контексте. «Не нашлось» — полный и правильный ответ.
- {style}
"""
        if context: prompt += f"\nКонтекст:\n{context}\n"
        if previous: prompt += f"\nПредыдущий черновик. Улучши только при наличии оснований:\n{previous}\n"
        return prompt
