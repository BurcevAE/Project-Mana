"""
mana.memory — persistent SQLite WAL multi-tier memory: events, facts, episodes, semantic search.
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

from .config import Config
from .version import PRODUCT_VERSION

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"


# ---------------------------------------------------------------------------
# MANA v5.0 Memory Architecture
# ---------------------------------------------------------------------------

class MemoryManager:
    """Persistent, tiered memory for conversations, facts, episodes and working context.

    SQLite is the source of truth. WAL + FTS5 are used when available. The class is
    deliberately independent from the existing KnowledgeBase so v4.x state remains
    compatible and can be reused as long-term semantic memory.
    """
    VERSION = PRODUCT_VERSION

    def __init__(self, config: Config, embedder=None):
        self.config = config
        self.embedder = embedder
        self.path = Path(config.memory_db_path)
        self.lock = threading.RLock()
        self.con = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30.0)
        self.con.row_factory = sqlite3.Row
        with self.con:
            if config.memory_wal:
                try:
                    self.con.execute("PRAGMA journal_mode=WAL")
                except Exception:
                    pass
            self.con.execute("PRAGMA synchronous=NORMAL")
            self.con.execute("PRAGMA foreign_keys=ON")
            self._init_schema()

    def _init_schema(self) -> None:
        with self.lock, self.con:
            self.con.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                topic TEXT DEFAULT '',
                active_task TEXT DEFAULT '',
                summary TEXT DEFAULT '',
                working_context TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                source TEXT NOT NULL DEFAULT 'conversation',
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_session_time ON events(session_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_events_kind_time ON events(kind, created_at DESC);
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.55,
                status TEXT NOT NULL DEFAULT 'candidate',
                provenance TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);
            CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status);
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                source_path TEXT DEFAULT '',
                source_type TEXT NOT NULL DEFAULT 'file',
                domain TEXT DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'ingested',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents(domain);
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                UNIQUE(document_id, chunk_index),
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id, chunk_index);
            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.5,
                provenance TEXT NOT NULL DEFAULT '{}',
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                provenance TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                UNIQUE(subject, predicate, object)
            );
            CREATE TABLE IF NOT EXISTS memory_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_type TEXT NOT NULL,
                ref_id INTEGER,
                text TEXT NOT NULL,
                session_id TEXT DEFAULT '',
                importance REAL NOT NULL DEFAULT 0.5,
                confidence REAL NOT NULL DEFAULT 0.5,
                provenance TEXT NOT NULL DEFAULT '{}',
                embedding BLOB,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_items_type ON memory_items(item_type);
            CREATE INDEX IF NOT EXISTS idx_memory_items_session ON memory_items(session_id);
            CREATE INDEX IF NOT EXISTS idx_memory_items_importance ON memory_items(importance DESC);
            CREATE TABLE IF NOT EXISTS acquisition_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                domain TEXT DEFAULT '',
                source_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                fact_count INTEGER NOT NULL DEFAULT 0,
                concept_count INTEGER NOT NULL DEFAULT 0,
                started_at REAL NOT NULL,
                finished_at REAL NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                outcome TEXT DEFAULT '',
                provenance TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS procedures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger_text TEXT NOT NULL,
                action_text TEXT NOT NULL,
                success_rate REAL NOT NULL DEFAULT 0.0,
                observations INTEGER NOT NULL DEFAULT 0,
                provenance TEXT NOT NULL DEFAULT '{}',
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT UNIQUE NOT NULL,
                path TEXT DEFAULT '',
                kind TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            );
            """)
            if self.config.memory_fts_enabled:
                try:
                    self.con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS event_fts USING fts5(content, kind, session_id, content='events', content_rowid='id')")
                    self.con.execute("INSERT INTO event_fts(event_fts) VALUES('rebuild')")
                except Exception:
                    pass

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if self.embedder is None or not self.config.memory_vector_enabled:
            return None
        try:
            return self.embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        except Exception:
            return None

    @staticmethod
    def _pack_embedding(vec: Optional[np.ndarray]) -> Optional[bytes]:
        return None if vec is None else np.asarray(vec, dtype=np.float32).tobytes()

    @staticmethod
    def _unpack_embedding(blob: Optional[bytes]) -> Optional[np.ndarray]:
        if not blob:
            return None
        try:
            return np.frombuffer(blob, dtype=np.float32)
        except Exception:
            return None

    def upsert_memory_item(self, item_type: str, text: str, session_id: str = "", importance: float = 0.5, confidence: float = 0.5, provenance: Optional[Dict[str, Any]] = None, ref_id: Optional[int] = None) -> int:
        text=(text or "").strip()
        if not text: return 0
        now=time.time(); emb=self._pack_embedding(self._embed(text))
        with self.lock, self.con:
            cur=self.con.execute("INSERT INTO memory_items(item_type,ref_id,text,session_id,importance,confidence,provenance,embedding,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",(item_type,ref_id,text,session_id,float(importance),float(confidence),json.dumps(provenance or {},ensure_ascii=False),emb,now,now))
            return int(cur.lastrowid)

    def semantic_search(self, query: str, limit: Optional[int] = None, session_id: str = "", cross_session: Optional[bool] = None) -> List[Dict[str, Any]]:
        limit=int(limit or self.config.memory_semantic_top_k)
        cross_session=self.config.memory_cross_session if cross_session is None else cross_session
        qv=self._embed(query)
        with self.lock:
            if cross_session or not session_id:
                rows=self.con.execute("SELECT * FROM memory_items ORDER BY importance DESC, updated_at DESC LIMIT ?",(max(limit*25,100),)).fetchall()
            else:
                rows=self.con.execute("SELECT * FROM memory_items WHERE session_id=? OR session_id='' ORDER BY importance DESC, updated_at DESC LIMIT ?",(session_id,max(limit*25,100))).fetchall()
        qwords=set(re.findall(r"\w+",query.lower())); scored=[]
        for row in rows:
            d=dict(row); v=self._unpack_embedding(d.get("embedding")); sim=0.0
            if qv is not None and v is not None and len(qv)==len(v): sim=float(np.dot(qv,v))
            else:
                words=set(re.findall(r"\w+",d["text"].lower())); sim=len(qwords & words)/max(1,len(qwords|words))
            score=0.72*sim+0.18*float(d.get("importance",0.5))+0.10*float(d.get("confidence",0.5))
            d["retrieval_score"]=float(score); scored.append((score,d))
        scored.sort(key=lambda x:x[0],reverse=True)
        return [d for _,d in scored[:limit]]

    def remember_user_claim(self, session_id: str, content: str) -> Optional[int]:
        text=(content or "").strip(); low=text.lower()
        patterns=[(r"я\s+тв[aо]й\s+создател[ья]",0.88),(r"меня зовут\s+(.+)",0.92),(r"я\s+(.{2,80})",0.72)]
        for pat,conf in patterns:
            if re.search(pat,low,re.I):
                item=self.upsert_memory_item("user_claim",f"Пользователь сообщил: {text}",session_id,0.96,conf,{"source":"conversation","status":"user_claim"})
                self.add_fact("пользователь","сообщил",text,conf,"user_claim",{"source":"conversation","session_id":session_id})
                return item
        return None

    def search_global(self, query: str, limit: int = 8) -> List[Dict[str, Any]]:
        results = self.semantic_search(query,limit=limit,session_id="",cross_session=True)
        return results

    def safe_search_global(self, query: str, limit: int = 8) -> List[Dict[str, Any]]:
        """Return retrieval results without binary embeddings or other internal payloads."""
        rows = self.search_global(query, limit)
        clean = []
        for row in rows:
            item = dict(row)
            item.pop("embedding", None)
            clean.append(item)
        return clean

    def store_explicit_memory(self, session_id: str, text: str, scope: str = "session", importance: float = 0.9) -> Dict[str, Any]:
        text=(text or "").strip()
        if not text:
            return {"stored":False,"reason":"empty"}
        scope = scope if scope in {"session","project","long_term"} else "session"
        item_type = "explicit_memory"
        prov={"source":"user","status":"explicit_memory","scope":scope,"session_id":session_id}
        iid=self.upsert_memory_item(item_type, text, session_id if scope=="session" else "", max(0.0,min(1.0,importance)), 0.92, prov)
        self.add_fact("пользователь","попросил запомнить",text,0.92,"user_explicit",prov)
        self._event(session_id,"MEMORY_UPDATE",text,{"scope":scope,"item_id":iid,"importance":importance},"user")
        return {"stored":True,"item_id":iid,"scope":scope,"text":text}

        return self.semantic_search(query,limit=limit,session_id="",cross_session=True)

    def _split_chunks(self, text: str) -> List[str]:
        text=re.sub(r"\r\n?","\n",text or ""); text=re.sub(r"[ \t]+"," ",text).strip()
        if not text: return []
        size=max(400,int(self.config.memory_chunk_chars)); overlap=max(0,min(size//3,int(self.config.memory_chunk_overlap)))
        out=[]; start=0
        while start<len(text) and len(out)<self.config.memory_ingest_max_chunks:
            end=min(len(text),start+size)
            if end<len(text):
                cut=text.rfind("\n",start+size//2,end)
                if cut>start: end=cut
            chunk=text[start:end].strip()
            if chunk: out.append(chunk)
            if end>=len(text): break
            start=max(start+1,end-overlap)
        return out

    def ingest_document(self,title: str,text: str,source_path: str="",source_type: str="file",domain: str="",metadata: Optional[Dict[str,Any]]=None)->Dict[str,Any]:
        raw=(text or "").replace("\x00"," ").strip()
        if not raw: return {"document_id":0,"chunks":0,"facts":0,"concepts":0,"status":"empty"}
        digest=hashlib.sha256(raw.encode("utf-8")).hexdigest(); now=time.time()
        with self.lock,self.con:
            row=self.con.execute("SELECT id FROM documents WHERE content_hash=?",(digest,)).fetchone()
            if row:
                did=int(row["id"]); existing=int(self.con.execute("SELECT COUNT(*) FROM document_chunks WHERE document_id=?",(did,)).fetchone()[0])
                return {"document_id":did,"chunks":existing,"facts":0,"concepts":0,"status":"already_ingested"}
            cur=self.con.execute("INSERT INTO documents(content_hash,title,source_path,source_type,domain,metadata,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",(digest,title,source_path,source_type,domain,json.dumps(metadata or {},ensure_ascii=False),"ingested",now,now)); did=int(cur.lastrowid)
        chunks=self._split_chunks(raw)
        for i,ch in enumerate(chunks):
            emb=self._pack_embedding(self._embed(ch)); meta=json.dumps({"domain":domain,"title":title,"chunk_index":i},ensure_ascii=False)
            with self.lock,self.con:
                self.con.execute("INSERT INTO document_chunks(document_id,chunk_index,text,embedding,metadata,created_at) VALUES(?,?,?,?,?,?)",(did,i,ch,emb,meta,now))
            self.upsert_memory_item("document_chunk",ch,"",0.70,0.55,{"source":source_type,"document_id":did,"title":title,"domain":domain},did)
        facts=self._extract_simple_facts(raw,did,title,domain) if self.config.memory_auto_extract_facts else 0
        concepts=self._extract_concepts(raw,did,title,domain) if self.config.memory_auto_extract_concepts else 0
        return {"document_id":did,"chunks":len(chunks),"facts":facts,"concepts":concepts,"status":"ingested"}

    def _extract_simple_facts(self,text:str,doc_id:int,title:str,domain:str)->int:
        made=0
        for line in text.splitlines():
            line=line.strip()
            if len(line)<30 or len(line)>500: continue
            m=re.match(r"^([A-ZА-ЯЁ][^:]{2,80}):\s*(.+)$",line)
            if m:
                self.add_fact(m.group(1).strip(),"описание",m.group(2).strip(),0.62,"source_extracted",{"source":"document","document_id":doc_id,"title":title,"domain":domain}); made+=1
        return made

    def _extract_concepts(self,text:str,doc_id:int,title:str,domain:str)->int:
        candidates=[]
        for line in text.splitlines():
            line=line.strip(" #-*\t")
            if line and len(line)<=100 and re.search(r"(?:1С|СКД|Запрос|Справочник|Документ|Регистр|Конфигуратор|Общий модуль)",line,re.I): candidates.append(line)
        made=0; now=time.time()
        for name in dict.fromkeys(candidates):
            with self.lock,self.con:
                self.con.execute("INSERT INTO concepts(name,description,confidence,provenance,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET confidence=max(concepts.confidence,excluded.confidence),updated_at=excluded.updated_at,provenance=excluded.provenance",(name,name,0.62,json.dumps({"source":"document","document_id":doc_id,"domain":domain},ensure_ascii=False),now))
            made+=1
        return made

    def acquisition_stats(self)->Dict[str,Any]:
        with self.lock:
            return {t:int(self.con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in ("documents","document_chunks","concepts","relations","memory_items","acquisition_runs")}

    def close(self) -> None:
        with self.lock:
            try:
                self.con.close()
            except Exception:
                pass

    def ensure_session(self, session_id: str) -> None:
        now = time.time()
        with self.lock, self.con:
            self.con.execute("INSERT OR IGNORE INTO sessions(session_id, created_at, updated_at) VALUES(?,?,?)", (session_id, now, now))

    def _event(self, session_id: str, kind: str, content: str, metadata: Optional[Dict[str, Any]] = None, source: str = "conversation") -> int:
        self.ensure_session(session_id)
        now = time.time()
        meta = json.dumps(metadata or {}, ensure_ascii=False)
        with self.lock, self.con:
            cur = self.con.execute("INSERT INTO events(session_id, kind, content, metadata, source, created_at) VALUES(?,?,?,?,?,?)", (session_id, kind, content, meta, source, now))
            self.con.execute("UPDATE sessions SET updated_at=?, message_count=message_count+1 WHERE session_id=?", (now, session_id))
            if self.config.memory_fts_enabled:
                try:
                    self.con.execute("INSERT INTO event_fts(rowid, content, kind, session_id) VALUES(?,?,?,?)", (cur.lastrowid, content, kind, session_id))
                except Exception:
                    pass
            return int(cur.lastrowid)

    def remember_user(self, session_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        if not self.config.memory_store_user_messages:
            return 0
        return self._event(session_id, "USER_MESSAGE", content, metadata, "user")

    def remember_assistant(self, session_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        if not self.config.memory_store_assistant_responses:
            return 0
        return self._event(session_id, "MANA_RESPONSE", content, metadata, "mana")

    def remember_tool(self, session_id: str, kind: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        return self._event(session_id, kind.upper(), content, metadata, "tool")

    def remember_decision(self, session_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        return self._event(session_id, "DECISION", content, metadata, "mana")

    def recent(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        limit = int(limit or self.config.memory_recent_messages)
        with self.lock:
            rows = self.con.execute("SELECT id,kind,content,metadata,source,created_at FROM events WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def search(self, session_id: str, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        limit = int(limit or self.config.memory_retrieval_limit)
        results: List[Dict[str, Any]] = []
        with self.lock:
            if self.config.memory_fts_enabled:
                try:
                    tokens = [re.sub(r"[^\w\-]", "", x) for x in re.findall(r"\w+", query.lower()) if len(x) > 2]
                    if tokens:
                        expr = " OR ".join(tokens[:12])
                        rows = self.con.execute("SELECT e.id,e.kind,e.content,e.metadata,e.source,e.created_at FROM event_fts f JOIN events e ON e.id=f.rowid WHERE e.session_id=? AND event_fts MATCH ? ORDER BY rank LIMIT ?", (session_id, expr, limit)).fetchall()
                        results.extend(dict(r) for r in rows)
                except Exception:
                    pass
            if len(results) < limit:
                rows = self.con.execute("SELECT id,kind,content,metadata,source,created_at FROM events WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, limit * 3)).fetchall()
                q=set(re.findall(r"\w+", query.lower()))
                scored=[]
                seen={r["id"] for r in results}
                for r in rows:
                    if r["id"] in seen: continue
                    words=set(re.findall(r"\w+", r["content"].lower()))
                    overlap=len(q & words)/max(1,len(q|words))
                    if overlap > 0:
                        scored.append((overlap, dict(r)))
                scored.sort(key=lambda x:x[0], reverse=True)
                results.extend(v for _,v in scored[:max(0,limit-len(results))])
        return results[:limit]

    def get_session(self, session_id: str) -> Dict[str, Any]:
        self.ensure_session(session_id)
        with self.lock:
            row=self.con.execute("SELECT * FROM sessions WHERE session_id=?",(session_id,)).fetchone()
        return dict(row) if row else {}

    def update_working_context(self, session_id: str, task: str, topic: str = "", context: Optional[Dict[str, Any]] = None) -> None:
        self.ensure_session(session_id)
        now=time.time()
        payload=json.dumps(context or {}, ensure_ascii=False)
        with self.lock, self.con:
            self.con.execute("UPDATE sessions SET active_task=?, topic=COALESCE(NULLIF(?,''),topic), working_context=?, updated_at=? WHERE session_id=?",(task,topic,payload,now,session_id))

    def set_summary(self, session_id: str, summary: str) -> None:
        self.ensure_session(session_id)
        with self.lock, self.con:
            self.con.execute("UPDATE sessions SET summary=?, updated_at=? WHERE session_id=?",(summary[:self.config.memory_max_summary_chars],time.time(),session_id))

    def add_fact(self, subject: str, predicate: str, obj: str, confidence: Optional[float] = None, status: str = "candidate", provenance: Optional[Dict[str, Any]] = None) -> int:
        now=time.time(); conf=float(confidence if confidence is not None else self.config.memory_fact_confidence_default)
        with self.lock, self.con:
            cur=self.con.execute("INSERT INTO facts(subject,predicate,object,confidence,status,provenance,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",(subject,predicate,obj,conf,status,json.dumps(provenance or {},ensure_ascii=False),now,now))
            return int(cur.lastrowid)

    def fact_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        q='%'+query+'%'
        with self.lock:
            rows=self.con.execute("SELECT * FROM facts WHERE subject LIKE ? OR predicate LIKE ? OR object LIKE ? ORDER BY confidence DESC, updated_at DESC LIMIT ?",(q,q,q,limit)).fetchall()
        return [dict(r) for r in rows]

    def add_episode(self, title: str, summary: str, outcome: str = "", session_id: Optional[str] = None, provenance: Optional[Dict[str, Any]] = None) -> int:
        with self.lock, self.con:
            cur=self.con.execute("INSERT INTO episodes(session_id,title,summary,outcome,provenance,created_at) VALUES(?,?,?,?,?,?)",(session_id,title,summary,outcome,json.dumps(provenance or {},ensure_ascii=False),time.time()))
            return int(cur.lastrowid)

    def maybe_compress(self, session_id: str) -> Optional[str]:
        state=self.get_session(session_id)
        if int(state.get("message_count",0)) < int(self.config.memory_auto_compress_after):
            return None
        recent=self.recent(session_id, min(self.config.memory_recent_messages, 8))
        if not recent: return None
        parts=[]
        for r in recent:
            prefix="Пользователь" if r["kind"]=="USER_MESSAGE" else "MANA"
            parts.append(f"{prefix}: {r['content'][:700]}")
        summary=(state.get("summary","") or "")
        new=(summary+"\n" if summary else "")+"\n".join(parts)
        new=new[-self.config.memory_max_summary_chars:]
        self.set_summary(session_id,new)
        self.add_episode("Автосжатие контекста", "Сводка последних событий сессии.", outcome="compressed", session_id=session_id, provenance={"method":"bounded_recent_context"})
        return new

    def context_for(self, session_id: str, task: str, include_recent: bool = True) -> Dict[str, Any]:
        state=self.get_session(session_id); recent=self.recent(session_id,self.config.memory_recent_messages) if include_recent else []
        relevant=self.search(session_id,task,self.config.memory_retrieval_limit)
        facts=self.fact_search(task,min(8,self.config.memory_retrieval_limit))
        semantic=self.semantic_search(task,self.config.memory_semantic_top_k,session_id=session_id,cross_session=self.config.memory_cross_session)
        with self.lock:
            claim_rows=self.con.execute("SELECT * FROM memory_items WHERE item_type='user_claim' AND (session_id=? OR session_id='') ORDER BY importance DESC, updated_at DESC LIMIT 6",(session_id,)).fetchall()
        claims=[dict(r) for r in claim_rows]
        return {"summary":state.get("summary","") or "","active_task":state.get("active_task","") or "","topic":state.get("topic","") or "","working_context":state.get("working_context","{}") or "{}","recent":recent,"relevant":relevant,"facts":facts,"semantic":semantic,"claims":claims}

    def render_prompt_context(self, session_id: str, task: str) -> Tuple[str, Dict[str, Any]]:
        data=self.context_for(session_id,task,True); chunks=[]
        if data["summary"]: chunks.append("[SESSION SUMMARY]\n"+data["summary"])
        if data["active_task"]: chunks.append("[ACTIVE TASK]\n"+data["active_task"])
        try:
            wc=json.loads(data.get("working_context") or "{}"); durable=wc.get("durable_memory",[]) if isinstance(wc,dict) else []
            if durable: chunks.append("[WORKING MEMORY]\n"+"\n".join(str(x) for x in durable[-8:]))
        except Exception: pass
        recent_lines=[]
        for r in data["recent"][-self.config.memory_recent_messages:]: recent_lines.append(("USER" if r["kind"]=="USER_MESSAGE" else "MANA")+": "+r["content"])
        if recent_lines: chunks.append("[RECENT CONVERSATION]\n"+"\n".join(recent_lines))
        if data["facts"]: chunks.append("[FACT MEMORY]\n"+"\n".join(f"{f['subject']} — {f['predicate']} — {f['object']} (confidence={float(f['confidence']):.2f}; status={f['status']})" for f in data["facts"]))
        if data.get("claims"): chunks.append("[USER CLAIM MEMORY]\n"+"\n".join(c["text"] for c in data["claims"][:6]))
        sem=[("[USER CLAIM] " if m.get("item_type")=="user_claim" else "[KNOWLEDGE] " if m.get("item_type")=="document_chunk" else "[SEMANTIC MEMORY] ")+m["text"] for m in data.get("semantic",[])]
        if sem: chunks.append("\n".join(sem[:self.config.memory_semantic_top_k]))
        recent_ids={x["id"] for x in data["recent"]}; rel=[f"[{r['kind']}] {r['content']}" for r in data["relevant"] if r["id"] not in recent_ids]
        if rel: chunks.append("[RELEVANT SESSION MEMORY]\n"+"\n".join(rel[:self.config.memory_retrieval_limit]))
        text="\n\n".join(chunks); budget=max(3000,int(self.config.memory_context_budget_chars))
        if len(text)>budget: text=text[:budget]+"\n[CONTEXT BUDGET EXHAUSTED]"
        return text,data
