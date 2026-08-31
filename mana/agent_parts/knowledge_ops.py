"""
mana.agent_parts.knowledge_ops — KnowledgeOpsMixin: file/topic knowledge acquisition entry points (acquire_knowledge, acquire_topic_from_web, knowledge_status).
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
from ..experience import ExperienceDB
from ..verifier import LocalVerifier
from ..memory import MemoryManager
from ..optional_deps import fitz, HAS_FITZ, HAS_SKLEARN, LogisticRegression, HAS_TORCH, DEVICE, HAS_WEB, WEB_BACKEND, torch

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"


class KnowledgeOpsMixin:
    def _read_knowledge_file(self,path:Path)->Tuple[str,str]:
        suf=path.suffix.lower()
        if suf in {".txt",".md",".py",".json",".csv",".xml",".log",".yaml",".yml",".rst"}:
            return path.read_text(encoding="utf-8",errors="ignore"),"text"
        if suf==".pdf" and HAS_FITZ:
            doc=fitz.open(str(path)); text="\n".join(page.get_text("text") for page in doc); doc.close(); return text,"pdf"
        raise ValueError(f"Неподдерживаемый тип файла: {suf or '<none>'}")

    def acquire_topic_from_web(self,topic:str,domain:str="")->Dict[str,Any]:
        topic=(topic or "").strip(); out={"topic":topic,"domain":domain or topic,"mode":"web_topic","sources":0,"chunks":0,"facts":0,"concepts":0,"errors":[]}
        if not topic: return out
        try:
            rows = self.tools.call("web_search", query=topic, max_results=self.config.knowledge_acquisition_max_sources).output or []
        except Exception as exc: out["errors"].append(str(exc)); return out
        for i,row in enumerate(rows[:self.config.knowledge_acquisition_max_sources],1):
            try:
                title=str(row.get("title") or f"Web source {i}"); body=str(row.get("body") or row.get("snippet") or "").strip(); href=str(row.get("href") or row.get("url") or "")
                if not body: continue
                r=self.persistent_memory.ingest_document(title,body,href,"web",domain or topic,{"url":href,"query":topic,"rank":i})
                out["sources"]+=1; out["chunks"]+=int(r.get("chunks",0)); out["facts"]+=int(r.get("facts",0)); out["concepts"]+=int(r.get("concepts",0))
            except Exception as exc: out["errors"].append(f"web[{i}]: {exc}")
        return out

    def acquire_knowledge(self,source:str,domain:str="",recursive:bool=True)->Dict[str,Any]:
        root=Path(source)
        if not root.exists():
            return self.acquire_topic_from_web(source,domain or source)
        paths=[root] if root.is_file() else sorted(root.rglob("*") if recursive else root.glob("*")) if root.is_dir() else []
        paths=[p for p in paths if p.is_file() and p.suffix.lower() in {".txt",".md",".py",".json",".csv",".xml",".log",".yaml",".yml",".rst",".pdf"}]
        started=time.time(); stats={"source":source,"domain":domain,"mode":"local","documents":0,"chunks":0,"facts":0,"concepts":0,"errors":[]}
        for path in paths[:self.config.knowledge_acquisition_max_sources]:
            try:
                text,stype=self._read_knowledge_file(path); r=self.persistent_memory.ingest_document(path.name,text,str(path.resolve()),stype,domain,{"size":path.stat().st_size})
                stats["documents"]+=1; stats["chunks"]+=int(r.get("chunks",0)); stats["facts"]+=int(r.get("facts",0)); stats["concepts"]+=int(r.get("concepts",0))
            except Exception as exc: stats["errors"].append(f"{path}: {exc}")
        with self.persistent_memory.lock,self.persistent_memory.con:
            self.persistent_memory.con.execute("INSERT INTO acquisition_runs(topic,domain,source_count,chunk_count,fact_count,concept_count,started_at,finished_at,metadata) VALUES(?,?,?,?,?,?,?,?,?)",(domain or root.name,domain,stats["documents"],stats["chunks"],stats["facts"],stats["concepts"],started,time.time(),json.dumps(stats,ensure_ascii=False)))
        return stats

    def knowledge_status(self)->Dict[str,Any]:
        return self.persistent_memory.acquisition_stats()
