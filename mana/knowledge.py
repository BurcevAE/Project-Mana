"""
mana.knowledge — legacy v3.4-compatible flat knowledge base (pickle-backed).
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
from . import events
from .optional_deps import (
    SentenceTransformer, HAS_SENTENCE_TRANSFORMERS,
    TfidfVectorizer, cosine_similarity, HAS_SKLEARN,
    DEVICE, HAS_TORCH,
)


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeEntry:
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    status: str = "unverified"
    quality: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = None
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "status": self.status,
            "quality": self.quality,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KnowledgeEntry":
        return cls(
            content=d.get("content", ""), source=d.get("source", "unknown"),
            metadata=d.get("metadata", {}), confidence=float(d.get("confidence", 0.5)),
            status=d.get("status", "unverified"), quality=float(d.get("quality", 0.0)),
            created_at=float(d.get("created_at", time.time())),
            last_used=float(d.get("last_used", time.time())),
            embedding=np.asarray(d["embedding"], dtype=np.float32) if d.get("embedding") is not None else None,
            id=d.get("id", ""),
        )

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.1"

def _out(*parts: Any, **_kw: Any) -> None:
    """print()-shaped adapter onto the event bus.

    Kept print-shaped on purpose: these call sites are status lines whose
    wording and formatting are already right, and rewriting each one into
    a bespoke emit() would have been a large diff with no behavioural
    gain. What changes is where the text goes -- a windowed build has no
    stdout, and a cp1251 console could not encode the markers these lines
    use. Severity is taken from the marker the line already carries.
    """
    text = " ".join(str(p) for p in parts)
    stripped = text.lstrip("\n ")
    if stripped.startswith(("⚠", "❌")):
        events.emit(events.WARNING, text)
    else:
        events.emit(events.STATUS, text)



class KnowledgeBase:
    def __init__(self, config: Config):
        self.config = config
        self.entries: List[KnowledgeEntry] = []
        self.embedder = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self._init_embeddings()
        self.load()

    def _init_embeddings(self) -> None:
        if not self.config.use_embeddings or not HAS_SENTENCE_TRANSFORMERS:
            return
        try:
            self.embedder = SentenceTransformer(self.config.embedding_model)
            if HAS_TORCH:
                self.embedder.to(DEVICE)
            _out(f"Эмбеддинги: {self.config.embedding_model}")
        except Exception as exc:
            _out(f"⚠️ Не удалось загрузить embeddings: {exc}")
            self.embedder = None

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if self.embedder is None:
            return None
        try:
            return self.embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        except Exception:
            return None

    def _rebuild_tfidf(self) -> None:
        if not HAS_SKLEARN or not self.entries:
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            return
        try:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=self.config.tfidf_max_features)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform([e.content for e in self.entries])
        except Exception:
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None

    def add(self, content: str, source: str, confidence: float = 0.5,
            metadata: Optional[Dict[str, Any]] = None, status: str = "unverified") -> None:
        content = (content or "").strip()
        if not content:
            return
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        if any(e.id == digest for e in self.entries):
            return
        self.entries.append(KnowledgeEntry(
            content=content, source=source, metadata=metadata or {},
            confidence=max(0.0, min(1.0, confidence)), status=status, embedding=self._embed(content), id=digest))
        if len(self.entries) > self.config.max_knowledge_entries:
            self.entries.sort(key=lambda x: x.last_used, reverse=True)
            self.entries = self.entries[:self.config.max_knowledge_entries]
        self._rebuild_tfidf()
        self.save()

    def search(self, query: str, top_k: int = 5, min_confidence: float = 0.0,
               min_relevance: Optional[float] = None) -> List[KnowledgeEntry]:
        """Retrieve entries relevant to `query`.

        NOTE (bugfix, found in a real session): this used to return
        `scored[:top_k]` with NO relevance floor. `min_confidence` filters
        on an entry's own stored confidence, not on its similarity to the
        query -- so with one item in memory, EVERY query retrieved it. A
        greeting ("Привет, Мана") pulled back a stored answer about AI news
        and the model dutifully recited it, which looked like the agent
        losing the thread of the conversation but was actually irrelevant
        memory being injected as context.

        NOTE (second bugfix, found on real hardware): the first attempt used
        ONE threshold for all three scoring paths. That was wrong -- they
        are not on the same scale. Embedding cosine similarity between
        totally unrelated sentences typically sits around 0.2-0.4, while
        TF-IDF cosine and word overlap sit near 0.0. A single 0.25 floor
        therefore filtered nothing at all on a machine with
        sentence-transformers installed. The bug was invisible in
        development because that environment had no embedding model and
        only ever exercised the lexical paths.

        NOTE (third fix, and the one that actually works -- measured with
        scripts/calibrate_memory_relevance.py on real hardware):
        per-path floors were still wrong, because NO embedding floor
        separates the groups at all. Measured with all-MiniLM-L6-v2
        against a stored news entry:

            "Привет, Мана. Я твой создатель"   embedding 0.577   lexical 0.000
            "напиши функцию сортировки"        embedding 0.546   lexical 0.000
            "что там было про ИИ в новостях"   embedding 0.563   lexical 0.354

        The greeting scores HIGHER than the weakest genuine query. That is
        the known anisotropy of sentence embeddings: same-language text
        collapses into a narrow cone, so absolute cosine thresholds carry
        almost no signal. Meanwhile the lexical score separated the groups
        perfectly (0.000 vs 0.354) -- and the old `if/elif` structure threw
        that working signal away the moment an embedding model was present.

        So the two signals are now used for what each is actually good at:

            lexical overlap  ->  GATES  (decides what may enter context)
            embedding cosine ->  RANKS  (orders what survived the gate)

        Trade-off, stated plainly: a true paraphrase sharing no words with
        a stored entry will now be rejected, which is exactly what
        embeddings were supposed to buy us. On the measured data that cost
        is worth paying, because the embedding signal was not delivering
        that benefit either -- it was admitting everything. Re-run the
        calibration script if you change embedding models; this conclusion
        is a measurement, not a law.

        `min_relevance`, when passed explicitly, overrides the lexical gate.
        """
        if not self.entries:
            return []
        floor = float(min_relevance) if min_relevance is not None else float(
            getattr(self.config, "memory_min_relevance_lexical", 0.20))
        q_emb = self._embed(query)

        # --- gate: lexical relevance decides ADMISSION ---
        lexical: Dict[str, float] = {}
        if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
            try:
                sims = cosine_similarity(self.tfidf_vectorizer.transform([query]),
                                          self.tfidf_matrix).ravel()
                for idx, score in enumerate(sims):
                    lexical[self.entries[idx].id] = float(score)
            except Exception:
                lexical = {}
        if not lexical:
            q_words = set(re.findall(r"\w+", query.lower()))
            for e in self.entries:
                words = set(re.findall(r"\w+", e.content.lower()))
                lexical[e.id] = len(q_words & words) / max(1, len(q_words | words))

        admitted = [e for e in self.entries
                    if e.confidence >= min_confidence and lexical.get(e.id, 0.0) >= floor]

        # --- rank: embeddings order what got through, when available ---
        scored: List[Tuple[float, KnowledgeEntry]] = []
        for e in admitted:
            if q_emb is not None and e.embedding is not None:
                scored.append((float(np.dot(q_emb, e.embedding)), e))
            else:
                scored.append((lexical.get(e.id, 0.0), e))
        scored.sort(key=lambda x: x[0], reverse=True)
        result = [e for _, e in scored[:top_k]]
        now = time.time()
        for e in result:
            e.last_used = now
        return result

    def save(self) -> None:
        path = Path(self.config.knowledge_db_path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as fh:
            pickle.dump([e.to_dict() for e in self.entries], fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def load(self) -> None:
        path = Path(self.config.knowledge_db_path)
        if not path.exists():
            return
        try:
            with path.open("rb") as fh:
                data = pickle.load(fh)
            self.entries = [KnowledgeEntry.from_dict(x) for x in data]
            self._rebuild_tfidf()
            _out(f"Память: {len(self.entries)} записей")
        except Exception as exc:
            _out(f"⚠️ Ошибка загрузки памяти: {exc}")
