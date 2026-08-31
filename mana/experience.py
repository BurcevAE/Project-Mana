"""
mana.experience — SQLite-backed experience/outcome store used for UCB-style pipeline selection.
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

from .pipeline import PipelineSpec

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"


# ---------------------------------------------------------------------------
# Experience store
# ---------------------------------------------------------------------------

class ExperienceDB:
    def __init__(self, path: str):
        import sqlite3
        self._lock = threading.RLock()
        self.path = path
        self.con = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.con.execute("CREATE TABLE IF NOT EXISTS experiences (id INTEGER PRIMARY KEY, task_hash TEXT, category TEXT, pipeline_key TEXT, pipeline_json TEXT, quality REAL, reliability REAL, latency REAL, created_at REAL)")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_exp_pipeline ON experiences(pipeline_key)")
        self.con.commit()

    def add(self, task: str, category: str, spec: PipelineSpec, quality: float, reliability: float, latency: float) -> None:
        h = hashlib.sha256(task.strip().lower().encode()).hexdigest()[:16]
        with self._lock:
            self.con.execute(
                "INSERT INTO experiences(task_hash,category,pipeline_key,pipeline_json,quality,reliability,latency,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (h, category, spec.key(), json.dumps(asdict(spec), ensure_ascii=False), quality, reliability, latency, time.time()),
            )
            self.con.commit()

    def best(self, category: str, limit: int = 4) -> List[PipelineSpec]:
        with self._lock:
            rows = self.con.execute(
                "SELECT pipeline_json FROM experiences WHERE category=? ORDER BY (quality*0.75+reliability*0.20) DESC, latency ASC LIMIT ?",
                (category, limit),
            ).fetchall()
        out = []
        for raw, in rows:
            try: out.append(PipelineSpec(**json.loads(raw)))
            except Exception: pass
        return out

    def stats(self, pipeline_key: str) -> Dict[str, float]:
        with self._lock:
            r = self.con.execute(
                "SELECT COUNT(*),AVG(quality),AVG(reliability),AVG(latency) FROM experiences WHERE pipeline_key=?",
                (pipeline_key,),
            ).fetchone()
        return {"count": int(r[0] or 0), "mean_quality": float(r[1] or 0),
                "mean_reliability": float(r[2] or 0), "mean_latency": float(r[3] or 0)}

    def ucb(self, key: str, total: int, coef: float = .35) -> float:
        s = self.stats(key)
        n = s["count"]
        if n <= 0:
            return float("inf")
        return s["mean_quality"] + coef * ((max(1, total) / n) ** .5)

    def count(self) -> int:
        with self._lock:
            return int(self.con.execute("SELECT COUNT(*) FROM experiences").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            try: self.con.close()
            except Exception: pass
