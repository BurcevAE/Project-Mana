"""
mana.llm — multi-provider LLM client (Ollama/Gemini/OpenRouter/OpenAI).
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
from .optional_deps import requests, HAS_REQUESTS

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

@dataclass
class LLMCallMeta:
    ok: bool = False
    timeout: bool = False
    fallback: bool = False
    provider: str = ""
    error: str = ""
    latency: float = 0.0


class LLMClient:
    def __init__(self, config: Config, vlog=None):
        self.config = config
        self.enabled = bool(config.enable_llm and HAS_REQUESTS)
        self.gemini_enabled = bool(config.gemini_api_key and HAS_REQUESTS)
        self.openrouter_enabled = bool(config.openrouter_api_key and HAS_REQUESTS)
        self.openai_enabled = bool(config.openai_api_key and HAS_REQUESTS)
        self.calls = 0
        self.failures = 0
        self.timeouts = 0
        self.vlog = vlog
        self._request_lock = threading.RLock()

    def _log(self, message: str) -> None:
        if self.vlog:
            self.vlog(message)

    def available_providers(self) -> List[str]:
        out = ["ollama"] if self.enabled else []
        if self.gemini_enabled: out.append("gemini")
        if self.openrouter_enabled: out.append("openrouter")
        if self.openai_enabled: out.append("openai")
        return out

    def _ollama(self, prompt: str, system: str, temperature: float, timeout: float) -> str:
        payload = {
            "model": self.config.ollama_model,
            "prompt": ((system + "\n\n") if system else "") + prompt,
            "stream": False,
            "options": {"temperature": float(temperature), "num_predict": int(self.config.llm_max_tokens)},
        }
        with self._request_lock:
            r = requests.post(self.config.ollama_url, json=payload, timeout=timeout)
        r.raise_for_status()
        text = (r.json().get("response") or "").strip()
        if not text:
            raise RuntimeError("Ollama returned empty response")
        return text

    def _gemini(self, prompt: str, system: str, temperature: float) -> str:
        url = self.config.gemini_url.format(model=self.config.gemini_model)
        payload = {"contents": [{"role": "user", "parts": [{"text": (system + "\n\n" if system else "") + prompt}]}],
                   "generationConfig": {"temperature": temperature, "maxOutputTokens": self.config.llm_max_tokens}}
        r = requests.post(url, params={"key": self.config.gemini_api_key}, json=payload, timeout=self.config.llm_timeout)
        r.raise_for_status()
        parts = []
        for c in r.json().get("candidates", []) or []:
            for part in c.get("content", {}).get("parts", []) or []:
                if part.get("text"): parts.append(part["text"])
        text = "\n".join(parts).strip()
        if not text: raise RuntimeError("Gemini returned empty response")
        return text

    def _openrouter(self, prompt: str, system: str, temperature: float) -> str:
        headers = {"Authorization": f"Bearer {self.config.openrouter_api_key}", "Content-Type": "application/json",
                   "HTTP-Referer": "http://localhost", "X-Title": "MANA v4.6"}
        payload = {"model": self.config.openrouter_model,
                   "messages": [{"role": "system", "content": system or "You are a helpful assistant."}, {"role": "user", "content": prompt}],
                   "temperature": temperature, "max_tokens": self.config.llm_max_tokens}
        r = requests.post(self.config.openrouter_url, headers=headers, json=payload, timeout=self.config.llm_timeout)
        r.raise_for_status()
        text = (r.json().get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        if not text: raise RuntimeError("OpenRouter returned empty response")
        return text

    def _openai(self, prompt: str, system: str, temperature: float) -> str:
        payload = {
            "model": self.config.openai_model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system or "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
            ],
            "max_output_tokens": self.config.openai_max_output_tokens,
        }
        r = requests.post(self.config.openai_url,
                          headers={"Authorization": f"Bearer {self.config.openai_api_key}", "Content-Type": "application/json"},
                          json=payload, timeout=self.config.openai_timeout)
        r.raise_for_status()
        data = r.json()
        if isinstance(data.get("output_text"), str) and data["output_text"].strip():
            return data["output_text"].strip()
        out = []
        for item in data.get("output", []) or []:
            for c in item.get("content", []) or []:
                if c.get("text"): out.append(c["text"])
        text = "\n".join(out).strip()
        if not text: raise RuntimeError("OpenAI returned empty response")
        return text

    def ask_detailed(self, prompt: str, system: str = "", temperature: float = 0.2,
                     provider: str = "auto", context_tag: str = "") -> Tuple[Optional[str], LLMCallMeta]:
        order = [provider] if provider != "auto" else ["ollama", "gemini", "openrouter", "openai"]
        timeout = float(getattr(self.config, "_active_llm_timeout", self.config.llm_timeout))
        last_meta = LLMCallMeta(ok=False)
        for name in order:
            enabled = (name == "ollama" and self.enabled) or (name == "gemini" and self.gemini_enabled) or \
                      (name == "openrouter" and self.openrouter_enabled) or (name == "openai" and self.openai_enabled)
            if not enabled:
                continue
            t0 = time.perf_counter()
            self._log(f"LLM START | provider={name} | tag={context_tag or '-'} | timeout={timeout:.0f}s")
            try:
                if name == "ollama":
                    text = self._ollama(prompt, system, temperature, timeout)
                elif name == "gemini":
                    text = self._gemini(prompt, system, temperature)
                elif name == "openrouter":
                    text = self._openrouter(prompt, system, temperature)
                else:
                    text = self._openai(prompt, system, temperature)
                elapsed = time.perf_counter() - t0
                self.calls += 1
                self.failures = 0
                meta = LLMCallMeta(ok=True, provider=name, latency=elapsed)
                self._log(f"LLM OK | provider={name} | tag={context_tag or '-'} | time={elapsed:.2f}s")
                return text, meta
            except requests.exceptions.Timeout as exc:  # type: ignore[union-attr]
                elapsed = time.perf_counter() - t0
                self.failures += 1
                self.timeouts += 1
                meta = LLMCallMeta(ok=False, timeout=True, provider=name, error=str(exc), latency=elapsed)
                self._log(f"LLM TIMEOUT | provider={name} | tag={context_tag or '-'} | time={elapsed:.2f}s")
                last_meta = meta
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                self.failures += 1
                meta = LLMCallMeta(ok=False, provider=name, error=str(exc), latency=elapsed)
                self._log(f"LLM ERROR | provider={name} | tag={context_tag or '-'} | {type(exc).__name__}: {exc}")
                last_meta = meta
        return None, last_meta

    def ask(self, prompt: str, system: str = "", temperature: float = 0.2,
            provider: str = "auto", context_tag: str = "") -> Optional[str]:
        text, _ = self.ask_detailed(prompt, system, temperature, provider, context_tag)
        return text

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "calls": self.calls,
            "failures": self.failures,
            "timeouts": self.timeouts,
            "providers": {
                "ollama": self.enabled,
                "gemini": self.gemini_enabled,
                "openrouter": self.openrouter_enabled,
                "openai": self.openai_enabled,
            },
        }
