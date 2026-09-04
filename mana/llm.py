"""
mana.llm — the LLM facade over the Brain Pool.

Since 5.10 this module contains no routing and no transport: both moved to
`mana.brains`, which selects among several models per call instead of
walking a fixed provider order ["ollama","gemini","openrouter","openai"].
What stays here is the *contract* every call site in MANA was already
written against -- `ask_detailed(prompt, ...) -> (text, LLMCallMeta)` --
so switching from one brain to a pool required changing none of them.

That is exactly what the Tool Engine choke point was for: the agent
reaches the LLM through one path (`ManaAgent._llm_call` -> the
`llm_generate` tool -> here), so replacing what sits behind that path is a
local change rather than a rewrite. `LLMCallMeta` gained brain/attempt
fields; nothing was removed from it, so every existing reader keeps
working and `meta.provider` still answers "who produced this text" -- it
now names the brain that actually did, not the first enabled backend.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .brains import BrainPool
from .config import Config
from .optional_deps import HAS_REQUESTS

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"


@dataclass
class LLMCallMeta:
    ok: bool = False
    timeout: bool = False
    fallback: bool = False
    provider: str = ""              # the brain that produced the text
    error: str = ""
    latency: float = 0.0
    # --- added in 5.10; every field below is optional so that older
    # readers (and pickled traces from earlier runs) stay valid ---
    brain: str = ""                 # same as provider, named for what it is
    model: str = ""
    attempts: Tuple[str, ...] = ()  # brains tried, in order, including the winner
    agreement: float = 0.0          # set by consensus calls, 0.0 for single-brain
    consensus: bool = False
    #: True only when the caller asked to avoid some brain (the critic
    #: avoiding the drafting brain) AND the router could honour it. False
    #: means the answer and its critique came from the same model, which is
    #: a materially weaker check -- see ExecutionMixin._critic.
    independent: bool = False


class LLMClient:
    """Thin compatibility facade. New code should use `self.pool` directly
    (it exposes selection, consensus and per-brain health); this class
    exists so that the ~30 existing call sites and their tests keep working
    unchanged."""

    def __init__(self, config: Config, vlog=None, pool: Optional[BrainPool] = None):
        self.config = config
        self.vlog = vlog
        self.pool = pool or BrainPool(config, vlog=vlog)
        self.calls = 0
        self.failures = 0
        self.timeouts = 0
        self._request_lock = threading.RLock()

    # ---------- capability reporting ----------

    @property
    def enabled(self) -> bool:
        """True when *any* brain can be called.

        Deliberately broader than before: MANA used to be "enabled" only if
        `enable_llm and HAS_REQUESTS`, i.e. only if the local backend was
        on. With a pool, a machine with no Ollama but a free Gemini key is
        a machine with a working LLM, and the agent should say so.
        """
        return bool(self.config.brain_pool_enabled) and bool(self.pool.configured())

    def available_providers(self) -> List[str]:
        """Legacy name; returns brain ids that are ready right now."""
        return self.pool.available()

    def _log(self, message: str) -> None:
        if self.vlog:
            self.vlog(message)

    # ---------- the call ----------

    def ask_detailed(self, prompt: str, system: str = "", temperature: float = 0.2,
                     provider: str = "auto", context_tag: str = "", *,
                     kind: str = "general", difficulty: Optional[float] = None,
                     task: str = "", policy: str = "", avoid: Sequence[str] = (),
                     consensus: int = 0) -> Tuple[Optional[str], LLMCallMeta]:
        """Route one prompt to the pool.

        `provider` keeps its old meaning and its old values ("auto",
        "ollama", "gemini", ...); the pool resolves legacy names onto brain
        ids so PipelineSpec genomes evolved before 5.10 still express a
        real preference instead of silently becoming "auto".
        """
        if not self.config.brain_pool_enabled:
            return None, LLMCallMeta(ok=False, error="brain pool disabled")
        # Per-call timeout override set by the evolution loop (it shortens
        # LLM timeouts while benchmarking). Preserved from the old client:
        # dropping it would have made every evolution cycle slower.
        override = getattr(self.config, "_active_llm_timeout", None)
        if consensus and int(consensus) > 1:
            res = self.pool.ask_consensus(prompt, n=int(consensus), system=system,
                                          temperature=temperature, kind=kind,
                                          difficulty=difficulty, task=task or prompt,
                                          policy=policy, context_tag=context_tag)
            meta = LLMCallMeta(
                ok=bool(res.get("ok")), provider=str(res.get("brain") or ""),
                brain=str(res.get("brain") or ""), error=str(res.get("error") or ""),
                latency=float(res.get("latency", 0.0)),
                attempts=tuple(res.get("brains") or ()),
                agreement=float(res.get("agreement", 0.0)),
                consensus=not bool(res.get("single")),
            )
        else:
            res = self.pool.ask(prompt, system=system, temperature=temperature, kind=kind,
                                difficulty=difficulty, task=task or prompt, brain=provider,
                                policy=policy, context_tag=context_tag, avoid=avoid,
                                timeout=None if override is None else float(override))
            meta = LLMCallMeta(
                ok=bool(res.get("ok")), timeout=bool(res.get("timeout")),
                provider=str(res.get("brain") or ""), brain=str(res.get("brain") or ""),
                model=str(res.get("model") or ""), error=str(res.get("error") or ""),
                latency=float(res.get("latency_total", res.get("latency", 0.0))),
                attempts=tuple(res.get("attempts") or ()),
                independent=bool(res.get("avoided")),
            )
        with self._request_lock:
            self.calls += 1
            if not meta.ok:
                self.failures += 1
            if meta.timeout:
                self.timeouts += 1
        return (res.get("text") if meta.ok else None), meta

    def ask(self, prompt: str, system: str = "", temperature: float = 0.2,
            provider: str = "auto", context_tag: str = "", **kwargs: Any) -> Optional[str]:
        text, _ = self.ask_detailed(prompt, system, temperature, provider, context_tag, **kwargs)
        return text

    def ask_consensus(self, prompt: str, n: int = 2, **kwargs: Any) -> Dict[str, Any]:
        """Full consensus result (answer + agreement + per-brain responses).
        `ask_detailed(consensus=n)` returns only the text and the summary;
        callers that need to *reason* about disagreement want this."""
        return self.pool.ask_consensus(prompt, n=n, **kwargs)

    def record_outcome(self, brain_id: str, quality: float) -> None:
        self.pool.record_outcome(brain_id, quality)

    # ---------- reporting ----------

    def status(self) -> Dict[str, Any]:
        pool_status = self.pool.status()
        return {
            "enabled": self.enabled,
            "calls": self.calls,
            "failures": self.failures,
            "timeouts": self.timeouts,
            "policy": pool_status["policy"],
            "brains_configured": pool_status["configured"],
            "brains_available": pool_status["available"],
            # Legacy shape: some reports/tests read status()["providers"] as
            # a {name: bool} map. Kept, now derived from the pool.
            "providers": {b["brain_id"]: bool(b["ready"]) for b in pool_status["brains"]},
            "pool": pool_status,
            "requests_available": HAS_REQUESTS,
        }
