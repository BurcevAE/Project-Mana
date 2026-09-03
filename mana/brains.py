"""
mana.brains — the Brain Pool: MANA thinks with several models, not one.

Why this module exists
----------------------
Until 5.9.2 a local Llama/Qwen through Ollama was effectively MANA's single
brain: `LLMClient.ask_detailed` walked a hardcoded order
["ollama","gemini","openrouter","openai"] and took the first *enabled*
one. That is not a pool, it is a preference list with a failover tail --
the second entry was only ever reached when the first was broken, never
because it was the better instrument for the task. Consequences that were
visible in practice: a 0.5B local model answered hard reasoning tasks that
a free-tier frontier model would have answered correctly; a slow local
call blocked while idle remote endpoints were available; and the whole
agent's competence was capped by whatever `ollama_model` happened to be
pulled.

This module replaces the preference list with an actual pool:

  * a **catalog** of brains (local + external), each declaring what it is
    good at, how fast/expensive it is, and what its free-tier limits are;
  * a **router** that picks per call, from task difficulty and category
    plus *measured* health/latency -- not from a fixed order;
  * **rate limiting and circuit breaking** per brain, so a free tier that
    is exhausted or throttled steps aside instead of failing every call;
  * **parallel consensus** -- ask several brains the same question and use
    their agreement as an independent signal (MANA already refuses to
    treat a single LLM answer as a fact; two independent models agreeing
    is weak evidence, disagreeing is a strong warning);
  * **load distribution** -- least-loaded selection among equally capable
    brains, so N free tiers behave like one larger budget.

Deliberate non-goal: scraping the consumer web UIs of ChatGPT/DeepSeek.
Those are against the providers' terms, break on every redesign and get
accounts banned. Every brain here talks to a documented API endpoint; the
free access the catalog targets is the providers' own free tier (a free
key), which reaches the same models.

Provider protocols, not provider count
--------------------------------------
Almost every provider worth adding speaks the OpenAI chat-completions
protocol at a different base URL, so there is ONE adapter for all of them
(`_call_openai_chat`) and a brain is mostly data. Adding DeepSeek, Groq,
Cerebras, Together, Fireworks, LM Studio or llama.cpp is a catalog entry,
not code. Only Gemini and Ollama's native endpoint need their own adapter,
and OpenAI's Responses API keeps the one it already had.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from .config import Config
from .optional_deps import requests, HAS_REQUESTS

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.4"


# ---------------------------------------------------------------------------
# What a brain is
# ---------------------------------------------------------------------------

#: Task kinds a brain can declare strength in. Deliberately the same
#: vocabulary `RoutingMixin._task_category` already produces, so routing a
#: task to a brain and routing it to a web/local arm speak one language.
KINDS = ("math", "programming", "reasoning", "current", "general", "planning", "synthesis")

#: Ordered small->large. `tier_rank` turns this into a number so a router
#: can ask for "at least medium".
TIERS = ("small", "medium", "large")


def tier_rank(tier: str) -> int:
    try:
        return TIERS.index(str(tier))
    except ValueError:
        return 0


@dataclass
class BrainSpec:
    """One addressable model. `provider` selects the wire protocol, not the
    company: several companies share `openai_chat`."""
    brain_id: str
    provider: str                       # openai_chat | gemini | ollama | openai_responses
    model: str
    base_url: str = ""
    api_key_env: str = ""               # env var holding the key ("" = no auth needed)
    tier: str = "medium"
    strengths: Tuple[str, ...] = ()
    local: bool = False
    free: bool = True                   # free tier / no per-token charge
    rpm: int = 0                        # requests per minute, 0 = unlimited
    rpd: int = 0                        # requests per day, 0 = unlimited
    timeout: float = 45.0
    max_tokens: int = 700
    weight: float = 1.0                 # manual preference nudge
    enabled: bool = True
    notes: str = ""
    #: What is still missing before this brain can be used, in the user's
    #: words. Shown by --list-brains and by the desktop app instead of a
    #: bare "disabled", which tells nobody what to do about it. Needed
    #: because not every brain is configured by an API key alone --
    #: Cloudflare also needs an account id, and a brain that silently
    #: reports "disabled" for a missing second value is a dead end.
    setup_hint: str = ""

    #: Resolved at build time from api_key_env; never serialized back out
    #: (see `public_dict`) so a status dump or an evolution report cannot
    #: leak a key into a file or a log.
    api_key: str = field(default="", repr=False)

    def needs_key(self) -> bool:
        return bool(self.api_key_env)

    def has_key(self) -> bool:
        return (not self.needs_key()) or bool(self.api_key)

    def public_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("api_key", None)
        d["key_present"] = self.has_key()
        return d


# ---------------------------------------------------------------------------
# Default catalog
# ---------------------------------------------------------------------------
#
# Model IDs move faster than this file does -- every one below is
# overridable by an env var, and the whole catalog is replaceable by a JSON
# file (Config.brains_file). A brain whose key env var is unset is simply
# absent from the pool: no error, no warning spam, MANA runs on whatever
# brains are actually reachable. That is the same graceful-degradation
# contract optional_deps.py already applies to libraries.


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def default_catalog(cfg: Config) -> List[BrainSpec]:
    env = os.environ.get
    return [
        # --- local, always free, no key --------------------------------
        BrainSpec(
            brain_id="ollama", provider="ollama", model=cfg.ollama_model,
            base_url=cfg.ollama_url, tier="small",
            strengths=("general", "synthesis"), local=True, free=True,
            timeout=float(cfg.llm_timeout), max_tokens=cfg.llm_max_tokens,
            notes="Local Ollama. Always available offline; smallest capability.",
        ),
        BrainSpec(
            brain_id="lmstudio", provider="openai_chat",
            model=env("MANA_LMSTUDIO_MODEL", "local-model"),
            base_url=env("MANA_LMSTUDIO_URL", "http://localhost:1234/v1/chat/completions"),
            tier="medium", strengths=("general", "programming"), local=True, free=True,
            enabled=_env_flag("MANA_ENABLE_LMSTUDIO", False),
            notes="LM Studio / llama.cpp OpenAI-compatible server. Off unless MANA_ENABLE_LMSTUDIO=1.",
        ),
        # --- external free tiers (free API key, official endpoints) -----
        BrainSpec(
            brain_id="gemini", provider="gemini",
            model=env("MANA_GEMINI_MODEL", cfg.gemini_model),
            base_url=cfg.gemini_url, api_key_env="GEMINI_API_KEY",
            tier="large", strengths=("reasoning", "general", "synthesis", "planning"),
            free=True, rpm=15, rpd=1000,
            setup_hint="ключ бесплатно на aistudio.google.com/apikey",
            notes="Google AI Studio free tier.",
        ),
        BrainSpec(
            brain_id="groq", provider="openai_chat",
            # Verified against a live free-tier account, not taken from
            # documentation: llama-3.3-70b-versatile (the previous default
            # here) returns 404 "model does not exist" -- Groq's catalogue
            # moved, which is exactly why every model id in this file is
            # env-overridable. gpt-oss-120b answered a Russian reasoning
            # prompt in 0.98s.
            model=env("MANA_GROQ_MODEL", "openai/gpt-oss-120b"),
            base_url=env("MANA_GROQ_URL", "https://api.groq.com/openai/v1/chat/completions"),
            api_key_env="GROQ_API_KEY", tier="large",
            strengths=("general", "reasoning", "synthesis"), free=True, rpm=30, rpd=1000,
            timeout=30.0,
            setup_hint="ключ бесплатно на console.groq.com/keys (начинается с gsk_)",
            notes="Groq free tier. Fastest remote brain -- preferred when latency matters.",
        ),
        BrainSpec(
            # A second Groq model from a different family, so consensus
            # between the two is worth something: two answers from one
            # model are one answer twice.
            brain_id="groq-qwen", provider="openai_chat",
            model=env("MANA_GROQ_QWEN_MODEL", "qwen/qwen3.8-27b"),
            base_url=env("MANA_GROQ_URL", "https://api.groq.com/openai/v1/chat/completions"),
            api_key_env="GROQ_API_KEY", tier="medium",
            strengths=("math", "programming", "general"), free=True, rpm=30, rpd=1000,
            timeout=30.0,
            notes="Second Groq model, different family -- gives the pool a real second opinion.",
        ),
        BrainSpec(
            brain_id="cerebras", provider="openai_chat",
            model=env("MANA_CEREBRAS_MODEL", "llama-3.3-70b"),
            base_url=env("MANA_CEREBRAS_URL", "https://api.cerebras.ai/v1/chat/completions"),
            api_key_env="CEREBRAS_API_KEY", tier="large",
            strengths=("general", "reasoning"), free=True, rpm=30, rpd=900, timeout=30.0,
            setup_hint="ключ бесплатно на cloud.cerebras.ai",
            notes="Cerebras free tier.",
        ),
        BrainSpec(
            brain_id="deepseek", provider="openai_chat",
            model=env("MANA_DEEPSEEK_MODEL", "deepseek-chat"),
            base_url=env("MANA_DEEPSEEK_URL", "https://api.deepseek.com/chat/completions"),
            api_key_env="DEEPSEEK_API_KEY", tier="large",
            strengths=("math", "programming", "reasoning"), free=False, rpm=60,
            notes="DeepSeek direct API (cheap, not free). Strongest math/code brain in the catalog.",
        ),
        BrainSpec(
            brain_id="mistral", provider="openai_chat",
            model=env("MANA_MISTRAL_MODEL", "mistral-small-latest"),
            base_url=env("MANA_MISTRAL_URL", "https://api.mistral.ai/v1/chat/completions"),
            api_key_env="MISTRAL_API_KEY", tier="medium",
            strengths=("general", "programming"), free=True, rpm=30,
            notes="Mistral free tier.",
        ),
        BrainSpec(
            brain_id="github-models", provider="openai_chat",
            model=env("MANA_GITHUB_MODEL", "gpt-4o-mini"),
            base_url=env("MANA_GITHUB_MODELS_URL", "https://models.inference.ai.azure.com/chat/completions"),
            api_key_env="GITHUB_TOKEN", tier="medium",
            strengths=("general", "programming"), free=True, rpm=15, rpd=150,
            setup_hint="любой GitHub PAT подойдёт",
            notes="GitHub Models -- free with any GitHub personal access token.",
        ),
        BrainSpec(
            # Cloudflare Workers AI. Two things make it worth a catalog
            # entry: a genuinely free daily allocation, and an
            # OpenAI-compatible route, so it needs no new adapter.
            #
            # It is also the first brain that needs more than a key: the
            # account id is part of the URL and differs per user, so it
            # cannot be a constant here. Without it the entry stays
            # disabled with a hint rather than sitting in the pool with a
            # URL that cannot resolve.
            brain_id="cloudflare", provider="openai_chat",
            model=env("MANA_CLOUDFLARE_MODEL", "@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
            base_url=(f"https://api.cloudflare.com/client/v4/accounts/"
                      f"{env('MANA_CLOUDFLARE_ACCOUNT_ID', '')}/ai/v1/chat/completions"),
            api_key_env="CLOUDFLARE_API_TOKEN", tier="large",
            strengths=("general", "reasoning", "programming"), free=True, rpm=30,
            enabled=bool(env("MANA_CLOUDFLARE_ACCOUNT_ID", "")),
            setup_hint=("нужны CLOUDFLARE_API_TOKEN (права Workers AI) и "
                        "MANA_CLOUDFLARE_ACCOUNT_ID — id аккаунта из URL панели Cloudflare"),
            notes="Cloudflare Workers AI free allocation, OpenAI-compatible route.",
        ),
        # --- aggregators: several brains behind one key -----------------
        BrainSpec(
            brain_id="openrouter-free", provider="openai_chat",
            model=env("MANA_OPENROUTER_FREE_MODEL", "deepseek/deepseek-chat-v3.1:free"),
            base_url=cfg.openrouter_url, api_key_env="OPENROUTER_API_KEY",
            tier="large", strengths=("reasoning", "math", "programming"),
            free=True, rpm=20, rpd=50,
            notes="OpenRouter ':free' model. Zero cost, hard daily cap.",
        ),
        BrainSpec(
            brain_id="openrouter-alt", provider="openai_chat",
            model=env("MANA_OPENROUTER_ALT_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
            base_url=cfg.openrouter_url, api_key_env="OPENROUTER_API_KEY",
            tier="large", strengths=("general", "synthesis"), free=True, rpm=20, rpd=50,
            notes="Second free OpenRouter model -- a different family, so consensus is meaningful.",
        ),
        BrainSpec(
            brain_id="openrouter", provider="openai_chat",
            model=cfg.openrouter_model, base_url=cfg.openrouter_url,
            api_key_env="OPENROUTER_API_KEY", tier="large",
            strengths=("general",), free=False,
            enabled=_env_flag("MANA_ENABLE_OPENROUTER_PAID", False),
            notes="Whatever Config.openrouter_model points at (may be paid). Off by default.",
        ),
        # --- paid, explicit ---------------------------------------------
        BrainSpec(
            brain_id="openai", provider="openai_responses",
            model=cfg.openai_model, base_url=cfg.openai_url,
            api_key_env="OPENAI_API_KEY", tier="large",
            strengths=("reasoning", "programming", "synthesis", "planning"),
            free=False, timeout=float(cfg.openai_timeout), max_tokens=cfg.openai_max_output_tokens,
            notes="OpenAI Responses API. Paid -- used only when policy allows non-free brains.",
        ),
    ]


#: Parameter-count boundaries for tiering a local model. Rough by nature:
#: what matters is that a 0.5B and a 32B do not compete for the same work,
#: not the exact cut-off.
_LOCAL_TIER_BOUNDS = ((2.0, "small"), (16.0, "medium"))


def _tier_for_parameters(param_size: str) -> str:
    """'7.6B' -> 'medium'. Anything unparseable stays 'small', which is the
    conservative answer: an unknown local model gets easy work rather than
    the reasoning tasks a wrong guess would waste."""
    try:
        billions = float(str(param_size).upper().rstrip("B").strip())
    except (TypeError, ValueError):
        return "small"
    for bound, tier in _LOCAL_TIER_BOUNDS:
        if billions < bound:
            return tier
    return "large"


def probe_ollama(base_url: str, timeout: float = 1.5) -> Dict[str, Any]:
    """Ask a local Ollama what it actually has.

    Two failures this prevents, both observed on real machines rather than
    imagined:

      * **A model name nobody pulled.** Config defaults to
        `qwen2.5:0.5b`; the machine had `qwen2.5:7b-instruct`. Every call
        would have returned 404 "model not found" while the pool happily
        listed the brain as ready.
      * **A brain with no server at all.** `usable()` checked config, key
        and flag but never reachability, so an uninstalled Ollama
        outranked working remote brains (it gets the local bonus) and
        quietly turned a two-brain consensus into one opinion.

    Never raises, and a refused connection to localhost returns
    immediately -- this runs once per pool, not once per call.
    """
    result: Dict[str, Any] = {"reachable": False, "models": [], "error": ""}
    if not HAS_REQUESTS:
        result["error"] = "requests not installed"
        return result
    # cfg.ollama_url points at /api/generate; the inventory lives next door.
    tags_url = (base_url.rsplit("/api/", 1)[0] + "/api/tags") if "/api/" in base_url else base_url
    try:
        r = requests.get(tags_url, timeout=timeout)
        r.raise_for_status()
        result["reachable"] = True
        for m in r.json().get("models", []) or []:
            result["models"].append({
                "name": m.get("name", ""),
                "parameters": (m.get("details") or {}).get("parameter_size", ""),
            })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return result


def adapt_local_brain(spec: BrainSpec, probe: Dict[str, Any]) -> None:
    """Reconcile a local brain with what the machine actually runs.

    Mutates in place. An installed configured model always wins -- an
    explicit `--llm-model` must never be silently overridden. Otherwise
    the first installed model is adopted and the tier comes from its real
    parameter count, because a 7B and a 0.5B should not be offered the
    same work.
    """
    if not probe.get("reachable"):
        spec.enabled = False
        spec.setup_hint = f"Ollama не отвечает на {spec.base_url} — запустите `ollama serve`"
        return
    installed = probe.get("models") or []
    if not installed:
        spec.enabled = False
        spec.setup_hint = "Ollama запущена, но моделей нет — `ollama pull qwen2.5:7b-instruct`"
        return
    names = [m["name"] for m in installed]
    if spec.model in names:
        chosen = next(m for m in installed if m["name"] == spec.model)
    else:
        chosen = installed[0]
        spec.setup_hint = f"модель {spec.model} не установлена, используется {chosen['name']}"
        spec.model = chosen["name"]
    spec.tier = _tier_for_parameters(chosen.get("parameters", ""))


def load_catalog(cfg: Config) -> List[BrainSpec]:
    """Built-in catalog, optionally extended/overridden by a JSON file.

    The file format is a list of objects with BrainSpec field names. An
    entry whose `brain_id` already exists *patches* that brain (so you can
    change one model name without restating the whole spec); an unknown
    `brain_id` adds a new brain. Unknown fields are ignored rather than
    raising -- a config file written for a newer MANA must not stop an
    older one from starting.
    """
    catalog = default_catalog(cfg)
    path = getattr(cfg, "brains_file", "") or ""
    if path and Path(path).exists():
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            by_id = {b.brain_id: b for b in catalog}
            valid = set(BrainSpec.__dataclass_fields__.keys()) - {"api_key"}
            for item in raw if isinstance(raw, list) else []:
                if not isinstance(item, dict) or "brain_id" not in item:
                    continue
                fields = {k: v for k, v in item.items() if k in valid}
                if "strengths" in fields:
                    fields["strengths"] = tuple(str(x) for x in fields["strengths"])
                bid = str(fields["brain_id"])
                if bid in by_id:
                    for k, v in fields.items():
                        setattr(by_id[bid], k, v)
                else:
                    fields.setdefault("provider", "openai_chat")
                    fields.setdefault("model", "")
                    spec = BrainSpec(**fields)
                    catalog.append(spec)
                    by_id[bid] = spec
        except Exception:
            # A broken brains file must not be fatal: fall back to the
            # built-in catalog, exactly like a broken state pickle does.
            pass
    for spec in catalog:
        if spec.api_key_env:
            spec.api_key = os.environ.get(spec.api_key_env, "")
    return catalog


# ---------------------------------------------------------------------------
# Health, quotas, circuit breaking
# ---------------------------------------------------------------------------

@dataclass
class BrainHealth:
    """Per-brain live state. Everything a router needs in order to prefer a
    brain is measured here, not guessed from the catalog."""
    calls: int = 0
    ok: int = 0
    failures: int = 0
    timeouts: int = 0
    rate_limited: int = 0
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    inflight: int = 0
    ewma_latency: float = 0.0
    ewma_quality: float = 0.5
    last_error: str = ""
    minute_window: Deque[float] = field(default_factory=deque)
    day_count: int = 0
    day_started: float = field(default_factory=time.time)

    def success_rate(self) -> float:
        return self.ok / self.calls if self.calls else 0.5


class BrainPool:
    """The pool itself: catalog + health + selection + transport.

    Thread-safe. `ask()` is what every existing LLM call site reaches
    through LLMClient; `ask_many()`/`ask_consensus()` are the new
    multi-brain primitives.
    """

    def __init__(self, config: Config, vlog: Optional[Callable[[str], None]] = None,
                 transport: Optional[Callable[..., str]] = None):
        self.config = config
        self.vlog = vlog
        #: Injectable transport, used by the tests to exercise routing,
        #: failover, quotas and consensus without a network. Production
        #: leaves it None and the real adapters run.
        self._transport = transport
        self._lock = threading.RLock()
        self.brains: Dict[str, BrainSpec] = {}
        self.health: Dict[str, BrainHealth] = {}
        self._rr_index = 0
        self.calls = 0
        self.failures = 0
        self.timeouts = 0
        catalog = load_catalog(config)
        # One probe per pool, not per call, and only when a real network
        # is in play: an injected transport means a test, and a test must
        # not depend on whether the developer happens to have Ollama
        # running. A refused localhost connection returns instantly, so
        # this costs nothing on a machine without it.
        if self._transport is None and getattr(config, "brain_probe_local", True):
            for spec in catalog:
                if spec.provider == "ollama" and spec.enabled:
                    adapt_local_brain(spec, probe_ollama(spec.base_url))
        for spec in catalog:
            self.add(spec)

    # ---------- catalog ----------

    def add(self, spec: BrainSpec) -> None:
        with self._lock:
            self.brains[spec.brain_id] = spec
            self.health.setdefault(spec.brain_id, BrainHealth())

    def _log(self, message: str) -> None:
        if self.vlog:
            self.vlog(message)

    def usable(self, spec: BrainSpec) -> bool:
        """Configured, keyed and transport-capable. Says nothing about
        whether it is *currently* healthy -- see `ready`."""
        if not spec.enabled or not spec.has_key():
            return False
        if spec.local and not self.config.enable_llm:
            # --no-llm has always meant "no local backend"; it must not
            # silently keep talking to remote brains either.
            return False
        if not spec.local and not self.config.brain_external_enabled:
            return False
        if self._transport is not None:
            return True
        return bool(HAS_REQUESTS)

    def _quota_ok(self, spec: BrainSpec, h: BrainHealth, now: float) -> bool:
        if spec.rpd:
            if now - h.day_started > 86400.0:
                h.day_started, h.day_count = now, 0
            if h.day_count >= spec.rpd:
                return False
        if spec.rpm:
            while h.minute_window and now - h.minute_window[0] > 60.0:
                h.minute_window.popleft()
            if len(h.minute_window) >= spec.rpm:
                return False
        return True

    def ready(self, brain_id: str, now: Optional[float] = None) -> bool:
        """Usable AND not in cooldown AND within its free-tier quota."""
        now = now or time.time()
        with self._lock:
            spec = self.brains.get(brain_id)
            if spec is None or not self.usable(spec):
                return False
            h = self.health[brain_id]
            if now < h.cooldown_until:
                return False
            return self._quota_ok(spec, h, now)

    def available(self) -> List[str]:
        return [b for b in self.brains if self.ready(b)]

    def configured(self) -> List[str]:
        return [b for b, s in self.brains.items() if self.usable(s)]

    # ---------- routing ----------

    @staticmethod
    def estimate_difficulty(task: str) -> float:
        """Cheap 0..1 difficulty proxy, no LLM involved.

        It exists so the router can send the easy things to a small local
        model and spend a scarce free-tier call on the hard ones. It is a
        heuristic and is labelled as one everywhere it surfaces -- it is
        not a calibrated measure, and `record_outcome` is what actually
        corrects brain choice over time.
        """
        t = (task or "").strip().lower()
        if not t:
            return 0.0
        score = 0.0
        words = len(re.findall(r"\w+", t))
        score += min(0.30, words / 200.0)
        if len(t) > 400:
            score += 0.10
        hard_markers = ("почему", "объясни", "сравни", "спроектируй", "докажи", "выведи",
                        "оптимизируй", "архитектур", "trade-off", "компромисс",
                        "why", "explain", "compare", "design", "prove", "derive")
        score += 0.12 * sum(1 for m in hard_markers if m in t)
        multi_part = t.count("?") + len(re.findall(r"(?:^|\n)\s*\d+[\.\)]\s", t))
        if multi_part > 1:
            score += 0.10 * min(3, multi_part - 1)
        if re.search(r"\bи\b.*\bи\b", t) and words > 25:
            score += 0.08
        if re.search(r"```|def |class |import |SELECT |\bpython\b|\b1с\b", t):
            score += 0.12
        if re.search(r"\d+\s*(?:\*\*|[\*\+\-/])\s*\d+", t) and words < 15:
            score -= 0.20      # bare arithmetic is easy, whatever its length
        return max(0.0, min(1.0, score))

    @staticmethod
    def difficulty_to_tier(difficulty: float) -> str:
        if difficulty >= 0.55:
            return "large"
        if difficulty >= 0.25:
            return "medium"
        return "small"

    def _score(self, spec: BrainSpec, h: BrainHealth, *, kind: str, min_tier: str,
               policy: str, allow_paid: bool) -> float:
        """Higher is better. Every term is either declared catalog data or
        measured health -- deliberately no constants tuned against
        BenchmarkSuite, because 21 substring checks are not the thing brain
        selection has to generalize to."""
        if not spec.free and not allow_paid:
            return float("-inf")
        score = 1.0 * spec.weight
        if kind and kind in spec.strengths:
            score += 0.55
        gap = tier_rank(spec.tier) - tier_rank(min_tier)
        if gap < 0:
            score -= 0.45 * abs(gap)      # under-powered for this task
        else:
            score -= 0.08 * gap           # over-powered: works, but wasteful
        score += 0.50 * (h.ewma_quality - 0.5)
        score += 0.30 * (h.success_rate() - 0.5)
        if h.ewma_latency > 0:
            score += 0.25 * (1.0 / (1.0 + h.ewma_latency / 5.0)) - 0.125
        score -= 0.20 * min(3, h.inflight)
        if spec.free:
            score += 0.20
        if spec.local:
            score += float(self.config.brain_local_bonus)
        if policy == "fastest":
            score += 1.2 * (1.0 / (1.0 + h.ewma_latency / 3.0)) + (0.4 if spec.local else 0.0)
        elif policy == "cheapest":
            score += 1.5 * (1.0 if spec.local else (0.6 if spec.free else 0.0))
        elif policy == "least_loaded":
            score += 1.0 / (1.0 + h.inflight)
        elif policy == "strongest":
            score += 0.6 * tier_rank(spec.tier)
        return score

    def select(self, *, kind: str = "general", difficulty: Optional[float] = None,
               task: str = "", policy: str = "", exclude: Sequence[str] = (),
               limit: int = 1, allow_paid: Optional[bool] = None) -> List[str]:
        """Rank ready brains for this call and return the top `limit` ids.

        Returns [] when nothing is ready -- callers must treat that exactly
        like "LLM unavailable" and fall back, which every call site in MANA
        already does. That existing contract is what let this be dropped in
        behind LLMClient without touching any of them.
        """
        policy = policy or self.config.brain_policy
        allow_paid = self.config.brain_allow_paid if allow_paid is None else allow_paid
        if difficulty is None:
            difficulty = self.estimate_difficulty(task)
        min_tier = self.difficulty_to_tier(float(difficulty))
        now = time.time()
        with self._lock:
            scored: List[Tuple[float, str]] = []
            for bid, spec in self.brains.items():
                if bid in exclude or not self.ready(bid, now):
                    continue
                value = self._score(spec, self.health[bid], kind=kind, min_tier=min_tier,
                                    policy=policy, allow_paid=bool(allow_paid))
                if value == float("-inf"):
                    continue
                scored.append((value, bid))
            if not scored:
                return []
            if policy == "round_robin":
                ordered = sorted(b for _, b in scored)
                self._rr_index = (self._rr_index + 1) % len(ordered)
                rotated = ordered[self._rr_index:] + ordered[:self._rr_index]
                return rotated[:max(1, limit)]
            scored.sort(key=lambda x: (-x[0], x[1]))
            return [b for _, b in scored[:max(1, limit)]]

    # ---------- outcome feedback ----------

    def record_outcome(self, brain_id: str, quality: float) -> None:
        """Feed a downstream quality judgement (critic score, verification
        result, benchmark grade) back into the brain's reputation. This is
        the loop that makes selection improve with use instead of staying
        at whatever the catalog declared."""
        with self._lock:
            h = self.health.get(brain_id)
            if h is None:
                return
            a = float(self.config.brain_quality_ewma_alpha)
            h.ewma_quality = (1 - a) * h.ewma_quality + a * max(0.0, min(1.0, float(quality)))

    def _note_success(self, brain_id: str, latency: float) -> None:
        with self._lock:
            h = self.health[brain_id]
            h.calls += 1
            h.ok += 1
            h.consecutive_failures = 0
            h.cooldown_until = 0.0
            a = float(self.config.brain_latency_ewma_alpha)
            h.ewma_latency = latency if h.ewma_latency <= 0 else (1 - a) * h.ewma_latency + a * latency
            self.calls += 1

    def _note_failure(self, brain_id: str, error: str, *, timeout: bool, rate_limited: bool,
                      retry_after: float = 0.0, unreachable: bool = False) -> None:
        with self._lock:
            h = self.health[brain_id]
            h.calls += 1
            h.failures += 1
            h.consecutive_failures += 1
            h.last_error = error[:300]
            if timeout:
                h.timeouts += 1
                self.timeouts += 1
            self.failures += 1
            if rate_limited:
                h.rate_limited += 1
                # A 429 is not a broken brain, it is a full one: respect the
                # server's own Retry-After when it sends one instead of
                # applying the generic failure backoff, and do not let it
                # count toward tripping the circuit breaker.
                # The server's own Retry-After wins when it sends one: it
                # knows when the window reopens, and taking the larger of
                # the two (what this did first) means idling a healthy free
                # tier for longer than the provider asked for.
                wait = retry_after if retry_after > 0 else float(self.config.brain_rate_limit_cooldown)
                h.cooldown_until = time.time() + wait
                h.consecutive_failures = max(0, h.consecutive_failures - 1)
                return
            if unreachable:
                # No server on the other end. Waiting for three of these
                # before stepping aside means three wasted attempts per
                # call -- and in a consensus it silently turns N opinions
                # into N-1. One refusal is enough evidence.
                h.cooldown_until = time.time() + float(self.config.brain_cooldown_seconds)
                return
            if h.consecutive_failures >= int(self.config.brain_failure_limit):
                backoff = float(self.config.brain_cooldown_seconds) * (
                    2 ** min(4, h.consecutive_failures - int(self.config.brain_failure_limit)))
                h.cooldown_until = time.time() + min(backoff, float(self.config.brain_max_cooldown_seconds))

    def _reserve(self, brain_id: str) -> None:
        with self._lock:
            h = self.health[brain_id]
            h.inflight += 1
            h.minute_window.append(time.time())
            h.day_count += 1

    def _release(self, brain_id: str) -> None:
        with self._lock:
            h = self.health[brain_id]
            h.inflight = max(0, h.inflight - 1)

    # ---------- transport ----------

    def _call_brain(self, spec: BrainSpec, prompt: str, system: str, temperature: float,
                    timeout: float) -> str:
        if self._transport is not None:
            return self._transport(spec=spec, prompt=prompt, system=system,
                                   temperature=temperature, timeout=timeout)
        if spec.provider == "ollama":
            return self._call_ollama(spec, prompt, system, temperature, timeout)
        if spec.provider == "gemini":
            return self._call_gemini(spec, prompt, system, temperature, timeout)
        if spec.provider == "openai_responses":
            return self._call_openai_responses(spec, prompt, system, timeout)
        return self._call_openai_chat(spec, prompt, system, temperature, timeout)

    def _call_ollama(self, spec: BrainSpec, prompt: str, system: str, temperature: float,
                     timeout: float) -> str:
        payload = {"model": spec.model,
                   "prompt": ((system + "\n\n") if system else "") + prompt,
                   "stream": False,
                   "options": {"temperature": float(temperature), "num_predict": int(spec.max_tokens)}}
        r = requests.post(spec.base_url, json=payload, timeout=timeout)
        r.raise_for_status()
        text = (r.json().get("response") or "").strip()
        if not text:
            raise RuntimeError("empty response")
        return text

    def _call_openai_chat(self, spec: BrainSpec, prompt: str, system: str, temperature: float,
                          timeout: float) -> str:
        headers = {"Content-Type": "application/json"}
        if spec.api_key:
            headers["Authorization"] = f"Bearer {spec.api_key}"
        if "openrouter" in spec.base_url:
            headers["HTTP-Referer"] = "http://localhost"
            headers["X-Title"] = "MANA"
        payload = {"model": spec.model,
                   "messages": [{"role": "system", "content": system or "You are a helpful assistant."},
                                {"role": "user", "content": prompt}],
                   "temperature": float(temperature), "max_tokens": int(spec.max_tokens)}
        r = requests.post(spec.base_url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices") or []
        text = ""
        if choices:
            text = (choices[0].get("message", {}).get("content") or "").strip()
        if not text:
            # Aggregators (OpenRouter especially) return upstream provider
            # errors inside a 200 body, so raise_for_status alone is not
            # enough to notice that the call failed.
            err = data.get("error") or {}
            raise RuntimeError(str(err.get("message") or "empty response"))
        return text

    def _call_gemini(self, spec: BrainSpec, prompt: str, system: str, temperature: float,
                     timeout: float) -> str:
        url = spec.base_url.format(model=spec.model) if "{model}" in spec.base_url else spec.base_url
        payload = {"contents": [{"role": "user",
                                 "parts": [{"text": (system + "\n\n" if system else "") + prompt}]}],
                   "generationConfig": {"temperature": float(temperature),
                                        "maxOutputTokens": int(spec.max_tokens)}}
        # The key goes in a header, not the query string: the old
        # LLMClient._gemini passed params={"key": ...}, which puts a live
        # credential into proxy logs, crash reports and any request-debug
        # output. Same endpoint, same auth, one fewer place it leaks.
        r = requests.post(url, headers={"x-goog-api-key": spec.api_key,
                                        "Content-Type": "application/json"},
                          json=payload, timeout=timeout)
        r.raise_for_status()
        parts = []
        for c in r.json().get("candidates", []) or []:
            for part in c.get("content", {}).get("parts", []) or []:
                if part.get("text"):
                    parts.append(part["text"])
        text = "\n".join(parts).strip()
        if not text:
            raise RuntimeError("empty response")
        return text

    def _call_openai_responses(self, spec: BrainSpec, prompt: str, system: str, timeout: float) -> str:
        payload = {"model": spec.model,
                   "input": [
                       {"role": "system",
                        "content": [{"type": "input_text", "text": system or "You are a helpful assistant."}]},
                       {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
                   ],
                   "max_output_tokens": int(spec.max_tokens)}
        r = requests.post(spec.base_url,
                          headers={"Authorization": f"Bearer {spec.api_key}",
                                   "Content-Type": "application/json"},
                          json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if isinstance(data.get("output_text"), str) and data["output_text"].strip():
            return data["output_text"].strip()
        out = []
        for item in data.get("output", []) or []:
            for c in item.get("content", []) or []:
                if c.get("text"):
                    out.append(c["text"])
        text = "\n".join(out).strip()
        if not text:
            raise RuntimeError("empty response")
        return text

    # ---------- the calls ----------

    def ask_brain(self, brain_id: str, prompt: str, *, system: str = "", temperature: float = 0.2,
                  timeout: Optional[float] = None, context_tag: str = "") -> Dict[str, Any]:
        """One brain, one attempt. Never raises: a failure is data, because
        the router has to keep going and the health record needs it."""
        spec = self.brains.get(brain_id)
        if spec is None:
            return {"ok": False, "brain": brain_id, "text": None, "error": "unknown brain",
                    "latency": 0.0, "timeout": False}
        eff_timeout = float(timeout or spec.timeout or self.config.llm_timeout)
        self._reserve(brain_id)
        t0 = time.perf_counter()
        self._log(f"BRAIN START | {brain_id} ({spec.model}) | tag={context_tag or '-'} | "
                  f"timeout={eff_timeout:.0f}s")
        try:
            text = self._call_brain(spec, prompt, system, temperature, eff_timeout)
            elapsed = time.perf_counter() - t0
            self._note_success(brain_id, elapsed)
            self._log(f"BRAIN OK | {brain_id} | tag={context_tag or '-'} | time={elapsed:.2f}s")
            return {"ok": True, "brain": brain_id, "model": spec.model, "text": text,
                    "latency": elapsed, "error": "", "timeout": False}
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            timed_out, limited, retry_after = classify_error(exc)
            gone = is_unreachable(exc)
            self._note_failure(brain_id, f"{type(exc).__name__}: {exc}",
                               timeout=timed_out, rate_limited=limited, retry_after=retry_after,
                               unreachable=gone)
            self._log(f"BRAIN FAIL | {brain_id} | tag={context_tag or '-'} | "
                      f"{'429' if limited else ('unreachable' if gone else ('timeout' if timed_out else type(exc).__name__))} | "
                      f"time={elapsed:.2f}s")
            return {"ok": False, "brain": brain_id, "model": spec.model, "text": None,
                    "latency": elapsed, "error": f"{type(exc).__name__}: {exc}",
                    "timeout": timed_out, "rate_limited": limited, "unreachable": gone}
        finally:
            self._release(brain_id)

    def ask(self, prompt: str, *, system: str = "", temperature: float = 0.2, kind: str = "general",
            difficulty: Optional[float] = None, task: str = "", brain: str = "auto",
            policy: str = "", context_tag: str = "", timeout: Optional[float] = None,
            avoid: Sequence[str] = (), max_attempts: Optional[int] = None) -> Dict[str, Any]:
        """Route to the best brain, failing over down the ranking.

        `brain` accepts a brain_id or a legacy provider name -- see
        `resolve_alias`, which is what keeps PipelineSpec.llm_provider
        genomes evolved before 5.10 meaningful.

        `avoid` is a *preference*, not a constraint: it asks the router to
        pick someone else (the critic uses it so a draft is not judged by
        the model that wrote it), but if excluding those brains would leave
        nothing ready, the request still goes through rather than failing.
        Independence is worth having when it is free; it is not worth
        turning a working answer into no answer.
        """
        attempts = max(1, int(max_attempts or self.config.brain_max_attempts))
        tried: List[str] = []
        errors: List[Dict[str, str]] = []
        order: List[str] = []
        avoided = False
        if brain and brain != "auto":
            resolved = self.resolve_alias(brain)
            if resolved and self.ready(resolved):
                order.append(resolved)
            elif self.config.brain_strict_selection:
                return {"ok": False, "text": None, "brain": "", "error": f"brain {brain!r} is not ready",
                        "latency": 0.0, "latency_total": 0.0, "attempts": [], "errors": [],
                        "timeout": False, "avoided": False}
        if avoid and not order:
            order += self.select(kind=kind, difficulty=difficulty, task=task or prompt,
                                 policy=policy, exclude=list(avoid), limit=attempts)
            avoided = bool(order)
        if not order:
            order += self.select(kind=kind, difficulty=difficulty, task=task or prompt,
                                 policy=policy, exclude=order, limit=attempts)
        elif len(order) < attempts:
            order += self.select(kind=kind, difficulty=difficulty, task=task or prompt,
                                 policy=policy, exclude=order, limit=attempts - len(order))
        if not order:
            return {"ok": False, "text": None, "brain": "", "error": "no brain available",
                    "latency": 0.0, "latency_total": 0.0, "attempts": [], "errors": [],
                    "timeout": False, "avoided": False}
        total_latency = 0.0
        last_timeout = False
        for brain_id in order[:attempts]:
            res = self.ask_brain(brain_id, prompt, system=system, temperature=temperature,
                                 timeout=timeout, context_tag=context_tag)
            tried.append(brain_id)
            total_latency += float(res.get("latency", 0.0))
            if res.get("ok"):
                res["attempts"] = tried
                res["errors"] = errors
                res["latency_total"] = total_latency
                # Did the caller's `avoid` request actually hold? The critic
                # needs to know: a critique from the same brain that wrote
                # the draft is worth less than one from a second opinion,
                # and pretending otherwise would overstate the check.
                res["avoided"] = avoided and res.get("brain") not in set(avoid)
                return res
            errors.append({"brain": brain_id, "error": str(res.get("error", ""))})
            last_timeout = bool(res.get("timeout"))
        return {"ok": False, "text": None, "brain": tried[-1] if tried else "",
                "error": errors[-1]["error"] if errors else "no brain available",
                "latency": total_latency, "latency_total": total_latency,
                "attempts": tried, "errors": errors, "timeout": bool(last_timeout),
                "avoided": False}

    def resolve_alias(self, name: str) -> str:
        """Map a legacy provider name onto a brain id.

        PipelineSpec.llm_provider used to hold "ollama"/"gemini"/
        "openrouter"/"openai". Those genomes live in users' state pickles
        and in ExperienceDB rows; treating them as unknown would silently
        discard every routing preference evolution has already learned.
        """
        if name in self.brains:
            return name
        aliases = {"openrouter": "openrouter-free", "local": "ollama", "llama": "ollama"}
        candidate = aliases.get(name, name)
        if candidate in self.brains:
            return candidate
        for bid, spec in self.brains.items():
            if spec.provider == name or spec.model == name:
                return bid
        return ""

    def ask_many(self, prompt: str, *, n: int = 2, system: str = "", temperature: float = 0.2,
                 kind: str = "general", difficulty: Optional[float] = None, task: str = "",
                 policy: str = "", context_tag: str = "") -> List[Dict[str, Any]]:
        """Ask `n` *different* brains in parallel. Fewer results than `n`
        simply means fewer brains were ready -- never an error."""
        ids = self.select(kind=kind, difficulty=difficulty, task=task or prompt, policy=policy,
                          limit=max(1, int(n)))
        if not ids:
            return []
        if len(ids) == 1:
            return [self.ask_brain(ids[0], prompt, system=system, temperature=temperature,
                                   context_tag=context_tag)]
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(ids)) as pool:
            futures = {pool.submit(self.ask_brain, bid, prompt, system=system,
                                   temperature=temperature, context_tag=f"{context_tag} ENS"): bid
                       for bid in ids}
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:   # a broken future must not kill the ensemble
                    results.append({"ok": False, "brain": futures[fut], "text": None,
                                    "error": f"{type(exc).__name__}: {exc}", "latency": 0.0})
        order = {bid: i for i, bid in enumerate(ids)}
        results.sort(key=lambda r: order.get(str(r.get("brain", "")), 99))

        # Top up. Asking N brains and accepting N-minus-the-failures
        # opinions quietly weakens the thing consensus exists to provide:
        # the first live run wanted two views, one selected brain had no
        # server behind it, and the result was a single answer reported --
        # correctly, but uselessly -- as `single`. If other brains are
        # ready, ask them rather than returning a thinner ensemble.
        # Bounded by construction: each round excludes everyone already
        # tried, so it terminates when the pool runs out.
        tried = set(ids)
        wanted = max(1, int(n))
        rounds = 0
        while sum(1 for r in results if r.get("ok")) < wanted and rounds < wanted:
            rounds += 1
            missing = wanted - sum(1 for r in results if r.get("ok"))
            extra = self.select(kind=kind, difficulty=difficulty, task=task or prompt,
                                policy=policy, exclude=sorted(tried), limit=missing)
            if not extra:
                break
            tried.update(extra)
            for bid in extra:
                results.append(self.ask_brain(bid, prompt, system=system,
                                              temperature=temperature,
                                              context_tag=f"{context_tag} ENS+"))
        return results

    def ask_consensus(self, prompt: str, *, n: int = 2, **kwargs: Any) -> Dict[str, Any]:
        """Ask several brains and report both the answer and how much the
        brains actually agreed.

        The agreement number is the point. MANA already refuses to treat a
        single LLM answer as a fact (`verification_kind` exists precisely
        for that); independent models converging is weak positive evidence
        and diverging is a strong signal to verify or to say so out loud.
        The selected answer is the *medoid* -- the response most similar to
        the others -- not the first or the longest, so one verbose outlier
        cannot carry the vote.
        """
        results = self.ask_many(prompt, n=n, **kwargs)
        latency = sum(float(r.get("latency", 0.0)) for r in results)
        ok = [r for r in results if r.get("ok") and r.get("text")]
        if not ok:
            return {"ok": False, "text": None, "brain": "", "agreement": 0.0,
                    "brains": [r.get("brain") for r in results], "responses": results,
                    "error": (results[-1].get("error") if results else "no brain available"),
                    "disagreement": False, "single": False, "latency": latency}
        if len(ok) == 1:
            return {"ok": True, "text": ok[0]["text"], "brain": ok[0]["brain"], "agreement": 0.0,
                    "brains": [ok[0]["brain"]], "responses": results, "single": True,
                    "disagreement": False, "error": "", "latency": latency}
        sims: Dict[int, float] = {}
        for i, a in enumerate(ok):
            others = [answer_similarity(a["text"], b["text"]) for j, b in enumerate(ok) if i != j]
            sims[i] = sum(others) / len(others)
        best_i = max(sims, key=lambda i: (sims[i], -len(ok[i]["text"])))
        agreement = float(sims[best_i])
        return {"ok": True, "text": ok[best_i]["text"], "brain": ok[best_i]["brain"],
                "agreement": agreement,
                "disagreement": agreement < float(self.config.brain_consensus_threshold),
                "brains": [r["brain"] for r in ok], "responses": results, "single": False,
                "error": "", "latency": latency}

    # ---------- reporting ----------

    def status(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            brains = []
            for bid, spec in self.brains.items():
                h = self.health[bid]
                brains.append({
                    **spec.public_dict(),
                    "usable": self.usable(spec), "ready": self.ready(bid, now),
                    "calls": h.calls, "ok": h.ok, "failures": h.failures, "timeouts": h.timeouts,
                    "rate_limited": h.rate_limited, "inflight": h.inflight,
                    "cooldown_for": round(max(0.0, h.cooldown_until - now), 1),
                    "ewma_latency": round(h.ewma_latency, 3), "ewma_quality": round(h.ewma_quality, 3),
                    "day_count": h.day_count, "last_error": h.last_error,
                })
            brains.sort(key=lambda b: (not b["ready"], b["brain_id"]))
            return {"calls": self.calls, "failures": self.failures, "timeouts": self.timeouts,
                    "configured": self.configured(), "available": self.available(),
                    "policy": self.config.brain_policy, "brains": brains}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


#: Verdict words, matched at the start of an answer. Two models that both
#: answer "нет" and then explain differently AGREE -- see `verdict_of`.
_VERDICT_RE = re.compile(r"^\W*(нет|да|yes|no|true|false|верно|неверно)\b", re.I | re.UNICODE)
_VERDICT_CANON = {"нет": "no", "no": "no", "false": "no", "неверно": "no",
                  "да": "yes", "yes": "yes", "true": "yes", "верно": "yes"}


def verdict_of(text: str) -> Optional[str]:
    """The yes/no verdict an answer opens with, if it opens with one.

    Only the opening is inspected on purpose. "Нет. Модель может ошибаться"
    is a verdict; a "нет" buried in the middle of an explanation is not,
    and treating it as one would make two opposite answers look identical.
    """
    match = _VERDICT_RE.match((text or "").strip())
    return _VERDICT_CANON.get(match.group(1).lower()) if match else None


def answer_similarity(a: str, b: str) -> float:
    """Agreement between two answers, 0..1.

    Token Jaccard alone is wrong in both directions: it rates "391" and
    "по моим расчётам получается 391" as barely similar, and rates two
    differently-worded refusals as very similar. So the substance is
    checked before the wording.

    Two substance channels, in priority order:

      1. **Verdict.** Found on the first live run against two real models:
         both answered "Нет" to "can an LLM answer be treated as a proven
         fact", explained it in different words, and scored 0.29 -- which
         the pool reported as disagreement. They agreed completely on the
         only thing that mattered. Wording overlap is a poor proxy for
         agreement on prose, and for a yes/no question it is the wrong
         measurement entirely.
      2. **Numbers.** Where both answers contain figures, agreeing on the
         figures IS the agreement.

    Wording still contributes, but only as a tiebreaker: two answers that
    reach the same verdict for visibly different reasons are less
    corroborating than two that reason alike, and that difference should
    be visible without flipping the verdict.
    """
    ta = {w.lower() for w in _WORD_RE.findall(a or "")}
    tb = {w.lower() for w in _WORD_RE.findall(b or "")}
    jaccard = len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0

    va, vb = verdict_of(a), verdict_of(b)
    if va and vb:
        # An explicit contradiction is the strongest signal available:
        # floor it rather than letting shared vocabulary soften it.
        return (0.80 + 0.20 * jaccard) if va == vb else min(0.15, jaccard)

    na = {x.replace(",", ".") for x in _NUM_RE.findall(a or "")}
    nb = {x.replace(",", ".") for x in _NUM_RE.findall(b or "")}
    if na and nb:
        numeric = len(na & nb) / len(na | nb)
        return 0.65 * numeric + 0.35 * jaccard
    return jaccard


def classify_error(exc: Exception) -> Tuple[bool, bool, float]:
    """(is_timeout, is_rate_limited, retry_after_seconds).

    String-tolerant on purpose: `requests` is an optional dependency and a
    test transport raises plain exceptions, so this must not depend on
    importing requests' exception classes.
    """
    name = type(exc).__name__
    text = str(exc)
    timed_out = "Timeout" in name or "timed out" in text.lower()
    status = getattr(getattr(exc, "response", None), "status_code", None)
    limited = (status == 429) or "429" in text or "rate limit" in text.lower() or "quota" in text.lower()
    retry_after = 0.0
    if limited:
        headers = getattr(getattr(exc, "response", None), "headers", None) or {}
        try:
            retry_after = float(headers.get("Retry-After", 0) or 0)
        except (TypeError, ValueError):
            retry_after = 0.0
    return timed_out, bool(limited), retry_after


def is_unreachable(exc: Exception) -> bool:
    """Nothing is listening, as opposed to something answering badly.

    The distinction earns its own function because the two deserve
    opposite treatment. A flaky endpoint should get the benefit of the
    doubt for a few tries (that is what brain_failure_limit is for); an
    endpoint that refuses the TCP connection will refuse the next one too,
    and retrying it wastes an attempt every call.

    Found on the first live run: with no Ollama server installed, the
    local brain still ranked first (it gets the local bonus) and was still
    picked for a two-brain consensus. The call failed, one opinion came
    back instead of two, and the pool honestly reported `single` -- correct
    behaviour built on a brain that never had a chance of answering.
    """
    name = type(exc).__name__
    text = str(exc).lower()
    if "ConnectionError" in name or "ConnectionRefused" in name or "NewConnectionError" in name:
        return True
    return any(marker in text for marker in (
        "connection refused", "failed to establish", "max retries exceeded",
        "actively refused", "name or service not known", "getaddrinfo failed",
    ))
