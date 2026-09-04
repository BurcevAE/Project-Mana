"""
mana.hardware — detect the machine MANA is running on and adapt Config to it.

Goal: the same MANA checkout should behave sensibly whether it lands on a
workstation or a constrained laptop after being copied to a new machine,
without the person hand-tuning worker counts, timeouts or population sizes
each time. Detection is best-effort and every signal is optional; on a
machine where nothing can be detected, Config's existing defaults are left
untouched rather than guessed at aggressively.

This module only *adjusts numbers already in Config* (worker counts,
timeouts, population sizes, whether to load the heavy embedding model). It
does not choose an LLM backend/model or touch anything the person set
explicitly on the command line -- see ManaAgent's wiring in
agent_parts/core.py for exactly when and how it's applied.
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, TYPE_CHECKING

from .optional_deps import torch, HAS_TORCH, psutil, HAS_PSUTIL

if TYPE_CHECKING:
    from .config import Config


@dataclass
class HardwareProfile:
    cpu_count: int
    total_ram_gb: float
    has_cuda: bool
    gpu_name: str
    platform: str
    tier: str  # "low" / "medium" / "high"
    detected_via: Dict[str, str] = field(default_factory=dict)


def _detect_ram_gb() -> Tuple[float, str]:
    if HAS_PSUTIL:
        try:
            return psutil.virtual_memory().total / (1024 ** 3), "psutil"
        except Exception:
            pass
    # Dependency-free fallback for Linux (covers most server/container
    # deployments even when psutil isn't installed).
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2), "/proc/meminfo"
    except Exception:
        pass
    # NOTE (bugfix, caught by real-machine testing on Windows -- the
    # sandbox this module was authored in is Linux-only, so this branch
    # never ran during development). Without psutil, Windows had no
    # fallback at all and silently reported 0.0 GB / tier "medium"
    # regardless of the actual machine -- exactly the "works on a
    # different machine" guarantee this module exists to provide,
    # silently failing on one of the two most common desktop OSes.
    if platform.system() == "Windows":
        try:
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullTotalPhys / (1024 ** 3), "GlobalMemoryStatusEx"
        except Exception:
            pass
    return 0.0, "unknown"


def _detect_gpu() -> Tuple[bool, str, str]:
    if HAS_TORCH:
        try:
            if torch.cuda.is_available():
                return True, torch.cuda.get_device_name(0), "torch"
        except Exception:
            pass
    return False, "", "none"


def detect_hardware() -> HardwareProfile:
    cpu_count = os.cpu_count() or 1
    ram_gb, ram_source = _detect_ram_gb()
    has_cuda, gpu_name, gpu_source = _detect_gpu()

    # NOTE (bugfix, caught by real-machine testing): the RAM-unknown check
    # used to come FIRST, so a machine with a CUDA GPU and 8 cores was
    # classified "medium" purely because psutil wasn't installed and RAM
    # read as 0.0 (exactly what happened on the reporter's GTX 1070 /
    # 8-core Windows box). An unreadable RAM figure is missing evidence
    # about one signal -- it must not discard the signals that WERE read.
    # Order the checks by signal strength instead: a present CUDA GPU is
    # decisive on its own, and only fall back to "medium" when nothing
    # conclusive is known.
    if has_cuda or (cpu_count >= 8 and ram_gb >= 16):
        tier = "high"
    elif ram_gb <= 0:
        # RAM couldn't be read and there's no GPU -- assume "medium"
        # rather than crippling a machine we simply failed to inspect.
        tier = "medium"
    elif cpu_count <= 2 or ram_gb <= 4:
        tier = "low"
    else:
        tier = "medium"

    return HardwareProfile(
        cpu_count=cpu_count,
        total_ram_gb=round(ram_gb, 1),
        has_cuda=has_cuda,
        gpu_name=gpu_name,
        platform=platform.platform(),
        tier=tier,
        detected_via={"ram": ram_source, "gpu": gpu_source},
    )

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"


def apply_hardware_profile(cfg: "Config", profile: HardwareProfile) -> Dict[str, Any]:
    """Mutate `cfg` in place for the detected machine.

    Returns {field_name: {"before": ..., "after": ...}} for every field
    actually changed, so the caller can log exactly what was adapted
    instead of this happening silently.
    """
    changes: Dict[str, Any] = {}

    def _set(field_name: str, value: Any) -> None:
        before = getattr(cfg, field_name)
        if before != value:
            changes[field_name] = {"before": before, "after": value}
            setattr(cfg, field_name, value)

    # Evolution worker threads must never exceed real cores, and should
    # leave the local LLM backend (often a separate process, e.g. Ollama)
    # headroom rather than agent-side threads saturating every core.
    _set("evolution_workers", max(1, min(cfg.evolution_workers, max(1, profile.cpu_count - 1))))

    if profile.tier == "low":
        _set("strategy_population", min(cfg.strategy_population, 4))
        _set("strategy_generations", min(cfg.strategy_generations, 2))
        _set("max_pipeline_evaluations_per_cycle", min(cfg.max_pipeline_evaluations_per_cycle, 8))
        _set("llm_timeout", max(cfg.llm_timeout, 60))
        _set("evolution_llm_timeout", max(cfg.evolution_llm_timeout, 30))
        # sentence-transformers is the single heaviest optional dependency
        # MANA can load (a real neural embedding model); on a constrained
        # machine prefer the much lighter TF-IDF path KnowledgeBase already
        # falls back to whenever embeddings are unavailable/disabled.
        _set("use_embeddings", False)
    elif profile.tier == "high":
        # A capable machine can afford to notice a slow/hung LLM call
        # sooner rather than waiting out the conservative default.
        _set("llm_timeout", min(cfg.llm_timeout, 45))

    return changes
