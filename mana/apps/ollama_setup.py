"""
mana.apps.ollama_setup — getting a local model onto a machine that has none.

A fresh installation answers every question with a refusal, because it has
no language model and no way to get one without the user knowing what to
install. The window now says so; this is the other half -- offering the
thing, sized to the machine it will run on.

The model is chosen from a measurement, not a table
-----------------------------------------------------
Decoding is bound by memory bandwidth: every token reads the whole model.
So the expected speed is roughly bandwidth divided by model size, and
both terms are knowable on the machine in front of us -- the bandwidth by
measuring it, the size from the catalogue.

That matters because the usual advice ("8 GB of RAM? take a 7B") ignores
the term that decides whether it is usable. Two machines with the same
RAM and different memory speed give different answers, and a
recommendation that cannot say "about 6 tokens a second" is a
recommendation nobody can judge.

The estimate is a bound, not a benchmark, and says so: it ignores
quantisation details, prompt processing and whatever else is running.
Reported alongside the choice so a person can disagree with it.

Nothing is installed without being asked
------------------------------------------
`recommend()` and `status()` only look. `install_runtime()` and
`pull_model()` change the machine and are never called on their own --
downloading several gigabytes and installing software is a decision that
belongs to whoever owns the computer.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

DEFAULT_URL = "http://127.0.0.1:11434"

#: Candidates, smallest first, with the size of the default (4-bit) build.
#: Sizes are what `ollama pull` actually downloads -- taken from the
#: catalogue rather than computed from parameter counts, because the
#: quantisation and the embedding tables move the real number.
CANDIDATES = (
    ("qwen2.5:1.5b-instruct", 1.0, "самая маленькая; отвечает быстро, "
                                   "рассуждает слабо"),
    ("qwen2.5:3b-instruct", 1.9, "разумный минимум для связного диалога"),
    ("qwen2.5:7b-instruct", 4.7, "рабочая лошадка: заметно сильнее 3B "
                                 "и ещё быстрая"),
    ("qwen2.5:14b-instruct", 9.0, "сильнее, но на процессоре уже медленно"),
)

#: Below this a model is technically working and practically unusable --
#: a sentence takes most of a minute. Chosen as the point where a person
#: stops waiting and starts doing something else.
MIN_TOKENS_PER_SECOND = 3.0

#: The speed a default should clear. "The largest that fits" is not the
#: same as "the best choice": on this machine it picked a 14B at 5.7
#: tokens a second over a 7B at 10.9, which is more capable and worse to
#: sit in front of. So the default is the largest model that is still
#: comfortable, and the bigger one stays in `alternatives` for anyone who
#: wants capability more than latency.
COMFORTABLE_TOKENS_PER_SECOND = 8.0

#: Left for the operating system, the browser and MANA itself. A model
#: that fits only if nothing else runs does not fit.
#:
#: Proportional, not fixed. A flat 8 GB gave a machine with 8 GB of RAM a
#: budget of zero and no recommendation at all -- while a 1.5B model, one
#: gigabyte, would have run there perfectly well. The floor keeps small
#: machines from being told they can run nothing; the fraction keeps big
#: ones from being stripped of headroom they actually have.
RAM_HEADROOM_MIN_GB = 3.0
RAM_HEADROOM_FRACTION = 0.25


def _headroom(ram_gb: float) -> float:
    return max(RAM_HEADROOM_MIN_GB, ram_gb * RAM_HEADROOM_FRACTION)


class OllamaError(RuntimeError):
    pass


# ------------------------------------------------------------- looking only


def executable() -> str:
    """Path to the ollama binary, or "" if it is not installed."""
    found = shutil.which("ollama") or shutil.which("ollama.exe")
    if found:
        return found
    for root in (os.environ.get("LOCALAPPDATA", ""),
                 os.environ.get("ProgramFiles", "")):
        if not root:
            continue
        candidate = os.path.join(root, "Programs", "Ollama", "ollama.exe")
        if os.path.isfile(candidate):
            return candidate
        candidate = os.path.join(root, "Ollama", "ollama.exe")
        if os.path.isfile(candidate):
            return candidate
    return ""


def running(base_url: str = DEFAULT_URL) -> Dict[str, Any]:
    """Whether the service answers, and what it has.

    Installed and running are different states: the binary can be present
    with nothing listening, and reporting that as "you have ollama" would
    send the user back to a window that still refuses everything.
    """
    try:
        from ..brains import probe_ollama
        return dict(probe_ollama(base_url))
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def installed_models(base_url: str = DEFAULT_URL) -> List[str]:
    probe = running(base_url)
    # `reachable`, not `ok`: probe_ollama names it that, and reading the
    # wrong key made every machine look like it had no service at all.
    if not probe.get("reachable"):
        return []
    names = probe.get("models") or []
    out = []
    for entry in names:
        name = entry.get("name") if isinstance(entry, dict) else entry
        if name:
            out.append(str(name))
    return out


# ------------------------------------------------------------- measuring


def memory_bandwidth_gbps(seconds: float = 0.35) -> float:
    """Streaming memory bandwidth, measured on this machine.

    A copy, counting the read and the write, on several threads: one core
    cannot saturate a modern memory controller, and a single-threaded
    figure understates the machine by several times. Short on purpose --
    this runs while somebody waits for a dialog.
    """
    try:
        import numpy
    except Exception:
        return 0.0

    size = 64 * 1024 * 1024
    threads = min(8, os.cpu_count() or 1)
    results = [0.0] * threads
    barrier = threading.Barrier(threads)

    def worker(index: int) -> None:
        source = numpy.ones(size // 8, dtype=numpy.float64)
        destination = numpy.empty_like(source)
        numpy.copyto(destination, source)
        barrier.wait()
        started = time.perf_counter()
        rounds = 0
        while time.perf_counter() - started < seconds:
            numpy.copyto(destination, source)
            rounds += 1
        elapsed = time.perf_counter() - started
        results[index] = 2 * rounds * source.nbytes / elapsed if elapsed else 0.0

    workers = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()
    return round(sum(results) / 1e9, 1)


def _usable_gpu() -> Dict[str, Any]:
    """A GPU ollama can actually offload to, if there is one.

    Conservative on purpose. Ollama accelerates on NVIDIA through CUDA and
    on a subset of AMD cards through ROCm; an RX 580 is gfx803, which ROCm
    dropped, so a machine can hold 8 GB of VRAM that will not be used.
    Counting it would produce a recommendation that runs at a fraction of
    the promised speed, and the person would have no way to know why.
    """
    if sys.platform != "win32":
        return {"present": False, "why": "проверка только для Windows"}
    try:
        import winreg
        base = (r"SYSTEM\CurrentControlSet\Control\Class"
                r"\{4d36e968-e325-11ce-bfc1-08002be10318}")
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as key:
            for index in range(64):
                try:
                    child = winreg.EnumKey(key, index)
                except OSError:
                    break
                if not child.isdigit():
                    continue
                try:
                    with winreg.OpenKey(key, child) as sub:
                        name = str(winreg.QueryValueEx(sub, "DriverDesc")[0])
                        try:
                            vram = int(winreg.QueryValueEx(
                                sub, "HardwareInformation.qwMemorySize")[0])
                        except OSError:
                            vram = 0
                except OSError:
                    continue
                if "nvidia" in name.lower():
                    return {"present": True, "name": name,
                            "vram_gb": round(vram / 1e9, 1),
                            "why": "NVIDIA: ollama использует CUDA"}
    except Exception:
        pass
    return {"present": False,
            "why": "ускорителя, который ollama может использовать, не найдено; "
                   "модель пойдёт на процессоре"}


@dataclass
class Recommendation:
    model: str
    size_gb: float
    tokens_per_second: float
    note: str
    reason: str
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    bandwidth_gbps: float = 0.0
    ram_gb: float = 0.0
    gpu: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"model": self.model, "size_gb": self.size_gb,
                "tokens_per_second": self.tokens_per_second,
                "note": self.note, "reason": self.reason,
                "alternatives": self.alternatives,
                "bandwidth_gbps": self.bandwidth_gbps, "ram_gb": self.ram_gb,
                "gpu": self.gpu}


def recommend(measure: bool = True) -> Optional[Recommendation]:
    """The largest model this machine can run at a usable speed.

    Two constraints, and both have to hold: it must fit in memory with
    room to spare, and it must decode fast enough that somebody waits for
    it. The second is the one usually left out, and it is the one that
    decides whether the recommendation was any good.
    """
    from ..hardware import detect_hardware

    profile = detect_hardware()
    ram = float(getattr(profile, "total_ram_gb", 0.0) or 0.0)
    gpu = _usable_gpu()
    bandwidth = memory_bandwidth_gbps() if measure else 0.0
    if bandwidth <= 0:
        # Could not measure. DDR4 dual channel, deliberately pessimistic:
        # a recommendation that overstates the speed is worse than one
        # that understates it.
        bandwidth = 25.0

    budget = max(0.0, ram - _headroom(ram))
    if gpu.get("present") and gpu.get("vram_gb"):
        budget = max(budget, float(gpu["vram_gb"]) - 1.0)

    considered: List[Dict[str, Any]] = []
    comfortable: Optional[Recommendation] = None
    workable: Optional[Recommendation] = None

    def make(name, size, speed, note, why):
        return Recommendation(
            model=name, size_gb=size, tokens_per_second=speed, note=note,
            reason=why, bandwidth_gbps=bandwidth, ram_gb=ram, gpu=gpu)

    for name, size, note in CANDIDATES:
        speed = round(bandwidth / size, 1) if size else 0.0
        fits = size <= budget
        considered.append({"model": name, "size_gb": size,
                           "tokens_per_second": speed, "fits": fits,
                           "fast_enough": speed >= MIN_TOKENS_PER_SECOND,
                           "comfortable": speed >= COMFORTABLE_TOKENS_PER_SECOND,
                           "note": note})
        if not fits:
            continue
        why = (f"влезает в {budget:.0f} ГБ свободной памяти и по измеренной "
               f"полосе {bandwidth:.0f} ГБ/с даёт около {speed:.0f} токенов "
               f"в секунду")
        if speed >= COMFORTABLE_TOKENS_PER_SECOND:
            comfortable = make(name, size, speed, note, why)
        elif speed >= MIN_TOKENS_PER_SECOND and workable is None:
            workable = make(name, size, speed, note, why)

    # The comfortable one wins where there is one. Falling back to merely
    # workable is for machines where nothing is fast -- there the choice
    # is between slow and nothing.
    best = comfortable or workable
    if best is None:
        return None
    best.alternatives = considered
    return best


def status(base_url: str = DEFAULT_URL) -> Dict[str, Any]:
    """Everything the window needs to decide what to offer."""
    path = executable()
    service = running(base_url)
    up = bool(service.get("reachable"))
    models = installed_models(base_url) if up else []
    return {
        "installed": bool(path),
        "path": path,
        "service_up": up,
        "service_error": str(service.get("error") or ""),
        "models": models,
        "has_model": bool(models),
        "url": base_url,
    }


# ------------------------------------------------------------- changing


def _run(command: List[str], timeout: float,
         on_line: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise OllamaError(f"не найдено: {exc}") from exc

    lines: List[str] = []
    deadline = time.time() + timeout
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            lines.append(line)
            if on_line is not None:
                try:
                    on_line(line)
                except Exception:
                    pass
        if time.time() > deadline:
            process.kill()
            raise OllamaError(f"не уложилось в {timeout:.0f} с")
    code = process.wait()
    return {"ok": code == 0, "code": code, "output": lines[-20:]}


def install_runtime(on_line: Optional[Callable[[str], None]] = None,
                    timeout: float = 900.0) -> Dict[str, Any]:
    """Install Ollama through winget. Never called without being asked.

    winget rather than downloading an installer ourselves: it verifies the
    package against Microsoft's catalogue, and a program that fetches an
    executable from the internet and runs it is a program nobody should
    have to trust.
    """
    if executable():
        return {"ok": True, "already": True, "path": executable()}
    if not (shutil.which("winget") or shutil.which("winget.exe")):
        raise OllamaError(
            "winget не найден. Поставьте Ollama вручную с https://ollama.com "
            "и нажмите «Проверить снова»")
    result = _run(["winget", "install", "--id", "Ollama.Ollama", "-e",
                   "--accept-package-agreements", "--accept-source-agreements",
                   "--disable-interactivity"], timeout, on_line)
    result["path"] = executable()
    result["ok"] = bool(result["path"])
    return result


def pull_model(model: str, on_line: Optional[Callable[[str], None]] = None,
               timeout: float = 3600.0) -> Dict[str, Any]:
    """Download a model. Gigabytes, so the caller gets progress lines."""
    binary = executable()
    if not binary:
        raise OllamaError("ollama не установлена")
    result = _run([binary, "pull", model], timeout, on_line)
    result["model"] = model
    result["present"] = model in installed_models()
    result["ok"] = bool(result["present"])
    return result
