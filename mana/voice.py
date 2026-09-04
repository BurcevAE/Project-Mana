"""
mana.voice — voice interface: wake-word routing, VAD capture, Whisper STT, Silero/pyttsx3 TTS.
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
from .agent import ManaAgent
from .optional_deps import (
    sd, HAS_SOUNDDEVICE, WhisperModel, HAS_WHISPER, pyttsx3, HAS_TTS,
    torch, DEVICE, HAS_TORCH,
)

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"


# ---------------------------------------------------------------------------
# Voice
# ---------------------------------------------------------------------------

class VoiceInterface:
    def __init__(self, agent: ManaAgent, config: Config):
        self.agent = agent
        self.config = config
        self.whisper = None
        self.tts_backend = None
        self.silero_model = None
        self.silero_sr = config.voice_silero_sample_rate
        self.pyttsx3_engine = None
        self.running = True
        self._last_listen_notice = 0.0
        if not (HAS_SOUNDDEVICE and HAS_WHISPER):
            raise RuntimeError("Для voice нужны: sounddevice и faster-whisper")
        device = "cuda" if HAS_TORCH and str(DEVICE) == "cuda" else "cpu"
        prefs = list(config.voice_compute_preferences) if device == "cuda" else ["int8", "float32"]
        for ct in prefs:
            try:
                self.whisper = WhisperModel(config.voice_whisper_model, device=device, compute_type=ct)
                self.compute_type = ct
                break
            except Exception:
                pass
        if self.whisper is None:
            raise RuntimeError("Whisper backend unavailable")
        self._init_tts()
        self._print_audio_devices()
        print(f"🎙️ Voice: Whisper={config.voice_whisper_model} device={device} compute={getattr(self, 'compute_type', '?')}")

    def _print_audio_devices(self):
        try:
            out = sd.default.device[1]
            print(f"🔈 Устройство вывода по умолчанию: [{out}] {sd.query_devices(out).get('name')}")
            if self.config.voice_output_device is not None:
                sd.default.device = (sd.default.device[0], self.config.voice_output_device)
                print(f"🔈 Переопределено устройством вывода: {self.config.voice_output_device}")
        except Exception as exc:
            print(f"⚠️ Не удалось определить аудио-устройство: {exc}")

    def _init_tts(self):
        backend = self.config.voice_tts_backend
        if backend in {"auto", "silero"} and HAS_TORCH:
            try:
                model, _ = torch.hub.load("snakers4/silero-models", "silero_tts", language="ru", speaker="v4_ru", trust_repo=True)
                model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                self.silero_model = model
                self.tts_backend = "silero"
                print(f"🔊 TTS: Silero v4_ru, голос={self.config.voice_silero_speaker}, device={'cuda' if torch.cuda.is_available() else 'cpu'}")
                return
            except Exception as exc:
                if backend == "silero":
                    print(f"❌ Silero TTS недоступен: {exc}")
                    return
        if backend in {"auto", "pyttsx3"} and HAS_TTS:
            try:
                self.pyttsx3_engine = pyttsx3.init()
                self.pyttsx3_engine.setProperty("rate", 175)
                self.pyttsx3_engine.setProperty("volume", 1.0)
                self.tts_backend = "pyttsx3"
                print("🔊 TTS: pyttsx3")
            except Exception as exc:
                print(f"❌ pyttsx3 initialization failed: {exc}")

    def speak(self, text: str):
        text = str(text or "").strip()
        if not text: return
        print(f"MANA (voice): {text}")
        # TTS-friendly normalization: pronounce digits and latin abbreviations reasonably in Russian.
        safe = self._tts_text(text)
        try:
            if self.tts_backend == "silero" and self.silero_model is not None:
                audio = self.silero_model.apply_tts(text=safe, speaker=self.config.voice_silero_speaker, sample_rate=self.silero_sr)
                arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
                sd.play(arr, samplerate=self.silero_sr, blocking=True)
            elif self.tts_backend == "pyttsx3":
                engine = pyttsx3.init()
                engine.setProperty("rate", 175); engine.setProperty("volume", 1.0)
                engine.say(safe); engine.runAndWait(); engine.stop()
        except Exception as exc:
            print(f"⚠️ TTS error: {exc}")

    @staticmethod
    def _tts_text(text: str) -> str:
        # Keep log text untouched, but make common status tokens speakable.
        repl = {
            "cycle": "цикл", "fitness": "качество", "p50": "пятьдесят процентиль",
            "p95": "девяносто пятый процентиль", "LLM": "языковая модель", "HOLDOUT": "контрольный тест",
            "YES": "да", "NO": "нет", "s": "секунд",
        }
        out = text
        for k, v in repl.items():
            out = re.sub(rf"\b{re.escape(k)}\b", v, out, flags=re.I)
        # Common decimal notation in fitness/statistics.
        def dec(m):
            return m.group(1).replace('.', ' точка ')
        out = re.sub(r"\b(\d+\.\d+)\b", dec, out)
        return out

    @staticmethod
    def _normalize(text: str) -> str:
        text = (text or "").lower().replace("ё", "е")
        text = text.replace("-", " ")
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _record_vad(self):
        sr = self.config.voice_sample_rate; block = .20; block_n = int(sr * block)
        pre_n = max(1, int(self.config.voice_pre_roll_sec / block))
        pre, frames = [], []
        try:
            ambient = sd.rec(int(sr * .35), samplerate=sr, channels=1, dtype="float32"); sd.wait()
            ambient_rms = float(np.sqrt(np.mean(np.square(ambient))) + 1e-7)
        except Exception:
            ambient_rms = .003
        threshold = max(.008, ambient_rms * 2.8)
        started = False; silence = 0.0; elapsed = 0.0
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=block_n) as stream:
            while elapsed < self.config.voice_seconds:
                data, _ = stream.read(block_n); data = np.asarray(data, dtype=np.float32)
                rms = float(np.sqrt(np.mean(np.square(data))) + 1e-9)
                pre.append(data.copy())
                if len(pre) > pre_n: pre.pop(0)
                if rms >= threshold:
                    if not started: frames.extend(pre); pre.clear(); started = True
                    frames.append(data.copy()); silence = 0.0
                elif started:
                    frames.append(data.copy()); silence += block
                    if silence >= self.config.voice_silence_sec: break
                elapsed += block
        if not started or not frames: return None
        return np.concatenate(frames, axis=0).reshape(-1)

    def listen_once(self):
        audio = self._record_vad()
        if audio is None: return None
        segs, _ = self.whisper.transcribe(audio, language=self.config.voice_language, vad_filter=False, beam_size=5, condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segs if s.text.strip()).strip()
        if text: print(f"👤 {text}")
        return text or None

    def _extract_cycles(self, text: str) -> int:
        nums = {"один":1,"два":2,"три":3,"четыре":4,"пять":5,"шесть":6,"семь":7,"восемь":8,"девять":9,"десять":10,"одиннадцать":11,"двенадцать":12,"тринадцать":13,"четырнадцать":14,"пятнадцать":15,"шестнадцать":16,"семнадцать":17,"восемнадцать":18,"девятнадцать":19,"двадцать":20}
        m = re.search(r"\b(\d{1,3})\b", text)
        if m: return max(1, min(100, int(m.group(1))))
        for word, val in nums.items():
            if word in text: return val
        return 3

    def command(self, text):
        raw = str(text or "").strip()
        low = self._normalize(raw)
        # All commands require direct address at the beginning. No wake word => IGNORE.
        wake = ("мана", "мама", "омана", "манна", "маманна", "мано")
        matched = None
        for w in wake:
            if low == w or low.startswith(w + " "):
                matched = w; break
        if matched is None:
            return "IGNORE", None
        low = low[len(matched):].strip()
        if not low: return "IGNORE", None

        if low in {"стоп", "остановись", "хватит", "отбой"}:
            return "EXIT", None
        if any(x in low for x in ["останови самоулучшение", "остановить самоулучшение", "прекрати самоулучшение", "останови самосовершенствование"]):
            self.agent.stop_evolution(); return "SAY", "Останавливаю самоулучшение."
        if "статус" in low or "состояние" in low or "что происходит" in low:
            st = self.agent.evolution_status()
            msg = (f"Самоулучшение {'запущено' if st['running'] else 'не запущено'}. "
                   f"Текущий цикл: {st['cycle']}. Лучший результат: {st['best_fitness']:.1f}. "
                   f"Время текущего цикла: {self.agent._fmt_duration(st['current_cycle_elapsed'])}. "
                   f"Общее время: {self.agent._fmt_duration(st['total_elapsed'])}.")
            return "SAY", msg
        start = any(x in low for x in ["начни", "запусти", "стартуй", "проведи"])
        evo = any(x in low for x in ["самоулучш", "самосовершен", "самоусоверш", "оптимизац"])
        if start and evo:
            cycles = self._extract_cycles(low)
            if self.agent.start_evolution_background(cycles):
                return "SAY", f"Запускаю самоулучшение на {cycles} циклов."
            return "SAY", "Самоулучшение уже запущено."
        return "TASK", raw

    def run(self):
        self.speak("Голосовое управление MANA запущено. Скажите стоп для выхода.")
        while self.running:
            try:
                text = self.listen_once()
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"⚠️ Voice input error: {exc}"); continue
            if not text: continue
            kind, payload = self.command(text)
            if kind == "IGNORE": continue
            if kind == "EXIT":
                self.speak("Голосовое управление остановлено."); break
            if kind == "SAY":
                self.speak(str(payload)); continue
            try:
                r = self.agent.solve_task(payload)
                self.speak(r.get("answer", "Не удалось получить ответ."))
            except Exception as exc:
                print(f"⚠️ Voice task error: {exc}")
                self.speak("Не удалось обработать запрос.")
