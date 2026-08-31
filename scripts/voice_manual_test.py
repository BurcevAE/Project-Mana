#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/voice_manual_test.py — MANUAL, interactive voice interface check.

Voice was never tested even once during this project's development (no
microphone/audio device in the sandbox it was built in) -- everything
about its real-world behavior is currently unverified. This can't be
automated into a pass/fail like the other diagnostics: it needs a human
to actually speak and listen. Run this, do what it asks, and paste your
observations (not just "it worked"/"it didn't") back.

Usage:
    python scripts/voice_manual_test.py
    python scripts/voice_manual_test.py --list-devices   # if audio setup looks wrong
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--voice-model", default="small")
    parser.add_argument("--voice-language", default="ru")
    parser.add_argument("--voice-tts-backend", choices=["auto", "silero", "pyttsx3"], default="auto")
    parser.add_argument("--voice-speaker", default="xenia")
    args = parser.parse_args()

    from mana.optional_deps import HAS_SOUNDDEVICE, HAS_WHISPER, HAS_TTS, sd

    print("=" * 70)
    print("Dependency check (all should be True for a real test):")
    print(f"  sounddevice available: {HAS_SOUNDDEVICE}")
    print(f"  faster-whisper available: {HAS_WHISPER}")
    print(f"  pyttsx3 available: {HAS_TTS}")
    print("=" * 70)
    if not HAS_SOUNDDEVICE:
        print("\nsounddevice not available -- install it and the system PortAudio "
              "library, then re-run. Nothing else here can work without it.")
        return 1

    if args.list_devices:
        print(sd.query_devices())
        return 0

    from mana import ManaAgent, Config
    from mana.voice import VoiceInterface

    cfg = Config(enable_llm=True, enable_web=True, voice_enabled=True,
                 voice_whisper_model=args.voice_model, voice_language=args.voice_language,
                 voice_tts_backend=args.voice_tts_backend, voice_silero_speaker=args.voice_speaker)
    agent = ManaAgent(cfg)
    voice = VoiceInterface(agent, cfg)

    print("""
MANUAL TEST STEPS -- please actually do each one and note what happened:

1. TTS check: MANA will now try to speak a short test phrase.
   -> Did you hear anything? Was it understandable? Which backend did it
      actually use (it will print that below)?
""")
    try:
        voice.speak("Проверка синтеза речи. Раз, два, три.")
        print(">>> Called voice.speak() -- note above what backend/errors printed.")
    except Exception as exc:
        print(f">>> speak() raised: {type(exc).__name__}: {exc}")

    print("""
2. STT check: MANA will now listen once for a few seconds.
   -> Say a short phrase clearly (e.g. "проверка распознавания речи").
   -> Compare what it prints below to what you actually said.
""")
    input("Press Enter when ready to speak, then talk immediately after...")
    try:
        heard = voice.listen_once()
        print(f">>> Recognized text: {heard!r}")
    except Exception as exc:
        print(f">>> listen_once() raised: {type(exc).__name__}: {exc}")

    print("""
3. Full loop check (optional, longer): run the wake-word loop for real
   interaction. This will run until you interrupt it (Ctrl+C).
""")
    if input("Run the full wake-word loop now? [y/N] ").strip().lower() == "y":
        try:
            voice.run()
        except KeyboardInterrupt:
            print("\n>>> Stopped.")
        except Exception as exc:
            print(f">>> run() raised: {type(exc).__name__}: {exc}")

    print("""
=== Please report back ===
- Which of the 3 steps worked, which didn't, and the exact error text if any.
- For TTS: which backend actually spoke (silero/pyttsx3), and audio quality/naturalness.
- For STT: recognized text vs. what you actually said (accuracy).
- Your OS and audio setup (e.g. "Ubuntu 22.04, USB headset").
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
