"""
mana.optional_deps
===================
Single source of truth for every optional third-party dependency MANA can use.

Every other module imports capability flags (HAS_X) and the objects
themselves from here instead of repeating try/except ImportError blocks.
This is the only file that is allowed to contain a bare `try: import ...`.
Adding support for a new optional library means editing this file only.
"""
from __future__ import annotations

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"

try:
    # PyMuPDF renamed its module; importing the old `fitz` alias prints a
    # deprecation warning on every single MANA start, including runs that
    # never touch a PDF. The alias still works, so this is cosmetic --
    # but a warning nobody can act on trains people to ignore warnings.
    import pymupdf as fitz  # optional, for PDF knowledge acquisition
    HAS_FITZ = True
except Exception:
    try:
        import fitz  # older PyMuPDF, before the rename
        HAS_FITZ = True
    except Exception:
        fitz = None
        HAS_FITZ = False

try:
    import requests
    HAS_REQUESTS = True
except Exception:
    requests = None
    HAS_REQUESTS = False

try:
    # NOTE (bugfix vs MANA_5_4.py): sounddevice raises OSError, not
    # ImportError, when the native PortAudio library is missing from the
    # host system. The original file caught only ImportError here, so on
    # any machine without PortAudio installed the whole script crashed at
    # import time instead of degrading gracefully. Catching Exception is
    # the correct, already-used pattern (see `fitz` below) for a truly
    # optional dependency.
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    sd = None
    HAS_SOUNDDEVICE = False

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except Exception:
    WhisperModel = None
    HAS_WHISPER = False

try:
    import pyttsx3
    HAS_TTS = True
except Exception:
    pyttsx3 = None
    HAS_TTS = False

try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HAS_TORCH = True
except Exception:
    torch = None
    DEVICE = None
    HAS_TORCH = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None
    LogisticRegression = None
    HAS_SKLEARN = False

try:
    from ddgs import DDGS
    WEB_BACKEND = "ddgs"
    HAS_WEB = True
except Exception:
    try:
        from duckduckgo_search import DDGS
        WEB_BACKEND = "duckduckgo_search"
        HAS_WEB = True
    except Exception:
        DDGS = None
        WEB_BACKEND = None
        HAS_WEB = False

try:
    # Used by mana.hardware for accurate RAM/CPU detection when adapting
    # Config to the host machine. Optional: mana.hardware falls back to
    # /proc/meminfo (Linux) or Config's untouched defaults if unavailable.
    import psutil
    HAS_PSUTIL = True
except Exception:
    psutil = None
    HAS_PSUTIL = False

__all__ = [
    "fitz", "HAS_FITZ",
    "requests", "HAS_REQUESTS",
    "sd", "HAS_SOUNDDEVICE",
    "WhisperModel", "HAS_WHISPER",
    "pyttsx3", "HAS_TTS",
    "torch", "DEVICE", "HAS_TORCH",
    "SentenceTransformer", "HAS_SENTENCE_TRANSFORMERS",
    "TfidfVectorizer", "cosine_similarity", "LogisticRegression", "HAS_SKLEARN",
    "DDGS", "WEB_BACKEND", "HAS_WEB",
    "psutil", "HAS_PSUTIL",
]
