"""
packaging_deps.py — the import list PyInstaller cannot discover on its own.

`mana` is deliberately excluded from PyInstaller's analysis (build_exe.py
explains why: a frozen copy of the package would win over the shipped
source, because PyInstaller's FrozenImporter sits in sys.meta_path and
beats any sys.path entry -- and a frozen copy cannot be patched). The cost
of that exclusion is that PyInstaller never walks mana's imports, so none
of the standard-library modules it needs get collected either. The first
packaged run failed on exactly that: `ModuleNotFoundError: No module named
'platform'`.

This module states those dependencies explicitly. It is imported by app.py
so the analyser sees it, and it does nothing at runtime beyond the imports
themselves.

Keeping it accurate: every module under mana/ opens with the same block of
standard-library imports (a leftover from the original single-file build).
That block is reproduced here. If a new third-party dependency is added to
mana, it belongs in this file too -- otherwise the failure appears only in
a packaged build, and only on the code path that needs it.
"""
from __future__ import annotations

# --- the standard-library block every mana module carries ---------------
import argparse            # noqa: F401
import ast                 # noqa: F401
import base64              # noqa: F401
import collections         # noqa: F401
import concurrent.futures  # noqa: F401
import copy                # noqa: F401
import csv                 # noqa: F401
import dataclasses         # noqa: F401
import datetime            # noqa: F401
import difflib             # noqa: F401
import enum                # noqa: F401
import functools           # noqa: F401
import gzip                # noqa: F401
import hashlib             # noqa: F401
import importlib           # noqa: F401
import inspect             # noqa: F401
import io                  # noqa: F401
import itertools           # noqa: F401
import json                # noqa: F401
import logging             # noqa: F401
import math                # noqa: F401
import os                  # noqa: F401
import pathlib             # noqa: F401
import pickle              # noqa: F401
import platform            # noqa: F401
import queue               # noqa: F401
import random              # noqa: F401
import re                  # noqa: F401
import shutil              # noqa: F401
import socket              # noqa: F401
import sqlite3             # noqa: F401
import ssl                 # noqa: F401
import statistics          # noqa: F401
import string              # noqa: F401
import subprocess          # noqa: F401
import sys                 # noqa: F401
import tempfile            # noqa: F401
import textwrap            # noqa: F401
import threading           # noqa: F401
import time                # noqa: F401
import traceback           # noqa: F401
import types               # noqa: F401
import typing              # noqa: F401
import unicodedata         # noqa: F401
import urllib.parse        # noqa: F401
import uuid                # noqa: F401
import warnings            # noqa: F401
import weakref             # noqa: F401
import zipfile             # noqa: F401

# --- what the installer is supposed to carry ----------------------------
#
# Declared, not discovered. Every entry below was previously an
# `import x` inside a try/except, which is right at runtime and wrong at
# build time: the except branch that lets MANA degrade gracefully on a
# user's machine also let the BUILD succeed with the package absent, and
# print "Готово" over an application missing web search. The build now
# resolves this list against the environment it is building in and
# refuses to ship a capability it silently dropped.


@dataclasses.dataclass(frozen=True)
class Bundled:
    """A third-party package the packaged application is meant to contain."""
    module: str                 # import name
    package: str                # pip name -- the two differ often enough
    disables: str               # what the user loses if it is not there
    required: bool = False      # True: MANA does not start without it

    def found(self) -> "Found":
        try:
            module = importlib.import_module(self.module)
        except Exception:
            return Found(self, False, "")
        version = getattr(module, "__version__", "") or ""
        if not version:
            # keyring and pywebview carry no __version__ attribute. An
            # empty column would read as "present but unknown build",
            # which is the state this manifest exists to eliminate.
            try:
                import importlib.metadata as _md
                version = _md.version(self.package)
            except Exception:
                version = "?"
        return Found(self, True, str(version))


@dataclasses.dataclass(frozen=True)
class Found:
    spec: Bundled
    present: bool
    version: str


#: Shipped. A missing entry here is a build error, not a footnote.
BUNDLED = (
    Bundled("numpy", "numpy", "MANA не запускается", required=True),
    Bundled("requests", "requests", "все LLM-провайдеры недоступны"),
    Bundled("psutil", "psutil", "RAM/CPU определяются неточно, машина может быть отнесена не к тому классу"),
    Bundled("ddgs", "ddgs", "веб-поиск недоступен навсегда"),
    Bundled("sklearn", "scikit-learn", "семантический поиск по памяти и обученный роутер отключаются"),
    Bundled("pymupdf", "pymupdf", "PDF не читаются"),
    Bundled("keyring", "keyring", "ключи API нельзя сохранить между запусками", required=True),
    Bundled("webview", "pywebview", "окно не открывается", required=True),
)

#: NOT shipped, on purpose, with the reason. Written down because the
#: difference between "we decided against it" and "we forgot" is invisible
#: in a list of things that are absent.
DELIBERATELY_ABSENT = (
    ("torch + sentence-transformers", "2-3 ГБ ради семантического поиска, "
     "у которого есть работающий TF-IDF-запасной путь"),
    ("faster-whisper + sounddevice + pyttsx3", "голос: требует PortAudio и "
     "загрузки модели, ставится отдельно"),
)


def resolve() -> "list[Found]":
    """Which of the declared packages this environment actually has."""
    return [spec.found() for spec in BUNDLED]


def missing(found: "typing.Iterable[Found]" = ()) -> "list[Found]":
    return [f for f in (found or resolve()) if not f.present]


def report(found: "typing.Iterable[Found]" = ()) -> str:
    lines = []
    for f in (found or resolve()):
        mark = "  есть " if f.present else "  НЕТ  "
        tail = f.version if f.present else f.spec.disables
        lines.append(f"{mark} {f.spec.package:16s} {tail}")
    return "\n".join(lines)


# --- the imports themselves ---------------------------------------------
#
# PyInstaller's analyser walks this file's AST, so these statements are
# what actually pulls each package into the bundle. `mana/` is excluded
# from that analysis (build_exe.py explains why), which means an import
# that appears only inside mana/optional_deps.py is invisible to the
# build -- sklearn was exactly that, and the packaged app had no semantic
# search for a reason nobody could see from the outside.
import dataclasses          # noqa: F401,E402
import importlib            # noqa: F401,E402
import typing               # noqa: F401,E402

import numpy                # noqa: F401,E402

# Each one named individually, never through a loop over strings: the
# analyser reads source, not runtime behaviour, and a module imported as
# importlib.import_module(variable) is a module PyInstaller never sees.
try:
    import requests         # noqa: F401,E402
except Exception:
    pass
try:
    import psutil           # noqa: F401,E402
except Exception:
    pass
try:
    import ddgs             # noqa: F401,E402
except Exception:
    pass
try:
    import sklearn          # noqa: F401,E402
    from sklearn.feature_extraction.text import TfidfVectorizer   # noqa: F401,E402
    from sklearn.metrics.pairwise import cosine_similarity        # noqa: F401,E402
    from sklearn.linear_model import LogisticRegression           # noqa: F401,E402
except Exception:
    pass
try:
    import pymupdf          # noqa: F401,E402
except Exception:
    pass
