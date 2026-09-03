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

# --- required third-party (requirements.txt) ----------------------------
import numpy               # noqa: F401

try:
    import requests        # noqa: F401
except Exception:          # optional at runtime, see mana/optional_deps.py
    pass

try:
    import psutil          # noqa: F401
except Exception:
    pass

try:
    import ddgs            # noqa: F401
except Exception:
    pass
