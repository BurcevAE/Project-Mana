#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin entry point: `python mana_run.py <args>` == old `python MANA_5_4.py <args>`."""
from __future__ import annotations

from mana.cli import main
from mana.console import speak_utf8

if __name__ == "__main__":
    # Same reason as in app.py: the messages are Russian and the stream
    # encoding is whatever the locale says unless somebody says otherwise.
    speak_utf8()
    raise SystemExit(main())
