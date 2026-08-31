#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin entry point: `python mana_run.py <args>` == old `python MANA_5_4.py <args>`."""
from __future__ import annotations

from mana.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
