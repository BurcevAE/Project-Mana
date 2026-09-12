"""
mana.research — asking the world on purpose, and saying how far the answer
goes.

The research contract (docs/RESEARCH_CONTRACT.md), the part outside the
immutable core. What an answer may claim -- verified for the task,
conditional, unexplained -- is decided in `core/standing.py`; everything
here is policy that may change: which probe next, when a question is worth
more looking, how an explanation is revised.

    contract.py        the shapes: explanation, probe, world, stakes, question
    loop.py            the loop: tell apart, check, close
    adapters/device.py the synthetic device as a world

Carried over from `cognition/inquiry.py` 1.14 -- only what its experiments
kept. That module stays as the record of the rest until step 5.
"""

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"
