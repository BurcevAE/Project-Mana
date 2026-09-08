"""
mana.world — a frame for what it means to be somewhere, not a description
of anywhere.

Why this is not a knowledge base
---------------------------------
Loading facts about Windows, 1C and the internet would produce an
encyclopedia: a great deal of text about the world, and no way to tell a
claim that has been checked from one that was read. The thing missing
underneath is smaller. An actor needs to be able to hold, in a form it can
be wrong about:

    что вообще существует        entities, in layers
    что о них верно и когда      state, with a time and a source
    что можно сделать            actions, with conditions and effects
    что из этого доступно мне    capabilities, and limits
    откуда я это знаю            observed / known / believed / unknown

`schema.py` is that frame. It contains no facts about any world.

Why there is a toy universe in here
------------------------------------
Because "MANA has a model of the world" is otherwise a sentence nobody
can check, and this project has learned what those are worth. `universe.py`
is a small world with layers whose true rules are written down, so that
what `explore.py` reconstructs from acting in it can be compared with what
is actually there and scored. Same shape as the chess capability test:
a domain where the right answer is known independently, measured honestly,
reported whichever way it comes out.

What this is not, yet
----------------------
It is not connected to the running agent. Nothing in `mana/agent_parts/`
consults a world model, and no answer changes because of one. That is
deliberate: the question being asked here is whether the frame can be
filled from experience at all. Wiring an unmeasured model into the live
path would be this project's oldest mistake in a new place.
"""
from .schema import (BELIEVED, CAN, CANNOT, Capability, Entity, KNOWN,
                     Observation, OBSERVED, Rule, StateFact, UNKNOWN,
                     WorldModel, situation_of)

__all__ = ["BELIEVED", "CAN", "CANNOT", "Capability", "Entity", "KNOWN",
           "Observation", "OBSERVED", "Rule", "StateFact", "UNKNOWN",
           "WorldModel", "situation_of"]
