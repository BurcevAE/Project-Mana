"""
mana.discovery — models MANA writes itself, in a language it may outgrow.

The experiment
--------------
Not "can MANA find a rule" -- the research loop does that over hypothesis
spaces somebody declared. The question here is harder: can a system with
no language model discover both a regularity and the way to look for it?

Two operations, and only two:

    change a program      so it predicts the world better
    change the language   when no program in it predicts well enough

What is fixed, said plainly so nothing is claimed that was handed in
--------------------------------------------------------------------
    the language     get, const, compare, add, sub, if -- and nothing
                     about what any world means
    the edits        syntactic: replace a node, change a constant or an
                     argument, swap an operator, add or remove a condition,
                     combine with a leaf. None knows what it is for.
    the currency     description length in bits: the program plus the
                     errors it leaves (two-part code). A table of
                     exceptions pays for every entry, so memorising loses
                     to structure whenever there is structure to find.
    the protocol     held-out states the search never saw, and a world
                     whose truth lives in `worlds.py` where no learner
                     module may import it.

What is NOT given: which edit to try where, and -- from step 2 on -- what
new variables or primitives the language needs. The claim under test is
that semantic "operations of thinking" (generalise, specialise, abstract)
emerge as edit patterns that pay, rather than being written in.

Step 1, measured 2026-09-13 (scripts/run_discovery.py, 10 seeds, 200
training states, 300 other states held out)
----------------------------------------------------------------------
    W0   the rule itself 10/10; 35.9 bits = the truth's 35.9, a table of
         the training states 2665; held-out 1.000 against a decision
         tree 0.746 (132 nodes) and the table 0.087
    W3   the rule itself 10/10 and not one exception bought: bits equal
         the truth's on every seed (163.0 on average); held-out 0.913 =
         the truth's 0.913 (the noise caps it), tree 0.603, table 0.094
    about 110 000 programs a run, 2.6 s

Re-measured after step 2 corrected the currency -- a wrong guess inside
the outcome alphabet now costs log2(K-1), not log2(K). The verdicts did
not move; W3's bits went from 165.4 to 163.0.

Said plainly: W0 is easy. Its rule is one edit away from a single
variable -- "add a condition" turns z into if(x < 6, z, y) -- so this step
checks the machinery and the resistance to noise, not depth. Depth is
what W2 is for.

Step 2, measured 2026-09-13 (scripts/run_invention.py, 10 seeds, 20
training episodes of 20 steps, 50 new episodes held out)
----------------------------------------------------------------------
    W2   a hidden switch: every a = 1 flips it, the outcome is x while
         it is on and y while it is off. The language changed 10/10,
         to the same thing every time:

             v1      = if(a == v1_prev, 0, 1)      starting at 0
             outcome = if(v1, x, y)

         which is the switch -- "v1 is 1 where a differs from its last
         value" -- written with a comparison where the world was written
         with a sum. 51.5 bits against 889.9 without it; on new episodes
         1.000 against 0.562 for the same search without the variable,
         a decision tree 0.542, a table 0.506; the switch itself recovered
         on 1.000 of new steps. About 22 s a seed.
    W0   language changed 0/10: nothing to explain.
    W3   language changed 0/10: an invented variable cost 148.8 bits
         against 129.2 without it -- noise earns no cause.

Two things this step found about itself, both kept in the code: the
currency of step 1 charged a wrong guess log2(K), which on two outcomes
made "always wrong" cheaper than "right seven times in ten"; and the pair
a hidden variable chooses between has to be picked for what it covers
together, not taken from the best program's errors (see invent.py).

Steps
-----
    1  language, edit search, description length; worlds W0 and W3
    2  inventing a variable; world W2 with hidden state
    3  compressing fragments into primitives; worlds W1 and W4, transfer
    4  statistics of accepted edits: do stable transformation classes form?
"""
from __future__ import annotations

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"
