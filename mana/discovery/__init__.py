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

Its boundary, measured on three worlds it was not built on (invent.py has
the numbers): with 10% noise on W2 the switch was found on 3 of 10 seeds;
a three-state counter 1 of 10; a hidden number added to x 3 of 10, and
only partly. What it does is find a binary hidden selector seen cleanly.

Two things this step found about itself, both kept in the code: the
currency of step 1 charged a wrong guess log2(K), which on two outcomes
made "always wrong" cheaper than "right seven times in ten"; and the pair
a hidden variable chooses between has to be picked for what it covers
together, not taken from the best program's errors (see invent.py).

Step 3, measured 2026-09-13 (scripts/run_library.py, 5 seeds; library.py
has the table)
----------------------------------------------------------------------
    The language grew a word on four seeds of five: the distance between
    two values, if(#0 < #1, #1 - #0, #0 - #1), on two of them as a general
    function, on two with one variable baked in. With it, questions that
    use it were found far sooner -- |x - y| at the 15th program instead of
    the 87 712th, |y - z| + x at the 32 161st instead of the 286 046th --
    and a question over other names was solved where it had not been. But
    the total did not move: fourteen of twenty-five questions solved with
    the word and without, because the word misled the search on a question
    whose shortest answer does not use it. Bits reward what repeats in what
    was seen; they do not pay for generality, and a larger vocabulary is
    not free for the questions that do not need it.

Step 4, measured 2026-09-13 (scripts/run_search_reliability.py, 7
questions x 10 seeds; the search measured, not changed)
----------------------------------------------------------------------
    A solution exists in all 70 runs: every world's rule, written in the
    same language, reproduces every state and fits within the size the
    search explores (the certificate; the bottom-up ceiling adds that no
    exact program of 7 nodes or fewer exists for T1..W4, so their smallest
    lies between 8 and the rule's own size). Every miss is the search's.

        question   found   +strong   budget  local end   held-out
        W0         10/10      -        0        0        1.000 +- 0.000
        W3         10/10      -        0        0        1.000 +- 0.000
        T1          8/10     +2        0        2        0.890 +- 0.221
        T2          4/10     +6        6        0        0.647 +- 0.292
        T3          4/10     +6        6        0        0.662 +- 0.289
        T4          4/10     +2        0        6        0.838 +- 0.133
        W4          4/10     +6        6        0        0.647 +- 0.292

    P(found | a solution exists) = 44/70 = 63% as the search stands; 66/70
    = 94% when every miss is retried with a beam of 16 and a budget of two
    million (about five times the evaluations). Of the 26 misses, 18 ran
    out of budget with their neighbourhood unexhausted -- all rescued -- and
    8 stopped at a local end, 4 of them rescued. The four left are all T4,
    whose rule is 15 nodes: exactly the size limit. Even the successes on
    T2 and T3 mostly stopped at the budget, the answer first seen around
    the 300 000th program of 400 000. W4 is T2 with other names and failed
    on the same seeds: the search does not depend on what things are called.

    What this bounds: everything built above the search. The library's
    14 of 25 and the invention's boundary were measured through a search
    that finds an existing solution six times in ten.

Step 5.0, measured 2026-09-14 -- the zero law (scripts/run_budget_curve.py,
the same 7 questions x 10 seeds at every budget; each budget read off one
profiled run, which a test shows is exactly what a separate run returns)
----------------------------------------------------------------------
    P(found | a solution exists) against programs evaluated:

        budget     10k  25k  50k  100k  200k  400k  800k  1.2M   2M
        beam 4     29%  29%  29%   39%   44%   63%   63%   63%  63%
        beam 16    29%  29%  29%   29%   29%   43%   51%   84%  94%

    (29% is W0 and W3; the family needs 100k and more.)

    The beam of 4 -- the search as it stands -- is flat from 400k: past
    800k every one of its 26 misses is a local end. More computation buys
    it nothing; its limit is structural. The beam of 16 buys completeness
    with computation: worse below the crossing between 800k and 1.2M (43%
    against 63% at 400k), 94% at 2M, at a median cost of 645 380 programs
    to the answer against 67 387, and a held-out error of 0.012 where the
    narrow beam's plateau stays at 0.187. T4 is where both stop: 4/10 and
    6/10, every remaining miss a local end.

    A correction to step 4: the 18 misses it called budget failures were
    cut by the budget at 400k, but none would have been found by the same
    narrow search with more -- by 800k all had stalled. What rescued them
    was the width of the beam, not the budget.

    The zero law for every later change to the search is the upper envelope
    of the known configurations at equal programs evaluated:
        10k 29, 25k 29, 50k 29, 100k 39, 200k 44, 400k 63, 800k 63,
        1.2M 84, 2M 94 (%).
    A mechanism earns its place only above it.

Cells, measured 2026-09-14 (scripts/run_cells.py; cells.py has the table)
----------------------------------------------------------------------
    The world understood one region at a time: a model per cell, a cell
    split where its model breaks and only when the split shortens the
    description, the global hypothesis the tree stitched with if. The
    search inside every cell is the search as it stands.

    Adaptive cells (B) with a local budget of 100k reach 70% -- past the
    narrow beam's plateau of 63% -- and where the first cut is right they
    find the world's own pieces unasked: if(z < y, (x + y) - z,
    (x + z) - y) on half of T2's seeds. But under the zero law they are
    above the envelope at one budget only (69 against 63 at 800k) and
    below the wide beam from 1.2M on. A grid laid down in advance (A) is
    worse than no cells: 29%, 0%, 0% for 8, 27 and 100 cells.

    Why: a cell makes the rule simpler, not the search for it cheaper. On
    the points of T2's own cell the five-node (y + x) - z was first seen at
    the 378 798th program -- an exact-match error gives x + y no credit
    over x, and 5 814 of a leaf's 5 889 neighbours are conditions. The
    limit is inside every cell, in the edits and the currency, not in the
    size of the region.

Step 6, the breakthrough experiment, predicted before the run (2026-09-14;
docs/АУДИТ_ПОРОЖДЕНИЯ.md has the question and the criterion)
----------------------------------------------------------------------
    Everything grown so far was about the world. Here the object compressed
    is how answers were reached: the search's derivations -- each program
    the one the next was made from, one edit apart -- written in the
    language the search already had, its edits. Runs of steps that several
    derivations share become macro-edits, moves of the search, kept only
    when they shorten the derivations in the currency a search pays in
    (log2 of each neighbourhood). derive.py.

    Family A, learning: T1, T2, T3, solved with a wide beam. Family B,
    transfer: T4 -- the distance inside a condition, another surface, and
    the search's structural limit (4/10 narrow, 6/10 wide). W4 is not B:
    it is A renamed. Arms on B at equal programs evaluated: the search as it
    stands; with the moves learnt from A; with the words learnt from the
    same A answers (step 3, compressing results); with moves learnt from
    W0 and W3 (control).

    Predicted: moves are learnt from A -- at least one pays -- and some are
    noise from the search's detours through conditions. In A on unseen
    seeds they help, as the words did. On T4 they do not: no more than one
    seed of ten above the search as it stands, and few T4 answers reached
    through a move -- every move widens every neighbourhood by the square
    of the vocabulary, and T4's shortest answer says |x - y| < 3 with two
    comparisons, which no distance-making move reaches. If so, compressing
    the way of solving carries no further than compressing the answers,
    and the boundary lies between the program of the world and the
    program of solving.

Step 6, measured 2026-09-14 (scripts/run_derive.py; derive.py has the rest)
----------------------------------------------------------------------
    Learning: the wide beam solved all 30 of T1..T3, derivations of 4 steps
    (median). Of 54 candidate moves two were kept -- one move, written both
    ways round: n -> |n - #0|, "put this node at a distance from a leaf" --
    and the derivations went from 1 655 to 843 bits. No noise from detours
    was kept. The words compressed from the same answers: the distance,
    and three whole answers with their variables baked in (f1(x, z) + y).

    solved at budget         10k  25k  50k  100k  200k  400k  800k   used
    A, T1..T3 unseen seeds (of 30)
        as it stands           0    0    0     6     8    23    26
        with the move          9    9   12    22    26    26    27   26/27
        with the words        30   30   30    30    30    30    30   30/30
        random moves           0    0    0     6     8    23    26
    B, T4 (of 10)
        as it stands           0    0    0     0     4     4     4
        with the move          0    0    0     1     4     4     4    0/4
        with the words         0    0    0     0     0     4     4    4/4
        random moves           0    0    0     0     4     4     4

    (used: answers whose derivation took a step by the move, or which
    call a word.)

    The chain the user set as the criterion -- tasks, programs of solving,
    their repeated parts, compression, a new primitive, transfer -- held to
    its last link and broke there. MANA grew a move of its own search from
    how its answers were reached, nobody wrote it, it pays in the family:
    on unseen data the answer came at a median 73 325 programs against
    297 715, and 26 of 27 answers were reached through it; random moves of
    the same size change nothing, so it is this move, not a wider
    neighbourhood. That is level 2 of the audit's scale, for a way of
    solving -- the first thing MANA has grown that is not about the world.
    It did not carry to T4: one seed earlier, none more, and not one T4
    answer took a step by it. The words did no better there and in the
    family won by memorising the answers.

    Why, read from T4's own numbers: the rule with a distance is 66 bits,
    far shorter than anything found, so it is expressible and would win.
    The move reaches it in one step from if(x < 3, z, y) -- and that
    program costs 470 to 544 bits, more than the bare leaf y (387 to 498).
    The currency never lets the beam hold the place where the move
    applies; on the failed seeds the search ends in round 1 at
    if(x == y, z, y), where no move leads anywhere. The same valley
    without a gradient that kept (y + x) - z out of reach in the cells
    experiment keeps a learnt move from being used. The boundary is not
    between compressing the world and compressing the way of solving --
    both compressed, and both carried in the family and not outside it. It
    is that a primitive of a method is only as good as the states the
    search can stand in to use it, and what states it can stand in is set
    by the currency, which nothing grown here changes.

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
