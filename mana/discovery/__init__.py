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

Step 6 control, predicted before the run (2026-09-14; scripts/run_frontier.py)
----------------------------------------------------------------------
    Before blaming the measure: H1, the answer is reachable and the beam,
    ranked by bits, drops the state it is reached from; H2, T4 needs
    something the language or the edits lack. The measure is left alone;
    the search is only allowed to keep one given program in its beam,
    if(x < 3, z, y) -- chosen because the answer is known, a diagnostic.
    Predicted: kept + move finds the 66-bit rule on 10 of 10, in the first
    rounds, through the move; kept without the move rarely does -- from the
    kept program the rule is three plain edits away, through states no
    better ranked; the kept program stands thousands of places below the
    beam in the first round. If so, H1, strictly: the path exists and the
    frontier throws it away. If kept + move fails, H2, and the measure was
    blamed too soon.

Step 6 control, measured 2026-09-14 (T4, seeds 0..9, beam 4, 800k)
----------------------------------------------------------------------
                       solved   programs to the answer   through the move
        as it stands    4/10          146 285                  0/4
        move            4/10          136 181                  0/4
        kept           10/10          104 111                  0/10
        kept + move    10/10           17 877                 10/10

    H1, and more strongly than predicted. With the measure unchanged,
    keeping one state the ranking drops -- 234th to 987th of 8 865 in the
    first round, where the beam keeps 4 -- solves T4 on every seed, and
    does so without the move: from the kept state three plain edits reach
    the 66-bit rule. The move makes it six times cheaper (one step instead
    of three). So T4's "structural limit" of step 4 and 5.0 was never the
    language or the edits: the rule, a way to it, and a state it is
    reached from all exist, and the frontier -- a beam ranked by the
    description length of each state on its own -- throws the state away.
    Predicted wrongly: that plain edits would rarely finish from the kept
    state, and that it stood thousands of places down rather than hundreds.
    What decides whether an answer is reached is which states are kept,
    and that is decided by a measure MANA cannot change.

Step 7, predicted before the run (2026-09-14; frontier.py,
scripts/run_frontier_score.py)
----------------------------------------------------------------------
    Which states the beam keeps, learnt from the search's own successful
    traces instead of fixed. Bounds fixed with the user: the score decides
    only what the beam keeps -- the answer is still the shortest
    description seen, judged by the same gates; T4 takes no part in the
    learning, the features or any setting; the space of scores is given --
    weighted sums of a state's program bits, error bits, size and wrong
    points, the measure as it stands being (1, 1, 0, 0) -- so this is a
    learnt measure in a space we wrote, not yet one MANA invents; the
    teacher is which of its own states led on, at every step of a
    derivation, weighted by how close to the answer it stood; two controls
    of the same size, the weights permuted and random; recall by budget on
    all seven questions against the search as it stands.

    Predicted: the learnt score orders the history's decisions better than
    bits, and on T1..T3's unseen seeds brings answers sooner. On T4 it
    does not reach the ceiling the kept state showed -- no more than 5 of
    10 at 800k. From numbers already measured: the one state known to lead
    to T4's rule, if(x < 3, z, y), is worse than the bare leaf y in every
    quantity a score may read -- more error bits, more program bits, more
    nodes -- so no weighting of them keeps it ahead of the leaf, and what a
    score gains on T4 by preferring larger or worse-fitting states it pays
    for on the others. The controls no better than the search as it
    stands. If so, the space of measures is too poor for this frontier,
    and the next question is the one the user set: can MANA see that the
    quantities it judges a state by are not enough.

Step 7, measured 2026-09-14
----------------------------------------------------------------------
    Teacher: 30 solved derivations of T1..T3, 110 decisions of the
    frontier. Where the measure as it stands put the state that led on,
    among the programs made from the same parent: first of about 17 000,
    at the median; outside the first four in 6% of decisions. It ordered
    the history's decisions right 100% of the time -- and so did the
    learnt weights, (0.10, 0.01, 0.11, 0.02) on program bits, error bits,
    size and wrong points, and so, at 99.9%, did the same weights moved
    onto other features.

    solved at budget     10k  25k  50k  100k  200k  400k  800k  (of 70)
        as it stands      20   20   20    26    32    51    54
        learnt            20   20   20    20    20    20    20
        permuted          20   20   20    20    20    20    20
        random             0    0    0     0     0     0     0
    T4: 4 of 10 as it stands, 0 with every learnt or control score. The 20
    the learnt score keeps are W0 and W3, answered by a leaf or one edit.

    A clean negative, and the user's warning come true: the score learnt
    the style of the derivations, not the promise of a state. The steps
    that led on were small edits, most of their siblings wrapped the
    parent in a condition, so "small and short" ordered the history
    perfectly -- with almost no weight left on errors, which in a search
    means never fitting better. The learnt weights did no better than
    their own permutation. Predicted rightly: T4 not reached; wrongly: that
    the learnt score would order the history better than bits (there was
    nothing to improve) and help T1..T3 (it lost all 26).

    Why the teacher had nothing to teach: a successful trace is the record
    of the frontier's successes. The paths it threw away are exactly the
    ones that never reach an answer, so they are never in the history it
    learns from -- the 6% of decisions it got wrong on the way to answers
    it still found are all it ever sees of its mistakes. Learning the
    frontier from one's own successes is blind, by construction, to what
    the frontier loses. What would carry the lesson is a comparison of
    runs: the same question failed by a cheap search and solved by a dear
    one, and the states of the dear one's path that the cheap frontier
    dropped -- regret, bought with the search's own compute, no answer
    shown.

Step 7b, the last control of this branch, predicted before the run
(2026-09-14; frontier.py, scripts/run_regret.py)
----------------------------------------------------------------------
    The question, fixed with the user: can MANA distil a dearer search of
    its own into a cheaper frontier, from a comparison of the two, within
    the space of scores already given? Not whether it can grow a measure.
    Regret: a state on a dear search's way to its answer (beam 16, 2M)
    that the cheap search (beam 4, 400k) made and never kept, against
    what the cheap frontier kept that round. A correction to the bits,
    held near them by a penalty fixed in advance. Controls: its weights
    permuted, random weights of its norm; and the dear policy itself.

    Fixed before the run: T4 cannot be won in this space. The state its
    answer is reached from is worse than a bare leaf in all four
    quantities; no weighting with sensible signs prefers it, and signs
    that prefer larger or worse-fitting states lose everywhere else.

    Predicted: regret exists on the T1..T3 seeds the cheap search fails;
    most of it is dominated -- the states the cheap frontier kept no worse
    in every quantity -- so the correction can move little; T1..T3 gain
    at most a few seeds of 30 at equal budget, W0, W3, W4 unchanged, T4 at
    4 of 10, the controls no better, and the wide beam above all of them
    from 800k where it can afford itself. Whatever comes, the branch stops
    here: a gain says MANA can distil its dearer search into a cheaper
    one; none, with the regret dominated, says the four quantities are
    not enough -- and whether MANA can see that, and make a new way of
    judging a state, is the next question, and a new one.

Step 7b, measured 2026-09-14
----------------------------------------------------------------------
    The cheap search solved 16 of T1..T3's 30, the dear one all 30. Of the
    140 states on the dear derivations the cheap frontier kept 57, made
    and dropped 31, and never made 52. The 31 dropped are the regret --
    and not one was dominated: for every one, the states kept instead were
    worse in some quantity. The features separate the regret; the
    predicted ceiling is not what stopped this.

    ordered the regret pairs right    measure as it stands  0%
                                      learnt correction    44%
                                      its weights permuted 78%
                                      random weights       41%

    solved at budget (of 70)   10k  25k  50k  100k  200k  400k  800k
        as it stands            20   20   20    26    32    51    54
        learnt from regret      20   20   26    28    28    28    28
        permuted                20   20   20    20    20    20    20
        random                   0    0    0     0     0     0     0
        the dear policy (16)    20   20   20    20    20    30    36
    T1..T3 on unseen seeds: 6 of 30 at 50k where the search as it stands
    has none -- the one gain -- then 8 at 800k against its 26. T4: 0.

    Predicted rightly: regret exists where the cheap search fails; the
    controls no better. Wrongly: that the regret would be dominated (none
    was), that W0, W3, W4 and T4 would hold (24 fell to 20, 4 to 0), and
    that the dear policy would lead at 800k (it needs 2M).

    What this control answers -- the user's third outcome: the new measure
    does not win even where the features separate the regret, so the
    trouble is not only the teacher. How well a score orders the
    frontier's known mistakes says nothing about how a search does with
    it: 0% gives 54, 44% gives 28, 78% gives 20. A frontier is not a
    ranking of states one at a time: keeping one state is dropping others
    in the same round and every round after, and a third of the dear
    path was never made by the cheap search at all -- its beam was
    elsewhere, a matter of where the frontier went, not of how it ranked
    what it saw. A score of a state's own quantities, learnt from pairs,
    is the wrong kind of object for the decision it is asked to make.

    This branch stops here, as agreed. MANA can grow a move of its own
    search (step 6), and the control showed the frontier is where answers
    are lost; a frontier learnt inside a given space of scores -- from
    successes or from regret -- does not recover them. The question left
    is not another feature: it is whether MANA can see that the way it
    represents a state's promise is the wrong kind of thing, and make
    another.

Stage R1, predicted before the run (2026-09-14; policy.py,
scripts/run_policy.py)
----------------------------------------------------------------------
    Not an improvement. The question: can the search itself be an object
    MANA reads, copies, compares, edits and runs, without touching the
    immutable core? A SearchPolicy is data -- eight rewrite rules for how
    candidates are made, a selection (order by a weighted sum of a state's
    quantities, keep the first k), resources -- run by a fixed interpreter
    that also keeps the verdict: the answer is the shortest description
    seen, whatever was kept.

    Predicted:
      equivalence  the current search, written as the policy CURRENT, gives
                   what search.search gives on 70 runs of 70 -- program,
                   bits, evaluations, rounds, history, where the answer was
                   first seen, why it stopped, the whole budget curve. The
                   interpreter slower, by a factor under 3
      keep 2       about half the programs per round; W0 and W3 as before;
                   the family solved no more often than with 4
      reversed     the same neighbours of every node, only in another
                   order, and the beam sorts the whole round: identical
                   runs wherever the search stopped by itself, before its
                   budget; differences only in runs the budget cut, where
                   the order decides which programs of the last round got
                   evaluated
      no combine   no sums or differences can be made: T1..T4 and W4 not
                   solved on any seed; W0 and W3, whose rule is one
                   condition, as before
    If all four hold, the policy is not only a faithful copy of the search
    but the thing that controls it.

Stage R1, measured 2026-09-14 (seven questions x seeds 0..9, 400k)
----------------------------------------------------------------------
    equivalence  70 of 70 identical: program, bits, evaluations, rounds,
                 history, where the answer was first seen, why it stopped,
                 and every point of the budget curve. The interpreter 1.1
                 times slower than search.py
                 solved of 10     W0  W3  T1  T2  T3  T4  W4
      as it stands                10  10   8   4   4   4   4
      keep 2                      10  10   2   3   1   3   3
      reversed rules              10  10   8   4   4   4   4
      no combine                  10  10   0   0   0   0   0
      keep 2       0.39 of the programs per round; the family solved less
                   often, as a narrower beam does (step 5.0)
      reversed     the same answer on 70 of 70; wherever the search stopped
                   by itself (40 runs) the same programs, evaluations and
                   rounds; where the budget cut it (30), the same answer
                   still. What moved, and was not predicted: where in a
                   round the answer was first evaluated (found_at, 4 of 70
                   unchanged) -- the order decides when inside a round, not
                   what a round holds
      no combine   no sum or difference made, the family and W4 never
                   solved, W0 and W3 as before

    The four predictions held, and one detail went further than predicted
    (the budget-cut runs kept their answers too). So the search of
    discovery is an object: written as data, read, copied, compared and
    edited, run by a fixed interpreter that keeps the verdict, and every
    edit of it changes the search as the edit says. The wall is where
    policy.py draws it: selection only "order and keep k", a state only
    one program, rounds only full expansions, rules only one-node rewrites
    -- and the loop itself outside the object.

Stage R2, predicted before the run (2026-09-14; reflect.py,
scripts/run_reflect.py)
----------------------------------------------------------------------
    MANA builds a change of its own search policy, from its own
    experience, and core.gates decides. No list of changes is given. One
    principle: the better policy is the one in which the derivations of
    MANA's own answers (dear runs, T1..T3 seeds 0..9) are written most
    briefly, with the definitions of what it added. From it come only two
    kinds of change -- a rule added from repeated structure (step 6's
    move, now part of the policy), a rule dropped the experience never
    needed -- and it cannot touch the selection or the resources: the
    length of the experience does not depend on them. That is R2's limit.
    Gates: 30 dev pairs (seeds 10..19), hidden seeds 20..29, W0, W3, W4 as
    counterexamples, T4 as a second, transfer claim; budget 400k, the
    policy's own, fixed in advance.

    Predicted for A, the search as it stands: the change adds the distance
    move and drops every rule the family's derivations do not use -- they
    are written with "add a condition" and "combine with a leaf" alone --
    so replace, the swaps and "drop or turn a condition" go. The dropped
    rules make almost no difference to the cost of a step (a condition
    wrapped is 5 800 programs a node; they are a few dozen), so the new
    policy is the old one with the move: more answers sooner, as in step 6
    (at 400k, 23 of 30 against 26 there). Verdict: not accepted -- three
    or four discordant pairs do not pass McNemar at 30, so significance
    fails, whatever the direction; the curves gain at 50k-100k; T4 no
    better, possibly worse without "replace by a leaf"; W0, W3 untouched.
    For B: the idle "double" dropped, the same move added, and B' the same
    policy as A' -- both are what the experience needs, so the changes
    differ where A and B differed (A' dropped "swap <", B' dropped
    "double") and end in the same place.

Stage R2, measured 2026-09-14
----------------------------------------------------------------------
    The change MANA built for A, from 30 derivations of its own dear runs
    (1 641 -> 539 bits to write them):
        added    n -> |n - #leaf|, both ways round        (-438, -376 bits)
        dropped  "add a condition"                         (-275 bits)
        dropped  "replace by a leaf", "swap add"           (-14, -0.2)
    Not predicted: with the grown move, the family's derivations no longer
    need "add a condition" -- the move carries its own condition -- and
    that rule makes 5 800 programs a node, so dropping it shortened the
    experience more than anything but the move itself.

    solved, of 30 (T4 of 10)   10k  25k  50k  100k  200k  400k
        unseen seeds, old         0    0    0     6     8    23
        unseen seeds, new        28   28   28    28    28    28
        hidden seeds, old         0    0    0     7     9    17
        hidden seeds, new        27   27   27    27    27    27
        W0, W3, W4, old          20   20   20    20    20    24
        W0, W3, W4, new           7    7    7     7     7     7
        T4, old / new           0/0  0/0  0/0   0/0   4/0   4/0

    core.gates, budget 400k fixed in advance: REJECTED -- significance
    (McNemar b = 0, c = 5, p = 0.074 over 30 pairs) and counterexamples:
    20 of the 30 runs on W0, W3, W4 the old policy solved, the new one did
    not. The transfer claim also failed on T4 (0.4 -> 0.0).

    B -- A with an idle "double" and without "swap <" -- got the same
    change and dropped "double" too; B' differs from A' only in "swap <",
    which A kept because the experience never paid for it and B never had.
    The change follows the policy it starts from and the experience it is
    given, not one fixed trick.

    Against the user's four conditions: the change was made by MANA, from
    no list; from its own experience; better on unseen data -- 28 of 30
    from 10k programs where the old policy has none, 27 of 30 hidden
    against 17; and not a matter of budget -- at equal programs the gap is
    largest where the budget is smallest. And the core refused it, rightly:
    written as briefly as possible, an experience of one family of
    questions is a specialist -- everything that family never needed goes,
    and with it the conditions W0 and W3 are made of. Compression of
    one's own successes rewards what they needed, as the words of step 3
    and the move of step 6 did; here, for the first time, it rewrote the
    search itself, and the gates -- counterexamples, not significance --
    are what stopped a better-looking search that could no longer answer
    a question it used to.

Stage R2b, predicted before the run (2026-09-14; scripts/run_reflect.py R2b)
----------------------------------------------------------------------
    The user's control: the same generator, the same gates, one change --
    the experience widened to every training family, T1, T2, T3, W0, W3
    (seeds 0..9). Counterexamples sought on W0, W3, W4 seeds 10..19, which
    now are unseen; W4 stays out of the experience.

    Predicted: W0's and W3's derivations are one step -- a leaf wrapped in
    a condition -- so "add a condition" can no longer be dropped without
    leaving part of the experience unwritable. The distance move is still
    added, "replace by a leaf" and "swap add" still dropped. The new
    policy is the old one with the move: as in step 6, answers four times
    sooner on the family, a few more by 400k. Counterexamples: none, or
    near none. Verdict: REJECTED on significance alone -- at 400k a
    handful of discordant pairs, below McNemar's floor at 30 -- unless the
    move does more than in step 6. T4 no better. If so, R2's specialist
    came from the narrow experience, not from the principle: what is
    dropped is what the experience does not need, and a wider experience
    needs more.

Stage R2b, measured 2026-09-14
----------------------------------------------------------------------
    From 50 derivations (T1..T3, W0, W3; 1 966 -> 1 152 bits) MANA added
    the same distance move and dropped only "replace by a leaf" and "swap
    add" -- "add a condition" stayed, as W0's and W3's experience needs it.

    solved                     10k  25k  50k  100k  200k  400k
        T1..T3 unseen, old       0    0    0     6     8    23   (of 30)
        T1..T3 unseen, new       9    9   13    22    26    26
        T1..T3 hidden, old       0    0    0     7     9    17
        T1..T3 hidden, new      10   10   15    17    18    18
        W0, W3, W4 unseen, old  20   20   20    20    20    28
        W0, W3, W4 unseen, new  20   20   23    28    29    29
        T4, old / new          0/0  0/0  0/0   0/1   4/2   4/2   (of 10)

    core.gates at 400k: REJECTED on significance alone (McNemar b = 0,
    c = 3, p = 0.25); no counterexample in 30 unseen runs of W0, W3, W4;
    the transfer claim failed on T4 (0.4 -> 0.2).

    The prediction held: R2's specialist came from the narrow experience,
    not from the principle. With every training family in it, the same
    generator made a change that breaks nothing it was shown or was
    tested on -- W4, T2 renamed, even gains -- and the one family outside
    it, T4, lost two answers. What it bought is speed: at 100k programs 22
    of 30 against 6, 17 hidden against 7; at the budget declared in
    advance, 400k, the old policy has nearly caught up (26 against 23, 18
    against 17), too few to call. The gates judged the claim that was
    made -- more answers at 400k -- and that claim is not proven. A claim
    of speed would have to be declared before a run on seeds nobody has
    looked at; choosing 100k now, after seeing the curves, is the thing
    the gates exist to refuse.

Stage R2c, declared before the run (2026-09-14; scripts/run_speed_claim.py)
----------------------------------------------------------------------
    The claim, fixed here before anything is run: the policy MANA built in
    R2b solves more questions of its family within 100 000 programs than
    the policy it replaced. Evidence on seeds no run has touched: 30 dev
    pairs, T1..T3 seeds 30..39; hidden, T1..T3 seeds 40..49;
    counterexamples, W0, W3, W4 seeds 20..29; transfer, a separate claim,
    T4 seeds 10..19. Verdict: core.gates.judge, as it stands. The policy
    is not written by hand: the same generator rebuilds it from the same
    experience, and the run stops if the change differs from R2b's.

    Predicted: ACCEPTED -- at 100k R2b's curves stood at 22 of 30 against
    6 and 17 against 7, so discordant pairs well over McNemar's floor, all
    one way; no counterexample (W0, W3, W4 at 100k: 28 against 20 in R2b);
    the transfer claim not shown -- T4 at 100k is at 0 or 1 of 10 for both.

Stage R2c, measured 2026-09-14 (declared in commit c1f85f6, before the run)
----------------------------------------------------------------------
    The generator rebuilt R2b's change exactly, from R2b's experience.

    solved at 100k               old   new
        dev pairs, T1..T3 s30-39    6    13   (of 30)
        hidden, T1..T3 s40-49       5    15   (of 30)
        W0, W3, W4 s20-29          20    23   (of 30)
        T4 s10-19                   0     0   (of 10)

    core.gates: R2c-speed ACCEPTED -- McNemar b = 0, c = 7, p = 0.023;
    hidden 0.17 -> 0.50; no counterexample in 30. R2c-speed-transfer
    REJECTED on transfer (T4 0.0 -> 0.0).

    Predicted rightly: accepted, and transfer not shown. Smaller than
    R2b's curves promised on their seeds (22 against 6 there, 13 against
    6 here): the gain measured on the seeds it was first seen on was
    partly those seeds, which is why the claim had to be made again on
    new ones.

    The first change of its own search MANA has made that the immutable
    core accepted: built from its own experience by a principle nobody
    turned into a list of changes, declared before it was measured,
    judged on seeds no run had touched, with nothing it used to answer
    lost. What it is, said plainly: a faster search for questions like
    the ones it learnt from -- the move it grew made a rule, and two rules
    that experience never used were dropped -- inside the language of
    policies R1 wrote. It does not carry to T4, and it does not change
    that language.

N3, probe P1, predicted before the run (2026-09-14; selection.py,
scripts/run_p1.py)
----------------------------------------------------------------------
    Before any search for a new construct of the policy language: can a
    language made only of the interpreter's own operations -- collections
    ordered, cut, repeated; a state's score, answers, and where it is
    right; pointwise and, or, not -- say a selection of another kind than
    R1's "order by a state's own score, keep k", and does one written by
    hand do better? A diagnostic: the two hand-written programs --
    coverage (keep the best, then the state right where the kept ones are
    wrong) and diversity (keep the best, then the state whose answers
    differ most from the kept) -- enter no N3 experiment.

    Predicted: R1 written in the language is the search, bit for bit;
    coverage and diversity are of another kind -- removing one kept state
    changes which others are kept, in most rounds tested -- and R1 never.
    On T4 no confident prediction, and a guess against: after the best,
    coverage takes the leaf that is right wherever the best is wrong -- on
    T4, y and z between them are right everywhere -- and from then on it
    ranks by score, like R1; diversity keeps states whose answers differ,
    garbage among them. So neither above R1's 4 of 10, diversity worse on
    the others. If so, P1 says the language can decide about a set, but the
    two natural decisions do not recover what the kept state recovered,
    and what sets that state apart is still unnamed.

N3, probe P1, measured 2026-09-14
----------------------------------------------------------------------
    of what kind (T4, rounds 1-3, one kept state removed and chosen again)
        R1 0 rounds of 9 with a reversal; coverage 3 of 9; diversity 6 of 9
    solved at 800k                  T4 (of 10)   the other six (of 60)
        R1, in the language             4              40
        coverage                        4              27
        diversity                       0              24
    time the selection itself took: R1 20%, coverage 60%, diversity 68%

    The language can say a selection of another kind -- a decision about
    the set that no score of a state on its own reproduces -- and R1
    written in it is the search. But neither decision of that kind written
    by hand does better: coverage ties R1 on T4 and loses a third of the
    others, diversity loses T4 and more, and both spend most of the run
    choosing. Predicted rightly on T4 and on diversity; coverage lost more
    of the others than predicted.

    What the state kept by hand in step 6's control has, read from round
    1 of T4: coverage takes the best, then the leaf z -- right wherever
    the best is wrong, 200 points of 200 covered after two picks -- and
    from there it orders by score. if(x < 3, z, y) is right on 12 or 13 of
    the points the best gets wrong, where z is right on all of them, and on
    fewer points than y in all. It is set apart neither by what it gets
    right nor by what it adds to the set: its worth is where it leads --
    one move from the 66-bit rule. A selection that reads only the present
    of the states it chooses among cannot see that.

    And an omission in the probe itself, owned: the rule for the language
    was "every operation the interpreter performs", and one of them was
    left out -- making a state's successors, which the interpreter does
    every round, and the only operation that relates a state to its
    future. The finding of P1 is where the gap is: not in deciding about
    sets, but in not being able to look at what a state becomes.

N3, probe P1b, predicted before the run (2026-09-14; scripts/run_p1b.py)
----------------------------------------------------------------------
    P1 again, with the operation it left out restored: a state's
    successors under the policy's rules. Its price fixed before the run:
    every successor a selection looks at is evaluated like any program,
    in the search's own budget, and can be the answer. The diagnostic,
    written by hand: keep the best by score; then, three times, of the
    first M by score (M = 32, 256), the state whose best successor is
    best. Not a decision about a set -- a score of one state that reads
    its future.

    Predicted: no reversals -- it is a ranking of states, if one R1's
    language cannot write. On T4 no better than R1's 4 of 10: the state
    that leads to the rule ranks 234th to 987th by score in round 1, so
    it is outside the first 32 on every seed and inside the first 256 on
    two (seeds 2 and 8); and a look costs the state's neighbourhood,
    about 6 000 programs, 5 800 of them wrappings in a condition -- 256
    looks are more than the whole budget of 800k in round 1, so M = 256
    stops after one round, and from that state the rule is three plain
    edits away, not one. The others: below R1 at equal budget, the looking
    spending what R1 spends on its next rounds. If so, P1b says the
    future is expressible and, at its honest price, unaffordable for the
    states that matter: they are hundreds deep, and seeing one step ahead
    of each costs the whole neighbourhood. What would make it cheap is a
    narrower neighbourhood -- R2 found MANA dropping "add a condition"
    when its experience allowed -- or knowing where to look, which is what
    the pin of step 6's control was.

N3, probe P1b, measured 2026-09-14
----------------------------------------------------------------------
    A first run measured a flaw of the interpreter, not the idea:
    successors a selection looked at were evaluated, and when a kept state
    made them again the next round skipped them as seen -- they never
    entered any pool. Fixed (policy.run: what was looked at is offered
    when a kept state makes it, evaluated once, counted once), a test
    added, and run again. The figures are from the second run.

    of what kind: no reversals for either look (0 of 3 rounds each), R1
        none -- a score of one state, if one reading its future
    solved at 800k                  T4 (of 10)   the other six (of 60)
        R1                               4              40
        look, first 32                   0              29
        look, first 256                  0              20
    budget spent looking: 98% and 100%; rounds run, median: 2 and 1 (R1 6)
    the state step 6's control kept, in round 1 of T4: in the pool, never
        kept -- 232nd to 986th by score, outside the first 32 on every
        seed, inside the first 256 on two (232nd, 238th), where 256 looks
        of about 6 000 programs each had spent the whole budget before
        they reached it

    Every prediction held, and T4 went further than "no better than 4":
    it fell to 0. The future is expressible -- successors in the language
    make a score no present quantity can write -- and at its honest price
    it is unaffordable for the states that matter: the one that leads to
    T4's rule is hundreds deep, and a look one step ahead of each state
    costs its whole neighbourhood, 5 800 of 6 000 of it wrappings in a
    condition.

    P1 and P1b together, the existence probes N3 was to start from: in a
    language made only of the interpreter's operations, both new kinds of
    construct can be written -- a decision about the set, a score of the
    future -- and none of the natural ones written by hand beats R1. So
    there is no existence proof of a useful construct in this language at
    this cost, and by the rule fixed with the user an N3 search over it
    would search a space in which nothing useful was shown to lie. What
    the probes point at instead is the price of a look: the width of the
    neighbourhood, set by the rules, not by any selection.

N3, probe P2, predicted before the run (2026-09-14; scripts/run_p2.py)
----------------------------------------------------------------------
    One change to P1b: the rules the look makes successors with. The
    search keeps its own; the look uses a policy MANA grew -- A' of R2,
    rebuilt by the generator from T1..T3 (the move added, "add a
    condition" dropped), the specialist core.gates refused as a search.
    What the look sees is still evaluated in the budget and can be the
    answer. Controls: the look through R2b' (the same move, "add a
    condition" kept -- as wide as the search's own), and through A'
    without its grown rules, written by hand (narrow, no move). T4 is in
    no experience. This asks step 6's question from the other side: the
    move was "only as good as the states the search can stand in to use
    it" -- can a look use it from states the beam does not hold?

    Measured before the run, on R1's first round of T4 (a diagnostic,
    with the rules as R2 and R2b logged them): a look at a typical state
    of the first 256 costs 15 761 programs through the search's rules,
    15 787 through R2b', 343 through A', 239 through A' without the move.
    States one step of A' from an exact answer -- if(x < 3, z, y) and one
    like it -- lie inside the first 256 on seeds 0, 2, 6, 8 (225th to
    248th), inside the first 32 on none; on 2, 6, 8 the look's three picks
    begin with them. On seed 0 the look keeps other leaves than R1 in
    round 0, so its first round is another. R1 solves seeds 0, 2, 3, 8
    (step 6's control), none of them by the 66-bit rule.

    Predicted:
      of what kind  no reversals in any arm: a look is a score of a state
      look 256, A'  T4 solved on seeds 2, 6 and 8 in round 1, each answer
                    the 66-bit rule and made only by a look, first within
                    100k programs, not within 50k; seed 6 is one R1 does
                    not solve. Seeds 0 and 3 open -- the beam is not R1's.
                    In all, 3 to 5 of 10. Most of the budget spent looking
                    (60-90%), several rounds
      look 32, A'   no state of T4 within reach in round 1: no better than
                    R1's 4
      A', no move   T4 no better than R1: from those states the rule is
                    three plain edits away, and a look sees one
      R2b'          as P1b's look: the whole budget spent in round 1
                    before the look reaches rank 225; T4 0
      the other six the family's answers made by a look through the move
                    (T1 from a leaf in round 0), far sooner than R1 --
                    step 6's gain, the move's, not a selection's; W0, W3 as
                    ever. The control without the move no better than R1
    If so: a look through a narrow neighbourhood MANA grew reaches states
    the beam drops, and uses a move there the search could not; width and
    the move both necessary, the one without the other nothing. That is an
    existence proof of a selection in the language that does better on T4
    -- but narrow: it rests on the states that happen to lie in the first
    256, three or four seeds of ten, and on a hand-written look. If T4
    does not move, the price of a look was not the only wall.

N3, probe P2, measured 2026-09-14 (declared in commit 4925392, before the run)
----------------------------------------------------------------------
    The generator rebuilt A' and R2b' exactly, from R2b's dear runs.

    of what kind: no reversals in any arm (0 of 3 to 6 rounds)
    solved                10k  25k  50k 100k 200k 400k 800k  looking  made by a look
      T4 (of 10)
        R1                  0    0    0    0    4    4    4      0%    0 of 4
        look 32, A'         0    0    0    0    5    6    6     17%    1 of 6
        look 256, A'        0    0    0    4    4    9    9     62%    5 of 9
        256, A', no move    0    0    0    0    0    9    9     62%    1 of 9
        look 256, R2b'      0    0    0    0    0    0    0    100%    --
      the other six (of 60)
        R1                 20   20   20   27   27   40   40      0%    0 of 40
        look 32, A'        30   55   60   60   60   60   60     18%   40 of 60
        look 256, A'       30   55   60   60   60   60   60     67%   40 of 60
        256, A', no move   20   20   20   21   30   30   47     48%    4 of 47
        look 256, R2b'     30   30   30   36   48   60   60    100%   40 of 60
    (made by a look: answers no round of the search made, only a look.)
    Step 6's kept state, round 1 of T4: kept by the two looks of 256
    through narrow rules on seeds 2 and 8, by no other arm.

    Held: no reversals; the look of 256 through A' first solved T4
    between 50k and 100k, 4 of 10, the answers made inside the look; the
    look through R2b' spent the whole budget in round 1 and solved
    nothing; the family's answers came far sooner through the move, 60 of
    60 by 50k against R1's 20.

    Wrong, and it is the result: the control without the move. Predicted
    no better than R1; it solved T4 on 9 of 10 by 400k, as the look
    through A' did, and 8 of its 9 answers were made by the search's own
    rounds, not by a look. Wrong too: the look of 32, 6 of 10, not at most
    4; the look of 256 went to 9, not 3 to 5.

    So what decided T4 was the width of the look, not the move. A look one
    step ahead through a neighbourhood without "add a condition" -- 239 to
    343 programs a state against 15 761 -- is cheap enough for 256 states
    a round, and the states it keeps lead the search's own plain edits to
    T4's rule, where R1's best four do not. The move made it sooner (4 of
    10 by 100k, inside the look); without it the 9 came between 200k and
    400k. The same move through the wide neighbourhood solved nothing.
    Width necessary, the move not.

    What this is: the first selection written in the meta-language that
    does better than R1 on T4 at equal programs, every look counted -- 9
    of 10 against 4 at 400k -- and so the existence proof N3 was to start
    from. What it is not: N3. The look is written by hand; its narrow
    rules are A' (MANA's, refused as a search) or A' with the move taken
    out by hand; seeds 0..9 are the ones the diagnostic before the run
    looked at, and nothing is claimed before core.gates on fresh seeds.
    Not measured: which states the narrow look keeps, and why they lead
    to the rule; wall time per arm (the run took 2 856 s in all).

N3, P2c, declared before the run (2026-09-14; scripts/run_look_claim.py)
----------------------------------------------------------------------
    The claims, fixed here before anything is run. Each against the
    search as it stands, within 400 000 programs, every look counted in
    them:
      P2c-look   a look at the first 256 states by score, one step ahead
                 through the rules of A', solves T4 more often
      P2c-width  the same look through A' with its grown move taken out
                 by hand solves T4 more often
    Evidence on seeds no run has touched: 30 dev pairs, T4 seeds 20..49;
    hidden, T4 seeds 50..79; counterexamples -- a question the search
    solved and the look did not -- on W0, W3, W4 seeds 30..39 and T1, T2,
    T3 seeds 50..59. Verdict: core.gates.judge, as it stands. A' is not
    written by hand: the generator rebuilds it from R2's experience
    (T1..T3, seeds 0..9), and the run stops if the change differs from
    R2's. The budget, 400k, was read off P2's curves -- which is why the
    seeds are new.

    Predicted, from P2 at 400k (search 4 of 10 on T4, both looks 9; on
    the other six, search 40 of 60, the look through A' 60, the look
    without the move 30):
      P2c-look   ACCEPTED -- dev about 0.4 -> 0.9, discordant pairs at
                 least 12 one way and at most 2 the other, p < 0.001;
                 hidden the same; no counterexample, the family's answers
                 coming through the move
      P2c-width  dev and hidden as P2c-look, and REJECTED on
                 counterexamples: about ten questions of the family (T1..T3,
                 W4) the search solves by 400k and the narrow look without
                 the move does not yet -- none on W0, W3
    If so: the look through the rules MANA grew is a better search for T4
    by the core's own gates, and the width alone, though it is what
    decided T4, costs the family answers the search had. Neither is N3:
    the look is written by hand, and MANA did not choose it.

N3, P2c, measured 2026-09-14 (declared in commit 4428a2c, before the run)
----------------------------------------------------------------------
    The generator rebuilt R2's change exactly, from R2's experience.

    core.gates, 400k:
      P2c-look   ACCEPTED -- dev 0.37 -> 0.90, McNemar b = 0, c = 16,
                 p = 0.00018; hidden 0.27 -> 0.83; no counterexample in 60
      P2c-width  REJECTED on counterexamples, 9 of 60 -- W4 3, T2 5, T3 1,
                 none on W0, W3; dev 0.37 -> 0.93 (b = 0, c = 17), hidden
                 0.27 -> 0.87

    solved         50k            100k           200k           400k
                   dev hid  W  T  dev hid  W  T  dev hid  W  T  dev hid  W  T
      search        0   0  20  0   0   0  20  6  11   8  20  9  11   8  23 15
      look          3   5  30 30   7   7  30 30   7   7  30 30  27  25  30 30
      width         0   0  20  0   0   0  20  0   0   0  20 10  28  26  20 10
    (dev, hid: T4 of 30; W: W0, W3, W4 of 30; T: T1..T3 of 30)

    Every prediction held. On T4, both looks nearly triple the search, on
    seeds no run had touched; the width alone does it, and the move adds
    speed (the look through A' has 7 of 30 by 100k, without the move
    none) and the family (all 60 by 50k). The look without the move pays
    for T4 with answers of the family the search had by 400k -- the core
    refused it on exactly those.

    What is accepted, said plainly: a better search for T4 by the core's
    gates -- a look one step ahead at the first 256 states, through the
    rules MANA grew in R2. The rules are MANA's, grown from its own
    experience with T4 in none of it; the look is written by hand. So
    this is not N3: the selection that made the difference was not
    constructed by MANA. It is what N3 was missing before it could start
    -- a construct in the meta-language, shown to be worth finding, by
    the core, on fresh seeds. The run took 1 293 s, 391 of them to rebuild
    A'; wall time per arm was not recorded.

N3, P2d, predicted before the run (2026-09-14; scripts/run_look_anatomy.py)
----------------------------------------------------------------------
    A diagnostic: nothing claimed, nothing learnt, the hypothesis left as
    it stands. The look P2c accepted, and the look without the move, on
    P2c's dev seeds (T4 20..49, 400k), traced; each round's choice
    recomputed from the log and checked against what was kept. Four
    questions, from the user:
      1  which states the look keeps, and by what they pass through A'
      2  what sets a kept state apart from a near twin that was dropped
      3  how much of the gain is "a state -> the good successor the look
         saw -> T4", and how much "a state -> the search's own way on from
         another point"
      4  where in MANA's experience a record "state -> worth of exploring
         it further" could come from

    Predicted:
      1  the picks lie deep by score -- the median past the 100th of 256,
         most outside the first 32 -- conditions on one variable against
         a leaf, if(v < c, a, b) and its kin; their best successor is
         made, through A', mostly by the move, and without the move mostly
         by "combine with a leaf" at the condition's variable
      2  twins are dropped with nearly the same score of their own (median
         gap under 10 bits) and a far worse look (median over 50 bits):
         what separates them is not in the state but in what one step
         makes of it -- which no score of a state's present can carry, as
         steps 7 and 7b found
      3  through A': about half the answers made inside a look, some by
         looking at states that were not then picked; most of the rest
         through a pick and on through the successor it was picked for;
         "another way from the picked point" a minority, "through no pick"
         at most two. Without the move: inside a look at most three; the
         seen successor at least half; another way the rest
      4  read from the code before the run: no such record exists.
         Experience and derive.Derivation hold the answer's path only;
         frontier.Step the features of a path state's siblings (worth of
         keeping, from successes); frontier.Regret the states a dearer run
         went through (worth of keeping, from regret, with no cost and no
         look); Found.looked is a count; log_rounds are pools and kept
         states in a script, nobody's experience; the look's own gain is
         computed for every state it looks at and thrown away. The one
         record of the kind is in the frozen methods corpus -- a PROBE
         entry of the Journal, the cost of an experiment on a method,
         predicted against actual -- about other objects in another world

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
