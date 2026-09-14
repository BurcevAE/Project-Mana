"""
mana.methods — learning which way of knowing a task calls for.

The experiment (M)
------------------
Discovery searched for one method and one currency that would solve
anything, and its own measurements argued against it: the narrow beam is
best to 400k, adaptive cells at 800k, the wide beam from 1.2M, and a grid
laid down in advance only hurts. Which way of looking works depended on
the task and the budget. What may be universal is not a method but the
mechanism that learns which method a task calls for -- and, later, builds
one it has not got.

What is fixed, said plainly
---------------------------
    the methods      four, written by hand, none knowing any world:
                     exhaust (ask everything), calculate (a linear model
                     from n + 1 questions), decompose (each input's effect
                     on its own), experiment (test pairs of inputs for
                     interaction, group them, tabulate each group). Each
                     has its own representation, its own way of asking and
                     its own check.
    the verdict      the world's, on inputs no method asked about. Methods
                     are compared by it and by what they spent -- the one
                     currency left, and it lives above the methods, not
                     inside them.
    the protocol     methods see the form of the question (how many
                     inputs, how many values each) and may ask; the world's
                     structure lives in `world.py`, which no method may
                     import.

What is NOT given, from M1 on: which method to use for which world, and
under what conditions a method works. Brain_factory's table ("exactly
computable -> algorithmic" ...) is the hand-written form of that
knowledge; this experiment is about learning it.

Steps
-----
    M0  the world, the methods, the controls -- no learning. Is the order
        of the methods really different in different worlds? And how
        strong is the trivial control: a cascade, cheapest first, moving on
        when a method's own check fails?
    M1  meta-knowledge: choosing on new instances, including from small n
        to large
    M2  transfer to another family, where the order changes
    M3  a method as a composition of parts, built when none fits

M0, predicted before the run (2026-09-14)
-----------------------------------------
    order     linear   calculate < decompose < experiment < exhaust
              additive decompose < experiment < exhaust; calculate wrong
              blocks   experiment, and exhaust while the table fits;
                       calculate and decompose wrong
              global   exhaust while the table fits; nothing from n = 6
    cascade   within 1.5x of the oracle almost everywhere: the checks are
              cheap and, without noise, reliable, and trying the cheap
              methods first costs little next to the one that works
    random    a cascade in a random order is far worse wherever exhaust is
              tried early: order matters, but the trivial order -- cheapest
              announced plan first -- is nearly all of it
    checks    agree with the verdict on 99% of attempts or more

If that holds, M0 is a world where knowing which method to use is worth
almost nothing, and M1 must be measured where it is worth something:
where a wrong try is expensive or a check cannot be trusted.

M0, measured 2026-09-14 (scripts/run_methods.py, 4 structures x n in 2..8
x 10 seeds; median distinct questions of the methods that were right)
----------------------------------------------------------------------
                       n=2    n=3    n=4     n=5    n=6   n=8
    linear   order     B<C<D<A everywhere A fits; B 12 .. 19 questions
    additive order     C<D<A;  B wrong on every seed
    blocks   order     A<D at n = 2 (one block is the whole box), then D:
                       100    124    221     242    344   479
    global   order     A, and D as dear as A (it finds one group of all
                       inputs); nobody from n = 6
    cascade / oracle   1.00 in every cell of the table
    random order       up to 1 500x the oracle where exhaust fits and is
                       tried early (linear n = 5: 25 039 against 16)
    checks             agree with the verdict on 880 of 880 attempts

The orders are different, as the world was built to make them -- and
choosing among them is worth nothing here. Better than predicted, and for
a reason worth keeping: the cascade remembers what it asked, and the
methods' questions nest. Calculate's steps are among decompose's moves,
which are among experiment's tables, which are among exhaust's; so a
cheap method that fails has asked only what the next one needed anyway,
and failing costs nothing. The one place knowledge would have paid: a
global box from n = 6, where the cascade spends 110 - 167 questions to
learn that nothing fits, and the oracle spends none.

So the trivial rule -- cheapest announced plan first, move on when your
own check fails -- is the whole of method choice when three things hold:
a method can announce its cost before it starts, its check is reliable,
and what a failed method learnt is reused by the next. M1 must break at
least one of them, or it measures nothing. Each is broken somewhere real:
the discovery searches cannot say in advance what they will cost, a
check on noisy answers can lie, and a question that changes the world
cannot be asked again.

M1, predicted before the run (2026-09-14)
-----------------------------------------
One condition broken: no method announces what it will cost, and nothing
declines for free -- a method asks until it is done or its budget is
spent. The chooser (choice.py) learns from what it could have seen:
questions asked, finished or not, own check passed or not. Training:
40 boxes of n = 2..4, learning as it goes; test: 36 boxes of n = 5, 6, 8
it has never seen, with what it learnt frozen. 10 streams. Against, on
the same boxes: the cascade in the writer's order (now knowledge handed
in, not derived), a random order, the oracle.

    training   starts near the random order, at the cascade's cost from
               about the 10th box: it learns the writer's order
    test       linear, additive, blocks at n = 5, 6 and global at n = 5:
               the cascade's methods at the cascade's cost
               global at n = 6: experiment (predicted to fit) runs to its
               budget, then exhaust is skipped -- extrapolated to a
               million questions -- so about half the cascade's cost; the
               oracle still spends nothing
               n = 8: the cost of experiment after calculate and decompose
               failed is learnt from blocks and global boxes together,
               cheap and dear averaged in logs; extrapolated to n = 8 it
               likely passes the budget, and the chooser gives up on the
               blocks the cascade solves -- a predicted loss, the price of
               a chooser that knows the box only by which methods failed

So the claim at stake is modest: learnt experience replaces the order the
writer handed in, and adds one thing the cascade lacks -- "this will not
finish". Telling blocks from global at n = 8 needs more than failures to
describe the box: a method's cheap first part -- experiment's pair tests
-- used as a probe. That is M3's matter, a method as parts.

M1, measured 2026-09-14 (scripts/run_choice.py, 10 streams: 400 training
boxes, 360 test boxes)
----------------------------------------------------------------------
    training, n = 2..4: questions over the oracle's, by tens of boxes
        chooser      1.12   1.23   1.19   1.24
        cascade      1.00   1.00   1.00   1.00
        random       1.47   2.12   1.82   1.80
    test, n = 5, 6, 8, frozen          solved    questions
        300 solvable     chooser         291     6 038 629
                         cascade         300     3 039 630   (= oracle)
                         random          300    21 239 192
        60 hopeless      chooser           0    13 803 798
        (global, n >= 6) cascade           0    24 004 440
                         oracle            0             0

Prediction against result: the writer's order learnt by the 10th box --
no, the chooser stays at 1.2 times the oracle to the end; the cascade's
cost on solvable boxes -- no, twice it, and 9 of 300 lost; half the
cascade on hopeless boxes -- yes (0.58); giving up on blocks at n = 8 --
in 3 streams of 10.

Why, read from the chooser's own beliefs (streams 0, 1, 2 re-run):
    one size says nothing of growth, and the chooser took it to mean no
        growth. Exhaust, after calculate and decompose had failed, had
        one record: 9 953 questions at n = 4. So it expected 9 953 at
        n = 5, 6 and 8, cheaper than experiment, and ran it first -- the
        orders B C A and B C A D. Experiment first on a linear box the
        same way: one record, 47 questions at n = 3, expected at n = 8
    greed froze what it knew of the alternatives: 353 of 400 training
        boxes began with calculate, and the others' records from an empty
        start stayed at the one to three made on the first boxes, never
        revisited. Hence the flat training curve
    boxes of different kinds pooled: experiment after two failures,
        learnt from blocks and global together, extrapolated to 1.27 and
        193 million questions at n = 8 -- and the chooser gave up
    the cascade still equals the oracle on solvable boxes: announcing was
        broken, reuse was not, and the questions still nest

So learnt experience did not replace the order the writer handed in: it
paid twice as much where there was an answer, and gained only the one
thing the cascade lacks, "this will not finish". What the writer's order
holds -- that a table grows with n, that calculating is cheapest -- is
exactly what cannot be learnt from experience gathered at one size. A
chooser that only solves boxes gathers it by accident; it would have to
try a method at another size to learn how it grows -- choosing
experiments about its methods, not only answers about boxes.

M1b, predicted before the run (2026-09-14)
------------------------------------------
The Explorer (choice.py): a second kind of action -- a method run on a
smaller version of the box, the rest held still -- weighed with applying
a method by one rule, the decision's expected improvement less the
questions spent; costs as beliefs, with a doubt of extrapolation learnt
from its own journal of predicted against actual.

    M1 world, M1's protocol (frozen at test)
        solvable   far below M1's 6.04M and near the cascade's 3.04M: an
                   exhaust seen at one size is now uncertain at the next,
                   so it is no longer run before experiment
        blocks 8   30 of 30: a probe of experiment on a few of the box's
                   inputs tells a box of pairs (hundreds of questions)
                   from a global one (10 ** size), and only then is
                   experiment applied
        hopeless   far below M1's 13.8M: the same probe says "global", and
                   it gives up after the probe instead of running
                   experiment to its budget
        probes     few in training, at the test's new sizes, mostly of
                   experiment after calculate and decompose failed
    M1 world, learning through the test
        the trajectory: after the first probes the kind's line for
        experiment-after-two-failures stays wide -- two kinds of box under
        one line -- and the decisions are carried by probes of the box
        itself, not by that line
    the trap (trap.py: explosive 20x cheaper at n = 2..4, dearer from 6,
    past the budget at 8; steady linear)
        the Chooser pays explosive's budget on n = 8 boxes more than once
        before its line catches up; the Explorer, whose priors call a
        clean line four sizes out safe, pays it on its first n = 8 box --
        then its drift rises from the surprise, and it does not pay it
        again. A probe before the first surprise is not predicted: its
        worth to this box is small under those priors, and its worth to
        later boxes is not counted

The criterion, fixed with the user before the run: part of the questions
go to knowledge about a method, not to the box, where the uncertainty is
large and the choice depends on it -- and the choice changes after the
knowledge.

M1b, measured 2026-09-14 (scripts/run_probes.py, 10 streams per part)
----------------------------------------------------------------------
    M1 world, frozen at test         solvable (300)       hopeless (60)
        M1 Chooser                   291   6 038 629          13 803 798
        Explorer                     300   3 131 684          10 812 128
        cascade (= oracle)           300   3 039 630          24 004 440
    M1 world, learning through it
        M1 Chooser                   300   3 240 942          11 204 327
        Explorer                     300   3 066 906          12 094 370
    the trap, 12 test boxes, questions (oracle 642 080)
        sizes 5, 6, 8 in turn        M1 Chooser 1 076 080  Explorer 1 645 769
        8 first, then 5, 6           M1 Chooser 2 880 080  Explorer 3 487 227

Where the criterion was met -- the trap as posed. After n = 2..4 only,
straight to n = 8: in 10 streams of 10 the Explorer ran explosive on six
of the box's inputs (20 017 questions), saw it explode, and applied
steady: 28 036 questions on that box, against the Chooser's 208 019,
which ran explosive to its budget first. Knowledge about a method, bought
where the choice hung on it, and the choice changed. What it expected of
explosive at n = 8 (stream 0): 743 questions before, 20 794 after that
probe, 149 576 after four, 193 661 at the end -- the truth is 2 million;
a line through a bend, censored at the budget, still undershoots.

Where it was not, and why -- read from its own numbers:
    the worth of the present was taken as the worth of the best single
        application. The alternative is a sequence -- M1's ratio rule,
        cheapest likely method first, was the better rule. Worth rewards
        the likeliest method: training applied exhaust first on 306 of
        400 boxes (2 to 2.9 times the oracle, M1 1.2), experiment was
        never run in training in most streams, and at test it was run as
        an unknown -- which is why blocks at n = 8 came out 30 of 30 (24
        of them without a probe), and why global boxes still paid its
        budget: the hopeless boxes cost 10.8M, not the "far below" that
        was predicted
    probes were weighed against that same single application, so the
        cost of the catch-all method looked worth measuring where a cheap
        sequence settles the box: exhaust on 2 and 3 inputs of every n = 5
        box, 1.3M questions of probes at test in the frozen protocol
    success was left out of the uncertainty: steady, applied once, is
        believed to pass its check with Laplace 0.67. So on the second
        n = 8 box of every stream a probe of explosive on 7 inputs looked
        worth it -- and the probe itself spent explosive's whole budget,
        200 000 questions. The Explorer paid for the trap with the
        experiment meant to avoid it
    a line through a bend reads as spread between boxes: every box looks
        individually uncertain, and explosive on 5 inputs was probed
        again on box after box where the journal already had it

So: an experiment about a method, weighed with acting, did what it was
for once, cleanly, where the question was posed for it; everywhere else
the value it was weighed against was wrong. The next edit is that value:
the worth of the present as the worth of the best ordered sequence of
methods, and the uncertainty about success in it, not only about cost.

M1c, predicted before the run (2026-09-14)
------------------------------------------
One hypothesis: information is worth how much it changes the best plan,
not how much better one method is known. The Planner (choice.py) keeps the
Explorer's beliefs and changes only what they are weighed as: a plan -- an
order of methods to the first success -- costs its expected questions plus
the value of an answer times the chance none succeeds; each step carries
cost and success together (finish within budget, and pass the check), its
chances taken after the steps before it failed; a probe is worth the cost
of the best plan now less the expected cost of the best plan after its
outcome, less its own questions. Every decision is kept: the plan, the
action, the plans a probe's outcomes lead to and with what chance, the
plan after the actual outcome, and whether the probe repeated one the
journal already held.

The five criteria, fixed with the user, all at once:
    1  the plan comes closer to the oracle -- M1 world, frozen: solvable
       within 1.1 of the cascade (~3.3M), training near M1's 1.2 of the
       oracle (a method never run is tried first again, as in M1);
       hopeless far below the Explorer's 10.8M -- after calculate and
       decompose fail on a global box at n >= 6, experiment probed on 3 or
       4 inputs turns the plan into giving up (under 2M)
    2  probes are made where they change the plan -- few where a cheap
       sequence settles the box (probe questions at test well under the
       Explorer's 1.3M); the plan changes after a probe about as often as
       the probe said it might, within 0.15 in each band of the predicted
       chance
    3  the same information is not bought twice -- not predicted to hold
       in the trap: a bend in explosive's line still reads as spread
       between boxes, so each box looks uncertain on its own and explosive
       on 5 inputs is probed again; those repeats should rarely change the
       plan. The belief model is the Explorer's and not under test
    4  the trap is found before the costly application -- after n = 2..4,
       no n = 8 box pays explosive's budget, first or later; the probe on
       7 inputs that burnt it in M1b no longer pays, the plan being steady
       first whatever it shows
    5  what is known of a method's scale carries to new sizes -- frozen
       at test, n = 5, 6, 8: every solvable box solved near the oracle,
       the hopeless ones given up after a probe

A second arm, added after the first run and predicted before its own
(2026-09-14). The first run reckoned a probe's worth as the user's formula
reads -- the cost of the best plan now, less the expected cost of the best
plan after -- and bought 926 probes that were expected to change the plan
less than one time in five: the Explorer's beliefs are not coherent (an
observation removes the doubt of extrapolation outright, and a cost's
expectation falls with its spread), so the same plan looks cheaper after
any outcome, and that spread was paid for. The hypothesis itself says a
probe is worth only what it changes: CHANGE reckons the expected saving of
the best plan after the outcome over the present plan, both weighed with
what the outcome would teach -- zero where the plan would not change.
Predicted for it: probes expected to change nothing disappear, and with
them most repeats and most questions spent on probes; the probe of
explosive on 7 inputs that burnt its budget in the trap (5, 6, 8 order,
learning) no longer pays; criteria 1, 4, 5 no worse than the first arm.

M1c, measured 2026-09-14 (scripts/run_plans.py, 10 streams per part;
Δ the first arm, изм the hypothesis proper)
----------------------------------------------------------------------
    M1 world, frozen at test         solvable (300)       hopeless (60)
        M1 Chooser                   291   6 038 629          13 803 798
        Explorer (M1b)               300   3 131 684          10 812 128
        Planner Δ                    294   3 351 084           2 552 506
        Planner изм                  294   3 291 945           2 629 602
        cascade (= oracle)           300   3 039 630          24 004 440
    M1 world, learning through it
        M1 Chooser                   300   3 240 942          11 204 327
        Planner Δ                    300   3 131 628          11 271 237
        Planner изм                  300   3 163 033           8 984 381
    probes at test, frozen           probes  questions   plan changed / expected
        Planner Δ                     1 783  3 237 263        20% / 29%
        Planner изм                   1 676  3 347 846        31% / 46%
    the trap, 12 test boxes: questions, boxes that paid explosive's budget
        (oracle 642 080)             M1 Chooser     Planner Δ     Planner изм
        5-6-8 frozen            9 202 080  40   2 392 305  0   2 332 113  0
        5-6-8 learning          1 076 080   0   5 154 827 19   9 165 629 39
        8-5-6 frozen            9 202 080  40   2 392 323  0   2 332 131  0
        8-5-6 learning          2 880 080  10   1 666 551  0   1 646 121  0

Against the five criteria, for the hypothesis proper:
    1  closer to the oracle -- in part. Frozen: 1.08 of the cascade where
       there is an answer, training at M1's 1.0 - 1.5 of the oracle, and
       the hopeless boxes 2.6M against the Explorer's 10.8M and M1's
       13.8M: after calculate and decompose, experiment on four inputs
       answers 10 000 on a global box, and the plan becomes giving up.
       But 6 blocks boxes of 300 lost, and learning through the test
       brings the hopeless back to 9.0M: blocks and global under one line
       teach it that experiment is cheap after two failures
    2  probes where they change the plan -- no. As many probes as the
       first arm, as many questions; the plan changed after 31% of them,
       46% had been expected, and in the band where 30% was expected, 4%.
       The rule counts only changes now, and the beliefs it counts them in
       expect changes the world does not make
    3  the same information bought once -- no: 389 repeats frozen, 570
       learning, where the plan changed after a quarter of them
    4  the trap found before the costly application -- yes, frozen in
       both orders and with 8 first while learning: not one n = 8 box
       paid explosive's budget, against 40 and 10 for the Chooser. And no
       in the 5, 6, 8 order while learning: 39 of 40 paid it. There the
       sizes 5 and 6 entered explosive's line before any n = 8 box; a
       straight line through a bend that steep says 10^5 at n = 8, not
       2 x 10^6, and the plan applies explosive -- then, steady known at
       one size only, gives up, and probes steady on 2, 3, 4 inputs to
       learn how it grows before applying it
    5  what is known of scale carries to new sizes -- for giving up, yes;
       for solving, 294 of 300

What the kept decisions say, box after box: the probes are chosen for the
plans their outcomes would lead to -- the hypothesis is doing what it
says -- and the outcomes it weighs are those of beliefs that are too wide
(a world of fixed costs read with spread between boxes and a doubt of
extrapolation) and of the wrong shape (one straight line per method, where
a method may be cheap and then explode). Its only systematic failure is
the wrong shape. How much information is worth cannot be measured apart
from how well the beliefs are calibrated, and the kept decisions measure
that too: plan changes expected against plan changes seen. The next
question is the one the user posed after M1c: can the kind of a method's
growth -- a line, a bend, a wall -- be told from experience, so that the
beliefs weighed are of the right shape.
"""
from __future__ import annotations

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"
