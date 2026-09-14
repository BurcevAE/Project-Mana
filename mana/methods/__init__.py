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
"""
from __future__ import annotations

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"
