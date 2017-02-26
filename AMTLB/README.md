# Atari Multitask & Transfer Learning Benchmark (AMTLB)

[![Join the chat at https://gitter.im/ai-open-network/multitask_and_transfer_learning](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/ai-open-network/multitask_and_transfer_learning)

This is a library to test how a reinforcement learning architecture
performs on all Atari games in OpenAI's gym. It performs two kinds of
tests, one for transfer learning and one for multitask
learning. Crucially, this benchmark tests how an *architecture*
performs. Training the architecture on games is part of the test, so
it does not aim to test how well a pre-trained network does, but
rather how quickly an architecture can learn to play the games (but
see note below for details).

Throughout this document, we'll refer to **architecture** as the
system being tested irrespective of individual weight values. An
**instance** will refer to the architecture instantiated with a
particular set of weights (either trained or untrained). The benchmark
trains several instances of the architecture to infer how well the
architecture itself learns.

## Transfer learning benchmark

The goal of the transfer learning benchmark is to see quickly an
architecture can learn a new game it has never seen before, just using
what it's learned from other games (so, how much knowledge is
transferred from one game to another).

The way it works is first it creates a fresh instance of the
architecture (call it instance `F`), and then measures its score over
time as it learns on ten million frames of a random Atari game (call
it game `X`). Next, we create another fresh instance of the
architecture, but this one we train on bunch of other Atari games (but
not on game `X` itself), we'll call it instance `F_b`. Finally, we let
`F_b` play ten million frames of game `X` and measure its score over
time.

For each time frame, we take the cumulative score of `A` and the
cumulative score of `B` and get the ratio `r = 1 - B / A`.

 * If `r` negative, then the architecture actually got worse from seeing other Atari games.
 * If `r` is about 0, then the architecture didn't really transfer knowledge well from having seen the other Atari games.
 * If `r` positive, then we're in the sweet spot and the architecture is successfully learning to play a new Atari game from other games.

We're not quite done though, because really this is just a measure of
how well the architecture did on game `X`. Some games may transfer
knowledge well, and other games may be so unlike other Atari games
that it's hard to transfer much knowledge. What we could do to get
around this is to then do the process above for each game in the
entire collection and average the scores.

This would take a really long time though, so as a compromise, instead
of just holding out one game in the above process, we hold out about
30% of all games as tests, and keep 70% of games for training. We then
do the above process to test, except we create a fresh instance for
each test game, and we save the state of network after it's been
trained on the training set of games. We reset it to that "freshly
trained" state before each test game (so it doesn't learn from the
other testing games). Then we shuffle the training and testing sets up
randomly and do this a few more times from scratch.

As an example, lets say there are five games `S`, `U`, `V`, `X`, and `Y`.

We'll measure the performance of a fresh instance on each of the games
for 10 million frames, getting `F(S)`, `F(U)`, `F(V)`, `F(X)`, and `F(Y)`
(`F` is for "fresh").

Then for the first trial, we'll randomly select `X` and `Y` as the test games.
We'll train a new instance `F` on `S`, `U`, and `V` and save its weights as `F_suv`.
Then we train `F_suv` on `X` for ten million frames, getting `F_suv(X)`.
Then we train `F_suv` on `Y` for ten million frames, getting `F_suv(Y)`.

To get the score for the first trial, we average their ratios:

    r_1 = (F_suv(X)/F(X) + F_suv(Y)/F(Y)) / 2

Now we do a couple more trials, maybe using `S` and `V` as the test
games, then maybe for the third trial `U` and `S` as the tests.

    r_2 = avg(F_uxy(S)/F(S) , F_uxy(V)/F(V))
    r_3 = avg(F_vxy(U)/F(U) , F_vxy(S)/F(S))

Finally, we average the scores from all three trials:

    r = avg(r_1, r_2, r_3)

`r(t)` is the final transfer learning score for the architecture
for each time step, though we may simply use  `r(t_max)` as a summary.

## Multitask learning benchmark

The multitask learning benchmark is most similar to existing
benchmarking that's been done on Atari games, in that we are concerned
with an absolute score on the games. Since absolute scores aren't
comparable across games, we have to keep each game's score separate in
the results rather than aggregating them.

How it works is we once again train a fresh instance of the
architecture for 10 million frames on each game separately, obtaining
baseline scores for each game. These instances we call the
"specialists" since they're trained on only one game a piece.

Then we train an instance of the architecture on every game in random
order, so that the new architecture has seen 10 million frames of
every game. This instance we call the "generalist".

We then compare the generalist's cumulative scores for each frame
against the specialists' scores for the same game and time step. On
the multitask benchmark, we're looking for the generalist to match the
scores of the specialists in the best case. Since the architecture is
the same, the presumption is that the specialist will nearly always
have better performance, and we can only minimize how much the
generalist loses. Though in practice, if the architecture transfers
knowledge well, the generalist may actually outperform the
specialists in some cases.

On the multitask benchmark, we also output the absolute scores for
comparison with other benchmarks etc, though note that it should only
be compared with scores that were obtained under the 10 million frame
limit.

## Note on pre-training

The benchmark doesn't have a strong opinion about how the weights are
initialized in a fresh instance of an architecture. It's reasonable to
not initialize the weights randomly, instead opting to come with some
prior training so that (for example) a deep convolutional network
doesn't waste part of its precious 10 million frames learning to
recognize edges and shapes etc.

The transfer learning benchmark is somewhat robust to this kind of
pre-training since it relies on measuring the amount of improvement in
the architecture before and after it is able to see other games. If a
"fresh" instance already has extensive training on Atari games
beforehand, we should expect this to simply eat into the improvement.

Nevertheless, as a rule of thumb, it's best if a fresh instance of an
architecture does not include any prior training on Atari games or
images from Atari games, to eliminate confusion.
