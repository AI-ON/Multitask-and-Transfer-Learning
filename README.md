# Multitask and Transfer Learning

* Benchmark and build RL architectures that can do multitask and transfer learning.
* Date: December 2016
* Category: Fundamental Research
* Contact: deontologician@gmail.com
* [![Join the chat at https://gitter.im/ai-open-network/multitask_and_transfer_learning](https://badges.gitter.im/Join%20Chat.svg)][gitter]
## Problem description

1. Create a benchmark for transfer learning and multitask learning.
    * Should measure improvement in learning that is directly attributable to knowledge transfer between games.
    * Should also be able to measure performance by a single agent on multiple games.
    * Should use cross-validation to mitigate the effects of a small number of games to test on.
2. Design and implement deep reinforcement learning architectures that do well on the benchmark.
    * For methodological reasons, we think it's important to design the ideal benchmark before getting too attached to a particular architecture.
    * It's important that we're sure the benchmark is measuring the crux of the transfer and multi-task problem rather than measuring something our architecture is good at.

## Contributing

We have a few different "threads" going on right now, so there are several different ways you can get involved if you're interested:
* [Reviewing relevant papers and summarizing for the community][papers project]
  - There are lots of papers coming out in this area and it's hard to keep on top of all of them!
  - Check out the [review rubric]
* [Replicating existing RL architectures on the benchmarks][replication project]
  - If you're new to machine learning / RL, this might be a good place to start, since you're reimplementing existing architectures.
  - We want to see what numbers current state-of-the-art architectures get on the benchmark, so this is important for measuring how well we're doing.
* [Design & implement novel architectures][architectures project]
  - This is the ultimate goal of the project. We're still in the preliminary stages, but if you have an idea you're itching to try, feel free to dive in.
  - We're always game to chat with you about ideas in the [gitter channel][gitter]
* [Building and testing the benchmarks][benchmark project]
  - This is mainly python coding, rather than ML, but if you're interested, contact @deontologician
* If none of these sound appealing, but you still feel like you want to get involved somehow, come chat with us in the [gitter channel][gitter]

[architectures project]: https://github.com/AI-ON/Multitask-and-Transfer-Learning/projects/5
[benchmark project]: https://github.com/AI-ON/Multitask-and-Transfer-Learning/projects/1
[replication project]: https://github.com/AI-ON/Multitask-and-Transfer-Learning/projects/3
[papers project]: https://github.com/AI-ON/Multitask-and-Transfer-Learning/projects/4
[review rubric]: ./paper-reviews.md

### A few notes on contributing

* **Be Kind** and **Be Respectful**
* Value other people's work: please reference them. This also helps other people in the project find valuable prior work. Don't just copy & paste what you find elsewhere when it comes to sharing information.
* Give constructive criticism. If you see something not working or wrong, [open an issue][new issue], or bring it up in the [chat][gitter]. Any ideas, practices, project direction or code are all open to any criticism. Try to avoid criticizing people or making things personal, but feel free to criticize code, ideas, project direction constructively. If you come with a proposed solution in hand, all the better!
* Please Ask Questions! An important part of this project is to open up the opportunity for everyone to contribute. We want anyone who wants to to be able to add value towards these research topics.
* Keep in mind that most of the researchers that are opening these projects have full-time work/research. If there is a specific question, use the [gitter channel][gitter] or [open an issue][new issue] rather than directly emailing them.

[gitter]: https://gitter.im/ai-open-network/multitask_and_transfer_learning
[issues]: https://github.com/AI-ON/Multitask-and-Transfer-Learning/issues
[new issue]: https://github.com/AI-ON/Multitask-and-Transfer-Learning/issues/new

## Project Status:

**See detailed status on the [project tracker](https://github.com/AI-ON/Multitask-and-Transfer-Learning/projects)**

* Currently writing and testing the benchmarks for measuring performance.
* Looking for people to review papers trawling for ideas, and to implement some existing architectures to benchmark their performance.
* Check the README in [AMTLB] directory to learn more about the tool/ library used.
  - *"This is a library to test how a reinforcement learning architecture performs on all Atari games in OpenAI's gym. It performs two kinds of tests, one for transfer learning and one for multitask learning."*

[AMTLB]: ./AMTLB

## Why this problem matters:

Generalizing across tasks is a crucial component of human intelligence. Current deep RL architectures get less effective the more tasks they are put to, whereas for humans, diversity of experience is a strength that improves performance on new tasks. Overcoming catastrophic forgetting and achieving one-shot learning are abilities that should fall out naturally if this task is solved convincingly.

At a more meta-level, this problem is both out of reach of current reinforcement learning architectures, but it seems reasonably within reach within a year or two. Much like ImageNet spurred innovation by creating a common target for researchers to aim for, this project could similarly provide a common idea of success for multitask and transfer learning. Many papers researching multi-task and transfer learning using Atari are doing it in ad-hoc ways that cherry-pick games that get good results.

## How to measure success:

Success is in degrees, since an architecture (in principle) could surpass human
ability in multi-task Atari, getting both higher scores on all games, and
picking up new games faster than a human does. Ideally, a good waterline would
be human level performance on the benchmark, but creating a robust dataset on
human performance is beyond the scope of this project.

The fundamental benchmark then will be two measures:

1. **Transfer Learning**: How much a given architecture improves on an unseen
game when it is untrained versus when it has been trained on other games firest.
Measured as a ratio of total score pre-trained vs. untrained. Ratio is averaged using
cross-validation given that there is a small number of available games and the
fact that high scores are not comparable across games.
2. **Multitask Learning**: How well a given architecture does across all games
with a single architecture and set of weights. Rather than an aggregate, this
result will be a vector of top scores achieved for each game.

In addition to the scores, the benchmark will also make some strict demands on
the architecture itself due to the testing/training regime:

- Training will happen on random games sequentially. After each loss a new
  random game from the training set will be selected to play next.
- **No out of band signal** will be given to indicate which game is being played, so
  architectures that need to allocate a set of extra weights for each game will
  have to be more clever.
- All games in ALE will be used, **even ones which standard DQNs perform poorly on**
  like Montezuma’s Revenge.

## Datasets:

Currently no datasets, but it’s possible the dataset being created at
[atarigrandchallenge.com](http://atarigrandchallenge.com/) will potentially be a useful comparison once
it’s available. Measuring human performance needs to be done with a large sample
size, both to control for pre-training (some people have played Atari games
before, or other video games before) and to control for individual human skill
levels (this could be seen as pre-training on non-Atari games, generalization
from real life, or natural ability etc).

Akin to a dataset will be the benchmark framework itself. Since this is a
reinforcement learning problem, the testing environment provides the data,
rather than a static dataset.

## Relevant/Related Work

Since the original Mnih paper, the Atari 2600 environment has been a popular
target for testing out RL architectures

- [Overcoming Catastrophic forgetting in neural networks [Dec 2016]](https://arxiv.org/abs/1612.00796)
    - Uses a technique they call “elastic weight consolidation” to dynamically adjust learning speed for each weight, avoiding catastrophic forgetting.
    - Chooses a subset of games to play
    - Avoids games like Montezuma’s Revenge that performed poorly in the original Atari paper.

- [Progressive Neural networks [June 2016]](https://arxiv.org/abs/1606.04671v3)
    - Addresses both multitask and transfer learning on a small number of games through freezing weights for a previous game when learning a new game.
    - Number of weights grows with the number of tasks, a very hands-on approach

- [Actor-Mimic: Deep Multitask and Transfer Reinforcement Learning [Nov 2015]](https://arxiv.org/abs/1511.06342v4)
    - Trains a common network, along with an expert network for each game.
    - Uses the transfer learning benchmark of random initialization vs. pre-trained on other games.
    - Trained mostly on games that DQNs perform well on, though Seaquest was included.

- [Modular Multitask Reinforcement Learning with Policy Sketches [Nov 2016]](https://arxiv.org/abs/1611.01796v1)
    - Uses as extra input a list of high level actions which need to be accomplished to complete a task. The network must learn to use this signal to create heirarchical representations.
    - Uses two custom test environments for which their approach is more amenable.

- [Human-level control through deep reinforcement [2015]](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
    - Original DeepMind Atari paper
        - [Related workshop paper from 2013](https://arxiv.org/abs/1312.5602v1)
    - No multi-task or transfer learning attempted, but has some reasonable baselines for human performance on the games (which are then re-used in many subsequent papers)

- [Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397v1)

*Note: More Work to be added to, always check the chat for latest related work for now*


