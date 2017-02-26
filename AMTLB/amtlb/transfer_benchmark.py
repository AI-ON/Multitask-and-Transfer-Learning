from __future__ import division

import random
import json
from math import ceil
import string
from datetime import datetime
from collections import defaultdict

import gym

from common import Agent, RandomAgent, GAME_NAMES, NUM_GAMES, BenchmarkParms


def fold_name(num):
    name = string.ascii_uppercase[num % 26]
    if num > 25:
        name += str(num - 25)
    return name


class BenchmarkResult(object):
    def __init__(self, agent, game=None):
        self.agent = agent
        self.rewards = []  # list of int rewards, index is round #
        self.dones = []  # list of rounds where the agent died
        self.games = []  # list of what games were played each round
        if game is not None:
            self.record_game(game, 0)
        # TODO: make serializable

    def record_reward(self, reward):
        self.rewards.append(reward)

    def record_done(self, round):
        self.dones.append(round)

    def record_game(self, game_name, round):
        self.games.append((game_name, round))


class EnvMaker(object):
    '''Mixin class'''
    def create_env(self, game_name):
        env = gym.make(game_name)
        # Ensure all being tested on this game get the same seed to
        # reduce variability. Crucially, this means an agent can't see
        # the same environment more than once!
        env.seed(self.parms.seed)
        return env


class TestRun(EnvMaker):
    def __init__(self, agent, game_name, parms):
        self.agent = agent
        self.parms = parms
        self.game = self.create_env(game_name)
        self.result = BenchmarkResult(agent, game_name)

    def __call__(self):
        '''Test an agent on the given game'''
        observation = self.game.reset()
        reward = 0
        for round_num in xrange(self.parms.max_rounds_per_game):
            action = self.agent(observation, reward)
            if action >= self.game.action_space.n:
                action = 0  # Map invalid actions to no-op
            observation, reward, done, _ = self.game.step(action)
            self.result.record_reward(reward)
            if done:
                observation = self.game.reset()
                self.result.record_done(round_num)
        return self.result


class TransferBenchmark(object):
    '''Benchmark for testing knowledge transfer.

    Uses k-fold cross-validation to test an agent's performance on a
    new game. Each fold is a set of games held out from the training
    set.  All other folds except that one are the training
    set. Performance is compared as a ratio between the cumulative
    score over time of a fresh agent vs. the cumulative score over
    time of an agent trained on the training set. Both agents will be
    seeing the game in the test set for the first time, but one of the
    agents will have preparation, and the other will not. The ratio
    measures how much that preparation helps.

    For the purposes of this class, the term `game_agent` is used to
    denote a fresh agent who has no preparation, but has trained on a
    particular game. Game agents are identified by the name of their
    game. A `fold_agent` is an agent who has been trained up on all
    the games in the training set, i.e. everything but the test game
    fold.  The folds are indexed by an integer, so the fold_agents are
    also indexed by an integer indicating which fold is their test
    set (all other folds are implicitly their training set).

    In practice, we only care about the results of the game agents,
    and don't need to keep the trained agent around once we have their
    scores, since they are intended to be fresh. We do need to keep
    around the fold agents' trained agent since we should start from
    the same baseline on each test game. In other words, we shouldn't
    allow the agent to train on the other test games first, since then
    results would be dependent on the order in which we did tested
    games in the fold. So we checkpoint the agent at the time it
    finishes all of its training, and reset to that point before
    testing on each game in the test fold.

    '''
    def __init__(self, parms, AgentClass, dir=None):
        self.parms = parms  # BenchmarkParms
        self.AgentClass = AgentClass
        self.untrained_agent = AgentClass()
        self.game_agents = {}  # indexed by game name, since 1 per game
        self.game_results = {}  # benchmarks for each game agent
        self.fold_agents = []  # indexed by fold number
        self.fold_results = []  # Each fold gets a result for each game
        self.dir = dir or self.default_dir()
        # TODO: make dir recursively
        # TODO: add checkpoints, benchmark is ephemeral now

    def test_set(self, test_index):
        '''Copy the test set of game names from the BenchmarkParms'''
        return set(self.parms.folds[test_index])

    def training_set(self, test_index):
        '''Aggregate the training set from the current BenchmarParms'''
        return {x
                for i, fold in enumerate(self.parms.folds)
                for x in fold if i != test_index}

    def default_dir(self):
        '''A reasonable default directory to store benchmark results in'''
        time = datetime.now().strftime('%Y-%m-%d_%H.%M')
        return 'benchmarks/' + time + '/'

    def game_agent_filename(self, game_name):
        '''Constructs a save filename for a game agent'''
        return self.dir + 'game_agent_' + game_name

    def fold_agent_filename(self, fold_num):
        '''Constructs a save filename for a fold agent'''
        return self.dir + 'fold_agent_' + str(fold_num)

    def tested_agent_filename(self, fold_num, game_name):
        '''Constructs a filename for an fold agent tested on a particular
        game'''
        return self.dir + 'tested_agent_' + str(fold_num) + '_' + game_name

    def ensure_game_agents(self):
        '''Ensures there is a game agent for every game.

        Game agents don't care about folds, and this function checks
        if an agent already exists, so there's no harm in running this
        function multiple times.
        '''
        for game_name in GAME_NAMES:
            if game_name not in self.game_agents:
                game_agent = self.untrained_agent.clone()
                self.game_results[game_name] = self.test(game_agent, game_name)
                game_agent.save(self.game_agent_filename(game_name))
                self.game_agents[game_name] = game_agent

    def test(self, game_agent, game_name):
        return TestRun(game_agent, game_name, self.parms)()

    def train(self, fold_agent, training_set):
        return TrainingRun(fold_agent, training_set, self.parms)()


    def do_folds(self):
        '''Runs through each fold, and trains an agent for each set.

        The outer loop of this function should run k times.
        '''
        self.ensure_game_agents()

        for fold_num in range(len(self.parms.folds)):
            training_set = self.training_set(fold_num)
            fold_agent = self.untrained_agent.clone()
            self.fold_agents[fold_num] = fold_agent
            self.train(fold_agent, training_set)
            fold_agent.save(self.fold_agent_filename(fold_num))

            test_set = self.test_set(fold_num)
            self.fold_results[fold_num] = fold_results = {}

            for game_name in test_set:
                tested_agent = fold_agent.clone()
                fold_results[game_name] = self.test(tested_agent, game_name)
                tested_agent.save(
                    self.tested_agent_filename(fold_num, game_name))


class TrainingRun(EnvMaker):
    def __init__(self, agent, training_set, parms):
        self.agent = agent
        self.training_set = training_set
        self.parms = parms
        self.envs = defaultdict(self.create_env)
        self.game_rounds_left = {game: self.max_test_game_rounds
                                 for game in training_set}
        self.trace_result = BenchmarkResult(agent)

    def total_rounds_left(self):
        return sum(self.game_rounds_left.values())

    def sample_env(self):
        '''Samples from remaining games in the training set, inversely
        proportional to how much they've already been played. It has
        the added benefit of not sampling games that have already
        exhausted their round quota. It also doesn't care what order
        the items of self.game_rounds_left arrive in.
        '''

        total_rounds_left = self.total_rounds_left()
        threshold = random.randint(0, total_rounds_left)
        cumulative_sum = 0
        for game, rounds_left in self.game_rounds_left.items():
            cumulative_sum += rounds_left
            if cumulative_sum >= threshold:
                break
        return game, self.envs[game]  # Will lazily initialize env

    def keep_playing(self, game_name, done, no_reward_turns):
        return

    def __call__(self):
        round = 0
        while self.total_rounds_left() > 0:
            game_name, game = self.sample_env()
            self.result.record_game(game_name, round)
            observation = game.reset()
            reward = 0
            done = False
            no_reward_turns = 0
            while (self.game_rounds_left[game_name] > 0 and
                   not done and
                   no_reward_turns < self.parms.max_rounds_w_no_reward):
                round += 1
                action = self.agent(observation, reward)
                if action >= game.action_space.n:
                    action = 0  # Map invalid actions to no-op
                observation, reward, done, _ = game.step(action)
                self.result.record_reward(reward)
                no_reward_turns = 0 if reward > 0 else no_reward_turns + 1
                self.game_rounds_left[game_name] -= 1
            self.result.record_done(round)
        return self.result
