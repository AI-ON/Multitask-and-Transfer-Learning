from functools import partial

import pytest
import mock

import amtlb.common
import amtlb.transfer_benchmark as mut  # module under test

from conftest import folds_equal

class TestTransferBenchmark(object):
    def test_test_set_returns_correct_fold(self, tb, bp, num_folds):
        '''Returns the correct fold for a given index'''
        for i in range(0, num_folds):
            obtained = tb.test_set(i)
            expected = bp.folds[i]
            assert folds_equal(tb.test_set(i), bp.folds[i])

    def test_test_set_returns_no_entries_from_training_set(self, tb, num_folds):
        '''Returns no entries from the training set.

        A sanity check about the relationship between test_set and
        training_set
        '''
        empty_set = set()
        for i in range(0, num_folds):
            test_set = tb.test_set(i)
            training_set = tb.training_set(i)
            assert test_set.intersection(training_set) == empty_set

    def test_training_set_returns_all_but_test_set(self, tb, bp, num_folds):
        '''Returns all but test set.

        Sanity check that test games + training_games == all games
        '''
        all_games = set(bp.game_identifiers)
        for i in range(0, num_folds):
            test_games = tb.test_set(i)
            training_games = tb.training_set(i)
            assert test_games.union(training_games) == all_games

    def test_do_folds_all_fold_agents_get_trained(self):
        '''All fold_agents need to be trained'''
        pass
 # .do_folds - high priority
  # all fold_agents get trained
  # fold agents are saved after training
  # a fold agent gets tested on each test game
  # a fold agent results on a game are saved
  # the "tested_agent' is saved
 # .ensure_game_agents - medium priority
  # if there's no agent for a game it makes one
  # if there is an agent, it does nothing
 # .default_dir  - low priority
  # check that it returns something matching a regex
 # .game_agent_filename  - low priority
  # check with a regex
 # .fold_agent_filename  - low priority
  # check with a regex
 # .tested_agent_filename - low priority
  # check with a regex
 # .test - low priority
  # returns a TestRun
 # .train  - low priority
  # returns a TrainingRun


# TestRun
 # When the game is done, reset is called
 # Respects the max rounds per game
 # Maps invalid actions for a game to 0
 # records every reward
 # gives action from the agent to the game.step method


# TrainingRun
 # max_rounds_w_no_reward is respected
 # max_rounds_per_game is respected
 # .total_rounds_left - low priority
   # Ensure it sums up the right property from self.game_rounds_left
 # .sample_env - High priority
   # create a fake distribution
     # [0 3 0], ensure index 1 is selected
     # [2,0, 0], ensure index 0 is selected
     # [0, 0, 3], ensure index 3 is selected
     # [1, 2, 0], ensure runs are proportional
   # ensure returns game name and an environment for the game
