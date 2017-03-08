import pytest

import amtlb
import amtlb.common


# TestRun
 # When the game is done, reset is called
 # Respects the max rounds per game
 # Maps invalid actions for a game to 0
 # records every reward
 # gives action from the agent to the game.step method

@pytest.fixture
def agent():
    return mock.Mock(spec=amtlb.Agent)

@pytest.fixture
def transfer_benchmark(agent):
    return TransferBenchmark

@pytest.fixture
def parms():
    return mock.Mock(spec=amtlb.BenchmarkParms)()

class TransferBenchmark(object):
    def test_test_set_returns_correct_fold(mock_transfer_benchmark):

 # .test_set - high priority
  # returns the correct fold
  # returns no entries from the training set
 # .training_set - high priority
  # returns all entries except from the test set
 # .default_dir  - low priority
  # check that it returns something matching a regex
 # .game_agent_filename  - low priority
  # check with a regex
 # .fold_agent_filename  - low priority
  # check with a regex
 # .tested_agent_filename - low priority
  # check with a regex
 # .ensure_game_agents - medium priority
  # if there's no agent for a game it makes one
  # if there is an agent, it does nothing
 # .test - low priority
  # returns a TestRun
 # .train  - low priority
  # returns a TrainingRun
 # .do_folds - high priority
  # all fold_agents get trained
  # fold agents are saved after training
  # a fold agent gets tested on each test game
  # a fold agent results on a game are saved
  # the "tested_agent' is saved

# TrainingRun
 # .total_rounds_left - low priority
   # Ensure it sums up the right property from self.game_rounds_left
 # .sample_env - High priority
   # create a fake distribution
     # [0 3 0], ensure index 1 is selected
     # [2,0, 0], ensure index 0 is selected
     # [0, 0, 3], ensure index 3 is selected
     # [1, 2, 0], ensure runs are proportional
   # ensure returns game name and an environment for the game
