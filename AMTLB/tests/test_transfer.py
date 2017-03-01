import amtlb


# TestRun
 # When the game is done, reset is called
 # Respects the max rounds per game
 # Maps invalid actions for a game to 0
 # records every reward
 # gives action from the agent to the game.step method

# TransferBenchmark
 # .test_set
  # returns the correct fold
  # returns no entries from the training set
 # .training_set
  # returns all entries except from the test set
 # .default_dir
  # check that it returns something matching a regex
 # .game_agent_filename
  # check with a regex
 # .fold_agent_filename
  # check with a regex
 # .tested_agent_filename
  # check with a regex
 # .ensure_game_agents
  #
