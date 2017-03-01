import random
import json
from abc import ABCMeta, abstractmethod


GAMES = ['air_raid', 'alien', 'amidar', 'assault', 'asterix',
         'asteroids', 'atlantis', 'bank_heist', 'battle_zone',
         'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout',
         'carnival', 'centipede', 'chopper_command', 'crazy_climber',
         'demon_attack', 'double_dunk', 'elevator_action', 'enduro',
         'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
         'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull',
         'kung_fu_master', 'montezuma_revenge', 'ms_pacman',
         'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
         'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank',
         'seaquest', 'skiing', 'solaris', 'space_invaders', 'star_gunner',
         'tennis', 'time_pilot', 'tutankham', 'up_n_down', 'venture',
         'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']


def _to_camel(name):
    return ''.join([n.capitalize() for n in name.split('_')])


def _add_vrsn(name, version):
    return name + '-v' + str(version)


def to_identifier(game_name, version=0):
    '''Returns game identifiers that can be used by the gym library to
    actually load the correct environment.
    '''
    return _add_vrsn(_to_camel(game_name), version)


class Agent(object):

    __metaclass__ = ABCMeta  # ensures subclasses implement everything

    @abstractmethod
    def __call__(self, observation, reward):
        '''Called every time a new observation is available.
        Should return an integer from 0 to 17 inclusive
        '''

    @abstractmethod
    def clone(self):
        '''Returns a deep copy of the agent and its weights.'''

    @classmethod
    @abstractmethod
    def load(cls, filename):
        '''Loads an agent (with weights) from a filename'''

    @abstractmethod
    def save(self, filename):
        '''Saves the agent (with weights) to a filename'''


class RandomAgent(Agent):
    '''Simple random agent that has no state'''

    def __call__(self, observation, reward):
        # The benchmark maps invalid actions to No-op (action 0)
        return random.randint(0, 17)

    def clone(self):
        return self  # RandomAgent has no state

    @classmethod
    def load(cls, filename):
        return cls()

    def save(self, filename):
        pass


class BenchmarkParms(object):
    def __init__(self,
                 num_folds=5,
                 max_rounds_w_no_reward=10000,
                 seed=None,
                 max_rounds_per_game=100000,
                 game_names=GAMES,
                 game_version=0,
                 ):
        self.num_folds = num_folds
        self.max_rounds_w_no_reward = max_rounds_w_no_reward
        self.seed = random.randint(0, 2**64-1) if seed is None else seed
        self.max_rounds_per_game = max_rounds_per_game
        self.game_version = game_version
        self.games = {game: to_identifier(game, game_version)
                      for game in game_names}

        num_games = len(self.games)

        games = set(self.game_identifiers)
        fold_size = num_games // num_folds
        remainder = num_games % num_folds
        self.folds = [None] * num_folds

        for i in range(num_folds):
            if i < remainder:
                # distribute the remainder games evenly among the folds
                self.folds[i] = random.sample(games, fold_size + 1)
            else:
                self.folds[i] = random.sample(games, fold_size)
            games -= set(self.folds[i])

        assert(len(games) == 0)

    def save(self, filename=None):
        '''Save the TestPlan to a file'''
        filedata = {
            'num_folds': self.num_folds,
            'folds': self.folds,
            'seed': self.seed,
            'max_rounds_w_no_reward': self.max_rounds_w_no_reward,
            'max_rounds_per_game': self.max_rounds_per_game,
            'games': self.games,
            'game_version': self.game_version,
        }
        with open(filename, 'w') as savefile:
            json.dump(filedata, savefile, sort_keys=True, indent=True)

    @staticmethod
    def load_from_file(filename):
        '''Load a BenchmarkParms from a file'''
        with open(filename, 'r') as savefile:
            filedata = json.load(savefile)
            parms = BenchmarkParms()
            # Just overwrite the original fields. A little wasteful but w/e
            parms.folds = filedata['folds']
            parms.num_folds = len(parms.folds)
            parms.max_rounds_w_no_reward = filedata['max_rounds_w_no_reward']
            parms.seed = filedata['seed']
            parms.max_rounds_per_game = filedata['max_rounds_per_game']
            parms.games = filedata['games']
            parms.game_version = filedata['game_version']
        return parms

    @property
    def game_names(self):
        return self.games.keys()

    @property
    def game_identifiers(self):
        return self.games.values()
