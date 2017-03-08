from random import randint, sample
from functools import partial
import os.path
from string import lowercase

import pytest
import mock

import amtlb


@pytest.fixture
def game_names():
    prospects = [
        'air_raid', 'alien', 'amidar', 'assault', 'asterix',
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
        'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon'
    ]
    return sample(prospects, randint(13, len(prospects)))


@pytest.fixture
def num_games(game_names):
    return len(game_names)


@pytest.fixture(params=[3, 5, 7, 11, 13])
def num_folds(request):
    return request.param

@pytest.fixture
def benchmark_parms(game_names, num_folds):
    return partial(
        amtlb.BenchmarkParms,
        game_names=game_names,
        num_folds=num_folds,
    )

@pytest.fixture
def bp(benchmark_parms):
    return benchmark_parms()


@pytest.fixture
def random_benchmark_parms(game_names):
    return partial(
        amtlb.BenchmarkParms,
        num_folds=randint(2, 11),
        max_rounds_w_no_reward=randint(0, 1000),
        seed=randint(0, 10000),
        game_names=game_names,
        game_version=randint(0, 3),
    )

@pytest.fixture
def random_filename(tmpdir):
    return os.path.join(str(tmpdir), ''.join(sample(lowercase, 10)))


@pytest.fixture
def MockAgent():
    return mock.Mock(spec=amtlb.Agent)


@pytest.fixture
def transfer_benchmark(MockAgent, bp):
    return partial(
        amtlb.TransferBenchmark,
        parms=bp,
        AgentClass=MockAgent,
    )


@pytest.fixture
def tb(transfer_benchmark):
    return transfer_benchmark()


def folds_equal(A, B):
    _A = {frozenset(a) for a in A}
    _B = {frozenset(b) for b in B}
    return _A == _B
