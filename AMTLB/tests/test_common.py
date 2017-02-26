from functools import partial
from random import randint, sample
from string import uppercase, lowercase
import json
import os.path

import pytest
from mock import MagicMock, patch

import amtlb


@pytest.fixture
def game_names():
    return list(sample(uppercase, randint(14, 26)))

@pytest.fixture(params=[1, 3, 5, 7, 11, 13])
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
def random_benchmark_parms(game_names):
    return partial(amtlb.BenchmarkParms,
        num_folds=randint(0, 11),
        max_rounds_w_no_reward=randint(0, 1000),
        seed=randint(0, 10000),
        game_names=game_names,
    )

@pytest.fixture
def random_filename(tmpdir):
    return os.path.join(str(tmpdir), ''.join(sample(lowercase, 10)))


def folds_equal(A, B):
    _A = {frozenset(a) for a in A}
    _B = {frozenset(b) for b in B}
    return _A == _B

class TestBenchmarkParms(object):

    def test_creates_right_num_folds(self, num_folds, benchmark_parms):
        bp = benchmark_parms()
        assert len(bp.folds) == num_folds

    def test_folds_are_all_close_in_size(
            self, game_names, num_folds, benchmark_parms):
        bp = benchmark_parms()

        fold_div = len(game_names) // num_folds
        fold_rem = len(game_names) % num_folds

        for fold in bp.folds:
            assert len(fold) in [fold_div, fold_div + 1]

    def test_all_games_go_in_a_fold(self, game_names, benchmark_parms):
        bp = benchmark_parms()

        all_games_in_folds = set()
        for fold in bp.folds:
            all_games_in_folds.update(set(fold))
        assert set(game_names) == all_games_in_folds

    def test_save_parms(self, random_benchmark_parms, random_filename):
        bp = random_benchmark_parms()
        bp.save(random_filename)

        with open(random_filename, 'r') as fileobj:
            j = json.load(fileobj)

        assert j.pop('num_folds') == bp.num_folds
        assert j.pop('max_rounds_w_no_reward') == bp.max_rounds_w_no_reward
        assert j.pop('seed') == bp.seed
        assert j.pop('max_rounds_per_game') == bp.max_rounds_per_game
        assert j.pop('game_names') == bp.game_names
        assert folds_equal(j.pop('folds'), bp.folds)
        assert not j

    def test_load_parms(self, random_benchmark_parms, random_filename):
        bp = random_benchmark_parms()
        test_data = {
            "num_folds": bp.num_folds,
            "max_rounds_w_no_reward": bp.max_rounds_w_no_reward,
            "seed": bp.seed,
            "max_rounds_per_game": bp.max_rounds_per_game,
            "game_names": bp.game_names,
            "folds": bp.folds,
        }
        with open(random_filename, 'w') as fileobj:
            json.dump(test_data, fileobj)

        bp2 = amtlb.BenchmarkParms.load_from_file(random_filename)

        assert bp.num_folds == bp2.num_folds
        assert bp.max_rounds_w_no_reward == bp2.max_rounds_w_no_reward
        assert bp.seed == bp2.seed
        assert bp.max_rounds_per_game == bp2.max_rounds_per_game
        assert bp.game_names == bp2.game_names
        assert folds_equal(bp.folds, bp2.folds)
