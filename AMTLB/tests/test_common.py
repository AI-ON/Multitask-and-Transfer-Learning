from functools import partial
from random import randint, sample
from string import uppercase, lowercase
import json
import os.path

import pytest
from mock import MagicMock, patch
import amtlb

from conftest import folds_equal


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
        assert set(map(amtlb.to_identifier, game_names)) == all_games_in_folds

    def test_save_parms(self, random_benchmark_parms, random_filename):
        bp = random_benchmark_parms()
        bp.save(random_filename)

        with open(random_filename, 'r') as fileobj:
            j = json.load(fileobj)

        assert j.pop('num_folds') == bp.num_folds
        assert j.pop('max_rounds_w_no_reward') == bp.max_rounds_w_no_reward
        assert j.pop('seed') == bp.seed
        assert j.pop('max_rounds_per_game') == bp.max_rounds_per_game
        assert j.pop('games') == bp.games
        assert j.pop('game_version') == bp.game_version
        assert folds_equal(j.pop('folds'), bp.folds)
        assert not j

    def test_load_parms(self, random_benchmark_parms, random_filename):
        bp = random_benchmark_parms()
        test_data = {
            "num_folds": bp.num_folds,
            "max_rounds_w_no_reward": bp.max_rounds_w_no_reward,
            "seed": bp.seed,
            "max_rounds_per_game": bp.max_rounds_per_game,
            "games": bp.games,
            "game_version": bp.game_version,
            "folds": bp.folds,
        }
        with open(random_filename, 'w') as fileobj:
            json.dump(test_data, fileobj)

        bp2 = amtlb.BenchmarkParms.load_from_file(random_filename)

        assert bp.num_folds == bp2.num_folds
        assert bp.max_rounds_w_no_reward == bp2.max_rounds_w_no_reward
        assert bp.seed == bp2.seed
        assert bp.max_rounds_per_game == bp2.max_rounds_per_game
        assert sorted(bp.game_names) == sorted(bp2.game_names)
        assert sorted(bp.game_identifiers) == sorted(bp2.game_identifiers)
        assert folds_equal(bp.folds, bp2.folds)


def test_to_identifier():
    expected = 'FooGameBar-v3'
    result = amtlb.to_identifier('foo_game_bar', version=3)
    assert result == expected
