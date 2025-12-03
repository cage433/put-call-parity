from numbers import Number
from typing import Tuple, Optional
from unittest import TestCase

import numpy as np
from numpy import ndarray

from process.vector_path_builder import BrownianPathBuilder
from process.vectorpath import VectorPath
from utils.random_number_generator import RandomNumberGenerator
from utils.random_test_case import RandomisedTest
from utils.statistic_test_utils import StatisticalTestUtils


class TestVectorPath(TestCase):

    def _random_brownian_builder(self, rng: RandomNumberGenerator,
                                 n_factors: Optional[int] = None) -> BrownianPathBuilder:
        n_times = rng.randint(1, 10)
        times = rng.random_times(n_times)
        n_factors = n_factors or rng.randint(1, 4)
        return BrownianPathBuilder(times, n_factors)

    @RandomisedTest(number_of_runs=10)
    def test_brownian_mean(self, rng: RandomNumberGenerator):
        bldr = self._random_brownian_builder(rng)
        i_time = rng.randint(len(bldr.times))
        T = bldr.times[i_time]
        i_factor = rng.randint(bldr.n_factors)

        def sample(brownians: VectorPath) -> ndarray:
            return brownians.factor_values(i_factor, i_time)

        def check_stat(msg, statistic_func, expected_value):
            self.assertTrue(
                StatisticalTestUtils.check_statistic(
                    lambda n_samples: bldr.build(rng, n_samples),
                    statistic_func,
                    expected=expected_value,
                ),
                msg
            )

        check_stat(
            "Mean should be 0",
            lambda br: (sample(br).mean(), 0.01),
            0.0
        )
        check_stat(
            f"Std dev at time {T:1.3f} should be {np.sqrt(T):1.3f}",
            lambda br: (sample(br).std(), 0.01),
            np.sqrt(T)
        )

    @RandomisedTest(number_of_runs=10)
    def test_brownian_correlations(self, rng: RandomNumberGenerator):
        num_factors = rng.randint(2, 4)
        bldr = self._random_brownian_builder(rng, num_factors)
        factor_1, factor_2 = rng.shuffle(list(range(num_factors)))[:2]
        i_time = rng.randint(bldr.n_times)

        def sample_rho(brownians: VectorPath) -> Tuple[float, float]:
            tol = 0.01
            path_1 = brownians.factor_values(factor_1, i_time)
            path_2 = brownians.factor_values(factor_2, i_time)
            rho_matrix: np.ndarray = np.corrcoef(path_1, path_2)
            return rho_matrix[0, 1], tol

        def check_stat(msg, statistic_func, expected_value):
            self.assertTrue(
                StatisticalTestUtils.check_statistic(
                    lambda n_samples: bldr.build(rng, n_samples),
                    statistic_func,
                    expected=expected_value,
                ),
                msg
            )

        check_stat("Rho should be 0", sample_rho, 0.0)
