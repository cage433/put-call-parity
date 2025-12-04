from typing import Tuple, Optional
from unittest import TestCase

import numpy as np
from numpy import ndarray

from process.vector_path_builder import BrownianPathBuilder, CorrelatedNormalPathsBuilder, LognormalPathsBuilder
from process.vector_path import VectorPath
from test_utils.random_correlation_matrix import RandomCorrelationMatrix
from utils.random_number_generator import RandomNumberGenerator
from test_utils.random_test_case import RandomisedTest
from test_utils.statistic_test_utils import StatisticalTestUtils


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

    def _random_correlated_brownian_builder(self, rng: RandomNumberGenerator,
                                            n_factors: Optional[int] = None) -> CorrelatedNormalPathsBuilder:
        n_times = rng.randint(1, 10)
        times = rng.random_times(n_times)
        n_factors = n_factors or rng.randint(1, 4)
        rho_matrix = RandomCorrelationMatrix.truly_random(rng, n_factors)
        return CorrelatedNormalPathsBuilder(times, rho_matrix)

    @RandomisedTest(number_of_runs=10)
    def test_correlated_brownians(self, rng: RandomNumberGenerator):
        num_factors = rng.randint(2, 4)
        bldr = self._random_correlated_brownian_builder(rng, num_factors)
        factor_1, factor_2 = rng.shuffle(list(range(num_factors)))[:2]
        i_time = rng.randint(bldr.n_times)
        expected_rho = bldr.rho_matrix[factor_1][factor_2]

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

        check_stat(f"Rho should be {expected_rho:1.3f}", sample_rho, expected_rho)

    @RandomisedTest(number_of_runs=3)
    def test_lognormals(self, rng):
        n_factors = rng.randint(2, 4)
        Fs = [rng.uniform(80, 120) for _ in range(n_factors)]
        n_times = rng.randint(1, 10)
        Ts = rng.random_times(n_times)
        vols = np.asarray([rng.uniform(0.5) for _ in range(n_factors)])
        drifts = np.einsum("f,f->f", vols, vols) * -0.5
        times = np.asarray(Ts)
        rho_matrix = RandomCorrelationMatrix.truly_random(rng, n_factors)
        bldr = LognormalPathsBuilder(
            prices = np.asarray(Fs),
            times = np.asarray(times),
            rho_matrix=rho_matrix,
            drifts=drifts,
            vols = vols
        )

        i_factor = rng.randint(n_factors)
        j_factor = rng.randint(n_factors)
        i_time = rng.randint(n_times)
        def check_stat(msg, statistic_func, expected_value):
            self.assertTrue(
                StatisticalTestUtils.check_statistic(
                    lambda n_samples: bldr.build(rng, n_samples),
                    statistic_func,
                    init_n_samples=10_000,
                    expected=expected_value,
                    n_tries=10,
                    log_on_try=7
                ),
                msg,
            )

        def sample_mean(vector_path: VectorPath) -> Tuple[float, float]:
            tol = Fs[i_factor] * 1e-3
            path = vector_path.factor_values(i_factor=i_factor, i_time=i_time)
            return path.mean(), tol

        def sample_vol(vector_path: VectorPath) -> Tuple[float, float]:
            tol = vols[i_factor] * 1e-3
            path = vector_path.factor_values(i_factor=i_factor, i_time=i_time)
            observed_vol = np.log(path).std() / np.sqrt(times[i_time])
            return observed_vol, tol

        def sample_rho(vector_path: VectorPath) -> Tuple[float, float]:
            tol = 0.02
            path_i = vector_path.factor_values(i_factor=i_factor, i_time=i_time)
            path_j = vector_path.factor_values(i_factor=j_factor, i_time=i_time)
            observed_rho = np.corrcoef(np.log(path_i), np.log(path_j))[0][1]
            return observed_rho, tol
        check_stat(f"Forward prices should be {Fs[i_factor]:1.3f}", sample_mean, Fs[i_factor])
        check_stat(f"Vols should be {vols[i_factor]:1.3f}", sample_vol, vols[i_factor])
        check_stat(f"Rho should be {rho_matrix[i_factor, j_factor]:1.3f}", sample_rho, rho_matrix[i_factor, j_factor])

