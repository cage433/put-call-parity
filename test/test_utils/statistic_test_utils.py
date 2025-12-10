from collections.abc import Callable
from typing import Tuple

from process.vector_path import VectorPath


class StatisticalTestUtils:

    @staticmethod
    def check_statistic(sample_generator,
                        statistic_func: Callable[[VectorPath], Tuple[float, float]],
                        expected: float,
                        init_n_samples=2_000,
                        log_on_try: int = 6,
                        n_tries: int = 6):
        has_passed = False
        i_try = 0
        n_samples = init_n_samples
        while not has_passed and i_try < n_tries:
            samples = sample_generator(n_samples)
            observed, tol = statistic_func(samples)
            error = (observed - expected)
            if abs(error) < tol:
                has_passed = True
            if i_try >= log_on_try:
                print(
                    f"i:{i_try}, N:{n_samples} O:{observed:1.3f}, EXP:{expected:1.3f}, ERR:{error:1.3f}, TOL:{tol:1.3f}")
            i_try += 1
            n_samples *= 2
        return has_passed
