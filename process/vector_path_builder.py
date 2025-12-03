from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray

from process.vectorpath import VectorPath
from utils.random_number_generator import RandomNumberGenerator
from utils.utils import checked_type


class VectorPathBuilder(ABC):
    @abstractmethod
    def build(self, rng: RandomNumberGenerator, n_paths: int):
        raise ValueError("implement 'build'")


class BrownianPathBuilder(VectorPathBuilder):
    def __init__(self, times: ndarray, n_factors: int):
        self.times: ndarray = checked_type(times, ndarray)
        self.n_factors: int = checked_type(n_factors, int)

    @property
    def n_times(self) -> int:
        return len(self.times)

    def build(self, rng: RandomNumberGenerator, n_paths: int):
        num_times = len(self.times)
        time_steps = [self.times[0]] + [self.times[i + 1] - self.times[i] for i in range(num_times - 1)]

        dZ = np.stack(
            [rng.normal(size=(self.n_factors, n_paths)) * np.sqrt(dt) for dt in time_steps],
            axis=1
        )
        Z = np.zeros(shape=(self.n_factors, num_times, n_paths))
        Z[:, 0, :] = dZ[:, 0, :]
        for i_t in range(1, num_times):
            Z[:, i_t, :] = Z[:, i_t - 1, :] + dZ[:, i_t, :]
        return VectorPath(self.times, Z)