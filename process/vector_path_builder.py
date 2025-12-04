from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray
from numpy.linalg import svd

from process.vectorpath import VectorPath
from utils.random_number_generator import RandomNumberGenerator
from utils.utils import checked_type


class VectorPathBuilder(ABC):
    def __init__(self, times: ndarray, n_factors: int):
        self.times: ndarray = checked_type(times, ndarray)
        self.n_factors: int = checked_type(n_factors, int)

    @property
    def n_times(self) -> int:
        return len(self.times)

    @abstractmethod
    def build(self, rng: RandomNumberGenerator, n_paths: int):
        raise ValueError("implement 'build'")


class BrownianPathBuilder(VectorPathBuilder):
    def __init__(self, times: ndarray, n_factors: int):
        super().__init__(times, n_factors)

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

class CorrelatedNormalPathsBuilder(VectorPathBuilder):
    def __init__(self, times: ndarray, rho_matrix: ndarray):
        super().__init__(times, n_factors=rho_matrix.shape[0])
        self.brownian_bldr = BrownianPathBuilder(times, self.n_factors)

        self.rho_matrix: ndarray = checked_type(rho_matrix, ndarray)
        assert self.rho_matrix.ndim == 2, "Expected square rho matrix"
        assert self.rho_matrix.shape == (self.n_factors, self.n_factors), "Expected square rho matrix"

        U, S, _ = svd(self.rho_matrix)
        self.left_correlating_matrix: ndarray = np.matmul(U, np.diag(np.sqrt(S)))


    def build(self, rng: RandomNumberGenerator, n_paths: int):
        uncorrelated_brownians = self.brownian_bldr.build(rng, n_paths)
        foo = uncorrelated_brownians.path # (factor, time, path)
        bar = np.einsum('kf,ftp->ktp', self.left_correlating_matrix, foo)
        return VectorPath(uncorrelated_brownians.times, bar)

class LognormalPathsBuilder(VectorPathBuilder):
    def __init__(
            self,
            prices: ndarray,
            times: ndarray,
            rho_matrix:
            ndarray, drifts: ndarray,
            vols: ndarray,
    ):
        super().__init__(times, n_factors=rho_matrix.shape[0])
        self.correlated_normals_builder = CorrelatedNormalPathsBuilder(times, rho_matrix)
        self.prices: ndarray = checked_type(prices, ndarray)    # (factor)
        self.drifts: ndarray = checked_type(drifts, ndarray)    # (factor)
        self.vols: ndarray = checked_type(vols, ndarray)        # (factor)

    def build(self, rng: RandomNumberGenerator, n_paths: int):
        correlated_paths = self.correlated_normals_builder.build(rng, n_paths).path     # (ftp)
        scaled_paths = np.einsum("f,ftp->ftp", self.vols, correlated_paths)             # (ftp)
        drift_matrix = np.expand_dims(np.einsum("t, f -> ft", self.times, self.drifts), axis=2)
        drift_matrix2 = np.broadcast_to(drift_matrix, (self.n_factors, self.n_times, n_paths))
        paths_with_drift = scaled_paths + drift_matrix2
        result = np.einsum("f, ftp -> ftp", self.prices, np.exp(paths_with_drift))
        return VectorPath(self.times, result)
