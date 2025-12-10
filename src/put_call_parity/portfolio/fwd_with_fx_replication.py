from numbers import Number

import numpy as np
from numpy import ndarray
from tp_random_tests.random_number_generator import RandomNumberGenerator
from tp_utils.type_utils import checked_type

from put_call_parity.process.vector_path_builder import LognormalPathsBuilder


# noinspection PyPep8Naming
class FwdWithFXReplication:
    def __init__(self, F: float, FX: float, vols: ndarray, rho_matrix: ndarray, T: float):
        self.F: float = checked_type(F, Number)
        self.FX: float = checked_type(FX, Number)
        self.vols: ndarray = checked_type(vols, ndarray)
        self.rho_matrix = checked_type(rho_matrix, ndarray)
        self.T: float = checked_type(T, Number)

    def simulation(self, rng: RandomNumberGenerator, n_time_steps: int, n_paths: int) -> np.ndarray:
        times = np.asarray([i * self.T / n_time_steps for i in range(n_time_steps + 1)])
        drifts = np.asarray([rng.uniform(-0.2, 0.2) for _ in range(2)])
        bldr = LognormalPathsBuilder(prices=np.asarray([self.F, self.FX]), times=times, rho_matrix=self.rho_matrix,
                                     drifts=drifts, vols=self.vols)
        price_paths = bldr.build(rng, n_paths)
        domestic_cash_position = np.ones(n_paths) * self.F * -1
        foreign_cash_position = domestic_cash_position * self.FX

        for i_time_step in range(n_time_steps):
            price1 = price_paths.factor_values(i_factor=0, i_time=i_time_step + 1)
            fx1 = price_paths.factor_values(i_factor=1, i_time=i_time_step + 1)

            foreign_cash_position = np.einsum("p,p->p", price1, fx1) * -1

        terminal_prices = price_paths.factor_values(i_factor=0, i_time=n_time_steps)
        terminal_fx = price_paths.factor_values(i_factor=1, i_time=n_time_steps)
        underlying_value = np.einsum("p, p -> p", terminal_prices, terminal_fx)
        pnl = underlying_value + foreign_cash_position

        return pnl
