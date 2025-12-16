from numbers import Number

import numpy as np
from numpy import ndarray
from tp_maths.vector_path.vector_path import VectorPath
from tp_quantity.quantity_array import QtyArray
from tp_random_tests.random_number_generator import RandomNumberGenerator
from tp_utils.type_utils import checked_type
from tp_quantity.quantity import Qty

from put_call_parity.process.vector_path_builder import LognormalPathsBuilder


# noinspection PyPep8Naming
class FwdWithFXReplication:
    def __init__(self, F: float, FX: float, vols: ndarray, rho_matrix: ndarray, T: float):
        self.F: Qty = checked_type(F, Qty)
        self.FX: Qty = checked_type(FX, Qty)
        self.vols: ndarray = checked_type(vols, ndarray)
        self.rho_matrix = checked_type(rho_matrix, ndarray)
        self.T: float = checked_type(T, Number)

    def simulation(self, rng: RandomNumberGenerator, n_time_steps: int, n_paths: int) -> QtyArray:
        times = np.asarray([i * self.T / n_time_steps for i in range(n_time_steps + 1)])
        drifts = np.asarray([rng.uniform(-0.2, 0.2) for _ in range(2)])
        price_paths = VectorPath.brownian_paths(n_variables=2, times=times, n_paths=n_paths).correlated(
            self.rho_matrix).scaled(self.vols).with_drifts(drifts).with_prices([self.F, self.FX])
        # bldr = LognormalPathsBuilder(prices=np.asarray([self.F, self.FX]), times=times, rho_matrix=self.rho_matrix,
        #                              drifts=drifts, vols=self.vols)
        # price_paths = bldr.build(rng, n_paths)
        domestic_cash_position = QtyArray([self.F.value * -1 for _ in range(n_paths)], self.F.uom)
        foreign_cash_position = domestic_cash_position * self.FX

        for i_time_step in range(n_time_steps):
            price1 = price_paths.variable_sample(i_variable=0, i_time=i_time_step + 1)
            fx1 = price_paths.variable_sample(i_variable=1, i_time=i_time_step + 1)
            foreign_cash_position = price1 * fx1 * Qty.to_qty(-1)

        terminal_prices = price_paths.variable_sample(i_variable=0, i_time=n_time_steps)
        terminal_fx = price_paths.variable_sample(i_variable=1, i_time=n_time_steps)
        underlying_value = terminal_prices * terminal_fx
        pnl = underlying_value + foreign_cash_position

        return pnl
