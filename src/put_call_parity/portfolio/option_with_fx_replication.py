from numbers import Number

import numpy as np
from tp_random_tests.random_number_generator import RandomNumberGenerator
from tp_utils.type_utils import checked_type

from put_call_parity.models import OptionRight, BlackScholes
from put_call_parity.process.vector_path_builder import LognormalPathsBuilder


# noinspection PyPep8Naming
class OptionWithFXReplication:
    def __init__(self, right: OptionRight, K: float, F: float, FX: float, F_vol: float, FX_vol: float, rho: float,
                 T: float):
        self.right: OptionRight = checked_type(right, OptionRight)
        self.K: float = checked_type(K, Number)
        self.F: float = checked_type(F, Number)
        self.FX: float = checked_type(FX, Number)
        self.F_vol: float = checked_type(F_vol, float)
        self.FX_vol: float = checked_type(FX_vol, float)
        self.rho: float = checked_type(rho, float)
        self.T: float = checked_type(T, Number)

        self.combined_vol = np.sqrt(F_vol * F_vol + 2 * rho * F_vol * FX_vol + FX_vol * FX_vol)

        self.vols = np.asarray([F_vol, FX_vol])
        self.rho_matrix = np.asarray(
            [
                [1.0, rho],
                [rho, 1.0]
            ]
        )

    def _delta(self, price: float, t: float):
        return BlackScholes(self.right, price, self.K, self.combined_vol, self.T - t).delta

    def _n2(self, price: float, t: float):
        return BlackScholes(self.right, price, self.K, self.combined_vol, self.T - t).N2

    def simulation(self, rng: RandomNumberGenerator, n_time_steps: int, n_paths: int) -> np.ndarray:
        times = np.asarray([i * self.T / n_time_steps for i in range(n_time_steps + 1)])
        drifts = np.asarray([rng.uniform(-0.2, 0.2) for _ in range(2)])
        bldr = LognormalPathsBuilder(prices=np.asarray([self.F, self.FX]), times=times, rho_matrix=self.rho_matrix,
                                     drifts=drifts, vols=self.vols)
        price_paths = bldr.build(rng, n_paths)
        underlying_position = np.ones(n_paths) * self._delta(self.F, t=0) * -1
        cash_position = underlying_position * self.F * self.FX * -1

        def foreign_prices_at_time(i_time: int):
            domestic_prices = price_paths.factor_values(i_factor=0, i_time=i_time)
            fx = price_paths.factor_values(i_factor=1, i_time=i_time)
            return np.einsum("p, p-> p", domestic_prices, fx)

        for i_time_step in range(n_time_steps):
            time_at_end_of_step = times[i_time_step + 1]
            foreign_prices = foreign_prices_at_time(i_time_step + 1)

            position_at_end_of_time_step = np.asarray([self._delta(price, time_at_end_of_step) * -1 for price in foreign_prices])
            change_in_position = position_at_end_of_time_step - underlying_position
            cost_of_position_change = np.einsum("p, p -> p", foreign_prices, change_in_position) * -1
            cash_position = cash_position + cost_of_position_change
            underlying_position = position_at_end_of_time_step

        terminal_foreign_prices = foreign_prices_at_time(i_time=n_time_steps)
        option_payoffs = np.asarray([max(p - self.K, 0) for p in terminal_foreign_prices])
        underlying_value = np.einsum("p, p -> p", terminal_foreign_prices, underlying_position)
        pnl = underlying_value + cash_position + option_payoffs

        return pnl
