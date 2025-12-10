from numbers import Number

import numpy as np

from models import OptionRight, BlackScholes
from process.vector_path_builder import LognormalPathsBuilder
from utils import checked_type
from utils.random_number_generator import RandomNumberGenerator


# noinspection PyPep8Naming
class OptionReplication:
    def __init__(self, right: OptionRight, K: float, F: float, vol: float, T: float):
        self.right: OptionRight = checked_type(right, OptionRight)
        self.K: float = checked_type(K, Number)
        self.F: float = checked_type(F, Number)
        self.vol: float = checked_type(vol, Number)
        self.T: float = checked_type(T, Number)

    def _delta(self, price: float, t: float):
        return BlackScholes(self.right, price, self.K, self.vol, self.T - t).delta

    def simulation(self, rng: RandomNumberGenerator, n_time_steps: int, n_paths: int) -> np.ndarray:
        times = np.asarray([i * self.T / n_time_steps for i in range(n_time_steps + 1)])
        drift = rng.uniform(-0.2, 0.2)
        bldr = LognormalPathsBuilder(prices=np.asarray([self.F]), times=times, rho_matrix=np.identity(1),
                                     drifts=np.asarray([drift]), vols=np.asarray([self.vol]))
        price_paths = bldr.build(rng, n_paths)
        underlying_position = np.ones(n_paths) * self._delta(self.F, t = 0) * -1
        cash_position = underlying_position * self.F * -1

        for i_time_step in range(n_time_steps):
            t1 = times[i_time_step + 1]
            price1 = price_paths.factor_values(i_factor=0, i_time=i_time_step + 1)

            position_at_end_of_time_step = np.asarray([self._delta(price, t1) * -1 for price in price1])
            change_in_position = position_at_end_of_time_step - underlying_position
            cost_of_position_change = np.einsum("p, p -> p", price1, change_in_position) * -1
            cash_position = cash_position + cost_of_position_change
            underlying_position = position_at_end_of_time_step

        terminal_prices = price_paths.factor_values(i_factor=0, i_time = n_time_steps)
        option_payoffs = np.asarray([max(p - self.K, 0) for p in terminal_prices])
        underlying_value = np.einsum("p, p -> p", terminal_prices, underlying_position)
        pnl = underlying_value + cash_position + option_payoffs

        return pnl
