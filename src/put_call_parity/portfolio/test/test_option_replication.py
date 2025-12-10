from unittest import TestCase

import numpy as np

from models import CALL, BlackScholes
from portfolio.option_replication import OptionReplication
from test_utils.random_test_case import RandomisedTest


class OptionReplicationTestCase(TestCase):
    @RandomisedTest()
    def test_deep_itm(self, rng):
        F = 100.0
        vol = 0.3
        K = 103.0
        T = 0.5
        replicator = OptionReplication(CALL, K, F, vol, T)
        n_time_steps = 100
        n_paths = 1000
        payoffs = replicator.simulation(rng, n_time_steps=n_time_steps, n_paths=n_paths)
        payoff = payoffs.mean()
        se = payoffs.std() / np.sqrt(n_paths)
        bs = BlackScholes(CALL, F, K, vol, T).value
        print(f"Mean {payoff:1.2f} ({se:1.2f}), bs {bs:1.2f}")

