from unittest import TestCase

import numpy as np
from tp_random_tests.random_test_case import RandomisedTest

from put_call_parity.models import CALL, BlackScholes
from put_call_parity.portfolio.option_with_fx_replication import OptionWithFXReplication


class OptionReplicationTestCase(TestCase):
    @RandomisedTest()
    def test_deep_itm(self, rng):
        F = 100.0
        FX = 1.1
        F_vol = 0.3
        FX_vol = 0.2
        rho = -0.5
        K = F * FX * rng.uniform(0.95, 1.05)
        T = 0.5
        combined_vol = np.sqrt(F_vol * F_vol + 2 * rho * F_vol * FX_vol + FX_vol * FX_vol)
        replicator = OptionWithFXReplication(CALL, K, F, FX, F_vol, FX_vol, rho, T)
        n_time_steps = 100
        n_paths = 2000
        payoffs = replicator.simulation(rng, n_time_steps=n_time_steps, n_paths=n_paths)
        payoff = payoffs.mean()
        se = payoffs.std() / np.sqrt(n_paths)
        bs = BlackScholes(CALL, F * FX, K, combined_vol, T).value
        print(f"Mean {payoff:1.2f} ({se:1.2f}), bs {bs:1.2f}")

