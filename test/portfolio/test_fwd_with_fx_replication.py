from unittest import TestCase

import numpy as np
from tp_maths.random.random_correlation_matrix import RandomCorrelationMatrix
from tp_quantity.quantity import Qty
from tp_quantity.uom import USD, MT, EUR
from tp_random_tests.random_test_case import RandomisedTest

from put_call_parity.portfolio.fwd_with_fx_replication import FwdWithFXReplication


class FwdWithFXReplicationTestCase(TestCase):
    @RandomisedTest()
    def test_deep_itm(self, rng):
        F = Qty(100.0, USD / MT)
        FX = Qty(1.4, EUR / USD)
        vols = np.asarray([0.3, 0.5])
        T = 0.5
        rho_matrix = RandomCorrelationMatrix.truly_random(rng, 2)
        replicator = FwdWithFXReplication(F, FX, vols, rho_matrix, T)
        n_time_steps = 200
        n_paths = 2000
        payoffs = replicator.simulation(rng, n_time_steps=n_time_steps, n_paths=n_paths)
        payoff = payoffs.mean
        se = payoffs.std_err
        expected = 0.0
        self.assertAlmostEqual(
            0.0,
            payoff.checked_value(EUR/MT),
            delta=1e-6
        )

