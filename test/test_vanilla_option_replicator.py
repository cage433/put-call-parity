import unittest

import numpy as np
from put_call_parity.models import CALL
from put_call_parity.portfolio.tradeable import OptionTrade
from put_call_parity.ref_data.commodity import WTI
from put_call_parity.replicator.vanilla_option_replicator import VanillaOptionPortfolio, VanillaOptionReplicator
from put_call_parity.valuation_context.valuation_context import ValuationContext
from tp_maths.brownians.uniform_generator import PseudoUniformGenerator
from tp_quantity.quantity import Qty
from tp_quantity.quantity_test_utils import QtyTestUtils
from tp_quantity.uom import MT, USD, SCALAR
from tp_random_tests.random_number_generator import RandomNumberGenerator
from tp_random_tests.random_test_case import RandomisedTest


class VanillaOptionReplicatorTestCase(unittest.TestCase, QtyTestUtils):
    def _random_option(self, rng: RandomNumberGenerator) -> OptionTrade:
        return OptionTrade(
            WTI,
            Qty(rng.uniform(100, 200), MT),
            CALL,
            strike = Qty(rng.uniform(95, 105), USD / MT),
            expiry_time=rng.uniform()
        )

    def _random_vc(self, rng: RandomNumberGenerator) -> ValuationContext:
        return ValuationContext(
            valuation_ccy=USD,
            time=0.0,
            commodity_prices={WTI: Qty(rng.uniform(95, 105), USD / MT)},
            commodity_vols={WTI: Qty(rng.uniform(0.5), SCALAR)},
        )

    @RandomisedTest()
    def test_initial_hedge(self, rng: RandomNumberGenerator):
        option = self._random_option(rng)
        vc = self._random_vc(rng)
        option_portfolio = VanillaOptionPortfolio(option)
        value1 = option_portfolio.value(vc)
        hedged_portfolio = option_portfolio.rehedge(vc)
        value2 = hedged_portfolio.value(vc)
        self.assertVeryClose(value1, value2)

        self.assertVeryClose(
            hedged_portfolio.delta(vc, WTI),
            Qty(0, MT),
        )

    @RandomisedTest(number_of_runs=10)
    def test_replicated_value(self, rng: RandomNumberGenerator):
        option = self._random_option(rng)
        vc = self._random_vc(rng)
        initial_value = option.value(vc)
        portfolio = VanillaOptionPortfolio(option)
        replicator = VanillaOptionReplicator(portfolio, vc)
        n_time_steps = 30
        n_paths = 1000
        portfolios, vcs = replicator.replicate(
            PseudoUniformGenerator(seed=rng.randint(999999)),
            n_time_steps,
            n_paths
        )
        self.assertEqual(len(portfolios), n_paths)
        terminal_values = np.asarray([p.value(vc).checked_value(vc.valuation_ccy) for p, vc in zip(portfolios, vcs)])
        print(f"\nInitial value {initial_value}")
        print(f"Average value {terminal_values.mean()}")

