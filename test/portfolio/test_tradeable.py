import unittest

from tp_quantity.quantity_test_utils import QtyTestUtils

from put_call_parity.models import CALL
from put_call_parity.portfolio.tradeable import OptionTrade
from put_call_parity.ref_data.commodity import WTI
from put_call_parity.valuation_context.valuation_context import ValuationContext
from tp_quantity.quantity import Qty
from tp_quantity.uom import MT, USD, SCALAR
from tp_random_tests.random_number_generator import RandomNumberGenerator
from tp_random_tests.random_test_case import RandomisedTest


class TradeableTestCase(unittest.TestCase, QtyTestUtils):
    @RandomisedTest(number_of_runs=10)
    def test_delta(self, rng: RandomNumberGenerator):
        option = OptionTrade(
            WTI,
            Qty(rng.uniform(100, 200), MT),
            CALL,
            strike = Qty(rng.uniform(95, 105), USD / MT),
            expiry_time=rng.uniform()
        )
        vc = ValuationContext(
            valuation_ccy=USD,
            time=0.0,
            commodity_prices={WTI: Qty(rng.uniform(95, 105), USD / MT)},
            commodity_vols={WTI: Qty(rng.uniform(0.5), SCALAR)},
        )
        bs_delta = option.delta(vc, WTI)
        numeric_delta = option.numeric_delta(vc, WTI)
        self.assertAlmostEqual(bs_delta, numeric_delta, delta=Qty(0.001, MT))

    @RandomisedTest(number_of_runs=10)
    def test_gamma(self, rng: RandomNumberGenerator):
        option = OptionTrade(
            WTI,
            Qty(rng.uniform(100, 200), MT),
            CALL,
            strike = Qty(rng.uniform(95, 105), USD / MT),
            expiry_time=rng.uniform(0.1, 0.5)
        )
        vc = ValuationContext(
            valuation_ccy=USD,
            time=0.0,
            commodity_prices={WTI: Qty(rng.uniform(95, 105), USD / MT)},
            commodity_vols={WTI: Qty(rng.uniform(0.1, 0.5), SCALAR)},
        )
        bs_gamma = option.gamma(vc, WTI)
        numeric_gamma = option.numeric_gamma(vc, WTI)
        self.assertVeryClose(bs_gamma, numeric_gamma, delta=Qty(0.001, MT  * MT / USD))

    @RandomisedTest(number_of_runs=10)
    def test_theta(self, rng: RandomNumberGenerator):
        option = OptionTrade(
            WTI,
            Qty(rng.uniform(100, 200), MT),
            CALL,
            strike = Qty(rng.uniform(95, 105), USD / MT),
            expiry_time=rng.uniform(0.1, 0.5)
        )
        vc = ValuationContext(
            valuation_ccy=USD,
            time=0.0,
            commodity_prices={WTI: Qty(rng.uniform(95, 105), USD / MT)},
            commodity_vols={WTI: Qty(rng.uniform(0.1, 0.5), SCALAR)},
        )
        bs_theta = option.theta(vc)
        dt = 0.0001
        numeric_theta = option.numeric_theta(vc, dt)
        tol = (option.value(vc) * 0.001).max(numeric_theta.abs * 0.001)
        self.assertVeryClose(bs_theta, numeric_theta, delta=tol)
