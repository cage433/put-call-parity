import unittest

from put_call_parity.models import CALL
from put_call_parity.portfolio.tradeable import OptionTrade
from put_call_parity.ref_data.commodity import WTI
from put_call_parity.valuation_context.valuation_context import ValuationContext
from tp_quantity.quantity import Qty
from tp_quantity.uom import MT, USD, SCALAR
from tp_random_tests.random_number_generator import RandomNumberGenerator
from tp_random_tests.random_test_case import RandomisedTest


class TradeableTestCase(unittest.TestCase):
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
