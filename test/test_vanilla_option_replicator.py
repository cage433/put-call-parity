import unittest

from put_call_parity.models import CALL
from put_call_parity.portfolio.tradeable import OptionTrade
from put_call_parity.ref_data.commodity import WTI
from put_call_parity.replicator.vanilla_option_replicator import VanillaOptionPortfolio
from put_call_parity.valuation_context.valuation_context import ValuationContext
from tp_quantity.quantity import Qty
from tp_quantity.quantity_test_utils import QtyTestUtils
from tp_quantity.uom import MT, USD, SCALAR
from tp_random_tests.random_number_generator import RandomNumberGenerator
from tp_random_tests.random_test_case import RandomisedTest


class VanillaOptionReplicatorTestCase(unittest.TestCase, QtyTestUtils):
    @RandomisedTest()
    def test_hedging(self, rng: RandomNumberGenerator):
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
        option_portfolio = VanillaOptionPortfolio(option)
        value1 = option_portfolio.value(vc)
        hedged_portfolio = option_portfolio.rehedge(vc)
        value2 = hedged_portfolio.value(vc)
        self.assertVeryClose(value1, value2)

        self.assertVeryClose(
            hedged_portfolio.delta(vc, WTI),
            Qty(0, MT),
        )

