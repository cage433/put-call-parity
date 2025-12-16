import unittest

from put_call_parity.portfolio.portfolio import Portfolio
from put_call_parity.portfolio.tradeable import CommodityTrade
from put_call_parity.ref_data.commodity import Commodity, WTI
from tp_quantity.quantity import Qty
from tp_quantity.uom import USD, MT
from tp_random_tests.random_test_case import RandomisedTest


class PortfolioTestCase(unittest.TestCase):
    @RandomisedTest()
    def test_netting(self, rng):
        portfolio = Portfolio.empty()
        tradeable = CommodityTrade(WTI, Qty(100, MT))
        portfolio += tradeable

        self.assertEqual(
            tradeable.amount,
            portfolio.volume(WTI)
        )