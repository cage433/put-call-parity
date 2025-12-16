import unittest

from put_call_parity.ref_data.ordered_fx_pair import OrderedFxPair
from tp_quantity.uom import USD, EUR


class OrderedFXPairTestCase(unittest.TestCase):
    def test_round_trip(self):
        for uom in [USD / EUR, EUR / USD]:
            pair = OrderedFxPair.from_uom(uom)
            self.assertEqual(uom, pair.uom)

