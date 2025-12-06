import unittest

from quantity import MWH, USD, Qty, MT, SCALAR


class QtyTestCase(unittest.TestCase):
    def test_str(self):
        self.assertEqual(str(Qty(1, USD)), '1 USD')
        self.assertEqual(str(Qty(1.5, SCALAR)), '1.5')
        self.assertEqual(str(Qty(1, SCALAR / USD)), '1 USD^-1')
        self.assertEqual(str(Qty(1, USD.inverse)), '1 USD^-1')
        self.assertEqual(str(Qty(1, USD / MWH)), '1 USD / MWH')
        self.assertEqual(str(Qty(1, USD * MT / MWH)), '1 MT USD / MWH')

    def test_multiply(self):
        self.assertEqual(
            Qty(2, USD / MWH) * Qty(3, MWH),
            Qty(6, USD)
        )
        self.assertEqual(
            Qty(2, MWH.inverse) * Qty(3, MWH),
            Qty(6, SCALAR)
        )

    def test_divide(self):
        self.assertEqual(
            Qty(6, USD * MWH) / Qty(3, MWH),
            Qty(2, USD)
        )
        self.assertEqual(
            Qty(6, SCALAR) / Qty(3, MWH),
            Qty(2, MWH.inverse)
        )

    def test_add(self):
        self.assertEqual(
            Qty(3, USD) + Qty(2, USD),
            Qty(5, USD)
        )
    def test_minus(self):
        self.assertEqual(
            Qty(3, USD) - Qty(2, USD),
            Qty(1, USD)
        )
