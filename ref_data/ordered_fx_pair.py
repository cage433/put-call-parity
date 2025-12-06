from quantity import UOM, CURRENCIES
from utils import checked_type


class OrderedFxPair:
    def __init__(self, from_ccy: UOM, to_ccy: UOM):
        from_ccy.assert_is_ccy()
        to_ccy.assert_is_ccy()
        assert from_ccy != to_ccy, f"require distinct currencies for OrderedFxPair, {from_ccy}, {to_ccy}"

        self.from_ccy = checked_type(from_ccy, UOM)
        self.to_ccy = checked_type(to_ccy, UOM)

    def __str__(self):
        return f"{self.from_ccy}{self.to_ccy}"

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.from_ccy == other.from_ccy and self.to_ccy == other.to_ccy

    def __hash__(self):
        return hash((self.from_ccy, self.to_ccy))

    @property
    def inverse(self) -> 'OrderedFxPair':
        return OrderedFxPair(self.to_ccy, self.from_ccy)

    @property
    def uom(self) -> UOM:
        return self.to_ccy / self.from_ccy