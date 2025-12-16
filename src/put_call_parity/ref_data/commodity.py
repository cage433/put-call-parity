from put_call_parity.ref_data.unit_of_account import UnitOfAccount
from tp_utils.type_utils import checked_type
from tp_quantity.uom import UOM, USD, MT
from tp_quantity.quantity import Qty


class Commodity(UnitOfAccount):
    def __init__(self, name: str, price_uom: UOM):
        super().__init__(name)
        self.price_uom: UOM = checked_type(price_uom, UOM)
        assert price_uom.numerator.is_ccy, f"Price uom {price_uom} should have a ccy numerator"

    def __eq__(self, other):
        return isinstance(other, Commodity) and self.name == other.name and self.price_uom == other.price_uom

    def __hash__(self):
        return hash((self.name, self.price_uom))

    @property
    def ccy(self):
        return self.price_uom.numerator

    @property
    def quantity_uom(self):
        return self.price_uom.denominator

    @property
    def default_dP(self):
        return Qty(0.01, self.price_uom)

WTI = Commodity("WTI", USD / MT)
