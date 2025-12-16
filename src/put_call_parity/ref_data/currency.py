from tp_quantity.uom import UOM

from put_call_parity.ref_data.unit_of_account import UnitOfAccount
from tp_utils.type_utils import checked_type


class Currency(UnitOfAccount):
    def __init__(self, ccy: UOM):
        super().__init__(f"CCY: {ccy}")
        self.ccy: UOM = checked_type(ccy, UOM)
        assert ccy.is_ccy, f"{ccy} is not a currency"

    def __eq__(self, other):
        return isinstance(other, Currency) and self.ccy == other.ccy

    def __hash__(self):
        return hash(self.ccy)
