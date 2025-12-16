from typing import Optional

from put_call_parity.valuation_context.valuation_context import ValuationContext
from tp_quantity.quantity import Qty
from tp_utils.list_utils import ListUtils
from tp_utils.type_utils import checked_list_type

from put_call_parity.portfolio.tradeable import Tradeable
from put_call_parity.ref_data.unit_of_account import UnitOfAccount


class Portfolio:
    def __init__(self, net_trades: list[Tradeable]):
        self.net_trades: list[Tradeable] = checked_list_type(net_trades, Tradeable)

    def __add__(self, other: Tradeable):
        matching, rest = ListUtils.partition(self.net_trades, lambda tr: tr.unit_of_account == other.unit_of_account)
        net_stuff = self.stuff.copy()
        if other.unit_of_account in self.stuff:
            net_tradeable = other + net_stuff[other.unit_of_account]
            net_stuff[other.unit_of_account] = net_tradeable
        else:
            net_stuff[other.unit_of_account] = other

        return Portfolio(net_stuff)

    def volume(self, unit_of_account: UnitOfAccount) -> Optional[Qty]:
        return self.stuff.get(unit_of_account)

    def value(self, vc: ValuationContext) -> Qty:
        v = Qty(0, vc.valuation_ccy)
        for t in self.net_trades:
            v += t.value(vc)
        return v

    @staticmethod
    def empty():
        return Portfolio([])