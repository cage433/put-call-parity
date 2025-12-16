from abc import abstractmethod, ABC
from numbers import Number
from typing import Optional

from put_call_parity.models import OptionRight, BlackScholes
from put_call_parity.ref_data.commodity import Commodity
from put_call_parity.ref_data.ordered_fx_pair import OrderedFxPair
from put_call_parity.ref_data.unit_of_account import UnitOfAccount
from put_call_parity.valuation_context.valuation_context import ValuationContext
from tp_utils.type_utils import checked_type
from tp_quantity.quantity import Qty


class Tradeable(ABC):
    @abstractmethod
    def value(self, vc: ValuationContext):
        pass

    @abstractmethod
    def delta(self, vc: ValuationContext, commodity: Commodity):
        pass

    def numeric_delta(self, vc: ValuationContext, commodity: Commodity, dP: Optional[Qty] = None):
        dP = commodity.default_dP if dP is None else dP
        up_vc = vc.shift_price(commodity, dP)
        up_value = self.value(up_vc)
        dn_vc = vc.shift_price(commodity, -dP)
        dn_value = self.value(dn_vc)
        return (up_value - dn_value) / (dP * 2)

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def is_nettable(self, other: 'Tradeable'):
        pass


class Cash(Tradeable):
    def __init__(self, amount: Qty):
        self.amount: Qty = checked_type(amount, Qty)

    def __str__(self):
        return f"Cash: {self.amount}"

    def value(self, vc: ValuationContext):
        fx_rate = vc.fx_rate(OrderedFxPair(self.amount.uom, vc.valuation_ccy))
        return self.amount * fx_rate

    def delta(self, vc: ValuationContext, commodity: Commodity):
        return Qty(0, commodity.quantity_uom)

    def __add__(self, other):
        assert self.is_nettable(other), f"Can't add {self} and {other}"
        return Cash(self.amount + other.amount)

    def is_nettable(self, other: Tradeable):
        return isinstance(other, Cash) and other.amount.uom == self.amount.uom


class CommodityTrade(Tradeable):
    def __init__(self, commodity: Commodity, amount: Qty):
        self.commodity: Commodity = commodity
        self.amount: Qty = checked_type(amount, Qty)
        assert amount.uom == commodity.quantity_uom, f"Unexpected amount {amount} for commodity {commodity}"

    @property
    def name(self) -> str:
        return self.commodity.name

    def __str__(self):
        return f"Commodity {self.name}: {self.amount}"

    def value(self, vc: ValuationContext):
        price = vc.price(self.commodity)
        from_ccy = price.uom.numerator
        fx_rate = vc.fx_rate(OrderedFxPair(from_ccy, vc.valuation_ccy))
        return self.amount * price * fx_rate

    def __add__(self, other):
        assert self.is_nettable(other), f"Can't add {self} and {other}"
        return CommodityTrade(self.commodity, self.amount + other.amount)

    def is_nettable(self, other: Tradeable):
        return isinstance(other, CommodityTrade) and other.commodity == self.commodity

    def delta(self, vc: ValuationContext, commodity: Commodity):
        if commodity != self.commodity:
            return Qty(0, commodity.quantity_uom)
        return self.amount


class OptionTrade(Tradeable):
    def __init__(self, commodity: Commodity, amount: Qty, right: OptionRight, strike: Qty, expiry_time: Number):
        self.commodity: Commodity = checked_type(commodity, Commodity)
        self.amount: Qty = checked_type(amount, Qty)
        self.right: OptionRight = checked_type(right, OptionRight)
        self.strike: Qty = checked_type(strike, Qty)
        self.expiry_time: float = checked_type(expiry_time, Number)

    def __add__(self, other):
        assert self.is_nettable(other), f"Can't add {self} and {other}"
        return OptionTrade(self.commodity, self.amount + other.amount, self.right, self.strike, self.expiry_time)

    def is_nettable(self, other: Tradeable):
        return isinstance(other, OptionTrade) and other.commodity == self.commodity and self.right == other.right and \
            self.strike == other.strike and self.expiry_time == other.expiry_time

    def _black_scholes(self, vc: ValuationContext) -> BlackScholes:
        F = vc.price(self.commodity).checked_value(self.commodity.price_uom)
        K = self.strike.checked_value(self.commodity.price_uom)
        vol = vc.vol(self.commodity).checked_scalar_value
        T = self.expiry_time - vc.time
        return BlackScholes(self.right, F, K, vol, T)

    def value(self, vc: ValuationContext):
        bs = self._black_scholes(vc)
        option_price = Qty(bs.value, self.commodity.price_uom)
        return option_price * self.amount

    def delta(self, vc: ValuationContext, commodity: Commodity):
        if commodity != self.commodity:
            return Qty(0, commodity.quantity_uom)
        bs = self._black_scholes(vc)
        price_delta = bs.delta
        return self.amount * price_delta
