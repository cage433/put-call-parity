from numbers import Number
from typing import Optional

from put_call_parity.ref_data.commodity import Commodity
from tp_quantity.quantity import Qty
from tp_quantity.uom import UOM, USD
from tp_utils.type_utils import checked_type, checked_dict_type

from put_call_parity.ref_data.ordered_fx_pair import OrderedFxPair


class ValuationContext:
    def __init__(
            self,
            valuation_ccy: UOM,
            time: Number,
            fx_rates: Optional[dict[OrderedFxPair, Qty]] = None,
            zero_rates: Optional[dict[UOM, Qty]] = None,
            commodity_prices: Optional[dict[Commodity, Qty]] = None,
            commodity_vols: Optional[dict[Commodity, Qty]] = None,
    ):
        self.valuation_ccy: UOM = checked_type(valuation_ccy, UOM)
        self.time: float = checked_type(time, Number)

        def dict_if_none(dict_or_none):
            return dict() if dict_or_none is None else dict_or_none

        self.fx_rates: dict[OrderedFxPair, Qty] = checked_dict_type(dict_if_none(fx_rates), OrderedFxPair, UOM)
        self.zero_rates = checked_dict_type(dict_if_none(zero_rates), UOM, Qty)
        self.commodity_prices = checked_dict_type(dict_if_none(commodity_prices), Commodity, Qty)
        self.commodity_vols = checked_dict_type(dict_if_none(commodity_vols), Commodity, Qty)

        valuation_ccy.assert_is_ccy()
        for ccy in self.zero_rates:
            ccy.assert_is_ccy()

    def zero_rate(self, ccy: UOM) -> Qty:
        return self.zero_rates[ccy]

    def fx_rate(self, pair: OrderedFxPair) -> Qty:
        if pair.is_degenerate:
            return Qty.to_qty(1)
        if not pair.is_usd_pair:
            uom = pair.uom
            return self.fx_rate(OrderedFxPair.from_uom(uom.numerator / USD)) * self.fx_rate(USD / uom.denominator)
        if pair in self.fx_rates:
            return self.fx_rates[pair]
        if pair.inverse in self.fx_rates:
            return self.fx_rates[pair.inverse].inverse
        raise ValueError(f"no fx rates for {pair}")

    def price(self, commodity: Commodity) -> Qty:
        return self.commodity_prices[commodity]

    def vol(self, commodity: Commodity) -> Qty:
        if commodity not in self.commodity_vols:
            raise ValueError(f"No vol for {commodity.name}")
        return self.commodity_vols[commodity]

    def copy(
            self,
            time: Optional[Number] = None,
            fx_rates: Optional[dict[OrderedFxPair, Qty]] = None,
            zero_rates: Optional[dict[UOM, Qty]] = None,
            commodity_prices: Optional[dict[Commodity, Qty]] = None,
            commodity_vols: Optional[dict[Commodity, Qty]] = None,
    ):
        return ValuationContext(
            self.valuation_ccy,
            time or self.time,
            fx_rates or self.fx_rates,
            zero_rates or self.zero_rates,
            commodity_prices or self.commodity_prices,
            commodity_vols or self.commodity_vols
        )

    def with_price(self, commodity: Commodity, price: Qty) -> 'ValuationContext':
        new_prices = self.commodity_prices.copy()
        new_prices[commodity] = price
        return self.copy(commodity_prices=new_prices)

    def shift_price(self, commodity: Commodity, dP: Qty) -> 'ValuationContext':
        return self.with_price(commodity, self.price(commodity) + dP)
