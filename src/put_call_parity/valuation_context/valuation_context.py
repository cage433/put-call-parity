from tp_quantity.quantity import Qty
from tp_quantity.uom import UOM
from tp_utils.type_utils import checked_type, checked_dict_type

from put_call_parity.ref_data.ordered_fx_pair import OrderedFxPair


class ValuationContext:
    def __init__(
            self,
            base_ccy: UOM,
            fx_rates: dict[OrderedFxPair, Qty],
            zero_rates: dict[UOM, Qty]
    ):
        self.base_ccy: UOM = checked_type(base_ccy, UOM)
        self.fx_rates: dict[OrderedFxPair, Qty] = checked_dict_type(fx_rates, OrderedFxPair, UOM)
        self.zero_rates = checked_dict_type(zero_rates, UOM, Qty)

        base_ccy.assert_is_ccy()
        for ccy in zero_rates:
            ccy.assert_is_ccy()

    def zero_rate(self, ccy: UOM) -> Qty:
        return self.zero_rates[ccy]

    def fx_rate(self, pair: OrderedFxPair) -> Qty:
        if pair in self.fx_rates:
            return self.fx_rates[pair]
        if pair.inverse in self.fx_rates:
            return self.fx_rates[pair.inverse].inverse
        raise ValueError(f"no fx rates for {pair}")


