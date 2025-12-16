from put_call_parity.portfolio.portfolio import Portfolio
from put_call_parity.valuation_context.valuation_context import ValuationContext
from put_call_parity.portfolio.tradeable import OptionTrade, Cash, CommodityTrade
from put_call_parity.valuation_context.valuation_context import  ValuationContext
from tp_quantity.quantity import Qty
from tp_utils.type_utils import checked_type


class HedgedOptionPortfolio:
    def __init__(self, option: OptionTrade, vc: ValuationContext):
        self.option_trade: OptionTrade = checked_type(option, OptionTrade)
        self.delta_hedge = CommodityTrade(option.commodity, option.delta(vc).negate())
        self.cash = Cash((self.delta_hedge.value(vc) + self.option_trade.value(vc)).negate())

        assert self.portolio.value(vc).abs.checked_value(vc.valuation_ccy) < 1e-6, \
            f"Portfolio value {self.portolio.value(vc)} non-zero"

    @property
    def portfolio(self):
        return Portfolio([self.option_trade, self.delta_hedge, self.cash])