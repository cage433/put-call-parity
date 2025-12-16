from typing import Optional

import numpy as np
from put_call_parity.ref_data.commodity import Commodity
from tp_maths.brownians.uniform_generator import UniformGenerator
from tp_maths.vector_path.vector_path import VectorPath
from tp_quantity.quantity import Qty
from tp_utils.type_utils import checked_type, checked_optional_type

from put_call_parity.portfolio.portfolio import Portfolio
from put_call_parity.portfolio.tradeable import OptionTrade, Cash, CommodityTrade, Tradeable
from put_call_parity.valuation_context.valuation_context import ValuationContext


class VanillaOptionPortfolio:
    def __init__(self, option: OptionTrade, commodity_trade: Optional[CommodityTrade] = None,
                 cash: Optional[Cash] = None):
        self.option: OptionTrade = checked_type(option, OptionTrade)
        self.commodity = option.commodity
        self.commodity_trade: CommodityTrade = checked_optional_type(commodity_trade, CommodityTrade) or CommodityTrade(
            self.commodity, Qty(0, self.commodity.quantity_uom))
        self.cash: Cash = checked_optional_type(cash, Cash) or Cash(Qty(0, self.commodity.ccy))
        self.trades: list[Tradeable] = [self.option, self.commodity_trade, self.cash]

    def value(self, vc: ValuationContext) -> Qty:
        return Qty.sum([t.value(vc) for t in self.trades])

    def delta(self, vc: ValuationContext, commodity: Commodity) -> Qty:
        return Qty.sum([t.delta(vc, commodity) for t in self.trades])

    def rehedge(self, vc: ValuationContext) -> 'VanillaOptionPortfolio':
        delta = self.option.delta(vc, self.commodity)
        change_in_hedge = -delta - self.commodity_trade.amount
        if change_in_hedge.is_zero:
            return self

        rehedge_trade = CommodityTrade(self.commodity, change_in_hedge)
        rehedge_trade_cost = vc.price(self.commodity) * change_in_hedge * -1
        current_value = self.value(vc)
        hedged_portfolio = VanillaOptionPortfolio(
            self.option,
            self.commodity_trade + rehedge_trade,
            self.cash + Cash(rehedge_trade_cost)
        )
        hedged_portfolio_value = hedged_portfolio.value(vc)
        assert abs(current_value - hedged_portfolio_value) < Qty(0.01, self.commodity.ccy)
        return hedged_portfolio


class VanillaOptionReplicator:
    def __init__(self, option: OptionTrade, initial_vc: ValuationContext):
        self.option: OptionTrade = checked_type(option, OptionTrade)
        self.initial_vc: ValuationContext = checked_type(initial_vc, ValuationContext)
        self.commodity = option.commodity

    def replicate(self, generator: UniformGenerator, n_time_steps: int, n_paths: int):
        option_value = self.option.value(self.initial_vc)
        portfolios = [
            Portfolio([self.option, Cash(option_value.negate())])
            for _ in range(n_paths)
        ]
        vcs = [self.initial_vc for _ in range(n_paths)]

        times = np.asarray(
            [i * self.option.expiry_time / n_time_steps for i in range(n_time_steps + 1)]
        )
        paths = VectorPath.brownian_paths(n_variables=1, times=times, n_paths=n_paths, uniform_generator=generator)
        for i_time_step in range(n_time_steps):
            time = times[i_time_step + 1]
            prices = paths.variable_sample(i_variable=0, i_time=i_time_step + 1).values
            shifted_vcs = [vc.copy(time=time).with_price(self.commodity, Qty(price, self.commodity.price_uom)) for
                           vc, price in zip(vcs, prices)]

            pass
