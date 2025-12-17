import numpy as np
from tp_quantity.quantity import Qty
from tp_quantity.uom import MT, USD, SCALAR
from tp_random_tests.random_number_generator import RandomNumberGenerator

from put_call_parity.models import CALL
from put_call_parity.portfolio.tradeable import OptionTrade
from put_call_parity.ref_data.commodity import WTI
from put_call_parity.replicator.vanilla_option_replicator import VanillaOptionPortfolio
from put_call_parity.valuation_context.valuation_context import ValuationContext


def replicate():
    F = Qty(100, USD/MT)
    K = Qty(100, USD/MT)
    T = 1.0
    vol = Qty(0.3, SCALAR)

    option = OptionTrade(WTI, Qty(100, MT), CALL, K, T)
    vc = ValuationContext(
        USD,
        time=0,
        commodity_prices={WTI:F},
        commodity_vols={WTI:vol},
    )
    portfolio = VanillaOptionPortfolio(option).rehedge(vc)

    dT = 0.05
    s = vol.checked_scalar_value * np.sqrt(dT)
    gamma = option.gamma(vc)
    rng = RandomNumberGenerator(seed=12345)
    for _ in range(100):
        dF = F * (np.exp(s * rng.normal() - 0.5 * s * s) - 1.0)
        shifted_vc = vc.copy(time = dT, commodity_prices={WTI:F + dF})
        shifted_value = portfolio.value(shifted_vc)
        expected_value = portfolio.gamma(vc, WTI) * dF * dF * 0.5
        pass
    print(f"option value {option.value(vc)}")
    print(f"portfolio value {portfolio.value(vc)}")
    print(f"option delta {option.delta(vc, WTI)}")
    print(f"option numeric delta {option.numeric_delta(vc, WTI)}")
    print(f"portfolio delta {portfolio.delta(vc, WTI)}")
    print(f"portfolio numeric delta {portfolio.numeric_delta(vc, WTI)}")
    print(f"option gamma {option.gamma(vc, WTI)}")
    print(f"option numeric gamma {option.numeric_gamma(vc, WTI)}")
    print(f"portfolio gamma {portfolio.gamma(vc, WTI)}")
    print(f"portfolio numeric gamma {portfolio.numeric_gamma(vc, WTI)}")

if __name__ == '__main__':
    replicate()