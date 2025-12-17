import unittest

import numpy as np
from tp_maths.vector_path.vector_path import VectorPath

from put_call_parity.models import CALL
from put_call_parity.portfolio.tradeable import OptionTrade
from put_call_parity.ref_data.commodity import WTI
from put_call_parity.replicator.vanilla_option_replicator import VanillaOptionPortfolio, VanillaOptionReplicator
from put_call_parity.valuation_context.valuation_context import ValuationContext
from tp_maths.brownians.uniform_generator import PseudoUniformGenerator, SOBOL_UNIFORM_GENERATOR
from tp_quantity.quantity import Qty
from tp_quantity.quantity_test_utils import QtyTestUtils
from tp_quantity.uom import MT, USD, SCALAR
from tp_random_tests.random_number_generator import RandomNumberGenerator
from tp_random_tests.random_test_case import RandomisedTest


class VanillaOptionReplicatorTestCase(unittest.TestCase, QtyTestUtils):
    def _random_option(self, rng: RandomNumberGenerator) -> OptionTrade:
        return OptionTrade(
            WTI,
            Qty(rng.uniform(100, 200), MT),
            CALL,
            strike = Qty(rng.uniform(95, 105), USD / MT),
            expiry_time=rng.uniform()
        )

    def _random_vc(self, rng: RandomNumberGenerator) -> ValuationContext:
        return ValuationContext(
            valuation_ccy=USD,
            time=0.0,
            commodity_prices={WTI: Qty(rng.uniform(95, 105), USD / MT)},
            commodity_vols={WTI: Qty(rng.uniform(0.5), SCALAR)},
        )

    @RandomisedTest()
    def test_initial_hedge(self, rng: RandomNumberGenerator):
        option = self._random_option(rng)
        vc = self._random_vc(rng)
        option_portfolio = VanillaOptionPortfolio(option)
        value1 = option_portfolio.value(vc)
        hedged_portfolio = option_portfolio.rehedge(vc)
        value2 = hedged_portfolio.value(vc)
        self.assertVeryClose(value1, value2)

        self.assertVeryClose(
            hedged_portfolio.delta(vc, WTI),
            Qty(0, MT),
        )

    @RandomisedTest()
    def test_replicated_value(self, rng: RandomNumberGenerator):
        option = self._random_option(rng)
        vc = self._random_vc(rng)
        vega = option.numeric_vega(vc, option.commodity)
        vol = vc.vol(option.commodity)
        F = vc.price(option.commodity)
        initial_value = option.value(vc)
        portfolio = VanillaOptionPortfolio(option).rehedge(vc)
        n_time_steps = 100
        n_paths = 1000
        times = np.asarray(
            [i * option.expiry_time / n_time_steps for i in range(n_time_steps + 1)]
        )
        vols = np.asarray([vol.checked_scalar_value])
        paths = (VectorPath.brownian_paths(
            n_variables=1,
            times=times,
            n_paths=n_paths,
            uniform_generator=PseudoUniformGenerator(seed=rng.randint(99999))
        ).scaled(vols)
                 .with_lognormal_adjustments(vols)
                 .exp()
                 .with_prices([F]))
        replicator = VanillaOptionReplicator(portfolio, vc, paths)
        portfolios, vcs = replicator.replicate(
            # SOBOL_UNIFORM_GENERATOR,
            PseudoUniformGenerator(seed=rng.randint(999999)),
            n_time_steps,
            n_paths
        )
        self.assertEqual(len(portfolios), n_paths)
        raw_errors = []
        vega_adjusted_errors = []
        for i_path in range(n_paths):
            p_value = portfolios[i_path].value(vcs[i_path])
            raw_errors.append((p_value - initial_value).checked_value(USD))
            observed_vol = Qty.to_qty(paths.observed_vol(i_variable=0, i_path=i_path))
            vega_diff = vega * (observed_vol - vol)
            vega_adjusted_errors.append(
                (p_value - (initial_value + vega_diff)).checked_value(USD)
            )
            # print(f"Vega {vega}, vol: {vol}, obs vol {observed_vol}, vega diff {vega_diff}")
            # print(f"Act: {p_value}, Init: {initial_value}, Vega adj: {initial_value + vega_diff}")
        print(f"Raw {np.std(raw_errors)}")
        print(f"Adj {np.std(vega_adjusted_errors)}")
        terminal_values = np.asarray([p.value(vc).checked_value(vc.valuation_ccy) for p, vc in zip(portfolios, vcs)])
        print(f"\nInitial value {initial_value}")
        print(f"Average value {terminal_values.mean()}")

