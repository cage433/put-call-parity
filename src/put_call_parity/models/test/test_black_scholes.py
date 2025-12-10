from unittest import TestCase

from models import BlackScholes, CALL, PUT
from test_utils.random_test_case import RandomisedTest


class BlackScholesTestCase(TestCase):

    @RandomisedTest(number_of_runs=100)
    def test_intrinsic(self, rng):
        F, K = [rng.uniform(90, 110) for _ in range(2)]
        vol, T = rng.choice((0, 1), (1, 0), (0, 0))   # Any of these will lead to the intrinsic value being returned
        bs = BlackScholes(rng.choice(CALL, PUT), F=F, K=K, vol=vol, T=T)
        expected = bs.right.intrinsic(F, K)
        self.assertAlmostEqual(expected, bs.value, delta=1e-6)

    def test_known_values(self):
        self.assertAlmostEqual(
            BlackScholes(right=CALL, F=100, K=100, vol=0.2, T=1).value,
            7.965567,
            delta=1e-6
        )
        self.assertAlmostEqual(
            BlackScholes(right=PUT, F=150, K=100, vol=0.2, T=1).value,
            0.192475,
            delta=1e-6
        )

    @RandomisedTest(number_of_runs=30)
    def test_put_call_parity(self, rng):
        F, K = [rng.uniform(90, 110) for _ in range(2)]
        vol = rng.uniform()
        T = rng.uniform()
        call_value = BlackScholes(CALL, F=F, K=K, vol=vol, T=T).value
        put_value = BlackScholes(PUT, F=F, K=K, vol=vol, T=T).value
        self.assertAlmostEqual(
            call_value - put_value,
            F - K,
            delta=1e-5
        )
