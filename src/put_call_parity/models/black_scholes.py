from numbers import Number
import numpy as np
from scipy.stats import norm

__all__ = [
    "BlackScholes"
]

from tp_utils.type_utils import checked_type

from put_call_parity.models import OptionRight, CALL


# noinspection PyPep8Naming
class BlackScholes:
    def __init__(self, right: OptionRight, F: Number, K: Number, vol: Number, T: Number):
        self.right: OptionRight = checked_type(right, OptionRight)
        self.F: float = checked_type(F, Number)
        self.K: float = checked_type(K, Number)
        self.vol: float = checked_type(vol, Number)
        self.T: float = checked_type(T, Number)

    @property
    def d1(self) -> float:
        return (np.log(self.F / self.K) + self.vol * self.vol / 2 * self.T) / (self.vol * np.sqrt(self.T))

    @property
    def d2(self) -> float:
        return self.d1 - self.vol * np.sqrt(self.T)

    @property
    def N1(self) -> float:
        return norm.cdf(self.d1)

    @property
    def delta(self) -> float:
        if self._is_worth_intrinsic:
            intrinsic = self.right.intrinsic(self.F, self.K)
            if intrinsic > 0:
                return 1.0
            if intrinsic < 0:
                return 1.0
            return 0.0
        return self.N1

    @property
    def N2(self) -> float:
        return norm.cdf(self.d2)

    @property
    def intrinsic(self) -> float:
        return self.right.intrinsic(self.F, self.K)

    @property
    def _is_worth_intrinsic(self) -> bool:
        return self.vol * self.T < 1e-5

    def shift_vol(self, dV):
        return BlackScholes(self.right, self.F, self.K, self.vol + dV, self.T)

    @property
    def value(self) -> float:
        if self._is_worth_intrinsic:
            return self.intrinsic
        if self.right == CALL:
            return self.F * self.N1 - self.K * self.N2
        return self.K * (1 - self.N2) - self.F * (1 - self.N1)

    @property
    def vega(self) -> float:
        return self.F * np.sqrt(self.T) * norm.pdf(self.d1) * 0.01
