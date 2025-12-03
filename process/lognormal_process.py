from numbers import Number

from utils.utils import checked_type


class LognormalProcess:
    def __init__(self, mu: Number, sigma: Number):
        self.mu: Number = checked_type(mu, Number)
        self.sigma: Number = checked_type(sigma, Number)

