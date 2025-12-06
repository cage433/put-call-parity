from numbers import Number

from quantity import UOM, SCALAR
from utils import checked_type


class Qty:
    def __init__(self, value: Number, uom: UOM):
        self.value: float = checked_type(value, Number)
        self.uom: UOM = checked_type(uom, UOM)

    def __str__(self):
        if self.uom == SCALAR:
            return f"{self.value}"
        return f"{self.value} {self.uom}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, Qty) and self.value == other.value and self.uom == other.uom

    def __mul__(self, other: 'Qty') -> 'Qty':
        return Qty(self.value * other.value, self.uom * other.uom)

    def __add__(self, other: 'Qty') -> 'Qty':
        if self.uom != other.uom:
            raise ValueError(f"Cannot add Qty with different uom, {self} vs {other}")
        return Qty(self.value + other.value, self.uom)

    def negate(self):
        return Qty(-self.value, self.uom)

    def __sub__(self, other: 'Qty') -> 'Qty':
        return self + other.negate()

    def __truediv__(self, other: 'Qty') -> 'Qty':
        return Qty(self.value / other.value, self.uom / other.uom)

    @property
    def inverse(self):
        return Qty(1 / self.value, self.uom.inverse)