from ref_data.unit_of_account import UnitOfAccount
from utils import checked_list_type


class Account:
    def __init__(self, units_of_account: list[UnitOfAccount]):
        self.units_of_account = checked_list_type(units_of_account, UnitOfAccount)

    