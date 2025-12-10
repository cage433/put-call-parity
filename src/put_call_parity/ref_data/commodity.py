from ref_data.unit_of_account import UnitOfAccount


class Commodity(UnitOfAccount):
    def __init__(self, name: str):
        super().__init__(name)

