from tp_utils.type_utils import checked_type


class UnitOfAccount:
    def __init__(self, name: str):
        self.name = checked_type(name, str)


    def __hash__(self):
        return self.name.__hash__()

    def __eq__(self, other):
        return type(self) == type(other) and self.name == other.name
