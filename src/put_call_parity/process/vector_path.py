import numpy as np
from tp_utils.type_utils import checked_type


class VectorPath:
    def __init__(self, times: np.ndarray, path: np.ndarray):
        self.times: np.ndarray = checked_type(times, np.ndarray)
        self.path: np.ndarray = checked_type(path, np.ndarray)

        assert self.times.ndim == 1, "Expected 1 dimension in times"
        assert self.path.ndim == 3, "Expected 3 dimensions in path"
        assert self.num_times == self.times.size, "path and times should have a consistent shape"

    @property
    def num_factors(self) -> int:
        return self.path.shape[0]

    @property
    def num_times(self) -> int:
        return self.path.shape[1]

    @property
    def num_paths(self) -> int:
        return self.path.shape[2]

    def factor_values(self, i_factor: int, i_time: int) -> np.ndarray:
        return self.path[i_factor, i_time, :]


