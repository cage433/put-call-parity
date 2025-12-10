import csv
from collections import defaultdict
from datetime import datetime
from numbers import Number
from pathlib import Path
from typing import Union, Callable, TypeVar, Optional

import numpy as np


def write_csv_file(path, table):
    with open(path, 'wt', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(table)


def read_csv_file(path):
    with open(path, 'rt', newline='') as f:
        return list(row for row in csv.reader(f))


def checked_subclass(obj_type, parent_type):
    assert issubclass(obj_type, parent_type), f"{obj_type} is not a subclass of {parent_type}"
    return obj_type



def parse_date(text):
    formats = [
        "%Y-%m-%d",
        "%d-%b-%y",
        "%d-%b-%Y",
        "%d-%m-%y",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d %b %y",
        "%Y-%b-%d",
        "%Y%m%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Can't parse '{text}' as date")



class LogTime:
    def __init__(self, name: str):
        self.name: str = name

    def __enter__(self):
        self.dt0 = datetime.now()
        print(f"Starting {self.name}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dt1 = datetime.now()
        time_taken = (self.dt1 - self.dt0).total_seconds() * 1000
        print(f"Finished {self.name} in {time_taken:1.0f} (ms)")


def check_shape(arr: np.ndarray, *expected):
    assert arr.shape == expected, f"Expected shape {expected}, got {arr.shape}"
    return arr


def checked_path(path: Union[Path, str]) -> Path:
    path = Path(path)
    assert path.exists(), f"{path} does not exist"
    return path


def stringify(thing, float_fmt=None):
    if isinstance(thing, list):
        return [stringify(x, float_fmt) for x in thing]
    if isinstance(thing, Number) and float_fmt is not None:
        return f"{thing:{float_fmt}}"
    return str(thing)


def delete_recursively(path: Path):
    ''' essentially rm -rf path'''
    if path.exists():
        # path.chmod(S_IWRITE)
        if path.is_dir():
            for item in path.iterdir():
                delete_recursively(item)
            path.rmdir()
        else:
            path.unlink()

    assert not path.exists(), f"Path '{path}' should not exist"


T = TypeVar("T")
R = TypeVar("R")


def map_optional(x: Optional[T], f: Callable[[T], R], default: Optional[R] = None) -> Optional[R]:
    if x is None:
        return default
    return f(x)


def or_else(x: Optional[T], default: T) -> T:
    if x is None:
        return default
    return x


