"""Identifiers and dimensions shared by feeding, queries and parity checks."""

import re
from collections.abc import Mapping

import numpy as np


def engine_key(key: str, slot_index: Mapping[str, int]) -> str:
    if not isinstance(key, str) or key.count("|") != 1:
        raise ValueError(f"Invalid slot|value key: {key!r}")
    slot, value = key.split("|")
    if slot not in slot_index or not re.fullmatch(r"n[0-9]+", value):
        raise ValueError(f"Unknown slot or invalid Prime value ID: {key!r}")
    index = slot_index[slot]
    if type(index) is not int or index < 0:
        raise ValueError("Slot indices must be nonnegative integers")
    return f"s{index}_{value}"


def document_number(value) -> int:
    if (isinstance(value, bool) or not isinstance(value, (str, int, np.integer))
            or not re.fullmatch(r"[0-9]+", str(value)) or not 0 <= int(value) <= 2147483647):
        raise ValueError("Vespa doc_id must fit a nonnegative signed 32-bit integer")
    return int(value)


def unit_vector(value, dim: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if (vector.shape != (dim,) or not np.isfinite(vector).all()
            or not np.isclose(np.linalg.norm(vector), 1, atol=1e-3)):
        raise ValueError(f"{name} must be a finite unit vector of dimension {dim}")
    return vector
