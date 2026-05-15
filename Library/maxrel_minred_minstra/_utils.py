from __future__ import annotations

import numpy as np


def validate_X_y(X, y) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        y = y.ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    if X.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")
    if np.unique(y).size < 2:
        raise ValueError("y must contain at least two classes.")

    return X, y


def min_max_normalize(values, lower: float = 1.0, upper: float = 2.0) -> np.ndarray:
    if lower >= upper:
        raise ValueError("lower must be smaller than upper.")

    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr

    min_value = np.nanmin(arr)
    max_value = np.nanmax(arr)
    if np.isclose(min_value, max_value):
        return np.full_like(arr, lower, dtype=float)

    return lower + (upper - lower) * (arr - min_value) / ((max_value - min_value) + 1e-10)
