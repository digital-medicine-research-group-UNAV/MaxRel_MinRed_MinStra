from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import digamma


def _pairwise_distance_array(data: pd.DataFrame, coords, discrete_dist: float = 1.0) -> np.ndarray:
    n_rows = data.shape[0]
    coords = list(coords)
    dist_array = np.empty((len(coords), n_rows, n_rows), dtype=float)

    for idx, coord in enumerate(coords):
        values = data.iloc[:, coord].to_numpy()
        if pd.api.types.is_numeric_dtype(data.iloc[:, coord].dtype):
            dist_array[idx, :, :] = np.abs(values[:, None] - values)
        else:
            dist_array[idx, :, :] = (values[:, None] != values).astype(float) * discrete_dist

    return dist_array


def _count_neighbors(coord_dists: np.ndarray, rho: float, coords) -> int:
    if not coords:
        coords = range(coord_dists.shape[1])
    dists = np.max(coord_dists[:, coords], axis=1)
    return int(np.count_nonzero(dists <= rho) - 1)


def _effective_k(k: int, n_samples: int) -> int:
    if n_samples < 2:
        raise ValueError("At least two samples are required to estimate mutual information.")
    return max(1, min(int(k), n_samples - 1))


def _mi_point(point_i: int, k: int, dist_array: np.ndarray) -> float:
    n_samples = dist_array.shape[1]
    coord_dists = np.transpose(dist_array[[0, 1], :, point_i])
    dists = np.max(coord_dists, axis=1)
    rho = np.partition(dists, k)[k]
    k_tilde = np.count_nonzero(dists <= rho) - 1

    nx = _count_neighbors(coord_dists, rho, [0])
    ny = _count_neighbors(coord_dists, rho, [1])
    return float(digamma(k_tilde) + digamma(n_samples) - digamma(nx) - digamma(ny))


def _cmi_point(point_i: int, k: int, dist_array: np.ndarray) -> float:
    coord_dists = np.transpose(dist_array[[0, 1, 2], :, point_i])
    dists = np.max(coord_dists, axis=1)
    rho = np.partition(dists, k)[k]
    k_tilde = np.count_nonzero(dists <= rho) - 1

    nxz = _count_neighbors(coord_dists, rho, [0, 2])
    nyz = _count_neighbors(coord_dists, rho, [1, 2])
    nz = _count_neighbors(coord_dists, rho, [2])
    return float(digamma(k_tilde) - digamma(nxz) - digamma(nyz) + digamma(nz))


def cmi(x, y, z, k: int, data: pd.DataFrame, discrete_dist: float = 1.0, minzero: bool = True) -> float:
    """Estimate I(x; y | z) with the kNN estimator used in the paper experiments."""
    n_samples = data.shape[0]
    k = _effective_k(k, n_samples)
    variables = [list(x), list(y), list(z)]

    for idx, cols in enumerate(variables):
        if cols and all(isinstance(col, str) for col in cols):
            variables[idx] = list(data.columns.get_indexer(cols))

    x, y, z = variables
    dist_array = _pairwise_distance_array(data, x + y + z, discrete_dist=discrete_dist)

    if z:
        point_estimates = (_cmi_point(obs, k, dist_array) for obs in range(n_samples))
    else:
        point_estimates = (_mi_point(obs, k, dist_array) for obs in range(n_samples))

    estimate = sum(point_estimates) / n_samples
    return float(max(estimate, 0.0) if minzero else estimate)
