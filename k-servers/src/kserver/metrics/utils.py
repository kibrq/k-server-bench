from __future__ import annotations

import numpy as np


def antipode_extension(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Build the antipode extension of a metric matrix.

    Given D (m x m), returns D_ext (2m x 2m):
        D_ext = [[D, R - D],
                 [R - D, D]]
    where R = max(D).

    In the extension, point (i + m) is the antipode of point i.
    """
    d = np.asarray(distance_matrix)
    if d.ndim != 2:
        raise ValueError("distance_matrix must be 2D")
    if d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be square")
    if d.shape[0] == 0:
        raise ValueError("distance_matrix must be non-empty")

    if np.any(d < 0):
        raise ValueError("distance_matrix must be non-negative")

    r = np.max(d)
    cross = r - d
    if np.any(cross < 0):
        raise ValueError("distance_matrix has entries larger than its maximum")

    top = np.concatenate([d, cross], axis=1)
    bottom = np.concatenate([cross, d], axis=1)
    return np.concatenate([top, bottom], axis=0)
