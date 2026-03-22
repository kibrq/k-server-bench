from __future__ import annotations

from typing import Callable

import numpy as np


class Circle:
    @classmethod
    def discrete(cls, m: int) -> np.ndarray:
        """Return the shortest-path distance matrix on a cycle of size m."""
        if m <= 0:
            raise ValueError("m must be positive")

        idx = np.arange(m, dtype=int)
        diff = np.abs(idx[:, None] - idx[None, :])
        return np.minimum(diff, m - diff)

    @classmethod
    def continous(cls, m: int) -> Callable[[float, float], float]:
        """Return circle distance over points on [0, m) with wrap-around."""
        if m <= 0:
            raise ValueError("m must be positive")
        period = float(m)

        def distance(x: float, y: float) -> float:
            dx = abs((float(x) % period) - (float(y) % period))
            return min(dx, period - dx)

        return distance
