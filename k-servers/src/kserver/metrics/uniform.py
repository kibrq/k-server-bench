from __future__ import annotations

import numpy as np


class Uniform:
    @classmethod
    def discrete(cls, m: int) -> np.ndarray:
        if m <= 0:
            raise ValueError("m must be positive")
        d = np.ones((m, m), dtype=int)
        np.fill_diagonal(d, 0)
        return d
