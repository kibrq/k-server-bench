# EVOLVE-BLOCK-START
"""Initial baseline PotentialFamily for k=4 canonical-potential search."""

import numpy as np
from typing import List, Dict, Tuple


def estimated_submissions(
    n_workers: int,
    search_timeout: float,
    min_worker_timeout: float,
    max_worker_timeout: float,
):
    return n_workers * search_timeout / max_worker_timeout


class PotentialFamily:
    def __init__(
        self,
        n_instances: int,
        n_workers: int,
        search_timeout: float,
        min_worker_timeout: float,
        max_worker_timeout: float,
    ):
        self.n_instances = n_instances
        self.n_workers = n_workers
        self.search_timeout = search_timeout
        self.min_worker_timeout = min_worker_timeout
        self.max_worker_timeout = max_worker_timeout

        self.estimated_budget = estimated_submissions(
            n_workers=n_workers,
            search_timeout=search_timeout,
            min_worker_timeout=min_worker_timeout,
            max_worker_timeout=max_worker_timeout,
        ) / n_instances

        self.potential_kwargs = {
            "n": 1,
            "coefs": (),
            "index_matrix": ((1, 1, 1, 1),) * 5,
        }

    def ask(self) -> List[Dict]:
        return []

    def tell(self, submission, results):
        pass

    def finalize(self) -> Tuple[List[Dict], Dict]:
        return [{"score": 0.0, "kwargs": self.potential_kwargs}], {}


# EVOLVE-BLOCK-END
