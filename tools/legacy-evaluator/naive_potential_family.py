from typing import List, Dict, Any, Tuple

def estimated_submissions(
    n_workers: int,
    search_timeout: float,
    min_worker_timeout: float,
    max_worker_timeout: float,
):
    return n_workers * search_timeout / max_worker_timeout


class PotentialFamily:
    
    def __init__(
        # These are the arguments that are always passed, so keep them as is. 
        self,
        n_instances: int,
        n_workers: int,
        search_timeout: float,
        min_worker_timeout: float,
        max_worker_timeout: float,
        *,
        # Here you can design any kwargs you want to use for the potential
        # And pass them via --potential_family_kwargs_path
        potential_kwargs: Dict[str, Any] = None,
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

        self.potential_kwargs = potential_kwargs
        if self.potential_kwargs is None:
            self.potential_kwargs = {}

    def ask(self) -> List[Dict]:
        return []

    def tell(self, submission, results):
        pass
    
    def finalize(self) -> Tuple[List[Dict], Dict]:
        return [{"score": 0.0, "kwargs": self.potential_kwargs}], {}

