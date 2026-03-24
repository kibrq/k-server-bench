# EVOLVE-BLOCK-START

from typing import List, Dict, Any, Tuple
import numpy as np


def estimated_submissions(
    n_workers: int,
    search_timeout: float,
    min_worker_timeout: float,
    max_worker_timeout: float,
):
    return n_workers * search_timeout / max_worker_timeout


def default_canonical_kwargs() -> Dict[str, Any]:
    return {
        "n": 1,
        "index_matrix": np.array(
            [
                (1, 1, 1, 1)
            ] * 5,
            dtype=int,
        ),
        "coefs": None,
    }


class PotentialFamily:
    def __init__(
        self,
        n_instances: int,
        n_workers: int,
        search_timeout: float,
        min_worker_timeout: float,
        max_worker_timeout: float,
        *,
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
            self.potential_kwargs = default_canonical_kwargs()

    def ask(self) -> List[Dict]:
        return []

    def tell(self, submission, results):
        pass

    def finalize(self) -> Tuple[List[Dict], Dict]:
        return [{"score": 0.0, "kwargs": self.potential_kwargs}], {}


import numpy as np
from itertools import product
import time

from typing import List, Dict, Any

import logging
logger = logging.getLogger(__name__)


class Potential:
    def __init__(self, context, **kwargs):
        self.context = context
        self.kwargs = kwargs

        self.k = context.k
        self.n = kwargs["n"]
        self.index_matrix = kwargs["index_matrix"]
        self.coefs = kwargs["coefs"]
        self.support = kwargs.get("support", None)
        if self.support is None:
            self.support = range(context.m)
        self.support = list(self.support)

        self.config_idxes_dtype = kwargs.get("config_idxes_dtype", np.uint16)
        self.coefs_dtype = kwargs.get("coefs_dtype", np.int32)

        self.validate_kwargs()

        self.distances = self._precompute_distances(context, self.n, self.coefs_dtype)
        self.config_idxes = self._precompute_config_idxes(context, self.n, self.index_matrix, self.config_idxes_dtype)
        self.penalties = self._precompute_distances_dot_coefs(context, self.n, self.coefs, self.coefs_dtype)

        self.tmp = np.zeros(self.config_idxes.shape[1], dtype=self.coefs_dtype)
        self.scratch = np.zeros(self.config_idxes.shape[1], dtype=self.coefs_dtype)


    def validate_kwargs(self):
        
        if self.coefs is not None:
            assert len(self.coefs) == self.n * (self.n - 1) // 2, "coefs must be of length n * (n - 1) // 2"

            try:
                for x in self.coefs:
                    float(x)
            except ValueError:
                assert False, "coefs must be a tuple of NUMBERS"

        for row in self.index_matrix:
            assert len(row) == self.k, "every row of index_matrix should be exactly k"

        assert len(self.index_matrix) >= self.k + 1, "index_matrix must be at least k + 1 rows"
        

    def _precompute_config_idxes(self, context, n, index_matrix, dtype):
        
        m = context.m
        k = context.k

        antipode = lambda x: (x + m // 2) % m

        config_idxes = []
        for points in product(self.support, repeat=n):
            config_idxes.append([])

            antipodes = [antipode(p) for p in points]

            for row in index_matrix:
                config_idxes[-1].append(context.config_to_idx(
                    tuple(points[r - 1] if r > 0 else antipodes[abs(r) - 1] for r in row)
                ))

        return np.ascontiguousarray(np.array(config_idxes, dtype=dtype).T)


    def _precompute_distances(self, context, n, dtype):
        distances = []
        for points in product(self.support, repeat=n):
            distances.append([])
            for i in range(n):
                for j in range(i + 1, n):
                    distances[-1].append(context.distance(points[i], points[j]))

        return np.array(distances, dtype=dtype)


    def _precompute_distances_dot_coefs(self, context, n, coefs, dtype=None):
        if coefs is None:
            coefs = np.zeros(n * (n - 1) // 2, dtype=dtype)
        else:
            coefs = np.array(coefs, dtype=dtype)
        if not hasattr(self, "distances"):
            self.distances = self._precompute_distances(context, n)
        return np.dot(self.distances, coefs)

    
    def _compute_candidate_values(self, wf):
        wf = np.asarray(wf, dtype=self.coefs_dtype)

        np.take(wf, self.config_idxes[0], out=self.tmp)
        for i in range(1, self.config_idxes.shape[0]):
            np.take(wf, self.config_idxes[i], out=self.scratch)
            self.tmp += self.scratch

        self.tmp -= self.penalties
        return self.tmp


    def __call__(self, wf):
        self._compute_candidate_values(wf)
        argmin_idx = np.argmin(self.tmp)
        return self.tmp[argmin_idx], {"idx": argmin_idx}


from kserver.cpp_compat import WFContext
import numpy as np
import time

from typing import List, Dict, Any

import logging
logger = logging.getLogger(__name__)


class SearchEvaluator:
    def __init__(
        self,
        instances: List["KServerInstance"],
        potential_cls: "Potential",
        potential_kwargs: Dict[str, Any],
        timeout: float = None,
        **kwargs,
    ):
        # Keep this unchanged
        self.start_time = time.time()
        self.timeout = timeout
        self.instances = instances
        

        # Here you can parse kwargs and prepare for __call__
        self.seed = kwargs.get("seed", None)
        self.instance_idx = kwargs.get("instance_idx", 0)

        instance = instances[self.instance_idx]

        self.nodes = instance.get_nodes()
        self.edges = instance.get_edges()
        self.context = instance.get_context()
        self.bellman = instance.get_bellman()

        self.potential = potential_cls(self.context, **potential_kwargs)

        self.rng = np.random.default_rng(self.seed)
        self.edges_idxes = self.rng.permutation(len(self.edges))        

        self.total_time_getting_edges = 0
        self.total_time_checking_violations = 0
        self.total_time_getting_wf = 0
        self.total_time_getting_nodes = 0
        self.total_time_computing_potential = 0
        self.total_time_bookkeeping = 0

    # You absolutely must keep KeyboardInterrupt handling and exit upon it
    # Otherwise, the whole evaluation will be terminated
    def __call__(self):
        state = {
            "edges_processed": 0,
            "violations": [],
            "edges_total": len(self.edges),
        }

        try:
            for idx in self.edges_idxes:
                # You can compute any metrics here
                # This example computes violations_k

                start_time = time.time()
                edge = self.edges[idx]
                end_time = time.time()
                self.total_time_getting_edges += end_time - start_time

                start_time = time.time()
                violation_info = self.check_violation(edge)
                end_time = time.time()
                self.total_time_checking_violations += end_time - start_time

                start_time = time.time()
                if violation_info["violation"]:
                    state["violations"].append(
                        dict(
                            edge_idx = idx,
                            edge_up = violation_info["up"],
                            edge_vp = violation_info["vp"],
                            edge_ext = violation_info["ext"],
                            edge_d_min = violation_info["d_min"],
                        )   
                    )

                state["edges_processed"] += 1

                end_time = time.time()
                self.total_time_bookkeeping += end_time - start_time

        except KeyboardInterrupt:
            pass

        logger.debug(f"Total time getting edges: {self.total_time_getting_edges}")
        logger.debug(f"Total time checking violations: {self.total_time_checking_violations}")
        logger.debug(f"Total time getting wf: {self.total_time_getting_wf}")
        logger.debug(f"Total time getting nodes: {self.total_time_getting_nodes}")
        logger.debug(f"Total time computing potential: {self.total_time_computing_potential}")
        logger.debug(f"Total time bookkeeping: {self.total_time_bookkeeping}")
        logger.debug(f"Ratio of processed edges: {state['edges_processed'] / state['edges_total']}")

        return state


    def check_violation(self, edge: Dict[str, Any]):
        start_time = time.time()
        u = edge["from"]
        v = edge["to"]
        end_time = time.time()
        self.total_time_getting_nodes += end_time - start_time

        
        up = self.compute_potential(u)
        vp = self.compute_potential(v)

        violation = vp - up + (self.context.k + 1) * edge["d_min"] < edge["ext"]

        return {
            "violation": violation,
            "up": up,
            "vp": vp,
            **edge,
        }


    def compute_potential(self, node_idx: int):
        start_time = time.time()
        wf = np.asarray(self.nodes[node_idx]["wf_norm"])
        end_time = time.time()
        self.total_time_getting_wf += end_time - start_time


        # You should keep caching which significantly speeds up the evaluation
        if not hasattr(self, "_potential_cache"):
            self._potential_cache = {}

        key = wf.data.tobytes()
        if not key in self._potential_cache:
            start_time = time.time()
            val = self.potential(wf)
            end_time = time.time()
            self.total_time_computing_potential += end_time - start_time

            if isinstance(val, tuple):
                val, info = val
            self._potential_cache[key] = val

        return self._potential_cache[key]

# EVOLVE-BLOCK-END
