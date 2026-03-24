# EVOLVE-BLOCK-START
"""Initial baseline for the legacy evaluator with all three components editable."""

from typing import Any, Dict, List, Tuple

import numpy as np


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
        ) / max(n_instances, 1)
        self.best_result = None
        self.best_kwargs: Dict[str, Any] = {}
        self.submitted = False

    def ask(self) -> List[Dict[str, Any]]:
        if self.submitted:
            return []
        self.submitted = True
        return [
            {
                "worker_input": {
                    "potential_kwargs": self.best_kwargs,
                    "search_evaluator_kwargs": {"instance_idx": 0},
                },
                "timeout": 1.0,
                "metadata": {"candidate_id": "baseline"},
            }
        ]

    def tell(self, submissions, results):
        for submission, result in zip(submissions, results):
            if result is None:
                continue
            score = float(len(result.get("violations", [])))
            candidate = {
                "score": score,
                "kwargs": dict(submission["worker_input"]["potential_kwargs"]),
            }
            if self.best_result is None or score < self.best_result["score"]:
                self.best_result = candidate
                self.best_kwargs = dict(candidate["kwargs"])

    def finalize(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        recommendation = self.best_result
        if recommendation is None:
            recommendation = {"score": 0.0, "kwargs": dict(self.best_kwargs)}
        summary = {
            "estimated_budget": self.estimated_budget,
            "submitted": self.submitted,
        }
        return [recommendation], summary


class Potential:
    def __init__(self, context, **kwargs):
        self.context = context

    def __call__(self, wf):
        wf = np.asarray(wf)
        return float(np.sum(wf))


class SearchEvaluator:
    def __init__(
        self,
        instances,
        potential_cls,
        potential_kwargs,
        timeout: float = None,
        **kwargs,
    ):
        self.timeout = timeout
        self.instance_idx = int(kwargs.get("instance_idx", 0))
        self.instance = instances[self.instance_idx]
        self.nodes = self.instance.get_nodes()
        self.edges = self.instance.get_edges()
        self.context = self.instance.get_context()
        self.potential = potential_cls(self.context, **potential_kwargs)
        self._potential_cache: Dict[bytes, float] = {}

    def __call__(self):
        state = {
            "edges_processed": 0,
            "violations": [],
            "edges_total": len(self.edges),
        }

        try:
            for edge_idx, edge in enumerate(self.edges):
                if self._is_violation(edge):
                    state["violations"].append(
                        {
                            "edge_idx": edge_idx,
                            "edge_up": self._compute_potential(edge["from"]),
                            "edge_vp": self._compute_potential(edge["to"]),
                            "edge_ext": edge["ext"],
                            "edge_d_min": edge["d_min"],
                        }
                    )
                state["edges_processed"] += 1
        except KeyboardInterrupt:
            pass

        return state

    def _is_violation(self, edge: Dict[str, Any]) -> bool:
        up = self._compute_potential(edge["from"])
        vp = self._compute_potential(edge["to"])
        return vp - up + (self.context.k + 1) * edge["d_min"] < edge["ext"]

    def _compute_potential(self, node_idx: int) -> float:
        wf = np.asarray(self.nodes[node_idx]["wf_norm"])
        key = wf.data.tobytes()
        if key not in self._potential_cache:
            value = self.potential(wf)
            if isinstance(value, tuple):
                value = value[0]
            self._potential_cache[key] = float(value)
        return self._potential_cache[key]


# EVOLVE-BLOCK-END
