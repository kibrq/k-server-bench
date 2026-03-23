from __future__ import annotations

from argparse import ArgumentParser, Namespace
import json
import signal
import time
import traceback
from itertools import product
from typing import Any

import numpy as np

from kserver.potential import KServerInstance



# EVOLVE-BLOCK-START


# Naive first version, should be changed
def default_canonical_kwargs() -> dict[str, Any]:
    return {
        "n": 5,
        "index_matrix": [
            [-1, -1, -1, -1],
            [-2, 1, 1, 2],
            [-3, 1, 1, 3],
            [-4, 1, 1, 4],
            [-5, 1, 1, 5],
        ],
        "coefs": [0] * 10,
    }


# Should be named "Potential"
class Potential:
    
    # Signature should stay unchanged
    def __init__(self, context, **kwargs):
        self.context = context
        self.kwargs = kwargs

        self.k = context.k
        self.n = int(kwargs["n"])
        self.index_matrix = np.asarray(kwargs["index_matrix"], dtype=np.int16)
        self.coefs = np.asarray(kwargs["coefs"], dtype=np.int32)
        self.support = kwargs.get("support", None)
        if self.support is None:
            self.support = range(context.m)
        self.support = list(self.support)

        self.config_idxes_dtype = kwargs.get("config_idxes_dtype", np.uint16)
        self.coefs_dtype = kwargs.get("coefs_dtype", np.int32)

        self.validate_kwargs()

        self.distances = self._precompute_distances(context, self.n, self.coefs_dtype)
        self.config_idxes = self._precompute_config_idxes(
            context, self.n, self.index_matrix, self.config_idxes_dtype
        )
        self.penalties = self._precompute_distances_dot_coefs(
            context, self.n, self.coefs, self.coefs_dtype
        )

        self.tmp = np.zeros(self.config_idxes.shape[1], dtype=self.coefs_dtype)
        self.scratch = np.zeros(self.config_idxes.shape[1], dtype=self.coefs_dtype)

    def validate_kwargs(self):
        if self.coefs is not None:
            assert len(self.coefs) == self.n * (self.n - 1) // 2, (
                "coefs must be of length n * (n - 1) // 2"
            )

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

        antipode = lambda x: (x + m // 2) % m

        config_idxes = []
        for points in product(self.support, repeat=n):
            config_idxes.append([])

            antipodes = [antipode(p) for p in points]

            for row in index_matrix:
                config_idxes[-1].append(
                    context.config_to_idx(
                        tuple(
                            points[r - 1] if r > 0 else antipodes[abs(r) - 1]
                            for r in row
                        )
                    )
                )

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
            self.distances = self._precompute_distances(context, n, dtype)
        return np.dot(self.distances, coefs)

    def _compute_candidate_values(self, wf):
        wf = np.asarray(wf, dtype=self.coefs_dtype)

        np.take(wf, self.config_idxes[0], out=self.tmp)
        for i in range(1, self.config_idxes.shape[0]):
            np.take(wf, self.config_idxes[i], out=self.scratch)
            self.tmp += self.scratch

        self.tmp -= self.penalties
        return self.tmp

    # Signature should stay unchangeds
    def __call__(self, wf):
        self._compute_candidate_values(wf)
        argmin_idx = int(np.argmin(self.tmp))
        return float(self.tmp[argmin_idx]), {"idx": argmin_idx}


def is_violation(
    u_potential: float,
    v_potential: float,
    d_min: float,
    ext: float,
    rho: float,
):
    return v_potential - u_potential + (rho + 1) * d_min < ext


# Naive, can be changed. Uses only one instance
def compute_candidate_score(
    instance: KServerInstance,
    potential_kwargs: dict[str, Any],
    *,
    max_edges: int | None = None,
) -> dict[str, Any]:
    context = instance.get_context()
    potential = Potential(context, **potential_kwargs)

    nodes = instance.get_nodes()
    edges = instance.get_edges()

    potentials = []
    for node_idx in range(len(nodes)):
        node = nodes[node_idx]
        value = potential(node["wf_norm"])
        if isinstance(value, tuple):
            value = value[0]
        potentials.append(float(value))

    n_violations = 0
    edges_processed = 0
    total_edges = len(edges)
    edge_limit = total_edges if max_edges is None else min(total_edges, max_edges)

    for edge_idx in range(edge_limit):
        edge = edges[edge_idx]
        u, v = edge["from"], edge["to"]

        if is_violation(
            u_potential=potentials[u],
            v_potential=potentials[v],
            d_min=edge["d_min"],
            ext=edge["ext"],
            rho=context.k,
        ):
            n_violations += 1

        edges_processed += 1

    return {
        "n_violations": n_violations,
        "edges_processed": edges_processed,
        "edges_total": total_edges,
    }


def sample_candidate_kwargs(rng: np.random.Generator) -> dict[str, Any]:
    candidate = default_canonical_kwargs()
    candidate["coefs"] = [int(x) for x in rng.integers(-3, 4, size=10)]
    return candidate


def main(args: Namespace):
    instances = [KServerInstance.load(path) for path in args.metrics]
    if not instances:
        raise ValueError("No metrics were provided")

    rng = np.random.default_rng(42)
    timeout_budget = float(args.timeout) if args.timeout is not None else 60.0
    deadline = time.time() + max(2.0, min(20.0, 0.5 * timeout_budget))

    # Naive usage of one instance, should be changed
    search_instance = instances[0]
    max_edges = 20_000

    opt_so_far = {
        "score": float("inf"),
        "kwargs": default_canonical_kwargs(),
        "details": None,
    }

    baseline_result = compute_candidate_score(
        search_instance,
        opt_so_far["kwargs"],
        max_edges=max_edges,
    )
    opt_so_far["score"] = baseline_result["n_violations"]
    opt_so_far["details"] = baseline_result

    n_attempts = 0
    # Naive random sampling
    while time.time() < deadline and n_attempts < 24:
        random_potential_kwargs = sample_candidate_kwargs(rng)
        result = compute_candidate_score(
            search_instance,
            random_potential_kwargs,
            max_edges=max_edges,
        )

        if result["n_violations"] < opt_so_far["score"]:
            opt_so_far["score"] = result["n_violations"]
            opt_so_far["kwargs"] = random_potential_kwargs
            opt_so_far["details"] = result

        n_attempts += 1

    return {
        "potential_kwargs": opt_so_far["kwargs"],
        "_search_summary": {
            "n_attempts": n_attempts,
            "best_score": opt_so_far["score"],
            "details": opt_so_far["details"],
        },
    }


# EVOLVE-BLOCK-END


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def _install_interrupt_handlers() -> None:
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)



if __name__ == "__main__":
    _install_interrupt_handlers()
    parser = ArgumentParser()
    parser.add_argument("--metrics", type=str, nargs="+", help="Path to .pickle files")
    parser.add_argument("--output", type=str, help="Path to the result file")
    parser.add_argument("--timeout", type=float, help="timeout in seconds")
    parser.add_argument("--n_cpus", type=int, default=None, help="CPU count hint")

    args = parser.parse_args()

    output_filename = str(args.output)

    try:
        result = main(args)
    except KeyboardInterrupt:
        result = dict(
            failure="Search Failed",
            reason="Not handled interruption",
        )
    except Exception:
        result = dict(
            failure="Search Failed",
            reason=traceback.format_exc(),
        )

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(result, f)
