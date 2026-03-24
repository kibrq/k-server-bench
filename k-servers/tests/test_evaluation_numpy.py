import pickle
from pathlib import Path

import numpy as np
import pytest

from kserver.context.numpy_wf_context import WFContext
from kserver.evaluation import NumpyKServerInstance, compute_potential_stats
from kserver.potential.canonical_potential import Potential as CanonicalPotential


def _line_metric(m: int) -> np.ndarray:
    d = np.zeros((m, m), dtype=int)
    for i in range(m):
        for j in range(m):
            d[i, j] = abs(i - j)
    return d


def _build_instance() -> NumpyKServerInstance:
    context = WFContext(k=2, distance_matrix=_line_metric(3))
    wf0 = np.asarray(context.initial_wf((0, 1)), dtype=float)
    wf1 = np.asarray(context.update_wf(wf0, 2), dtype=float)
    wf1_norm = wf1 - wf1.min()

    nodes = [
        {"id": 0, "depth": 0, "wf_norm": wf0},
        {"id": 1, "depth": 1, "wf_norm": wf1_norm},
    ]
    edges = [
        {"from": 0, "to": 1, "ext": 0.0, "d_min": float(wf1.min()), "weight": 1.0},
    ]
    payload = {
        "k": 2,
        "distance_matrix": context.distance_matrix,
        "nodes": nodes,
        "edges": edges,
        "bellman": np.asarray([0.0, 1.0]),
    }
    return NumpyKServerInstance.from_legacy_dict(payload)


def _unifying_hyperparams(k: int) -> tuple[int, list[list[int]], list[int]]:
    index_matrix = []
    for i in range(k + 1):
        row = []
        row.extend([-i] * i)
        for j in range(i, k):
            row.append(j + 1)
        index_matrix.append(row)
    n = k
    coefs = [0] * (n * (n - 1) // 2)
    return n, index_matrix, coefs


ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "metrics"

LEGACY_CIRCLE_K4_M6 = METRICS_DIR / "circle_k4_m6.pickle"
LEGACY_CIRCLE_TAXI_K4_M6 = METRICS_DIR / "circle_taxi_k4_m6.pickle"


def test_numpy_instance_roundtrip_npz(tmp_path) -> None:
    instance = _build_instance()
    path = tmp_path / "instance.npz"

    instance.dump_numpy(path)
    loaded = NumpyKServerInstance.load(path)

    np.testing.assert_array_equal(loaded.distance_matrix, instance.distance_matrix)
    np.testing.assert_array_equal(loaded.node_wf_norm, instance.node_wf_norm)
    np.testing.assert_array_equal(loaded.edge_from, instance.edge_from)
    np.testing.assert_array_equal(loaded.get_bellman(), instance.get_bellman())


def test_numpy_instance_loads_legacy_pickle(tmp_path) -> None:
    instance = _build_instance()
    path = tmp_path / "instance.pickle"
    payload = {
        "k": instance.k,
        "distance_matrix": instance.distance_matrix,
        "nodes": [
            {"id": int(instance.node_id[i]), "depth": int(instance.node_depth[i]), "wf_norm": instance.node_wf_norm[i]}
            for i in range(len(instance.node_id))
        ],
        "edges": [
            {
                "from": int(instance.edge_from[i]),
                "to": int(instance.edge_to[i]),
                "ext": float(instance.edge_ext[i]),
                "d_min": float(instance.edge_d_min[i]),
                "weight": float(instance.edge_weight[i]),
            }
            for i in range(len(instance.edge_from))
        ],
        "bellman": instance.get_bellman(),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    loaded = NumpyKServerInstance.load(path)

    np.testing.assert_array_equal(loaded.node_id, instance.node_id)
    np.testing.assert_array_equal(loaded.edge_weight, instance.edge_weight)


def test_compute_potential_stats_without_graph_construction() -> None:
    instance = _build_instance()

    def potential_factory(context):
        del context
        return lambda wf: float(np.min(wf))

    stats = compute_potential_stats(
        potential_factory,
        instance,
        compute_bellman_closure=True,
        include_wf_columns=True,
    )

    assert "bellman_closure_iterations" not in stats.metrics
    assert stats.metrics["total_nodes"] == 2
    assert stats.metrics["total_edges"] == 1
    assert "wf_norm" in stats.df_nodes.columns
    assert "u_wf" in stats.df_edges.columns


def test_compute_potential_stats_mp_backend_on_small_instance() -> None:
    instance = _build_instance()

    def potential_factory(context):
        del context
        return lambda wf: float(np.min(wf))

    stats = compute_potential_stats(
        potential_factory,
        instance,
        compute_potential_backend="mp",
        compute_potential_kwargs={"n_processes": 2, "chunk_size": 1},
        opt_upper_bound_estimate_sample_size=None,
        include_info_columns=False,
    )

    assert stats.metrics["total_nodes"] == 2
    assert stats.metrics["total_edges"] == 1


def test_compute_potential_stats_robustness_uses_custom_idx_to_config() -> None:
    instance = _build_instance()

    seen = []

    def potential_factory(context):
        seen.append(tuple(tuple(cfg) for cfg in context._idx_to_config))
        return lambda wf: float(np.min(wf))

    stats = compute_potential_stats(
        potential_factory,
        instance,
        robustness_check=True,
    )

    assert stats.metrics["robustness"] is True
    assert len(seen) >= 2
    assert seen[0] != seen[1]


@pytest.mark.legacy_compat
def test_compute_potential_stats_circle_taxi_k4_m6_canonical_has_17_violations() -> None:
    if not LEGACY_CIRCLE_TAXI_K4_M6.exists():
        pytest.skip(f"Legacy instance pickle not found: {LEGACY_CIRCLE_TAXI_K4_M6}")

    instance = NumpyKServerInstance.load(LEGACY_CIRCLE_TAXI_K4_M6)
    n, index_matrix, coefs = _unifying_hyperparams(instance.k)

    def potential_factory(context):
        return CanonicalPotential(context, n=n, index_matrix=index_matrix, coefs=coefs)

    stats = compute_potential_stats(
        potential_factory,
        instance,
        compute_potential_backend="mp",
        compute_potential_kwargs={"n_processes": 5, "chunk_size": 200},
        opt_upper_bound_estimate_sample_size=None,
        robustness_check=False,
        compute_bellman_closure=False,
        include_info_columns=False,
        include_wf_columns=False,
        include_renormalized_metrics=False,
    )

    assert stats.metrics["violations_k"] == 17


@pytest.mark.legacy_compat
def test_bellman_lookup_potential_has_zero_violations_and_fails_robustness_on_small_real_instance() -> None:
    if not LEGACY_CIRCLE_K4_M6.exists():
        pytest.skip(f"Legacy instance pickle not found: {LEGACY_CIRCLE_K4_M6}")

    instance = NumpyKServerInstance.load(LEGACY_CIRCLE_K4_M6)
    wf_lookup = {
        tuple(float(x) for x in instance.node_wf_norm[i]): float(instance.get_bellman()[i])
        for i in range(instance.node_wf_norm.shape[0])
    }

    def potential_factory(context):
        del context
        return lambda wf: wf_lookup.get(tuple(float(x) for x in wf), 0.0)

    stats = compute_potential_stats(
        potential_factory,
        instance,
        compute_potential_backend="simple",
        compute_potential_robustness_backend="simple",
        opt_upper_bound_estimate_sample_size=None,
        robustness_check=True,
        compute_bellman_closure=False,
        include_info_columns=False,
        include_wf_columns=False,
    )

    assert stats.metrics["violations_k"] == 0
    assert stats.metrics["robustness"] is False
