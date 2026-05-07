from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "k-servers" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kserver.evaluation import NumpyKServerInstance


def _import_networkx():
    return pytest.importorskip("networkx")


def _build_generated_subinstance(metric_name: str, max_nodes: int) -> NumpyKServerInstance:
    metric_path = ROOT / "metrics" / metric_name
    if not metric_path.exists():
        pytest.skip(f"Metric fixture not found: {metric_path}")

    instance = NumpyKServerInstance.load(metric_path)
    node_count = min(max_nodes, int(instance.node_id.shape[0]))
    keep_node_ids = instance.node_id[:node_count]
    reindex = {int(node_id): idx for idx, node_id in enumerate(keep_node_ids.tolist())}
    keep_node_set = set(reindex)

    nodes = [
        {
            "id": idx,
            "depth": int(instance.node_depth[idx]),
            "wf_norm": np.asarray(instance.node_wf_norm[idx], dtype=float),
        }
        for idx in range(node_count)
    ]

    edges = []
    for idx in range(instance.edge_from.shape[0]):
        src = int(instance.edge_from[idx])
        dst = int(instance.edge_to[idx])
        if src not in keep_node_set or dst not in keep_node_set:
            continue
        edges.append(
            {
                "from": reindex[src],
                "to": reindex[dst],
                "ext": float(instance.edge_ext[idx]),
                "d_min": float(instance.edge_d_min[idx]),
                "weight": float(instance.edge_weight[idx]),
            }
        )

    assert edges, f"Generated subinstance from {metric_name} had no internal edges"

    payload = {
        "k": instance.k,
        "distance_matrix": instance.distance_matrix,
        "nodes": nodes,
        "edges": edges,
        "bellman": np.zeros(len(nodes), dtype=float),
    }
    return NumpyKServerInstance.from_legacy_dict(payload)


def _assert_bellman_feasible(instance: NumpyKServerInstance) -> None:
    nx = _import_networkx()

    graph = nx.DiGraph()
    for node in instance.get_nodes():
        graph.add_node(int(node["id"]))

    for edge in instance.get_edges():
        u = int(edge["from"])
        v = int(edge["to"])
        weight = float(edge["weight"])
        if graph.has_edge(u, v):
            graph[u][v]["weight"] = min(float(graph[u][v]["weight"]), weight)
        else:
            graph.add_edge(u, v, weight=weight)

    source = "__source__"
    graph.add_node(source)
    for node_id in instance.node_id.tolist():
        graph.add_edge(source, int(node_id), weight=0.0)

    try:
        distances, _ = nx.single_source_bellman_ford(graph, source, weight="weight")
    except nx.NetworkXUnbounded as exc:
        pytest.fail(f"Bellman-Ford found a negative cycle: {exc}")

    for edge in instance.get_edges():
        u = int(edge["from"])
        v = int(edge["to"])
        weight = float(edge["weight"])
        assert distances[v] <= distances[u] + weight + 1e-9


@pytest.mark.parametrize(
    ("metric_name", "max_nodes"),
    [
        ("circle_k3_m6.pickle", 16),
        ("circle_k4_m6.pickle", 32),
        ("circle_taxi_k4_m6.pickle", 32),
    ],
)
def test_networkx_bellman_ford_generated_instances_are_feasible(metric_name: str, max_nodes: int) -> None:
    instance = _build_generated_subinstance(metric_name=metric_name, max_nodes=max_nodes)
    _assert_bellman_feasible(instance)


@pytest.mark.parametrize(
    "metric_name",
    [
        "circle_k3_m6.pickle",
        "circle_k4_m6.pickle",
        "circle_taxi_k4_m6.pickle",
        "circle_taxi_k4_m8.pickle",
    ],
)
def test_networkx_bellman_ford_legacy_metric_instances_are_feasible(metric_name: str) -> None:
    metric_path = ROOT / "metrics" / metric_name
    if not metric_path.exists():
        pytest.skip(f"Metric fixture not found: {metric_path}")

    instance = NumpyKServerInstance.load(metric_path)
    _assert_bellman_feasible(instance)
