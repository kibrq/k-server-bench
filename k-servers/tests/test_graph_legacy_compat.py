import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from kserverclean.context import WFContext, all_multicombinations
from kserverclean.graph.hash_utils import create_normalized_sha256_hash_fn
from kserverclean.graph.parallel_bfs_exploration import parallel_bfs_exploration
from kserverclean.metrics.circle import Circle

pytestmark = pytest.mark.legacy_compat


LEGACY_PATHS = {
    (3, 6): Path("/home/brillian/rl-k-servers/k-servers/potential-evaluation/metrics/circle_k3_m6.pickle"),
    (3, 8): Path("/home/brillian/rl-k-servers/k-servers/potential-evaluation/metrics/circle_k3_m8.pkl"),
    (4, 6): Path("/home/brillian/rl-k-servers/k-servers/potential-evaluation/metrics/circle_k4_m6.pickle"),
    (4, 8): Path("/home/brillian/rl-k-servers/k-servers/potential-evaluation/metrics/circle_k4_m8.pickle"),
}

STORE_CASES = [
    (True, True),
    (True, False),
    (False, True),
]


@pytest.mark.parametrize("k,m", [(3, 6), (3, 8), (4, 6), (4, 8)])
@pytest.mark.parametrize("store_wfs,store_paths", STORE_CASES)
def test_circle_legacy_compat_single_worker(k: int, m: int, store_wfs: bool, store_paths: bool) -> None:
    _assert_circle_legacy_compat(
        k=k,
        m=m,
        n_workers=0,
        store_wfs=store_wfs,
        store_paths=store_paths,
    )


@pytest.mark.parametrize("k,m", [(3, 6), (3, 8), (4, 6), (4, 8)])
@pytest.mark.parametrize("store_wfs,store_paths", STORE_CASES)
def test_circle_legacy_compat_ray_workers(k: int, m: int, store_wfs: bool, store_paths: bool) -> None:
    pytest.importorskip("ray")
    _assert_circle_legacy_compat(
        k=k,
        m=m,
        n_workers=10,
        store_wfs=store_wfs,
        store_paths=store_paths,
        ray_kwargs={"address": "local"},
    )


def _assert_circle_legacy_compat(
    k: int,
    m: int,
    n_workers: int,
    store_wfs: bool,
    store_paths: bool,
    ray_kwargs=None,
) -> None:
    legacy_path = LEGACY_PATHS[(k, m)]
    if not legacy_path.exists():
        pytest.skip(f"Legacy instance pickle not found: {legacy_path}")

    distance_matrix = Circle.discrete(m=m)
    initial_nodes = list(all_multicombinations(m, k=k))
    transitions = [lambda context, wf, r=r: context.update_wf(wf, r) for r in range(m)]

    result = parallel_bfs_exploration(
        k=k,
        distance_matrix=distance_matrix,
        n_workers=n_workers,
        initial_nodes=initial_nodes,
        transitions=transitions,
        create_hash_fn=create_normalized_sha256_hash_fn,
        return_wfs=store_wfs,
        return_paths=store_paths,
        ray_kwargs=ray_kwargs,
    )

    with legacy_path.open("rb") as f:
        legacy = pickle.load(f)

    context = WFContext(k=k, distance_matrix=distance_matrix)
    wf_norm_by_hash = {}
    for node in result["node_bookkeeper"].nodes.values():
        wf = node.get_wf(context=context, transitions=transitions, cache=False)
        wf = np.asarray(wf)
        wf_norm_by_hash[node.hsh] = tuple((wf - wf.min()).tolist())

    legacy_wfs = {tuple(node["wf_norm"]) for node in legacy["nodes"]}
    new_wfs = set(wf_norm_by_hash.values())

    assert len(new_wfs) == len(legacy_wfs)
    assert new_wfs == legacy_wfs

    id_to_wf_norm = {node["id"]: tuple(node["wf_norm"]) for node in legacy["nodes"]}
    legacy_edges = Counter(
        (id_to_wf_norm[edge["from"]], id_to_wf_norm[edge["to"]])
        for edge in legacy["edges"]
    )
    new_edges = Counter(
        (wf_norm_by_hash[u.hsh], wf_norm_by_hash[v.hsh])
        for u, v, _ in result["edge_bookkeeper"].edges
    )

    assert sum(legacy_edges.values()) == len(legacy["edges"])
    assert sum(new_edges.values()) == len(result["edge_bookkeeper"].edges)
    assert new_edges == legacy_edges
