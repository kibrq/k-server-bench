import numpy as np
import pytest

from kserver.context.numpy_wf_context import WFContext, k_taxi_update
from kserver.context.support import SupportWFAContext, k_taxi_update_support
from kserver.metrics.circle import Circle


def _random_line_metric(rng: np.random.Generator, m: int) -> np.ndarray:
    coords = np.sort(rng.integers(0, 50, size=m))
    return np.abs(coords[:, None] - coords[None, :]).astype(int)


@pytest.mark.parametrize("seed", list(range(25)))
def test_fuzz_support_matches_numpy_point_requests(seed: int) -> None:
    rng = np.random.default_rng(seed)
    m = int(rng.integers(2, 6))
    k = int(rng.integers(1, 4))
    dist = _random_line_metric(rng, m)

    np_ctx = WFContext(k=k, distance_matrix=dist)
    sup_ctx = SupportWFAContext(k=k, distance_matrix=dist)

    init_cfg = tuple(int(x) for x in rng.integers(0, m, size=k))
    np_wf = np_ctx.initial_wf(init_cfg)
    sup_wf = sup_ctx.initial_work_function(init_cfg)

    np.testing.assert_allclose(
        np_wf.astype(float),
        sup_wf.dense_values(np_ctx._idx_to_config),
        atol=1e-9,
    )

    # Use full metric support so both implementations model the same state space.
    seen = set(range(m))
    n_steps = int(rng.integers(1, 6))
    for _ in range(n_steps):
        req = int(rng.integers(0, m))
        np_wf = np_ctx.update_wf(np_wf, req)
        sup_wf = sup_wf.update(req, seen=seen)

        np.testing.assert_allclose(
            np_wf.astype(float),
            sup_wf.dense_values(np_ctx._idx_to_config),
            atol=1e-9,
        )


@pytest.mark.parametrize("seed", list(range(20)))
def test_fuzz_support_matches_numpy_k_taxi_updates(seed: int) -> None:
    rng = np.random.default_rng(10_000 + seed)
    m = int(rng.integers(2, 6))
    k = int(rng.integers(1, 4))
    dist = _random_line_metric(rng, m)

    np_ctx = WFContext(k=k, distance_matrix=dist)
    sup_ctx = SupportWFAContext(k=k, distance_matrix=dist)

    init_cfg = tuple(int(x) for x in rng.integers(0, m, size=k))
    np_wf = np_ctx.initial_wf(init_cfg)
    sup_wf = sup_ctx.initial_work_function(init_cfg)

    np.testing.assert_allclose(
        np_wf.astype(float),
        sup_wf.dense_values(np_ctx._idx_to_config),
        atol=1e-9,
    )

    seen = set(range(m))
    n_steps = int(rng.integers(1, 5))
    for _ in range(n_steps):
        s = int(rng.integers(0, m))
        t = int(rng.integers(0, m))
        req = (s, t)

        np_wf = k_taxi_update(np_ctx, np_wf, req)
        sup_wf = k_taxi_update_support(sup_wf, req, seen=seen)

        np.testing.assert_allclose(
            np_wf.astype(float),
            sup_wf.dense_values(np_ctx._idx_to_config),
            atol=1e-9,
        )


@pytest.mark.parametrize("seed", list(range(20)))
def test_support_distance_matrix_vs_distance_fn_equivalence(seed: int) -> None:
    rng = np.random.default_rng(20_000 + seed)
    m = int(rng.integers(2, 7))
    k = int(rng.integers(1, 4))
    dist = _random_line_metric(rng, m)

    ctx_matrix = SupportWFAContext(k=k, distance_matrix=dist)
    ctx_fn = SupportWFAContext(k=k, distance_fn=lambda a, b: float(dist[a, b]))

    for _ in range(12):
        a = int(rng.integers(0, m))
        b = int(rng.integers(0, m))
        assert ctx_matrix.distance(a, b) == pytest.approx(ctx_fn.distance(a, b))

    for _ in range(10):
        cfg_a = tuple(int(x) for x in rng.integers(0, m, size=k))
        cfg_b = tuple(int(x) for x in rng.integers(0, m, size=k))
        d1 = ctx_matrix.distance_between_sets(cfg_a, cfg_b)
        d2 = ctx_fn.distance_between_sets(cfg_a, cfg_b)
        assert d1 == pytest.approx(d2)


@pytest.mark.parametrize("seed", list(range(20)))
def test_fuzz_support_with_distance_fn_matches_numpy(seed: int) -> None:
    rng = np.random.default_rng(30_000 + seed)
    m = int(rng.integers(2, 6))
    k = int(rng.integers(1, 4))
    dist = _random_line_metric(rng, m)

    np_ctx = WFContext(k=k, distance_matrix=dist)
    sup_ctx = SupportWFAContext(k=k, distance_fn=lambda a, b: float(dist[a, b]))

    init_cfg = tuple(int(x) for x in rng.integers(0, m, size=k))
    np_wf = np_ctx.initial_wf(init_cfg)
    sup_wf = sup_ctx.initial_work_function(init_cfg)

    np.testing.assert_allclose(
        np_wf.astype(float),
        sup_wf.dense_values(np_ctx._idx_to_config),
        atol=1e-9,
    )

    seen = set(range(m))
    n_steps = int(rng.integers(2, 7))
    for _ in range(n_steps):
        req = int(rng.integers(0, m))
        np_wf = np_ctx.update_wf(np_wf, req)
        sup_wf = sup_wf.update(req, seen=seen)

        np.testing.assert_allclose(
            np_wf.astype(float),
            sup_wf.dense_values(np_ctx._idx_to_config),
            atol=1e-9,
        )


@pytest.mark.parametrize("seed", list(range(20)))
def test_fuzz_support_k_taxi_with_distance_fn_matches_numpy(seed: int) -> None:
    rng = np.random.default_rng(40_000 + seed)
    m = int(rng.integers(2, 6))
    k = int(rng.integers(1, 4))
    dist = _random_line_metric(rng, m)

    np_ctx = WFContext(k=k, distance_matrix=dist)
    sup_ctx = SupportWFAContext(k=k, distance_fn=lambda a, b: float(dist[a, b]))

    init_cfg = tuple(int(x) for x in rng.integers(0, m, size=k))
    np_wf = np_ctx.initial_wf(init_cfg)
    sup_wf = sup_ctx.initial_work_function(init_cfg)

    np.testing.assert_allclose(
        np_wf.astype(float),
        sup_wf.dense_values(np_ctx._idx_to_config),
        atol=1e-9,
    )

    seen = set(range(m))
    n_steps = int(rng.integers(1, 6))
    for _ in range(n_steps):
        req = (int(rng.integers(0, m)), int(rng.integers(0, m)))
        np_wf = k_taxi_update(np_ctx, np_wf, req)
        sup_wf = k_taxi_update_support(sup_wf, req, seen=seen)

        np.testing.assert_allclose(
            np_wf.astype(float),
            sup_wf.dense_values(np_ctx._idx_to_config),
            atol=1e-9,
        )


@pytest.mark.parametrize("m", [6, 8])
@pytest.mark.parametrize("seed", list(range(20)))
def test_fuzz_support_matches_numpy_on_circle_point_requests(m: int, seed: int) -> None:
    rng = np.random.default_rng(50_000 + 100 * m + seed)
    k = 3
    dist = Circle.discrete(m).astype(int)

    np_ctx = WFContext(k=k, distance_matrix=dist)
    sup_ctx = SupportWFAContext(k=k, distance_matrix=dist)

    init_cfg = tuple(int(x) for x in rng.integers(0, m, size=k))
    np_wf = np_ctx.initial_wf(init_cfg)
    sup_wf = sup_ctx.initial_work_function(init_cfg)

    np.testing.assert_allclose(np_wf.astype(float), sup_wf.dense_values(np_ctx._idx_to_config), atol=1e-9)

    seen = set(range(m))
    for _ in range(8):
        req = int(rng.integers(0, m))
        np_wf = np_ctx.update_wf(np_wf, req)
        sup_wf = sup_wf.update(req, seen=seen)
        np.testing.assert_allclose(np_wf.astype(float), sup_wf.dense_values(np_ctx._idx_to_config), atol=1e-9)


@pytest.mark.parametrize("m", [6, 8])
@pytest.mark.parametrize("seed", list(range(20)))
def test_fuzz_support_matches_numpy_on_circle_k_taxi(m: int, seed: int) -> None:
    rng = np.random.default_rng(60_000 + 100 * m + seed)
    k = 3
    dist = Circle.discrete(m).astype(int)

    np_ctx = WFContext(k=k, distance_matrix=dist)
    sup_ctx = SupportWFAContext(k=k, distance_matrix=dist)

    init_cfg = tuple(int(x) for x in rng.integers(0, m, size=k))
    np_wf = np_ctx.initial_wf(init_cfg)
    sup_wf = sup_ctx.initial_work_function(init_cfg)

    np.testing.assert_allclose(np_wf.astype(float), sup_wf.dense_values(np_ctx._idx_to_config), atol=1e-9)

    seen = set(range(m))
    for _ in range(6):
        req = (int(rng.integers(0, m)), int(rng.integers(0, m)))
        np_wf = k_taxi_update(np_ctx, np_wf, req)
        sup_wf = k_taxi_update_support(sup_wf, req, seen=seen)
        np.testing.assert_allclose(np_wf.astype(float), sup_wf.dense_values(np_ctx._idx_to_config), atol=1e-9)


def test_support_matches_numpy_on_counterexample_via_halfstep_mapping() -> None:
    # Counterexample source (continuous half-steps): archived notebook example.
    # We map x -> 2x onto a discrete circle of size 16 so both contexts
    # operate on identical integer states and distances.
    m_cont = 8
    m_disc = 2 * m_cont
    k = 4

    servers = (0.0, 0.0, 3.0, 4.0)
    requests = [
        (1.5, 2.0),
        (2.5, 5.0),
        (4.5, 2.0),
        (6.5, 7.0),
        (7.5, 3.0),
        (2.5, 5.0),
        (4.5, 1.0),
        2.0,
    ]

    map_point = lambda x: int(round(2 * x)) % m_disc
    mapped_servers = tuple(map_point(x) for x in servers)
    mapped_requests = []
    for r in requests:
        if isinstance(r, tuple):
            mapped_requests.append((map_point(r[0]), map_point(r[1])))
        else:
            mapped_requests.append(map_point(r))

    dist = Circle.discrete(m_disc).astype(int)
    np_ctx = WFContext(k=k, distance_matrix=dist)
    sup_ctx = SupportWFAContext(k=k, distance_matrix=dist)

    np_wf = np_ctx.initial_wf(mapped_servers)
    sup_wf = sup_ctx.initial_work_function(mapped_servers)
    np.testing.assert_allclose(np_wf.astype(float), sup_wf.dense_values(np_ctx._idx_to_config), atol=1e-9)

    seen = set(range(m_disc))
    for req in mapped_requests:
        if isinstance(req, tuple):
            np_wf = k_taxi_update(np_ctx, np_wf, req)
            sup_wf = k_taxi_update_support(sup_wf, req, seen=seen)
        else:
            np_wf = np_ctx.update_wf(np_wf, req)
            sup_wf = sup_wf.update(req, seen=seen)
        np.testing.assert_allclose(np_wf.astype(float), sup_wf.dense_values(np_ctx._idx_to_config), atol=1e-9)
