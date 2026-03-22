from itertools import product

import numpy as np
import pytest

from kserverclean.context.numpy_wf_context import WFContext
from kserverclean.context.support import SupportWFAContext, k_taxi_update_support
from kserverclean.metrics.circle import Circle
from kserverclean.potential.canonical_potential import Potential


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


def _huang_k3_hyperparams() -> tuple[int, list[list[int]], list[int]]:
    n = 4
    index_matrix = [
        [-4, -4, -4],
        [4, 1, -2],
        [4, 2, -3],
        [4, 3, -1],
    ]
    coefs_mapping = {
        (1, 2): 1,
        (1, 3): 1,
        (2, 3): 1,
    }
    coefs = []
    for i in range(n):
        for j in range(i + 1, n):
            coefs.append(coefs_mapping.get((i + 1, j + 1), 0))
    return n, index_matrix, coefs


def _potential_value_discrete(wf: np.ndarray, ctx: WFContext, pot: Potential) -> float:
    val, _info = pot(wf)
    return float(val)


def _extended_cost_discrete(wf_prev: np.ndarray, wf_curr: np.ndarray) -> float:
    return float(np.max(wf_curr - wf_prev))


def _assert_unifying_upperbounds_ext_on_requests(
    distance_matrix: np.ndarray,
    k: int,
    init_cfg: tuple[int, ...],
    requests: list[int],
) -> None:
    ctx = WFContext(k=k, distance_matrix=distance_matrix)
    n, index_matrix, coefs = _unifying_hyperparams(k)
    potential = Potential(ctx, n=n, index_matrix=index_matrix, coefs=coefs)

    wf = ctx.initial_wf(init_cfg)
    for req in requests:
        wf_next = ctx.update_wf(wf, req)
        delta = _potential_value_discrete(wf_next, ctx, potential) - _potential_value_discrete(wf, ctx, potential)
        ext = _extended_cost_discrete(wf, wf_next)
        assert delta + 1e-9 >= ext
        wf = wf_next


@pytest.mark.parametrize("m", [6, 8])
@pytest.mark.parametrize("seed", list(range(20)))
def test_fuzz_unifying_potential_circle_k3_upperbounds_extended_cost(m: int, seed: int) -> None:
    rng = np.random.default_rng(50_000 + 100 * m + seed)
    k = 3
    dist = Circle.discrete(m).astype(int)
    init_cfg = tuple(int(x) for x in rng.integers(0, m, size=k))
    requests = [int(rng.integers(0, m)) for _ in range(7)]
    _assert_unifying_upperbounds_ext_on_requests(dist, k, init_cfg, requests)


def _circle_distance_continuous(m: float):
    period = float(m)

    def dist(x, y):
        x = float(x)
        y = float(y)
        low, high = (x, y) if x <= y else (y, x)
        return min(abs(high - low), (period + low - high) % period)

    return dist


def _canonical_potential_support(
    wf,
    context: SupportWFAContext,
    m: int,
    n: int,
    index_matrix: list[list[int]],
    coefs: list[float],
) -> float:
    antipode = lambda x: x + m // 2 if x + m // 2 < m else x + m // 2 - m

    opt = float("inf")
    for points in product(range(2 * m), repeat=n):
        halved_points = tuple(x / 2 for x in points)
        antipodes = [antipode(p) for p in halved_points]

        ws = 0.0
        for row in index_matrix:
            row_points = tuple(
                halved_points[abs(x) - 1] if x > 0 else antipodes[abs(x) - 1]
                for x in row
            )
            ws += wf[row_points]

        penalty = 0.0
        idx = 0
        for i in range(len(halved_points)):
            for j in range(i + 1, len(halved_points)):
                penalty += coefs[idx] * context.distance(halved_points[i], halved_points[j])
                idx += 1

        opt = min(opt, ws - penalty)
    return float(opt)


def _handle_mixed_requests_support(
    context: SupportWFAContext,
    servers: tuple[float, ...],
    requests: list[float | tuple[float, float]],
):
    wf = context.initial_work_function(tuple(servers))
    seen = set(servers)
    for req in requests:
        if isinstance(req, tuple):
            wf = k_taxi_update_support(wf, req, seen)
            seen.add(req[0])
            seen.add(req[1])
        else:
            wf = wf.update(req, seen)
            seen.add(req)
    return wf, seen


def test_k_taxi_counterexample_breaks_unifying_potential_from_notebook() -> None:
    # Source: draft-v2/260112/main.ipynb counter_examples (k=4, m=8).
    m = 8
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

    context = SupportWFAContext(k=k, distance_fn=_circle_distance_continuous(m))
    wf_u, seen = _handle_mixed_requests_support(context, servers, requests[:-1])
    wf_v, _seen = _handle_mixed_requests_support(context, servers, requests)

    n, index_matrix, coefs = _unifying_hyperparams(k)
    pu = _canonical_potential_support(wf_u, context, m, n, index_matrix, coefs)
    pv = _canonical_potential_support(wf_v, context, m, n, index_matrix, coefs)
    delta = pv - pu

    ext = 0.0
    for config in product(range(2 * m), repeat=k):
        cfg = tuple(x / 2 for x in config)
        ext = max(ext, wf_v[cfg] - wf_u[cfg])

    assert delta < ext
    assert delta == pytest.approx(1.0)
    assert ext == pytest.approx(2.0)


def test_k3_m8_counterexample_breaks_unifying_potential_from_notebook() -> None:
    # Source: draft-v2/260112/main.ipynb ("k = 3 case reproduction").
    m = 8
    k = 3
    servers = (1.0, 6.0, 7.0)
    requests_prefix = [(6.5, 6.0), 4.0, (2.5, 2.0), 3.0, 4.0, (3.5, 5.0)]
    last_request = 4.0

    context = SupportWFAContext(k=k, distance_fn=_circle_distance_continuous(m))
    wf_u, seen = _handle_mixed_requests_support(context, servers, requests_prefix)
    wf_v, _seen = _handle_mixed_requests_support(context, servers, requests_prefix + [last_request])

    n, index_matrix, coefs = _unifying_hyperparams(k)
    pu = _canonical_potential_support(wf_u, context, m, n, index_matrix, coefs)
    pv = _canonical_potential_support(wf_v, context, m, n, index_matrix, coefs)
    delta = pv - pu

    ext = 0.0
    for config in product(range(2 * m), repeat=k):
        cfg = tuple(x / 2 for x in config)
        ext = max(ext, wf_v[cfg] - wf_u[cfg])

    assert delta < ext
    assert delta == pytest.approx(1.0)
    assert ext == pytest.approx(2.0)


def test_huang_k3_solves_same_m8_counterexample_from_notebook() -> None:
    m = 8
    k = 3
    servers = (1.0, 6.0, 7.0)
    requests_prefix = [(6.5, 6.0), 4.0, (2.5, 2.0), 3.0, 4.0, (3.5, 5.0)]
    last_request = 4.0

    context = SupportWFAContext(k=k, distance_fn=_circle_distance_continuous(m))
    wf_u, _seen = _handle_mixed_requests_support(context, servers, requests_prefix)
    wf_v, _seen = _handle_mixed_requests_support(context, servers, requests_prefix + [last_request])

    n, index_matrix, coefs = _huang_k3_hyperparams()
    pu = _canonical_potential_support(wf_u, context, m, n, index_matrix, coefs)
    pv = _canonical_potential_support(wf_v, context, m, n, index_matrix, coefs)
    delta = pv - pu

    ext = 0.0
    for config in product(range(2 * m), repeat=k):
        cfg = tuple(x / 2 for x in config)
        ext = max(ext, wf_v[cfg] - wf_u[cfg])

    assert delta >= ext - 1e-9
    assert delta == pytest.approx(2.0)
    assert ext == pytest.approx(2.0)
