import numpy as np
import pytest

from kserverclean.context.numpy_wf_context import (
    WFContext,
    _build_preimage_for_request,
    _ensure_preimage_cache,
    k_taxi_update,
)


def _line_metric(m: int) -> np.ndarray:
    d = np.zeros((m, m), dtype=int)
    for i in range(m):
        for j in range(m):
            d[i, j] = abs(i - j)
    return d


def _bruteforce_k_taxi_update(context: WFContext, wf: np.ndarray, request: tuple[int, int]) -> np.ndarray:
    s, t = request
    delta = int(context.distance_matrix[s, t])
    inf = int(1e8)

    tilde = np.full_like(wf, inf, dtype=wf.dtype)
    for didx, D in enumerate(context._idx_to_config):
        if t in D:
            lst = list(D)
            first_pos = lst.index(t)
            lst[first_pos] = s
            src = tuple(sorted(lst))
            src_idx = context.config_to_idx(src)
            tilde[didx] = int(wf[src_idx]) + delta

    out = np.zeros_like(wf)
    for uidx, U in enumerate(context._idx_to_config):
        candidates = []
        for i in range(context.k):
            V = list(U)
            V[i] = t
            V = tuple(sorted(V))
            vidx = context.config_to_idx(V)
            candidates.append(int(tilde[vidx]) + int(context.distance_matrix[U[i], t]))
        out[uidx] = min(candidates)
    return out


def test_ensure_preimage_cache_initializes_dict() -> None:
    context = WFContext(k=2, distance_matrix=_line_metric(4))
    cache = _ensure_preimage_cache(context)
    assert isinstance(cache, dict)
    assert cache == {}


def test_ensure_preimage_cache_reuses_existing_dict() -> None:
    context = WFContext(k=2, distance_matrix=_line_metric(4))
    cache1 = _ensure_preimage_cache(context)
    cache1[(0, 1)] = np.array([1, 2, 3], dtype=int)
    cache2 = _ensure_preimage_cache(context)
    assert cache1 is cache2
    assert (0, 1) in cache2


def test_build_preimage_marks_missing_t_as_negative_one() -> None:
    context = WFContext(k=2, distance_matrix=_line_metric(4))
    s, t = 0, 3
    pre = _build_preimage_for_request(context, s=s, t=t)
    for idx, cfg in enumerate(context._idx_to_config):
        if t not in cfg:
            assert pre[idx] == -1


def test_build_preimage_matches_expected_mapping() -> None:
    context = WFContext(k=3, distance_matrix=_line_metric(5))
    s, t = 1, 4
    pre = _build_preimage_for_request(context, s=s, t=t)
    for idx, cfg in enumerate(context._idx_to_config):
        if t not in cfg:
            assert pre[idx] == -1
            continue
        lst = list(cfg)
        lst[lst.index(t)] = s
        expected_idx = context.config_to_idx(tuple(sorted(lst)))
        assert pre[idx] == expected_idx


@pytest.mark.parametrize("m,k,req_pair", [(4, 2, (0, 3)), (5, 2, (2, 4)), (5, 3, (1, 3))])
def test_k_taxi_update_matches_bruteforce(m: int, k: int, req_pair: tuple[int, int]) -> None:
    context = WFContext(k=k, distance_matrix=_line_metric(m))
    wf = np.array([5 * i + 7 for i in range(len(context._idx_to_config))], dtype=int)
    expected = _bruteforce_k_taxi_update(context, wf, req_pair)
    got = k_taxi_update(context, wf, req_pair)
    np.testing.assert_array_equal(got, expected)


def test_k_taxi_update_populates_cache_key_once_and_reuses() -> None:
    context = WFContext(k=2, distance_matrix=_line_metric(5))
    wf = np.array([i for i in range(len(context._idx_to_config))], dtype=int)
    key = (0, 4)

    out1 = k_taxi_update(context, wf, key)
    cache = _ensure_preimage_cache(context)
    assert key in cache
    first_obj = cache[key]

    out2 = k_taxi_update(context, wf, key)
    assert cache[key] is first_obj
    np.testing.assert_array_equal(out1, out2)


def test_k_taxi_update_returns_expected_shape_and_dtype() -> None:
    context = WFContext(k=2, distance_matrix=_line_metric(4))
    wf = np.arange(len(context._idx_to_config), dtype=np.int64)
    out = k_taxi_update(context, wf, (0, 2))
    assert out.shape == wf.shape
    assert out.dtype == wf.dtype


def test_k_taxi_update_is_deterministic() -> None:
    context = WFContext(k=2, distance_matrix=_line_metric(5))
    wf = np.array([11 * i + 3 for i in range(len(context._idx_to_config))], dtype=int)
    req = (1, 4)
    out1 = k_taxi_update(context, wf, req)
    out2 = k_taxi_update(context, wf, req)
    np.testing.assert_array_equal(out1, out2)


def test_k_taxi_update_outputs_non_negative_for_non_negative_metric() -> None:
    context = WFContext(k=2, distance_matrix=_line_metric(4))
    wf = np.array([2 * i for i in range(len(context._idx_to_config))], dtype=int)
    out = k_taxi_update(context, wf, (0, 3))
    assert np.all(out >= 0)
