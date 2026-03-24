import itertools

import numpy as np
import pytest

from kserver.context.numpy_wf_context import WFContext


def _line_metric(m: int) -> np.ndarray:
    d = np.zeros((m, m), dtype=int)
    for i in range(m):
        for j in range(m):
            d[i, j] = abs(i - j)
    return d


def _bruteforce_set_distance(distance_matrix: np.ndarray, C: tuple[int, ...], D: tuple[int, ...]) -> int:
    k = len(C)
    best = None
    for perm in itertools.permutations(range(k)):
        cost = sum(int(distance_matrix[C[i], D[perm[i]]]) for i in range(k))
        best = cost if best is None else min(best, cost)
    assert best is not None
    return best


def _bruteforce_update(context: WFContext, wf: np.ndarray, request: int) -> np.ndarray:
    out = np.zeros_like(wf)
    for uidx, uconfig in enumerate(context._idx_to_config):
        cand = []
        for i in range(context.k):
            v = list(uconfig)
            v[i] = request
            v = tuple(sorted(v))
            vidx = context.config_to_idx(v)
            cand.append(int(wf[vidx]) + int(context.distance_matrix[uconfig[i], request]))
        out[uidx] = min(cand)
    return out


def test_config_to_idx_is_order_invariant() -> None:
    context = WFContext(k=3, distance_matrix=_line_metric(4))
    assert context.config_to_idx((2, 0, 2)) == context.config_to_idx((0, 2, 2))


def test_neighbors_and_move_cost_shapes() -> None:
    context = WFContext(k=2, distance_matrix=_line_metric(5))
    num_cfg = len(context._idx_to_config)
    assert context.neighbors.shape == (5, num_cfg, 2)
    assert context.move_cost.shape == (5, num_cfg, 2)


def test_neighbors_indices_are_valid() -> None:
    context = WFContext(k=2, distance_matrix=_line_metric(4))
    num_cfg = len(context._idx_to_config)
    assert np.all(context.neighbors >= 0)
    assert np.all(context.neighbors < num_cfg)


def test_move_cost_matches_distance_matrix() -> None:
    context = WFContext(k=3, distance_matrix=_line_metric(5))
    for r in range(context.m):
        for uidx, uconfig in enumerate(context._idx_to_config):
            for i in range(context.k):
                assert context.move_cost[r, uidx, i] == context.distance_matrix[uconfig[i], r]


def test_distance_method_forwards_to_matrix() -> None:
    d = _line_metric(6)
    context = WFContext(k=2, distance_matrix=d)
    assert context.distance(1, 4) == d[1, 4]


def test_distance_between_sets_identity_is_zero() -> None:
    context = WFContext(k=3, distance_matrix=_line_metric(6))
    C = (0, 2, 5)
    assert context.distance_between_sets(C, C) == 0


def test_distance_between_sets_matches_bruteforce_small_case() -> None:
    context = WFContext(k=3, distance_matrix=_line_metric(5))
    C = (0, 2, 4)
    D = (1, 1, 3)
    expected = _bruteforce_set_distance(context.distance_matrix, C, D)
    assert context.distance_between_sets(C, D) == expected


def test_distance_between_sets_symmetric_for_symmetric_metric() -> None:
    context = WFContext(k=3, distance_matrix=_line_metric(5))
    C = (0, 1, 4)
    D = (1, 2, 3)
    assert context.distance_between_sets(C, D) == context.distance_between_sets(D, C)


def test_initial_wf_is_zero_at_source_configuration() -> None:
    context = WFContext(k=2, distance_matrix=_line_metric(5))
    src = (1, 3)
    wf = context.initial_wf(src)
    assert wf[context.config_to_idx(src)] == 0


def test_initial_wf_matches_distance_between_sets_everywhere() -> None:
    context = WFContext(k=2, distance_matrix=_line_metric(4))
    src = (0, 2)
    wf = context.initial_wf(src)
    for vidx, vcfg in enumerate(context._idx_to_config):
        assert wf[vidx] == context.distance_between_sets(src, vcfg)


def test_update_wf_output_shape_and_dtype() -> None:
    context = WFContext(k=2, distance_matrix=_line_metric(4))
    wf = np.arange(len(context._idx_to_config), dtype=np.int64)
    out = context.update_wf(wf, request=3)
    assert out.shape == wf.shape
    assert out.dtype == wf.dtype


@pytest.mark.parametrize("m,k,req_node", [(3, 2, 0), (4, 2, 2), (4, 3, 1)])
def test_update_wf_matches_bruteforce(m: int, k: int, req_node: int) -> None:
    context = WFContext(k=k, distance_matrix=_line_metric(m))
    wf = np.arange(len(context._idx_to_config), dtype=int) * 3 + 1
    expected = _bruteforce_update(context, wf, req_node)
    got = context.update_wf(wf, req_node)
    np.testing.assert_array_equal(got, expected)


def test_update_wf_is_deterministic() -> None:
    context = WFContext(k=2, distance_matrix=_line_metric(5))
    wf = np.array([7 * i + 2 for i in range(len(context._idx_to_config))], dtype=int)
    out1 = context.update_wf(wf, request=4)
    out2 = context.update_wf(wf, request=4)
    np.testing.assert_array_equal(out1, out2)


def test_custom_idx_to_config_preserves_order() -> None:
    idx_to_config = [(0, 2), (0, 1), (1, 2), (2, 2), (1, 1), (0, 0)]
    context = WFContext(k=2, distance_matrix=_line_metric(3), idx_to_config=idx_to_config)
    assert context._idx_to_config == idx_to_config


def test_custom_idx_to_config_still_canonicalizes_lookup() -> None:
    context = WFContext(
        k=2,
        distance_matrix=_line_metric(3),
        idx_to_config=[(0, 2), (0, 0), (1, 2), (1, 1), (0, 1), (2, 2)],
    )
    assert context.config_to_idx((2, 0)) == 0
    assert context.config_to_idx((1, 1)) == 3


def test_invalid_custom_idx_to_config_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="closed under single-server request updates"):
        WFContext(k=2, distance_matrix=_line_metric(3), idx_to_config=[(0, 2), (1, 1), (0, 1)])


@pytest.mark.parametrize(
    "idx_to_config,match",
    [
        ([], "must not be empty"),
        ([(0,)], "must have length k"),
        ([(0, 3)], "must use indices in \\[0, m\\)"),
        ([(0, 1), (0, 1)], "must be unique"),
    ],
)
def test_invalid_custom_idx_to_config_shape_raises(idx_to_config, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        WFContext(k=2, distance_matrix=_line_metric(3), idx_to_config=idx_to_config)


def test_default_idx_to_config_matches_multicombinations() -> None:
    context = WFContext(k=3, distance_matrix=_line_metric(4))
    assert context._idx_to_config == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 0, 3),
        (0, 1, 1),
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 2),
        (0, 2, 3),
        (0, 3, 3),
        (1, 1, 1),
        (1, 1, 2),
        (1, 1, 3),
        (1, 2, 2),
        (1, 2, 3),
        (1, 3, 3),
        (2, 2, 2),
        (2, 2, 3),
        (2, 3, 3),
        (3, 3, 3),
    ]
