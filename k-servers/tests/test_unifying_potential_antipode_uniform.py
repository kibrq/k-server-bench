import numpy as np
import pytest

from kserver.context.numpy_wf_context import WFContext
from kserver.metrics.uniform import Uniform
from kserver.metrics.utils import antipode_extension
from kserver.potential.canonical_potential import Potential


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


def _run_sequence(k: int, m: int, init_cfg: tuple[int, ...], requests: list[int]):
    dist = antipode_extension(Uniform.discrete(m))
    ctx = WFContext(k=k, distance_matrix=dist)
    n, index_matrix, coefs = _unifying_hyperparams(k)
    potential = Potential(ctx, n=n, index_matrix=index_matrix, coefs=coefs)

    wf = ctx.initial_wf(init_cfg)
    first_fail = None
    for step, req in enumerate(requests):
        wf_next = ctx.update_wf(wf, req)
        delta = float(potential(wf_next)[0] - potential(wf)[0])
        ext = float(np.max(wf_next - wf))
        if first_fail is None and delta + 1e-9 < ext:
            first_fail = {
                "step": step,
                "request": req,
                "delta": delta,
                "ext": ext,
            }
        wf = wf_next
    return first_fail


@pytest.mark.parametrize(
    "k,m,init_cfg,requests",
    [
        (
            2,
            4,
            (1, 3),
            [0, 1, 3, 1, 1, 2, 3, 0, 0, 1],
        ),
        (
            3,
            6,
            (0, 0, 4),
            [3, 2, 0, 0, 0, 3, 3, 4, 0, 4],
        ),
        (
            4,
            6,
            (1, 0, 4, 4),
            [2, 1, 4, 5, 4, 4, 5, 4, 0, 2],
        ),
    ],
)
def test_antipode_uniform_sequences_with_restricted_init_and_requests(
    k: int,
    m: int,
    init_cfg: tuple[int, ...],
    requests: list[int],
) -> None:
    # Restrict both initial positions and requests to original points [0, m-1].
    assert all(0 <= p < m for p in init_cfg)
    assert all(0 <= r < m for r in requests)
    fail = _run_sequence(k, m, init_cfg, requests)
    assert fail is None


@pytest.mark.parametrize("k,m", [(2, 4), (2, 6), (3, 6), (4, 6)])
def test_antipode_uniform_fuzz_with_restricted_init_and_requests(k: int, m: int) -> None:
    dist = antipode_extension(Uniform.discrete(m))
    ctx = WFContext(k=k, distance_matrix=dist)
    n, index_matrix, coefs = _unifying_hyperparams(k)
    potential = Potential(ctx, n=n, index_matrix=index_matrix, coefs=coefs)

    fail_count = 0
    seed_count = 20
    steps = 8
    for seed in range(seed_count):
        rng = np.random.default_rng(40_000 + 1000 * k + 100 * m + seed)
        init = tuple(int(x) for x in rng.integers(0, m, size=k))  # restricted init
        requests = [int(rng.integers(0, m)) for _ in range(steps)]  # original points only

        wf = ctx.initial_wf(init)
        failed_here = False
        for req in requests:
            wf_next = ctx.update_wf(wf, req)
            delta = float(potential(wf_next)[0] - potential(wf)[0])
            ext = float(np.max(wf_next - wf))
            if delta + 1e-9 < ext:
                fail_count += 1
                failed_here = True
                break
            wf = wf_next
        if failed_here:
            continue

    assert fail_count == 0
