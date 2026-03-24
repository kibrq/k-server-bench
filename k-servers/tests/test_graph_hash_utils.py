import numpy as np

from kserver.context.numpy_wf_context import WFContext
from kserver.graph.hash_utils import create_circle_symmetry_hash_fn
from kserver.metrics.circle import Circle


def _rotate_config(cfg, r, m):
    return tuple(sorted(((x - r) % m) for x in cfg))


def test_circle_symmetry_hash_invariant_under_even_rotation() -> None:
    k, m = 3, 8
    context = WFContext(k=k, distance_matrix=Circle.discrete(m=m))
    hash_fn = create_circle_symmetry_hash_fn(context, rotation_step=2)

    cfg = (0, 2, 4)
    wf = context.initial_wf(cfg)
    h0, _ = hash_fn(wf)

    for r in range(0, m, 2):
        cfg_r = _rotate_config(cfg, r, m)
        wf_r = context.initial_wf(cfg_r)
        hr, _ = hash_fn(wf_r)
        assert hr == h0
