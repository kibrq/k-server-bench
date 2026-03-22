from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def create_normalized_sha256_hash_fn(context: Any):
    def hash_fn(wf):
        wf_ = np.asarray(wf)
        wf_ = wf_ - wf_.min()
        return hashlib.sha256(wf_.tobytes()).hexdigest(), {}

    return hash_fn


def _build_circle_symmetry_cache(context: Any) -> dict[str, np.ndarray]:
    m = context.m

    point_permutations = np.array([[(x - r) % m for x in range(m)] for r in range(m)], dtype=int)

    config_permutations: list[list[int]] = []
    for r in range(m):
        cfg_perm_for_r: list[int] = []
        for config in context._idx_to_config:
            permuted_config = [point_permutations[r][p] for p in config]
            new_idx = context.config_to_idx(tuple(sorted(permuted_config)))
            cfg_perm_for_r.append(new_idx)
        config_permutations.append(cfg_perm_for_r)
    config_permutations_arr = np.array(config_permutations, dtype=int)

    reflect_points = [(-x) % m for x in range(m)]
    config_reflection: list[int] = []
    for config in context._idx_to_config:
        reflected = [reflect_points[p] for p in config]
        new_idx = context.config_to_idx(tuple(sorted(reflected)))
        config_reflection.append(new_idx)
    config_reflection_arr = np.array(config_reflection, dtype=int)

    return {
        "config_permutations": config_permutations_arr,
        "config_reflection": config_reflection_arr,
    }


def create_circle_symmetry_hash_fn(context: Any, rotation_step: int = 2):
    """
    Canonical hash under circle symmetries (rotation + reflection).

    This mirrors the behavior from draft-v2/260122/circle.py:
    - normalize by subtracting min(wf)
    - iterate rotations in steps of 2 (default) for k-taxi circle parity
    - compare rotated and reflected candidates and keep lexicographically
      smallest SHA256 hash.
    """
    cache = _build_circle_symmetry_cache(context)
    config_permutations = cache["config_permutations"]
    config_reflection = cache["config_reflection"]

    if rotation_step <= 0:
        raise ValueError("rotation_step must be > 0")

    def hash_fn(wf):
        wf = np.asarray(wf)
        shift = wf.min()
        wf0 = wf - shift

        canonical_hash = None
        canonical_wf = None
        canonical_meta = None

        for r in range(0, context.m, rotation_step):
            wf_rot = wf0[config_permutations[r]]
            wf_ref = wf_rot[config_reflection]

            hash_rot = hashlib.sha256(wf_rot.tobytes()).hexdigest()
            hash_ref = hashlib.sha256(wf_ref.tobytes()).hexdigest()

            if canonical_hash is None or hash_rot < canonical_hash:
                canonical_hash = hash_rot
                canonical_wf = wf_rot
                canonical_meta = {
                    "rotation": int(r),
                    "reflected": False,
                    "shift": float(shift),
                }

            if canonical_hash is None or hash_ref < canonical_hash:
                canonical_hash = hash_ref
                canonical_wf = wf_ref
                canonical_meta = {
                    "rotation": int(r),
                    "reflected": True,
                    "shift": float(shift),
                }

        if canonical_hash is None:
            canonical_wf = wf0
            canonical_hash = hashlib.sha256(canonical_wf.tobytes()).hexdigest()
            canonical_meta = {
                "rotation": 0,
                "reflected": False,
                "shift": float(shift),
            }

        return canonical_hash, canonical_meta

    return hash_fn
