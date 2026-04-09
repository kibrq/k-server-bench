from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement, product
from typing import Iterable

import numpy as np


Ref = int
PenaltyPair = tuple[Ref, Ref, float]


@dataclass(frozen=True)
class StageSpec:
    n_inner: int
    row: tuple[Ref, ...]
    penalty_pairs: tuple[PenaltyPair, ...]
    inner_multiset: bool = True


@dataclass(frozen=True)
class ParametrizedEvaderSpec:
    n_outer: int
    stages: tuple[StageSpec, ...]
    constant: float = 0.0
    outer_distance_coefs: tuple[float, ...] = tuple()


def reduced_evader_spec(k: int) -> ParametrizedEvaderSpec:
    stages: list[StageSpec] = [
        StageSpec(
            n_inner=0,
            row=tuple(range(1, k + 1)),
            penalty_pairs=tuple(),
        )
    ]
    for t in range(1, k + 1):
        stages.append(
            StageSpec(
                n_inner=t,
                row=tuple([-i for i in range(1, t + 1)] + list(range(t + 1, k + 1))),
                penalty_pairs=tuple((-i, t, -1.0) for i in range(1, t + 1)),
                inner_multiset=True,
            )
        )
    return ParametrizedEvaderSpec(n_outer=k, stages=tuple(stages))


def _enumerate_assignments(
    support: Iterable[int],
    arity: int,
    *,
    multiset: bool,
) -> tuple[tuple[int, ...], ...]:
    support = tuple(support)
    if arity == 0:
        return (tuple(),)
    if multiset:
        return tuple(combinations_with_replacement(support, arity))
    return tuple(product(support, repeat=arity))


def _resolve_ref(ref: Ref, outer: tuple[int, ...], inner: tuple[int, ...]) -> int:
    if ref > 0:
        return outer[ref - 1]
    return inner[-ref - 1]


class Potential:
    def __init__(self, context, **kwargs):
        self.context = context
        self.kwargs = kwargs

        self.k = context.k
        self.spec = self._build_spec(context, kwargs)
        self.support = kwargs.get("support", None)
        if self.support is None:
            self.support = range(context.m)
        self.support = tuple(self.support)

        self.config_idxes_dtype = kwargs.get("config_idxes_dtype", np.uint16)
        self.value_dtype = kwargs.get("value_dtype", np.float64)

        self.validate_kwargs()

        self.outer_assignments = _enumerate_assignments(
            self.support,
            self.spec.n_outer,
            multiset=False,
        )
        self.n_outer = len(self.outer_assignments)
        self.stage_config_idxes: list[np.ndarray] = []
        self.stage_penalties: list[np.ndarray] = []
        self.outer_distances = self._precompute_outer_distances()
        self.outer_penalties = self._precompute_outer_penalties()
        self._compile()

        self.tmp = np.full(self.n_outer, self.spec.constant, dtype=self.value_dtype)
        self.stage_best = np.empty(self.n_outer, dtype=self.value_dtype)
        self.scratch = np.empty(self.n_outer, dtype=self.value_dtype)

    def _build_spec(self, context, kwargs) -> ParametrizedEvaderSpec:
        if "spec" in kwargs:
            return kwargs["spec"]

        if {"n_outer", "stages"} <= set(kwargs):
            return ParametrizedEvaderSpec(
                n_outer=int(kwargs["n_outer"]),
                stages=tuple(kwargs["stages"]),
                constant=float(kwargs.get("constant", 0.0)),
                outer_distance_coefs=tuple(kwargs.get("outer_distance_coefs", tuple())),
            )

        return reduced_evader_spec(context.k)

    def validate_kwargs(self) -> None:
        assert self.spec.n_outer >= 0, "n_outer must be nonnegative"
        n_outer_pairs = self.spec.n_outer * (self.spec.n_outer - 1) // 2
        if self.spec.outer_distance_coefs:
            assert len(self.spec.outer_distance_coefs) == n_outer_pairs, (
                "outer_distance_coefs must have length n_outer * (n_outer - 1) // 2"
            )
        for stage in self.spec.stages:
            assert len(stage.row) == self.k, "every stage row must have length k"
            for ref in stage.row:
                assert ref != 0, "row references are 1-based for outer and negative for inner"
                if ref > 0:
                    assert ref <= self.spec.n_outer, "outer reference out of range"
                else:
                    assert -ref <= stage.n_inner, "inner reference out of range"
            for left_ref, right_ref, _coef in stage.penalty_pairs:
                assert left_ref != 0 and right_ref != 0, "penalty references cannot be zero"
                if left_ref > 0:
                    assert left_ref <= self.spec.n_outer, "left outer reference out of range"
                else:
                    assert -left_ref <= stage.n_inner, "left inner reference out of range"
                if right_ref > 0:
                    assert right_ref <= self.spec.n_outer, "right outer reference out of range"
                else:
                    assert -right_ref <= stage.n_inner, "right inner reference out of range"

    def _compile(self) -> None:
        for stage in self.spec.stages:
            inner_assignments = _enumerate_assignments(
                self.support,
                stage.n_inner,
                multiset=stage.inner_multiset,
            )
            config_idxes = np.empty(
                (len(inner_assignments), len(self.outer_assignments)),
                dtype=self.config_idxes_dtype,
            )
            penalties = np.empty(
                (len(inner_assignments), len(self.outer_assignments)),
                dtype=self.value_dtype,
            )
            for outer_idx, outer in enumerate(self.outer_assignments):
                for inner_idx, inner in enumerate(inner_assignments):
                    cfg = tuple(
                        sorted(_resolve_ref(ref, outer, inner) for ref in stage.row)
                    )
                    config_idxes[inner_idx, outer_idx] = self.context.config_to_idx(cfg)
                    penalty = 0.0
                    for left_ref, right_ref, coef in stage.penalty_pairs:
                        left = _resolve_ref(left_ref, outer, inner)
                        right = _resolve_ref(right_ref, outer, inner)
                        penalty += coef * self.context.distance(left, right)
                    penalties[inner_idx, outer_idx] = penalty
            self.stage_config_idxes.append(np.ascontiguousarray(config_idxes))
            self.stage_penalties.append(np.ascontiguousarray(penalties))

    def _precompute_outer_distances(self) -> np.ndarray:
        n_pairs = self.spec.n_outer * (self.spec.n_outer - 1) // 2
        if n_pairs == 0:
            return np.empty((self.n_outer, 0), dtype=self.value_dtype)

        distances = np.empty((self.n_outer, n_pairs), dtype=self.value_dtype)
        for outer_idx, outer in enumerate(self.outer_assignments):
            pair_idx = 0
            for i in range(self.spec.n_outer):
                for j in range(i + 1, self.spec.n_outer):
                    distances[outer_idx, pair_idx] = self.context.distance(
                        outer[i],
                        outer[j],
                    )
                    pair_idx += 1
        return np.ascontiguousarray(distances)

    def _precompute_outer_penalties(self) -> np.ndarray:
        if not self.spec.outer_distance_coefs:
            return np.zeros(self.n_outer, dtype=self.value_dtype)
        coefs = np.asarray(self.spec.outer_distance_coefs, dtype=self.value_dtype)
        return np.ascontiguousarray(np.dot(self.outer_distances, coefs))

    def __call__(self, wf):
        wf = np.asarray(wf, dtype=self.value_dtype)
        self.tmp.fill(self.spec.constant)
        self.tmp += self.outer_penalties
        for config_idxes, penalties in zip(self.stage_config_idxes, self.stage_penalties):
            np.take(wf, config_idxes[0], out=self.stage_best)
            self.stage_best += penalties[0]
            for row_idx in range(1, config_idxes.shape[0]):
                np.take(wf, config_idxes[row_idx], out=self.scratch)
                self.scratch += penalties[row_idx]
                np.minimum(self.stage_best, self.scratch, out=self.stage_best)
            self.tmp += self.stage_best

        argmin_idx = int(np.argmin(self.tmp))
        return float(self.tmp[argmin_idx]), {"idx": argmin_idx}
