from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement, product
from typing import Iterable

import numpy as np


Ref = tuple[bool, int]
PenaltyPair = tuple[Ref, Ref, float]


def outer_ref(index: int) -> tuple[bool, int]:
    assert index != 0, "outer refs use nonzero 1-based indices"
    return (False, index)


def inner_ref(index: int) -> tuple[bool, int]:
    assert index != 0, "inner refs use nonzero 1-based indices"
    return (True, index)


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


def _coerce_ref(value) -> Ref:
    assert isinstance(value, (list, tuple)) and len(value) == 2, (
        "refs must decode to [is_inner, index] pairs"
    )
    is_inner, index = value
    return (bool(is_inner), int(index))


def _coerce_stage_spec(value) -> StageSpec:
    if isinstance(value, StageSpec):
        return value
    assert isinstance(value, dict), "stages must decode to StageSpec objects or dicts"
    return StageSpec(
        n_inner=int(value["n_inner"]),
        row=tuple(_coerce_ref(ref) for ref in value["row"]),
        penalty_pairs=tuple(
            (_coerce_ref(left), _coerce_ref(right), float(coef))
            for left, right, coef in value.get("penalty_pairs", ())
        ),
        inner_multiset=bool(value.get("inner_multiset", True)),
    )


def _coerce_spec(value) -> ParametrizedEvaderSpec:
    if isinstance(value, ParametrizedEvaderSpec):
        return value
    assert isinstance(value, dict), "spec must decode to a ParametrizedEvaderSpec or dict"
    return ParametrizedEvaderSpec(
        n_outer=int(value["n_outer"]),
        stages=tuple(_coerce_stage_spec(stage) for stage in value["stages"]),
        constant=float(value.get("constant", 0.0)),
        outer_distance_coefs=tuple(float(x) for x in value.get("outer_distance_coefs", ())),
    )


def reduced_evader_spec(k: int) -> ParametrizedEvaderSpec:
    stages: list[StageSpec] = [
        StageSpec(
            n_inner=0,
            row=tuple(outer_ref(i) for i in range(1, k + 1)),
            penalty_pairs=tuple(),
        )
    ]
    for t in range(1, k + 1):
        stages.append(
            StageSpec(
                n_inner=t,
                row=tuple(
                    [inner_ref(i) for i in range(1, t + 1)]
                    + [outer_ref(i) for i in range(t + 1, k + 1)]
                ),
                penalty_pairs=tuple(
                    (inner_ref(i), outer_ref(t), -1.0) for i in range(1, t + 1)
                ),
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


def _normalize_ref(ref: Ref) -> tuple[bool, int]:
    assert isinstance(ref, tuple) and len(ref) == 2, (
        "refs must be tuples of the form (is_inner, index)"
    )
    is_inner, index = ref
    assert isinstance(is_inner, bool), "refs must use a bool is_inner flag"
    assert isinstance(index, int), "refs must use an integer index"
    assert index != 0, "refs use nonzero 1-based indices"
    return is_inner, index


def _validate_ref(ref: Ref, *, n_outer: int, n_inner: int, label: str) -> None:
    is_inner, index = _normalize_ref(ref)
    abs_index = abs(index)
    if is_inner:
        assert abs_index <= n_inner, f"{label} inner reference out of range"
    else:
        assert abs_index <= n_outer, f"{label} outer reference out of range"


def _resolve_ref(ref: Ref, outer: tuple[int, ...], inner: tuple[int, ...], *, modulus: int) -> int:
    is_inner, index = _normalize_ref(ref)
    ref_idx = abs(index) - 1
    if is_inner:
        point = inner[ref_idx]
    else:
        point = outer[ref_idx]
    if index < 0:
        return _antipode_point(point, modulus=modulus)
    return point


def _antipode_point(point: int, *, modulus: int) -> int:
    assert modulus % 2 == 0, "antipode support requires an even-size point set"
    return (point + modulus // 2) % modulus


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
            return _coerce_spec(kwargs["spec"])

        if {"n_outer", "stages"} <= set(kwargs):
            return ParametrizedEvaderSpec(
                n_outer=int(kwargs["n_outer"]),
                stages=tuple(_coerce_stage_spec(stage) for stage in kwargs["stages"]),
                constant=float(kwargs.get("constant", 0.0)),
                outer_distance_coefs=tuple(
                    float(x) for x in kwargs.get("outer_distance_coefs", tuple())
                ),
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
                _validate_ref(
                    ref,
                    n_outer=self.spec.n_outer,
                    n_inner=stage.n_inner,
                    label="row",
                )
            for left_ref, right_ref, _coef in stage.penalty_pairs:
                _validate_ref(
                    left_ref,
                    n_outer=self.spec.n_outer,
                    n_inner=stage.n_inner,
                    label="left penalty",
                )
                _validate_ref(
                    right_ref,
                    n_outer=self.spec.n_outer,
                    n_inner=stage.n_inner,
                    label="right penalty",
                )

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
                        sorted(
                            _resolve_ref(
                                ref,
                                outer,
                                inner,
                                modulus=self.context.m,
                            )
                            for ref in stage.row
                        )
                    )
                    config_idxes[inner_idx, outer_idx] = self.context.config_to_idx(cfg)
                    penalty = 0.0
                    for left_ref, right_ref, coef in stage.penalty_pairs:
                        left = _resolve_ref(
                            left_ref,
                            outer,
                            inner,
                            modulus=self.context.m,
                        )
                        right = _resolve_ref(
                            right_ref,
                            outer,
                            inner,
                            modulus=self.context.m,
                        )
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
