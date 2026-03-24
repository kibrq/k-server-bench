from __future__ import annotations

from functools import lru_cache
from itertools import product

import numpy as np
from scipy.optimize import linear_sum_assignment


class SupportWFAContext:
    def __init__(
        self,
        k: int,
        distance_matrix: np.ndarray | None = None,
        distance_fn=None,
        eps: float = 1e-9,
    ):
        self.k = int(k)
        if distance_matrix is None and distance_fn is None:
            raise ValueError("Either distance_matrix or distance_fn must be provided")

        self.distance_matrix = None
        self._distance_fn = None
        self.m = None

        if distance_matrix is not None:
            dm = np.asarray(distance_matrix)
            if dm.ndim != 2:
                raise ValueError("distance_matrix must be 2D")
            if dm.shape[0] != dm.shape[1]:
                raise ValueError("distance_matrix must be square")
            self.distance_matrix = dm
            self.m = int(dm.shape[0])

        if distance_fn is not None:
            self._distance_fn = distance_fn

        if self._distance_fn is None:
            self._distance_fn = lambda a, b: float(self.distance_matrix[int(a), int(b)])

        self.eps = float(eps)

    def distance(self, a, b) -> float:
        return float(self._distance_fn(a, b))

    def clique(self, config) -> float:
        total = 0.0
        for i in range(len(config)):
            for j in range(i + 1, len(config)):
                total += self.distance(config[i], config[j])
        return total

    @lru_cache(maxsize=10_000)
    def _distance_between_sorted_sets(self, a_sorted: tuple, b_sorted: tuple) -> float:
        cost = np.zeros((len(a_sorted), len(b_sorted)), dtype=float)
        for i, a in enumerate(a_sorted):
            for j, b in enumerate(b_sorted):
                cost[i, j] = self.distance(a, b)
        row_ind, col_ind = linear_sum_assignment(cost)
        return float(cost[row_ind, col_ind].sum())

    def distance_between_sets(self, a, b) -> float:
        return self._distance_between_sorted_sets(tuple(sorted(a)), tuple(sorted(b)))

    def is_dominated(
        self,
        wf: "SupportWorkFunction",
        a,
        b,
        tol: float | None = None,
        wa: float | None = None,
        wb: float | None = None,
    ) -> bool:
        if tol is None:
            tol = self.eps
        if wa is None:
            wa = wf[a]
        if wb is None:
            wb = wf[b]
        return max(wa, wb) - min(wa, wb) >= self.distance_between_sets(a, b) - tol

    def initial_work_function(self, config) -> "SupportWorkFunction":
        canonical = tuple(sorted(config))
        if len(canonical) != self.k:
            raise ValueError("config length must match k")
        return SupportWorkFunction(self, [canonical], [0.0])


class SupportWorkFunction:
    def __init__(self, context: SupportWFAContext, supp: list[tuple], vals: list[float]):
        if len(supp) != len(vals):
            raise ValueError("supp and vals must have same length")
        self.context = context
        self.supp = [tuple(sorted(s)) for s in supp]
        self.vals = [float(v) for v in vals]

    @lru_cache(maxsize=None)
    def _get_value(self, config: tuple) -> float:
        opt = float("inf")
        for supp_cfg, val in zip(self.supp, self.vals):
            cand = val + self.context.distance_between_sets(config, supp_cfg)
            if cand < opt:
                opt = cand
        return opt

    def __getitem__(self, config) -> float:
        canonical = tuple(sorted(config))
        if len(canonical) != self.context.k:
            raise ValueError("config length must match k")
        return self._get_value(canonical)

    def normalized(self) -> tuple[float, "SupportWorkFunction"]:
        shift = min(self.vals) if self.vals else 0.0
        return shift, SupportWorkFunction(self.context, self.supp, [v - shift for v in self.vals])

    def update(self, request, seen: set, tol: float | None = None) -> "SupportWorkFunction":
        req = request
        seen_points = sorted(seen)
        if req not in seen_points:
            seen_points.append(req)
            seen_points.sort()

        # Canonicalize candidates early to remove duplicates from repeated product tuples.
        candidate_best: dict[tuple, float] = {}
        for points in product(seen_points, repeat=self.context.k - 1):
            candidate = tuple(sorted(tuple(points) + (req,)))
            val = self[candidate]
            prev = candidate_best.get(candidate)
            if prev is None or val < prev:
                candidate_best[candidate] = val

        candidate_supp = list(candidate_best.keys())
        candidate_vals = [candidate_best[cfg] for cfg in candidate_supp]

        dominated: set[int] = set()
        for i in range(len(candidate_supp)):
            if i in dominated:
                continue
            for j in range(i):
                if j in dominated:
                    continue
                if self.context.is_dominated(
                    self,
                    candidate_supp[i],
                    candidate_supp[j],
                    tol=tol,
                    wa=candidate_vals[i],
                    wb=candidate_vals[j],
                ):
                    dominated.add(i if candidate_vals[i] > candidate_vals[j] else j)

        new_supp, new_vals = [], []
        for i, cfg in enumerate(candidate_supp):
            if i not in dominated:
                new_supp.append(cfg)
                new_vals.append(candidate_vals[i])

        return SupportWorkFunction(self.context, new_supp, new_vals)

    def dense_values(self, configs: list[tuple]) -> np.ndarray:
        return np.array([self[cfg] for cfg in configs], dtype=float)


def k_taxi_update_support(
    wf: SupportWorkFunction,
    request: tuple,
    seen: set,
) -> SupportWorkFunction:
    s, t = request[0], request[1]
    new_wf = wf.update(s, seen)

    new_supp: list[tuple] = []
    new_vals: list[float] = []
    delta = wf.context.distance(s, t)
    for cfg, val in zip(new_wf.supp, new_wf.vals):
        lst = list(cfg)
        lst.remove(s)
        lst.append(t)
        new_supp.append(tuple(sorted(lst)))
        new_vals.append(float(val + delta))
    return SupportWorkFunction(wf.context, new_supp, new_vals)
