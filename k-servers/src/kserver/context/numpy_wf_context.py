import numpy as np
from scipy.optimize import linear_sum_assignment

from .common import all_multicombinations


class WFContext:
    def __init__(
        self,
        k: int,
        distance_matrix: np.ndarray,
        idx_to_config: list[tuple[int, ...]] | None = None,
    ):
        self.k = k
        self.distance_matrix = np.asarray(distance_matrix)
        self.m = distance_matrix.shape[0]
        self.use_multicombinations = idx_to_config is None

        if idx_to_config is None:
            self._idx_to_config = all_multicombinations(self.m, k)
        else:
            self._idx_to_config = [tuple(sorted(config)) for config in idx_to_config]
            self._validate_idx_to_config()
        self._config_to_idx = {c: i for i, c in enumerate(self._idx_to_config)}

        num_configs = len(self._idx_to_config)
        self.neighbors = np.zeros((self.m, num_configs, self.k), dtype=int)
        self.move_cost = np.zeros((self.m, num_configs, self.k), dtype=int)

        for r in range(self.m):
            for uidx, uconfig in enumerate(self._idx_to_config):
                for i in range(self.k):
                    vconfig = (*uconfig[:i], r, *uconfig[i + 1 :])
                    vconfig = tuple(sorted(vconfig))
                    vidx = self._config_to_idx.get(vconfig)
                    if vidx is None:
                        raise ValueError(
                            "custom idx_to_config must be closed under single-server request updates"
                        )
                    self.neighbors[r, uidx, i] = vidx
                    self.move_cost[r, uidx, i] = self.distance_matrix[uconfig[i], r]

    def _validate_idx_to_config(self) -> None:
        if len(self._idx_to_config) == 0:
            raise ValueError("custom idx_to_config must not be empty")

        seen = set()
        for config in self._idx_to_config:
            if len(config) != self.k:
                raise ValueError("every custom idx_to_config entry must have length k")
            if any((x < 0 or x >= self.m) for x in config):
                raise ValueError("custom idx_to_config entries must use indices in [0, m)")
            if tuple(config) != tuple(sorted(config)):
                raise ValueError("custom idx_to_config entries must be sorted")
            if config in seen:
                raise ValueError("custom idx_to_config entries must be unique")
            seen.add(config)

    def distance(self, a: int, b: int) -> int:
        return self.distance_matrix[a, b]

    def distance_between_sets(self, C: tuple[int, ...], D: tuple[int, ...]) -> int:
        cost = np.zeros((self.k, self.k), dtype=int)
        for i in range(self.k):
            for j in range(self.k):
                cost[i, j] = self.distance_matrix[C[i], D[j]]
        row_ind, col_ind = linear_sum_assignment(cost)
        return int(cost[row_ind, col_ind].sum())

    def config_to_idx(self, config: tuple[int, ...]) -> int:
        return self._config_to_idx[tuple(sorted(config))]

    def initial_wf(self, uconfig: tuple[int, ...]) -> np.ndarray:
        wf = np.zeros(len(self._idx_to_config), dtype=int)
        for vidx, vconfig in enumerate(self._idx_to_config):
            wf[vidx] = self.distance_between_sets(uconfig, vconfig)
        return wf

    def update_wf(self, wf: np.ndarray, request: int) -> np.ndarray:
        wf_neighbors = wf[self.neighbors[request]]
        candidates = wf_neighbors + self.move_cost[request]
        return np.min(candidates, axis=1)


def _ensure_preimage_cache(context: WFContext) -> dict[tuple[int, int], np.ndarray]:
    if not hasattr(context, "_preimage_cache"):
        context._preimage_cache = {}
    return context._preimage_cache


def _build_preimage_for_request(context: WFContext, s: int, t: int) -> np.ndarray:
    num_cfgs = len(context._idx_to_config)
    preimage = np.full(num_cfgs, -1, dtype=int)

    for idx, D in enumerate(context._idx_to_config):
        if t not in D:
            continue

        lst = list(D)
        for pos, val in enumerate(lst):
            if val == t:
                lst[pos] = s
                break

        new_config = tuple(sorted(lst))
        preimage[idx] = context.config_to_idx(new_config)

    return preimage


def k_taxi_update(context: WFContext, wf: np.ndarray, request: tuple[int, int]) -> np.ndarray:
    s, t = request
    delta_st = context.distance_matrix[s, t]
    inf = int(1e8)

    cache = _ensure_preimage_cache(context)
    key = (s, t)
    if key not in cache:
        cache[key] = _build_preimage_for_request(context, s, t)
    preimage_idx = cache[key]

    tilde_wf = np.full_like(wf, inf, dtype=wf.dtype)
    valid_mask = preimage_idx >= 0
    tilde_wf[valid_mask] = wf[preimage_idx[valid_mask]] + delta_st

    return context.update_wf(tilde_wf, t)
