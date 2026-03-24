from itertools import product

import numpy as np


class Potential:
    def __init__(self, context, **kwargs):
        self.context = context
        self.kwargs = kwargs

        self.k = context.k
        self.n = kwargs["n"]
        self.index_matrix = kwargs["index_matrix"]
        self.coefs = kwargs["coefs"]
        self.support = kwargs.get("support", None)
        if self.support is None:
            self.support = range(context.m)
        self.support = list(self.support)

        self.config_idxes_dtype = kwargs.get("config_idxes_dtype", np.uint16)
        self.coefs_dtype = kwargs.get("coefs_dtype", np.int32)

        self.validate_kwargs()

        self.distances = self._precompute_distances(context, self.n, self.coefs_dtype)
        self.config_idxes = self._precompute_config_idxes(
            context,
            self.n,
            self.index_matrix,
            self.config_idxes_dtype,
        )
        self.penalties = self._precompute_distances_dot_coefs(
            context,
            self.n,
            self.coefs,
            self.coefs_dtype,
        )

        self.tmp = np.zeros(self.config_idxes.shape[1], dtype=self.coefs_dtype)
        self.scratch = np.zeros(self.config_idxes.shape[1], dtype=self.coefs_dtype)

    def validate_kwargs(self):
        if self.coefs is not None:
            assert len(self.coefs) == self.n * (self.n - 1) // 2, (
                "coefs must be of length n * (n - 1) // 2"
            )
            try:
                for x in self.coefs:
                    float(x)
            except ValueError:
                assert False, "coefs must be a tuple of NUMBERS"

        for row in self.index_matrix:
            assert len(row) == self.k, "every row of index_matrix should be exactly k"

        assert len(self.index_matrix) >= self.k + 1, "index_matrix must be at least k + 1 rows"

    def _precompute_config_idxes(self, context, n, index_matrix, dtype):
        m = context.m
        antipode = lambda x: (x + m // 2) % m

        config_idxes = []
        for points in product(self.support, repeat=n):
            config_idxes.append([])
            antipodes = [antipode(p) for p in points]

            for row in index_matrix:
                cfg = tuple(points[r - 1] if r > 0 else antipodes[abs(r) - 1] for r in row)
                config_idxes[-1].append(context.config_to_idx(cfg))

        return np.ascontiguousarray(np.array(config_idxes, dtype=dtype).T)

    def _precompute_distances(self, context, n, dtype):
        distances = []
        for points in product(self.support, repeat=n):
            distances.append([])
            for i in range(n):
                for j in range(i + 1, n):
                    distances[-1].append(context.distance(points[i], points[j]))
        return np.array(distances, dtype=dtype)

    def _precompute_distances_dot_coefs(self, context, n, coefs, dtype=None):
        if coefs is None:
            coefs = np.zeros(n * (n - 1) // 2, dtype=dtype)
        else:
            coefs = np.array(coefs, dtype=dtype)

        if not hasattr(self, "distances"):
            self.distances = self._precompute_distances(context, n, dtype)

        return np.dot(self.distances, coefs)

    def _compute_candidate_values(self, wf):
        wf = np.asarray(wf, dtype=self.coefs_dtype)

        np.take(wf, self.config_idxes[0], out=self.tmp)
        for i in range(1, self.config_idxes.shape[0]):
            np.take(wf, self.config_idxes[i], out=self.scratch)
            self.tmp += self.scratch

        self.tmp -= self.penalties
        return self.tmp

    def __call__(self, wf):
        self._compute_candidate_values(wf)
        argmin_idx = np.argmin(self.tmp)
        return self.tmp[argmin_idx], {"idx": int(argmin_idx)}
