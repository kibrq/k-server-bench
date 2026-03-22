from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence as SeqABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from kserverclean.context.numpy_wf_context import WFContext


class RowMappingProxy(Mapping):
    __slots__ = ("_cols", "_i", "_schema")

    def __init__(self, cols: dict[str, np.ndarray], i: int, schema: dict[str, Any]):
        self._cols = cols
        self._i = i
        self._schema = schema

    def __getitem__(self, key: str) -> Any:
        if key not in self._schema:
            raise KeyError(key)
        return self._schema[key](self._cols[key], self._i)

    def __iter__(self):
        yield from self._schema.keys()

    def __len__(self) -> int:
        return len(self._schema)


class ColumnarListProxy(SeqABC):
    __slots__ = ("_cols", "_n", "_schema", "_length_key")

    def __init__(self, cols: dict[str, np.ndarray], schema: dict[str, Any], length_key: str):
        self._cols = cols
        self._schema = schema
        self._length_key = length_key
        self._n = int(cols[length_key].shape[0])

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            rng = range(*idx.indices(self._n))
            return [RowMappingProxy(self._cols, i, self._schema) for i in rng]
        if idx < 0:
            idx += self._n
        if idx < 0 or idx >= self._n:
            raise IndexError(idx)
        return RowMappingProxy(self._cols, int(idx), self._schema)

    def __iter__(self):
        for i in range(self._n):
            yield RowMappingProxy(self._cols, i, self._schema)


def as_int(col: np.ndarray, i: int) -> int:
    return int(col[i])


def as_float(col: np.ndarray, i: int) -> float:
    return float(col[i])


def as_wf_view(col: np.ndarray, i: int) -> np.ndarray:
    return col[i]


@dataclass
class NumpyKServerInstance:
    k: int
    distance_matrix: np.ndarray
    node_id: np.ndarray
    node_depth: np.ndarray
    node_wf_norm: np.ndarray
    edge_from: np.ndarray
    edge_to: np.ndarray
    edge_ext: np.ndarray
    edge_d_min: np.ndarray
    edge_weight: np.ndarray
    bellman: Optional[np.ndarray] = None
    context: Optional[WFContext] = None

    def get_context(self) -> WFContext:
        if self.context is None:
            self.context = WFContext(k=self.k, distance_matrix=self.distance_matrix)
        return self.context

    def get_nodes(self) -> ColumnarListProxy:
        return ColumnarListProxy(
            {
                "id": self.node_id,
                "depth": self.node_depth,
                "wf_norm": self.node_wf_norm,
            },
            schema={
                "id": as_int,
                "depth": as_int,
                "wf_norm": as_wf_view,
            },
            length_key="id",
        )

    def get_edges(self) -> ColumnarListProxy:
        return ColumnarListProxy(
            {
                "from": self.edge_from,
                "to": self.edge_to,
                "ext": self.edge_ext,
                "d_min": self.edge_d_min,
                "weight": self.edge_weight,
            },
            schema={
                "from": as_int,
                "to": as_int,
                "ext": as_float,
                "d_min": as_float,
                "weight": as_float,
            },
            length_key="from",
        )

    def get_bellman(self) -> np.ndarray:
        if self.bellman is None:
            raise ValueError("bellman must be precomputed for NumpyKServerInstance")
        return np.asarray(self.bellman)

    def dump_numpy(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            k=np.asarray(self.k, dtype=np.int64),
            distance_matrix=np.asarray(self.distance_matrix),
            node_id=np.asarray(self.node_id),
            node_depth=np.asarray(self.node_depth),
            node_wf_norm=np.asarray(self.node_wf_norm),
            edge_from=np.asarray(self.edge_from),
            edge_to=np.asarray(self.edge_to),
            edge_ext=np.asarray(self.edge_ext),
            edge_d_min=np.asarray(self.edge_d_min),
            edge_weight=np.asarray(self.edge_weight),
            bellman=np.asarray(self.bellman) if self.bellman is not None else np.array([], dtype=float),
        )

    @classmethod
    def from_legacy_dict(cls, payload: dict[str, Any]) -> "NumpyKServerInstance":
        nodes = payload["nodes"]
        edges = payload["edges"]

        node_id = np.asarray([node["id"] for node in nodes], dtype=np.int32)
        node_depth = np.asarray([node["depth"] for node in nodes], dtype=np.int32)
        node_wf_norm = np.asarray([node["wf_norm"] for node in nodes], dtype=float)

        edge_from = np.asarray([edge["from"] for edge in edges], dtype=np.int32)
        edge_to = np.asarray([edge["to"] for edge in edges], dtype=np.int32)
        edge_ext = np.asarray([edge["ext"] for edge in edges], dtype=float)
        edge_d_min = np.asarray([edge["d_min"] for edge in edges], dtype=float)
        edge_weight = np.asarray([edge["weight"] for edge in edges], dtype=float)

        bellman = payload.get("bellman")
        if bellman is not None:
            bellman = np.asarray(bellman, dtype=float)

        return cls(
            k=int(payload["k"]),
            distance_matrix=np.asarray(payload["distance_matrix"]),
            node_id=node_id,
            node_depth=node_depth,
            node_wf_norm=node_wf_norm,
            edge_from=edge_from,
            edge_to=edge_to,
            edge_ext=edge_ext,
            edge_d_min=edge_d_min,
            edge_weight=edge_weight,
            bellman=bellman,
        )

    @classmethod
    def load(cls, path: str | Path) -> "NumpyKServerInstance":
        path = Path(path)
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                bellman = data["bellman"]
                if bellman.size == 0:
                    bellman = None
                return cls(
                    k=int(data["k"].item()),
                    distance_matrix=np.asarray(data["distance_matrix"]),
                    node_id=np.asarray(data["node_id"]),
                    node_depth=np.asarray(data["node_depth"]),
                    node_wf_norm=np.asarray(data["node_wf_norm"]),
                    edge_from=np.asarray(data["edge_from"]),
                    edge_to=np.asarray(data["edge_to"]),
                    edge_ext=np.asarray(data["edge_ext"]),
                    edge_d_min=np.asarray(data["edge_d_min"]),
                    edge_weight=np.asarray(data["edge_weight"]),
                    bellman=None if bellman is None else np.asarray(bellman),
                )

        with open(path, "rb") as f:
            payload = pickle.load(f)
        return cls.from_legacy_dict(payload)
