from collections.abc import Sequence as SeqABC, Mapping as MapABC
from typing import Any, Iterator, Optional
import numpy as np
import ray
import sys
from pathlib import Path

LEGACY_EVALUATOR_HOME = Path(__file__).resolve().parent
KSERVERCLEAN_SRC = LEGACY_EVALUATOR_HOME.parent.parent / "k-servers" / "src"

if str(KSERVERCLEAN_SRC) not in sys.path:
    sys.path.insert(0, str(KSERVERCLEAN_SRC))

from kserver.context.numpy_wf_context import WFContext
from kserver.evaluation import NumpyKServerInstance

class RowMappingProxy(MapABC):
    __slots__ = ("_cols", "_i", "_schema")

    def __init__(self, cols: dict[str, np.ndarray], i: int, schema: dict[str, Any]):
        self._cols = cols
        self._i = i
        self._schema = schema  # converters per key

    def __getitem__(self, k: str) -> Any:
        if k not in self._schema:
            raise KeyError(k)
        conv = self._schema[k]
        return conv(self._cols[k], self._i)

    def __iter__(self):
        yield from self._schema.keys()

    def __len__(self):
        return len(self._schema)

    def __repr__(self):
        # lightweight repr for debugging
        return "{" + ", ".join(f"{k}: {self[k]!r}" for k in self._schema.keys()) + "}"


class RayColumnarListProxy(SeqABC):
    """Sequence of dict-like rows backed by Ray object store column arrays."""
    __slots__ = ("_refs", "_cols", "_n", "_schema", "_length_key")

    def __init__(self, refs: dict[str, "ray.ObjectRef"], schema: dict[str, Any], length_key: str):
        self._refs = refs
        self._schema = schema
        self._cols: Optional[dict[str, np.ndarray]] = None
        self._n: Optional[int] = None
        self._length_key = length_key

    def _ensure(self):
        if self._cols is None:
            keys = list(self._refs.keys())
            arrays = ray.get([self._refs[k] for k in keys])
            self._cols = dict(zip(keys, arrays))
            self._n = int(self._cols[self._length_key].shape[0])

    def __len__(self):
        self._ensure()
        return self._n  # type: ignore

    def _get_as_np_array(self, key: str):
        self._ensure()
        return self._cols[key]

    def __getitem__(self, idx):
        self._ensure()
        if isinstance(idx, slice):
            rng = range(*idx.indices(self._n))  # type: ignore
            return [RowMappingProxy(self._cols, i, self._schema) for i in rng]  # type: ignore
        if idx < 0:
            idx += self._n  # type: ignore
        if idx < 0 or idx >= self._n:  # type: ignore
            raise IndexError(idx)
        return RowMappingProxy(self._cols, int(idx), self._schema)  # type: ignore

    def __iter__(self) -> Iterator[MapABC]:
        self._ensure()
        for i in range(self._n):  # type: ignore
            yield RowMappingProxy(self._cols, i, self._schema)  # type: ignore


from dataclasses import dataclass
from typing import Optional, Any

def as_int(col, i): return int(col[i])
def as_float(col, i): return float(col[i])
def as_wf_view(col, i): return col[i]                     # numpy view
# def as_wf_tuple(col, i): return tuple(map(float, col[i]))  # if you truly need tuple

@dataclass
class KServerInstanceRayFriendly:
    k: int
    distance_matrix: np.ndarray  # already numpy, good

    # Node column refs
    node_id_ref: "ray.ObjectRef"
    node_depth_ref: "ray.ObjectRef"
    node_wf_norm_ref: "ray.ObjectRef"

    # Edge column refs
    edge_from_ref: "ray.ObjectRef"
    edge_to_ref: "ray.ObjectRef"
    edge_ext_ref: "ray.ObjectRef"
    edge_d_min_ref: "ray.ObjectRef"
    edge_weight_ref: "ray.ObjectRef"

    # Optional cached/heavy computed fields (keep like before)
    context: Optional[Any] = None
    graph: Optional[Any] = None
    bellman: Optional[Any] = None

    def get_context(self):
        if self.context is None:
            self.context = WFContext(k=self.k, distance_matrix=self.distance_matrix)
        return self.context


    def get_nodes(self):
        refs = {
            "id": self.node_id_ref,
            "depth": self.node_depth_ref,
            "wf_norm": self.node_wf_norm_ref,
        }
        schema = {
            "id": as_int,
            "depth": as_int,
            "wf_norm": as_wf_view,  # or as_wf_tuple for exact compatibility
        }
        return RayColumnarListProxy(refs, schema=schema, length_key="id")

    def get_edges(self):
        refs = {
            "from": self.edge_from_ref,
            "to": self.edge_to_ref,
            "ext": self.edge_ext_ref,
            "d_min": self.edge_d_min_ref,
            "weight": self.edge_weight_ref,
        }
        schema = {
            "from": as_int,
            "to": as_int,
            "ext": as_float,
            "d_min": as_float,
            "weight": as_float,
        }
        return RayColumnarListProxy(refs, schema=schema, length_key="from")

    def get_bellman(self):
        if self.bellman is None:
            raise ValueError("Bellman is not computed yet")
        return self.bellman


import numpy as np
import ray

def pack_nodes(nodes: list[dict]):
    n = len(nodes)
    node_id = np.empty(n, dtype=np.int32)
    depth = np.empty(n, dtype=np.int32)
    wfs = []
    for i, d in enumerate(nodes):
        node_id[i] = d["id"]
        depth[i] = d["depth"]
        wfs.append(np.asarray(d["wf_norm"], dtype=np.float32))
    return node_id, depth, np.stack(wfs)

def pack_edges(edges: list[dict]):
    m = len(edges)
    ef = np.empty(m, dtype=np.int32)
    et = np.empty(m, dtype=np.int32)
    ext = np.empty(m, dtype=np.float32)
    dmin = np.empty(m, dtype=np.float32)
    w = np.empty(m, dtype=np.float32)
    for i, e in enumerate(edges):
        ef[i] = e["from"]
        et[i] = e["to"]
        ext[i] = e["ext"]
        dmin[i] = e["d_min"]
        w[i] = e["weight"]
    return ef, et, ext, dmin, w

def to_ray_friendly(inst: NumpyKServerInstance) -> KServerInstanceRayFriendly:
    node_id, node_depth, node_wf = pack_nodes(inst.get_nodes())
    ef, et, ext, dmin, w = pack_edges(inst.get_edges())

    return KServerInstanceRayFriendly(
        k=inst.k,
        distance_matrix=inst.distance_matrix,
        node_id_ref=ray.put(node_id),
        node_depth_ref=ray.put(node_depth),
        node_wf_norm_ref=ray.put(node_wf),
        edge_from_ref=ray.put(ef),
        edge_to_ref=ray.put(et),
        edge_ext_ref=ray.put(ext),
        edge_d_min_ref=ray.put(dmin),
        edge_weight_ref=ray.put(w),
        bellman=ray.put(inst.get_bellman()),  # optional; consider packing too if big
    )
