import numpy as np

from kserverclean.context import all_multicombinations
from kserverclean.graph import create_worker_potential_hooks, parallel_bfs_exploration
from kserverclean.metrics.circle import Circle


class SumPotential:
    def __init__(self, context):
        self.context = context

    def __call__(self, wf):
        wf = np.asarray(wf)
        return float(wf.sum()), {"argmin": int(np.argmin(wf))}


def create_hash_fn(context):
    import hashlib

    def hash_fn(wf):
        wf = np.asarray(wf)
        wf_norm = wf - wf.min()
        return hashlib.sha256(wf_norm.tobytes()).hexdigest(), {"wf_norm": tuple(wf_norm.tolist())}

    return hash_fn


def test_worker_potential_hook_computes_on_nodes() -> None:
    k, m = 2, 4
    result = parallel_bfs_exploration(
        k=k,
        distance_matrix=Circle.discrete(m=m),
        n_workers=0,
        initial_nodes=list(all_multicombinations(m, k=k)),
        transitions=[lambda context, wf, r=r: context.update_wf(wf, r) for r in range(m)],
        create_hash_fn=create_hash_fn,
        worker_hook_constructors=create_worker_potential_hooks(SumPotential),
    )

    for node in result["node_bookkeeper"].nodes.values():
        assert "potential" in node.metadata
        assert "potential_meta" in node.metadata
