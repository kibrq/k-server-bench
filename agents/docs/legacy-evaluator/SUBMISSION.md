# Potential Evaluation Guide

`evaluate.py` expects a three-component system:

- `Potential`: defines the potential template and builds a concrete potential from hyperparameters.
- `PotentialFamily`: runs the outer search over candidates (ask/tell strategy) and produces final recommendations.
- `SearchEvaluator`: assigns scores to candidate potentials on search instances and provides feedback for guiding the search.

## Evaluation flow

In a loop:
1. call `PotentialFamily.ask()` to propose candidate submissions.
2. Evaluate those candidates in parallel using `SearchEvaluator`.
3. Each evaluator runs within `SEARCH_WORKER_TIMEOUT` and returns results.
4. Send back `submissions + results` using `PotentialFamily.tell(...)`.

At timeout SEARCH_TIMEOUT:
5.  call `PotentialFamily.finalize()` to get recommendations.
6. Evaluate the top-k recommendations on test instances and select the best test score as the final score.

## Interfaces

### PotentialFamily

```py
# Not Python syntax, this is the expected shape.
Submission = {
    "worker_input": {
        # Required
        "potential_kwargs": Dict[str, Any],

        # Optional (defaults shown)
        "potential_cls": Optional[str] = "Potential",
        "search_evaluator_kwargs": Optional[Dict[str, Any]] = {},
        "search_evaluator_cls": Optional[str] = "SearchEvaluator",
    },

    # Fraction of worker timeout budget in [0.01, 1.0].
    # Total worker budget is:
    # min_worker_timeout + timeout * (max_worker_timeout - min_worker_timeout)
    "timeout": Optional[float] = 1.0,

    # Extra metadata to store with your submission (recommended: unique signature key).
    "metadata": Optional[Dict[str, Any]] = {},
}

Recommendation = {
    # Score used for ranking before final test evaluation.
    "score": float,

    # Hyperparameters passed to the potential constructor.
    "kwargs": Dict[str, Any],

    # Optional potential class name.
    "potential_cls": Optional[str] = "Potential",
}

class PotentialFamily:
    def __init__(
        self,
        # Number of instances available for SearchEvaluator
        n_instances: int,

        # The rest of the information is for the search logic

        # Number of workers
        n_workers: int,

        # Total search timeout for the procedure
        # Typically 60 to 600 seconds
        search_timeout: float,

        # Minimum per-candidate evaluator budget, typically 3-5 seconds
        min_worker_timeout: float,

        # Maximum per-candidate evaluator budget, typically 10-60 seconds
        max_worker_timeout: float,
    )

    # Returns candidates to evaluate. Can be empty.
    def ask(self) -> List[Submission]:
        pass

    # Receives evaluator results for the proposed candidates.
    # Result structure is defined by your SearchEvaluator.
    def tell(self, submissions: List[Submission], results: List[Result]):
        pass

    # Called at the end of search.
    # First argument: recommendations, second: JSON-serializable summary.
    # Must return a non-empty list; empty is a contract violation.
    def finalize(self) -> Tuple[List[Recommendation], Dict[str, Any]]:
        pass
```

Naive implementation might not do any search and return some kwargs right away assigning them zero score.

### Potential

```py
# Metric points are integers and have pairwise distances.
# WFContext provides all distance/work-function helpers.
class WFContext:
    # Number of servers
    k: int

    # Number of metric points
    m: int

    # Distance between two points
    def distance(i, j):
        pass

    # Map a configuration (length `k`) to its canonical configuration index.
    def config_to_idx(config: Sequence[int]):
        pass

    # Inverse mapping from index to canonical configuration
    idx_to_config: List[int]

    # Sum of pairwise distances within a configuration index
    def distance_within_config(idx: int):
        pass

    # Pairwise-distance sum for explicit point indices
    def distance_within_clique(S):
        pass

    # Distance between two configuration indices
    def distance_between_configs(idx1: int, idx2: int):
        pass

    # Distance between two explicit sets
    def distance_between_sets(S: Sequence[int], T: Sequence[int]):
        pass

    # Initial work function for a configuration
    def initial_wf(servers: Sequence[int]) -> np.ndarray:
        pass

    # Updated work function after request
    def update_wf(wf: np.ndarray, request: int) -> np.ndarray:
        pass


"""
Important:

* `wf` is indexed by configuration index, not raw point labels.
* Always use `config_to_idx(...)` or `idx_to_config[...]` for indexing; never assume a manual formula.
"""


class Potential:
    # Optional precomputation belongs here.
    def __init__(self, context: WFContext, **kwargs):
        pass

    # Core potential evaluation. Must be fast.
    # This is called for all nodes in the instances (hundreds of thousands).
    # wf might be a tuple, so start your implentation with np.asarray(wf)
    # Can return additional meta information
    def __call__(self, wf: Union[tuple, np.ndarray]) -> Union[Tuple[float, Dict[str, Any]], float]:
        pass
```

Naive implementation might just return 0 for all wf or for example sum of all work function values.

### SearchEvaluator

```py
Node = Dict[
    # Node index in the graph
    "idx": int

    # Normalized work function of the node
    "wf_norm": np.ndarray
]


# Edge is violated with competitive ratio c iff:
# Phi(v) - Phi(u) + (c + 1) * d_min < ext
Edge = Dict[
    # Start node
    "u": int

    # End node
    "v": int

    # Optimum work-function delta before/after request
    "d_min": float

    # Extended cost of the request
    "ext": float
]


class Instance:
    def get_nodes() -> List[Node]:
        pass

    def get_edges() -> List[Edge]:
        pass

    def get_context() -> WFContext:
        pass


"""
Note that get_edges() and get_nodes() is slightly more complicated and
you can access get as readonly numpy array by asking 

#```py
nodes = instance.get_nodes()
edges = instance.get_edges()

vj = edges._get_as_np_array('to')
uj = edges._get_as_np_array('from')

d_min = edges._get_as_np_array('d_min')
ext = edges._get_as_np_array('ext')

wf_norms = nodes._get_as_np_array('wf_norm')
#```

Be careful, some of the instances are too large (7 millions of edges), you have strict memory and time constraints.
"""


class SearchEvaluator:
    def __init__(
        self,

        # Instances used during search
        instance: List[Instance],

        # Potential class selected by PotentialFamily
        # It is not a string, but a cls object --
        # potential_cls(context, **kwargs) would create a potential object
        potential_cls: cls,

        # Potential hyperparameters selected by PotentialFamily
        potential_kwargs: Dict[str, Any],

        # Runtime budget for this evaluator run
        timeout: float,

        # Extra kwargs from `search_evaluator_kwargs`
        # Example: naive implementation supports `instance_idx`.
        **kwargs,
    ):
        pass

    # Evaluate the candidate hyperparameters.
    # KeyboardInterrupt may happen on timeout; handle it and return a partial result.
    # Result can be any structure your PotentialFamily expects.
    def __call__(self) -> Result:
        pass
```

Naive implentation of the SearchEvaluator might go over all edges until the timeout, compute Phi(u), Phi(v), test the inequality and accumulate the number of violations and return the number of edges it managed to process until the timeout and number of incurred violations.

### Additional Notes

- You can define multiple potential classes and search evaluators in one file. If you use non-standard class names, set `potential_cls` / `search_evaluator_cls` explicitly.
- Usually one `Potential` and one `SearchEvaluator` is enough.
- `Potential.__call__` should be fast; it may run on the order of hundreds of thousands of nodes.
- SearchEvaluator can optimize additional hyperparameters. Example: PotentialFamily may pass partial `potential_kwargs` and SearchEvaluator fills in the rest and returns to PotentialFamily.
- SearchEvaluator may use early stopping or do accumulatation (e.g., hard edges to test first).
- `PotentialFamily.finalize()` must return at least one recommendation, otherwise the run receives the worst score.
- Some pipeline components may be fixed by platform instructions. If a component is fixed, do not redefine it.
