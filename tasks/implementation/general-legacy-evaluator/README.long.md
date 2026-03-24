## General Legacy Evaluator Guide

This component targets the legacy evaluator regime where your program supplies the full three-part system:

- `Potential`: defines the potential template and evaluates a concrete potential.
- `PotentialFamily`: runs the outer ask/tell search and returns final recommendations.
- `SearchEvaluator`: scores candidate potentials on search instances and returns feedback.

The contract below is adapted from `k-servers/potential-evaluation/EVALUATE_SUBMISSION.md` and rewritten into the release task-component format.

## Evaluation Flow

The evaluator runs an outer search loop:

1. `PotentialFamily.ask()` proposes candidate submissions.
2. Each candidate is evaluated in parallel by `SearchEvaluator`.
3. Each evaluator run receives a bounded worker timeout.
4. Results are passed back through `PotentialFamily.tell(...)`.
5. When the global search timeout is exhausted, `PotentialFamily.finalize()` returns ranked recommendations.
6. The evaluator tests the top recommendations and uses the best final score as the run score.

## PotentialFamily Interface

```python
Submission = {
    "worker_input": {
        "potential_kwargs": Dict[str, Any],
        "potential_cls": Optional[str],           # default "Potential"
        "search_evaluator_kwargs": Optional[Dict[str, Any]],
        "search_evaluator_cls": Optional[str],    # default "SearchEvaluator"
    },
    "timeout": Optional[float],                   # fraction in [0.01, 1.0]
    "metadata": Optional[Dict[str, Any]],
}

Recommendation = {
    "score": float,
    "kwargs": Dict[str, Any],
    "potential_cls": Optional[str],
}

class PotentialFamily:
    def __init__(
        self,
        n_instances: int,
        n_workers: int,
        search_timeout: float,
        min_worker_timeout: float,
        max_worker_timeout: float,
    ):
        ...

    def ask(self) -> list[Submission]:
        ...

    def tell(self, submissions, results):
        ...

    def finalize(self) -> tuple[list[Recommendation], dict]:
        ...
```

Contract notes:

- `n_instances` is the number of search instances available during search.
- `n_workers` is the evaluator-side search parallelism.
- `search_timeout` is the total search budget.
- `min_worker_timeout` and `max_worker_timeout` define the per-submission budget range.
- `finalize()` must return a non-empty list.

## Potential Interface

The evaluator constructs `Potential(context, **kwargs)` and then calls it on many normalized work functions.

```python
class Potential:
    def __init__(self, context, **kwargs):
        ...

    def __call__(self, wf):
        ...
```

Practical notes:

- `wf` may arrive as a tuple, so start with `wf = np.asarray(wf)`.
- This method may be called hundreds of thousands of times.
- Heavy precomputation belongs in `__init__`, not in `__call__`.
- The return value may be either a scalar or `(value, metadata)`.

## SearchEvaluator Interface

```python
class SearchEvaluator:
    def __init__(
        self,
        instances,
        potential_cls,
        potential_kwargs,
        timeout: float = None,
        **kwargs,
    ):
        ...

    def __call__(self):
        ...
```

`SearchEvaluator` is responsible for using the selected potential on one or more search instances and returning feedback that `PotentialFamily.tell(...)` can use.

Typical inputs:

- `instances`: search instances available for the evaluator run
- `potential_cls`: the selected class object, not just its string name
- `potential_kwargs`: candidate hyperparameters
- `timeout`: runtime budget for this evaluator call
- `**kwargs`: extra evaluator-specific fields from `search_evaluator_kwargs`

Typical result shape:

```python
{
    "edges_processed": int,
    "violations": [
        {
            "edge_idx": int,
            "edge_up": float,
            "edge_vp": float,
            "edge_ext": float,
            "edge_d_min": float,
        },
        ...
    ],
    "edges_total": int,
}
```

The exact result schema is up to you, but it should be stable and useful for `tell()`.

## Useful Context and Instance Operations

`Potential` usually receives a `WFContext` through the selected instance. Useful methods include:

- `context.k`
- `context.m`
- `context.distance(i, j)`
- `context.config_to_idx(config)`
- `context.idx_to_config[idx]`
- `context.distance_within_config(idx)`
- `context.distance_between_configs(idx1, idx2)`
- `context.initial_wf(servers)`
- `context.update_wf(wf, request)`

Important indexing note:

- `wf` is indexed by canonical configuration index, not by raw point labels.
- Use `config_to_idx(...)` or `idx_to_config[...]` rather than assuming a manual indexing formula.

Instances may expose node and edge arrays. Some large instances have millions of edges, so memory and time discipline matter.

## Search Design Guidance

The quality of this regime depends on how well the three pieces cooperate.

You should:

- make `Potential.__call__` cheap enough for repeated evaluation
- cache repeated potential calls inside `SearchEvaluator`
- design `SearchEvaluator` to provide informative partial feedback
- implement a real ask/tell search in `PotentialFamily`
- avoid repeatedly evaluating the exact same candidate without reason
- return a non-empty ranked list from `finalize()`

## Common Failure Modes

Avoid these mistakes:

- returning an empty recommendation list from `finalize()`
- writing a `Potential` that is too slow for large edge sets
- ignoring timeout interruptions in `SearchEvaluator`
- coupling `PotentialFamily.tell(...)` to a result format that `SearchEvaluator` does not actually return
- doing no search at all when the regime allows all three components to improve together
