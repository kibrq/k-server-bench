## PotentialFamily-Only Evaluation Guide

This component is for the regime where only `PotentialFamily` is editable.

- `Potential` is fixed by the evaluator.
- `SearchEvaluator` is fixed by the evaluator.
- Your job is to implement a real outer search over candidate hyperparameters and return good final recommendations.

## Evaluation Flow

The evaluator runs an outer ask/tell loop:

1. `PotentialFamily.ask()` proposes candidate submissions.
2. Each submission is evaluated in parallel by `SearchEvaluator`.
3. Each evaluator run receives only a bounded worker timeout.
4. Results are passed back to `PotentialFamily.tell(...)`.
5. When the total search budget is exhausted, `PotentialFamily.finalize()` returns ranked recommendations.
6. The evaluator tests the top recommendations and the best final score becomes the run score.

## PotentialFamily Interface

```python
Submission = {
    "worker_input": {
        "potential_kwargs": Dict[str, Any],
        "potential_cls": Optional[str],           # default "Potential"
        "search_evaluator_kwargs": Optional[Dict[str, Any]],
        "search_evaluator_cls": Optional[str],    # default "SearchEvaluator"
    },
    "timeout": Optional[float],                   # in [0.01, 1.0]
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

    def tell(self, submission, result):
        ...

    def finalize(self) -> tuple[list[Recommendation], dict]:
        ...
```

Contract notes:

- `n_instances` is the number of instances available during search.
- `n_workers` is the number of concurrent evaluator workers.
- `search_timeout` is the total search budget.
- `min_worker_timeout` and `max_worker_timeout` define the per-submission budget range.
- `finalize()` must return a non-empty list, otherwise the run receives the worst score.

## Submission Semantics

Each submission should provide `worker_input["potential_kwargs"]`.

Typical optional fields:

- `potential_cls`: if omitted, evaluator usually uses `"Potential"`
- `search_evaluator_kwargs`: evaluator-specific settings
- `search_evaluator_cls`: if omitted, evaluator usually uses `"SearchEvaluator"`
- `timeout`: fraction of the worker budget in `[0.01, 1.0]`
- `metadata`: any extra bookkeeping, ideally including a unique signature for the candidate

What matters is how `PotentialFamily` chooses candidates and budgets.

Typical `search_evaluator_kwargs` may include things like:

- `instance_idx`

That means `PotentialFamily` can still choose which instance to probe.

## Result Shape

The fixed evaluator typically returns feedback of the form:

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

Your `tell()` logic should be prepared for partial-search feedback, not just a single scalar.

## Practical Search Guidance

`PotentialFamily` is the important part of this regime, not just a thin wrapper around fixed `potential_kwargs`.

You should:

- implement a real ask/tell loop
- avoid repeatedly proposing the same candidate
- use feedback from `tell()` to influence future proposals
- return the best discovered candidates from `finalize()`

You can still start simple, but a fixed baseline with no adaptation leaves a lot of performance on the table.


## Common Failure Modes

Avoid these mistakes:

- returning no recommendations from `finalize()`
- resubmitting the same candidate many times without learning
- using `PotentialFamily` as a fixed constant wrapper
- ignoring timeout scaling entirely
- proposing candidates that are obviously malformed for the canonical potential contract
