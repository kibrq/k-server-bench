## PotentialFamily-Only Contract

In this regime, only `PotentialFamily` should be modified.

- `Potential` is fixed by the evaluator.
- `SearchEvaluator` is fixed by the evaluator.
- Treat the task as a search problem over canonical `potential_kwargs`, especially `{"n", "coefs", "index_matrix"}`.

## Required interface

```python
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

    def ask(self) -> list[dict]:
        ...

    def tell(self, submission, result):
        ...

    def finalize(self) -> tuple[list[dict], dict]:
        ...
```

Submission shape:

```python
{
    "worker_input": {
        "potential_kwargs": {...},
        "potential_cls": "Potential",              # optional
        "search_evaluator_kwargs": {},             # optional
        "search_evaluator_cls": "SearchEvaluator", # optional
    },
    "timeout": 1.0,   # optional fraction in [0.01, 1.0]
    "metadata": {},
}
```

Recommendation shape:

```python
{
    "score": float,
    "kwargs": {...},
    "potential_cls": "Potential",  # optional
}
```

Typical `SearchEvaluator` result shape:

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

## Search behavior

- `ask()` should propose candidates.
- `tell()` should use evaluator feedback to guide future proposals.
- `finalize()` must return at least one recommendation.
- Prefer a real ask/tell loop over a fixed baseline.
- Avoid repeatedly submitting the same candidate.
