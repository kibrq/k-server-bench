## General Legacy Evaluator Contract

This component targets the legacy evaluator regime where all three components are editable:

- `Potential`
- `PotentialFamily`
- `SearchEvaluator`

Your program should define those classes in one Python file.

## Evaluation Flow

1. `PotentialFamily.ask()` proposes submissions.
2. `SearchEvaluator` evaluates those submissions on search instances.
3. `PotentialFamily.tell(...)` receives the search feedback.
4. At timeout, `PotentialFamily.finalize()` returns ranked recommendations.
5. The evaluator runs final evaluation on the top recommendations.

## Required Interfaces

`PotentialFamily` must provide:

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

    def tell(self, submissions, results):
        ...

    def finalize(self) -> tuple[list[dict], dict]:
        ...
```

`Potential` must provide:

```python
class Potential:
    def __init__(self, context, **kwargs):
        ...

    def __call__(self, wf):
        ...
```

`SearchEvaluator` must provide:

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

## Submission Shape

```python
{
    "worker_input": {
        "potential_kwargs": {...},
        "potential_cls": "Potential",              # optional
        "search_evaluator_kwargs": {...},          # optional
        "search_evaluator_cls": "SearchEvaluator", # optional
    },
    "timeout": 1.0,   # optional fraction in [0.01, 1.0]
    "metadata": {},
}
```

## Recommendation Shape

```python
{
    "score": float,
    "kwargs": {...},
    "potential_cls": "Potential",  # optional
}
```

## Practical Notes

- `Potential.__call__` is on the hot path and must be fast.
- `SearchEvaluator.__call__` may be interrupted on timeout; return partial progress.
- Cache potential values by node/work-function when possible.
- `finalize()` must return a non-empty recommendation list.
