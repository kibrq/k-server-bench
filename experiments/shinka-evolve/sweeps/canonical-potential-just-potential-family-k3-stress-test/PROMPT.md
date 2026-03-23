# k-Server Problem

You are proposing a *potential* for the k-server problem. Recall that a state in the k-server problem can be described by its current work function: a function mapping every configuration (a multiset of k points) to a real number representing the minimum cost required to serve all previous requests and end in that configuration.
Your goal is to propose a potential: a function that maps each work function to a real value.

## Scoring

Your potential will be evaluated based on the number of violations it incurs. Given a work function ( w ), you can compute the updated work function after serving request ( r ), denoted ( w' ). The potential must satisfy:

[
\Phi(w') - \Phi(w) >= \max_X \big( w'(X) - w(X) \big)
]

or normalized version:

[
\Phi(w'_norm) - \Phi(w_norm) + (k + 1) (\min_X w'(X) - \min_X w(X)) >= \max_X \big( w'(X) - w(X) \big)
]

Your score, which you want to minimize, is called `violations_k`. Ideally, this number should be 0 (equivalently, `violations_k_score` should be 1).

Metrics also include `processed_normalized_edges_score` (from 0 to 1) which is the score of how many violations were actually processed. This depends on the computation speed of your proposed potential. Usually this will be 1. If `processed_normalized_edges_score` is low, the `violations_k_score` might be less accurate.

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

## Canonical Potential

The canonical potential has the form:

\[
\min_{a_1, \ldots, a_n}
\sum_{r} wf[a_{\text{index\_matrix}[r,1]}, \ldots, a_{\text{index\_matrix}[r,k]}]
- \sum_{i<j} \text{coef}_{i,j} \, d(i,j)
\]

If `index_matrix[r, c] < 0`, the implementation uses the antipode of `a_{abs(index_matrix[r, c])}`.

The canonical `Potential` requires these keys:

- `n`: number of abstract points `a_1, ..., a_n`
- `index_matrix`: tuple/list of rows, where every row has length exactly `k`
- `coefs`: tuple/list of pairwise coefficients of length `n * (n - 1) // 2`, or `None` (all zeros)

If these keys are missing or malformed, construction will fail.

The implementation does not enforce stronger mathematical canonicalization for you. In particular, you should still treat these as your responsibility:

- sorting entries inside each row when you want equivalent rows to collapse
- sorting rows lexicographically when you want equivalent matrices to collapse
- avoiding duplicate exploration of the same `index_matrix`
- keeping `n` small enough (avoid using n > 8) for the search to remain computationally feasible

## Row Count Guidance

For the k-server inequality, using fewer than `k + 1` rows is mathematically invalid if your goal is to achieve `violations_k = 0`. Such a potential should not be expected to score zero violations.

Using exactly `k + 1` rows is the necessary regime to target `violations_k = 0`.

Using more than `k + 1` rows is not expected to achieve `violations_k = 0` either, but it can still reduce `violations_k`. In practice, allowing extra rows can help you identify which rows are useless or redundant before shrinking back to a tighter structure.

So the practical guidance is:

- if you want a plausible zero-violation candidate, keep the number of rows exactly `k + 1`
- if you want to explore or diagnose structure, it can still make sense to test more than `k + 1` rows

## Semantics Of `index_matrix`

Each row of `index_matrix` describes one term of the work-function sum.

- positive entry `r > 0` means use point `a_r`
- negative entry `r < 0` means use the antipode of `a_{abs(r)}`
- entries are 1-based with respect to `a_1, ..., a_n`

This means every absolute index appearing in `index_matrix` must be between `1` and `n`. The implementation expects that; it does not add its own bounds-checking layer around malformed indices.

Good ways to use it:

- search over `n`
- search over `coefs`
- search over `index_matrix`
- canonicalize candidate structures before submitting them

Avoid assumptions that are not guaranteed by the implementation, such as:

- automatic deduplication of equivalent `index_matrix` values
- automatic clipping or validation of out-of-range row indices
- cheap scaling to very large `n`

Usually `coefs` that work best are small integers from -5 to 5.

## Metrics Hint: k3 Stress Test

In this regime, you are given exactly two benchmark instance files:

- `circle_k3_m6.pickle`
- `circle_k3_m8.pickle`

Treat these as:

- circle metrics
- `k = 3`
- `m = 6` and `m = 8`
- `circle_k3_m6.pickle`: `350` nodes and `2100` inequalities
- `circle_k3_m8.pickle`: `5240` nodes and `41920` inequalities

This is a multi-instance stress-test setting.

Practical implication for the agent:

- the solution should achieve `violations_k = 0` on both instances
- the combined score is the product of the `violations_k_score` values on the two instances
- improving only one instance is not enough if the other instance still has many violations
- you should optimize for robustness across both `m = 6` and `m = 8`, not just for the smaller instance
