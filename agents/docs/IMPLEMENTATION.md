## Implementation Details


### PotentialFamily

You are not proposing just a single potential function, but rather a **search procedure over a family of potentials**. Your implementation should define three main components:

---

#### `PotentialFamily(n_instances: int, n_workers: int, search_timeout: float, min_worker_timeout: float, max_worker_timeout: float)`

This class controls the **search algorithm** over the space of potentials.

**Methods:**

* `ask() -> List[Submission]`
This method should suggest what hyperparameters of the `Potential` we should test. Every submission is:

```python
     {
        "worker_input": {
          "instance_idx": int, # index in [0, n_instances)
          "potential_cls": str, # usually "Potential"
          "potential_kwargs": dict, # parameters for Potential
          "search_evaluator_cls": str, # usually "SearchEvaluator"
          "search_evaluator_kwargs": dict, # extra kwargs for evalutor
        },
        # a fraction of the max allowed timeout for the submission ([0.01, 1.0])
        # the total timeout is min_worker_timeout + timeout_ratio * (max_worker_timeout - min_worker_timeout)
        "timeout": float,
        "metadata": {}, # anything you like to be kept alongside your submission, it is good to have unique key for potential_kwargs
     }
```


* `tell(Submission, Results)`
This method is used to tell you the results of the submission. Submission has the same format as the submissions you provide in `ask` and Results is the dict that is the output of the worker (`SearchEvaluator`).


* `finalize() -> Summary, ScoredAttempts`
This method symbolizes the end of the search. You should return as summary any dict with metrics you like and `ScoredAttempts` is the list of dicts with keys being `score` and `kwargs`. `score` is the score you assign to the given `kwargs` of the potential.


The complete search procedure is as follows:

```python
while Timeout:

  if len(in_flight_tasks) > 0:
    wait(in_flight_tasks)
    submission, result = ready.pop()
    potential_family.tell(submission, result)

  while len(in_flight_tasks) + len(queue) < n_workers:
    submissions = potential_family.ask()
    if not submissions:
      break
    queue.extend(submissions)

  while len(in_flight_tasks) < n_workers:
    submit(queue.pop())

return potential_family.finalize()
```

Then we take top_k attempts from ScoredAttempts and evaluate them on the test instances and return the best score.

#### Practical tips for `PotentialFamily`

- Put a **unique stable key** in `metadata` for dedup/caching, e.g.
  - `metadata={"key": json.dumps(potential_kwargs, sort_keys=True)}`
- Maintain an internal cache:
  - key -> best observed score, plus diagnostics (violations, processed edges)
- A simple robust ranking score often works: minimize `violations_k`.



#### `Potential(context, **kwargs)`

This class represents a single potential function.
**Methods:**

* `__call__(wf) -> float`: Computes the potential value for a given work function `wf` using the provided `context`.
  **Notes:**
* You may perform precomputation in `__init__` to make repeated evaluations fast.
* The `__call__` method will be invoked millions of times, so it must be extremely efficient.

`Potential.__call__(wf)` is called millions of times.

- Avoid allocations, Python loops, and attribute lookups inside `__call__`.
- Prefer:
  - precomputed numpy arrays in `__init__`
  - local variable binding inside `__call__`
  - simple reductions (`min`, `max`, dot products) over vectorized data
- Any slowdown reduces `processed_normalized_edges_score`, which makes `violations_k` less reliable.


---

#### `SearchEvaluator(nodes, edges, context, potential_kwargs, **kwargs)`

This class evaluates a single potential over a given set of edges.
**Methods:**

* `__call__() -> Dict[str, Any]`: Evaluates the specified potential and returns its performance metrics.

**Expected workflow:**

1. In `__init__`, instantiate `Potential(context, **potential_kwargs)`.
2. In `__call__`, iterate over the edges:

   * Compute `Φ(from)` and `Φ(to)` for each edge.
   * Check the inequality

     ```
     Φ(to) - Φ(from) + (k + 1) * d_min >= ext
     ```

     to determine violations.
3. Return a summary dictionary of the evaluation results.

**Additional requirements:**

* Must gracefully handle `KeyboardInterrupt` and return partial results.
* The evaluator runs under a strict timeout (typically ~15 seconds).
* `nodes` is a list of dicts with keys:

  * `id`
  * `wf_norm`: the normalized work function.
* `edges` is a list of dicts with keys:

  * `id`
  * `from`, `to`: node indices
  * `d_min`, `ext`: edge metadata describing optimal and extended costs.

#### Handling timeouts and interrupts

The evaluator must handle `KeyboardInterrupt` and return partial results.

A practical pattern:
- keep counters incrementally
- periodically check time budget (if available)
- return the best-known metrics so far

