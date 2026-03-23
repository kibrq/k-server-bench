# Potential Evaluation Guide

`evaluate.py` expects a single potential class, optionally paired with a candidate program that emits `potential_kwargs`.

## Evaluation flow

1. Resolve the requested metric instances.
2. Resolve `program_path` and load the potential class from that module.
3. Obtain `potential_kwargs` either by:
   - running the candidate program as a subprocess, or
   - loading `potential_kwargs_json` directly.
4. Instantiate `resolve_qualname(module, potential_qualname)(context, **potential_kwargs)` for each metric instance.
5. Run `compute_potential_stats(...)` on each instance.
6. Aggregate the per-instance scores into `combined_score` and write `correct.json` and `metrics.json`.

## Interfaces

### Candidate program output

If you let the evaluator run your program as a subprocess, it must write a JSON object to the requested `--output` path.

Multi-file programs are fine. The evaluator only requires that the selected subprocess entrypoint (`program_path`) is runnable and that the same loaded module resolves `potential_qualname` to the intended class.

Accepted shapes:

```py
{
    "potential_kwargs": Dict[str, Any]
}
```

or

```py
{
    "kwargs": Dict[str, Any]
}
```

or any JSON object whose non-private top-level keys are the desired potential kwargs.

Special failure shape:

```py
{
    "failure": str,
    "reason": Optional[str]
}
```

If `failure` is present, evaluation fails immediately with that reason.

### Potential

```py
# Metric points are integers and have pairwise distances.
# WFContext provides all distance/work-function helpers.
class WFContext:
    k: int
    m: int

    def distance(i, j):
        pass

    def config_to_idx(config: Sequence[int]):
        pass

    idx_to_config: List[int]

    def distance_within_config(idx: int):
        pass

    def distance_within_clique(S):
        pass

    def distance_between_configs(idx1: int, idx2: int):
        pass

    def distance_between_sets(S: Sequence[int], T: Sequence[int]):
        pass

    def initial_wf(servers: Sequence[int]) -> np.ndarray:
        pass

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
    # This is called for many nodes in the instances.
    # wf might be a tuple, so start your implementation with np.asarray(wf)
    # Can return additional meta information.
    def __call__(self, wf: Union[tuple, np.ndarray]) -> Union[Tuple[float, Dict[str, Any]], float]:
        pass
```

Naive implementation might just return `0` for all work functions, or for example the sum of all work-function values.

The evaluator looks up the class by `potential_qualname` (default: `Potential`) inside `program_path`. If your class lives in another file, import or re-export it from the entry module, or expose it under a nested exported object. Examples:

```text
Potential
MyPotential
factory.MyPotential
```

## Candidate subprocess contract

When subprocess mode is used, the evaluator runs roughly:

```bash
<exec_cmd> <program_path> \
  --metrics <metric1> <metric2> ... \
  --timeout <timeout> \
  --n_cpus <n_cpus> \
  --output <candidate_output.json>
```

Your program should:
- parse those arguments
- compute or choose candidate hyperparameters
- write a JSON object to `--output`
- exit with status 0 on success

You may implement that entrypoint as a thin wrapper over other files in the same solution directory.

If the subprocess exits successfully but does not create the output JSON file, the evaluator falls back to empty potential kwargs `{}` by default. This can make a non-entrypoint module appear to "work" if your potential constructor needs no kwargs. Set `raise_on_missing_candidate_output` to turn that into an error.

The subprocess is subject to:
- wall-clock timeout via `--timeout`
- extra termination grace period via `--kill_after`
- memory cap via `--memory_limit_gb`
- CPU time cap derived from `timeout * n_cpus`

## Additional Notes

- Usually one `Potential` class is enough.
- `Potential.__call__` should be fast; it may run on the order of hundreds of thousands of nodes.
- `potential_kwargs_json` is useful when you already know the desired kwargs and want to skip the subprocess phase.
- `potential_qualname` lets you select a non-default class exported by the entry module.
- If your subprocess emits private helper keys, prefix them with `_`; those are ignored when the evaluator falls back to top-level key extraction.
