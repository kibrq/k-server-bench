# Evaluating a solution (`evaluate_directory`)

`evaluate_directory` runs the evaluator on a solution stored in `directory`.

## What it does
1. Creates a new `<attempt_id>/` subfolder inside `directory` (default: `%Y-%m-%d_%H-%M-%S`)
   - saves `log.out` / `log.err`
   - saves `correct.json` and `metrics.json`
2. Loads your solution components:
   - `PotentialFamily`
   - `Potential`
   - `SearchEvaluator`
3. Runs a search over potential hyperparameters for `search_timeout` seconds.
4. Re-evaluates the best candidates on the full instance(s) and reports final metrics.

> Do not edit the evaluation code or the evaluation directory.

## Component loading / resolution rules
`evaluate_directory` uses `program_filename` inside `directory` (default: `main.py`) as `program_path`.

For `PotentialFamily`, `Potential`, and `SearchEvaluator` the resolution order is:
1. `{component}_path` (if provided)
2. `{component}_filename` joined with `directory` (if provided)
3. `program_path` (default)

If you pass `home` and set `use_default_*` flags, `evaluate.py` swaps the default implementation paths to:
- `naive_potential_family.py` (PotentialFamily)
- `canonical_potential.py` (Potential)
- `naive_search_evaluator.py` (SearchEvaluator)

All environment-based defaults come from `K_SERVER_EVALUATE_*` variables in `evaluate.py` (for example: `PROGRAM_PATH`, `METRICS_PATH`, `METRICS_NAMES`, `SEARCH_TIMEOUT`, etc.).

## Minimal examples

### A) Everything in `main.py`
```MCP
evaluate_directory(
  directory="solutions/010_softmax_gap",
  program_filename="main.py",
)
```

### B) main.py has PotentialFamily and SearchEvaluator, but you want default Potential

```MCP
evaluate_directory(
  directory="solutions/020_family_only",
  program_filename="main.py",
  search_evaluator_filename="main.py",
)
```

### C) Split files inside the folder
```MCP
evaluate_directory(
  directory="solutions/030_split_files",
  program_filename="main.py",
  potential_filename="potential.py",
  potential_family_filename="family.py",
  search_evaluator_filename="evaluator.py",
)
```


## Search loop: how timeouts work

The evaluator runs `n_workers = search_max_concurrent` workers. Each `ask()` submission can provide a timeout ratio in [0.01, 1.0], which maps to:

`worker_timeout = min_worker_timeout + timeout * (max_worker_timeout - min_worker_timeout)`

The system keeps generating/evaluating candidates until `search_timeout` expires.

## Common parameters (MCP tool)
- `ask_tell_max_queue_size` controls the max queued candidates in the ask/tell loop.
- `search_timeout`, `search_timeout_buffer`, `search_max_concurrent`, `search_min_worker_timeout`, `search_max_worker_timeout` control the search phase.
- `final_evaluation_timeout`, `final_evaluation_timeout_buffer`, `final_evaluation_max_concurrent`, `top_k_final_evaluation` control final evaluation.
- `skip_final_evaluation` skips final evaluation and returns search-only metrics.
- `metrics_names` specifies what metrics would be as `train_instances`. You can look up the choice at `metrics/*.pickle`

## JSON config inputs
### `potential_family_kwargs_path`
Path to a JSON file of keyword args passed into `PotentialFamily(**kwargs)` during the search phase. The evaluator also injects:
- `n_instances`
- `n_workers`
- `search_timeout`
- `min_worker_timeout`
- `max_worker_timeout`

Example file:
```json
{
  "seed": 123,
  "temperature": 0.2,
  "budget": 500
}
```

### `shared_readonly_ray_resources_kwargs_path`
Path to a JSON file of keyword args passed into `load_shared_readonly_ray_resources(**kwargs)` and then injected into worker kwargs. Common usage is to provide extra numpy payloads to the `SearchEvaluator`.

Expected structure (example):
```json
{
  "extra_features": "path/to/features.npz",
  "lookup_table": "path/to/lookup.npy"
}
```

Notes:
- `train_instances` is always supplied by the evaluator from `metrics_names`/`metrics_path`, so you usually only add extra keys.
- Each non-`train_instances` value should be a path to a `.npy`/`.npz`; these are loaded with `numpy.load` and stored as Ray object refs.
- The extra keys are forwarded into `SearchEvaluator(..., **search_evaluator_kwargs)` on each worker.
