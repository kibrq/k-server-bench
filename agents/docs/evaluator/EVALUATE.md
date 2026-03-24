# Evaluating a solution (`evaluate_directory`)

`evaluate_directory` runs the current evaluator on a solution stored in `directory`.

## What it does
1. Creates a new `<attempt_id>/` subfolder inside `directory` (default: `%Y-%m-%d_%H-%M-%S`)
   - saves `log.out` / `log.err`
   - saves `correct.json` and `metrics.json`
2. Resolves your program file and `potential_qualname`.
3. Either:
   - runs your candidate program as a subprocess to obtain `potential_kwargs`, or
   - directly uses `potential_kwargs_json` if you provide it
4. Loads `Potential` and runs `compute_potential_stats(...)` on each requested metric.
5. Aggregates per-metric results into the final score payload.

> Do not edit the evaluation code or the evaluation directory.

## Component loading / resolution rules
`evaluate_directory` uses `program_filename` inside `directory` (default: `main.py`) as `program_path`.

The evaluator always loads the potential class from `program_path`.
Inside that loaded module, it resolves `potential_qualname` (default: `Potential`). This can be a dotted attribute path such as `factory.Potential`.

`program_filename` must be a relative filename. Path-like overrides such as `results_dir`, `evaluate_home`, and `metrics_path` may be absolute or relative to `directory`.

`program_filename` has two responsibilities:
- it is executed as the candidate subprocess entrypoint when `potential_kwargs_json` is not provided
- it is also imported as the module from which `potential_qualname` is resolved

So if your potential class lives in another file, the selected entry module must re-export it, unless you point `program_filename` directly at the defining module and set `potential_qualname` accordingly.

Multi-file solutions are supported. The only requirement is that:
- `program_filename` points to the candidate subprocess entrypoint
- that same loaded module exports the evaluated potential under `potential_qualname`, directly or via imports/re-exports from sibling files

All environment-based defaults come from `K_SERVER_EVALUATE_*` variables in `evaluate.py` (for example: `PROGRAM_PATH`, `METRICS_PATH`, `METRICS_NAMES`, `TIMEOUT`, etc.).

## Minimal examples

### A) Everything in `main.py`
```MCP
evaluate_directory(
  directory="solutions/010_softmax_gap",
  program_filename="main.py",
  metrics_names="circle_k3_m6.pickle",
)
```

### B) Non-default class name exported from `main.py`
```MCP
evaluate_directory(
  directory="solutions/020_split_potential",
  program_filename="main.py",
  potential_qualname="MyPotential",
)
```

### C) Dotted qualname exported from `main.py`
```MCP
evaluate_directory(
  directory="solutions/025_nested_potential",
  program_filename="main.py",
  potential_qualname="custom.MyPotential",
)
```

### D) Skip candidate subprocess and pass kwargs directly
```MCP
evaluate_directory(
  directory="solutions/030_direct_kwargs",
  program_filename="main.py",
  potential_kwargs_json='{"alpha": 0.5, "beta": 2}',
)
```

## Evaluation flow: subprocess -> kwargs -> final evaluation

If `potential_kwargs_json` is not provided, the evaluator runs:

```text
<exec_cmd> <program_path> --metrics ... --timeout ... --n_cpus ... --output ...
```

Your program must create the requested JSON output file. The evaluator then extracts `potential_kwargs` from:
- `potential_kwargs`, if present
- otherwise `kwargs`, if present
- otherwise all top-level non-private keys

This means you can keep helper logic in sibling files and import it from the chosen entrypoint, as long as the selected entry module is wired up correctly.

If the candidate subprocess does not create an output JSON file, the evaluator uses empty potential kwargs `{}` by default. Set `raise_on_missing_candidate_output=true` to make that a hard error instead.

After that it instantiates:

```py
resolve_qualname(module, potential_qualname)(context, **potential_kwargs)
```

and evaluates it on the requested metric instances using `compute_potential_stats(...)`.

When `potential_kwargs_json` is provided, subprocess execution is skipped. In that mode you should not expect subprocess artifacts such as `subprocess.stdout`, `subprocess.stderr`, or subprocess-derived candidate output details.

## Common parameters (MCP tool)
- `metrics_names` selects which metric files under `metrics/` are used.
- `timeout`, `kill_after`, `memory_limit_gb`, `n_cpus`, and `exec_cmd` control the candidate subprocess.
- `potential_qualname` selects which class inside `program_path` is used. The default is `Potential`.
- `potential_kwargs_json` skips subprocess execution and injects kwargs directly.
- `raise_on_missing_candidate_output` turns missing candidate output files into errors instead of falling back to empty potential kwargs.
- `final_evaluation_timeout`, `final_evaluation_num_processes`, `compute_stats_round_digits`, `rho`, `keep_only_violations_k`, and `robustness_check` are forwarded into the final `compute_potential_stats(...)` call.
- `results_dir` overrides the default `<directory>/<attempt_id>` output folder.
- `evaluate_home` changes the default metrics root; `metrics_path` overrides it directly.

Some metric files may not include precomputed Bellman data. Those files can fail during scoring with an error such as `bellman must be precomputed for NumpyKServerInstance`. Use a metric known to contain Bellman data, for example `circle_k3_m6.pickle`. `uniform_small.pickle` is not evaluation-ready here.

## JSON config inputs

### `potential_kwargs_json`
JSON object or path to a JSON file containing the kwargs passed into:

```py
resolve_qualname(module, potential_qualname)(context, **potential_kwargs)
```

Example file:
```json
{
  "alpha": 0.25,
  "temperature": 2.0,
  "index_matrix": [[1, 2, 3], [1, -2, 3]]
}
```

Notes:
- If this is provided, the evaluator does not run the candidate subprocess.
- If a file path is provided, it is resolved from the evaluator process working directory, not from `directory`. Prefer an explicit path if the caller's working directory may vary.
- The JSON object must decode to a dictionary.
