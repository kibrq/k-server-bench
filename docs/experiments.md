# Experiments

## Layout

Released runs live under [`experiments/`](../experiments/README.md).

There are three method roots:

- [`experiments/best-of-n/`](../experiments/best-of-n/README.md)
- [`experiments/loong-flow/`](../experiments/loong-flow/README.md)
- [`experiments/shinka-evolve/`](../experiments/shinka-evolve/README.md)

Each method root contains:

- a method-specific launcher
- a `run-sweep.sh` wrapper
- a `sweeps/` folder of released sweep definitions

## Sweep Structure

A sweep directory usually contains:

- `generate_task.py`
- `sweep.sh`
- generated task text such as `TASK.md` or `PROMPT.md`
- a seed `initial.py`
- method-specific config such as `base.env` or `config.yaml`

The generated files are committed so the release surface is visible, but `generate_task.py` remains the authoritative regeneration step.

## Execution Flow

When you run:

```bash
./run-sweep.sh <sweep-name>
```

the method wrapper:

1. validates the sweep directory
2. runs `generate_task.py`
3. executes the derived `sweep.sh` via [`experiments/run_grid.py`](../experiments/run_grid.py)

`run_grid.py` is the common executor that expands the sweep file into repeated method invocations.

## Shared Task Source Of Truth

Sweep folders are intentionally thin. They should point back to shared assets in:

- `tasks/goals/`
- `tasks/implementation/`
- `tasks/hints/`

This keeps released methods aligned while still allowing each method to have its own prompt, config, and environment conventions.
