# loong-flow

This method root runs the released loong-flow workflow.

Key files:

- [`run-sweep.sh`](./run-sweep.sh): regenerate a sweep and execute it through `run_grid.py`
- [`run_math.sh`](./run_math.sh): shell entrypoint used by generated sweeps
- [`eval_program.py`](./eval_program.py): evaluation bridge used by the method
- `sweeps/`: released sweep definitions for this method

Compared with `best-of-n`, this method carries more method-specific runtime configuration inside per-sweep `config.yaml` and `base.env` files.
