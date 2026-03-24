# best-of-n

This method root runs the released best-of-n workflow.

Key files:

- [`run-sweep.sh`](./run-sweep.sh): regenerate a sweep and execute it through `run_grid.py`
- [`run_best_of_n.sh`](./run_best_of_n.sh): shell entrypoint used by generated sweeps
- [`run_best_of_n.py`](./run_best_of_n.py): Python driver for the method
- `sweeps/`: released sweep definitions for this method

The sweep directories under `sweeps/` are thin wrappers around shared task assets. `generate_task.py` inside each sweep is the place where shared task components are composed into method-local prompt files and launch commands.
