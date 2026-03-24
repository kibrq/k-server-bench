# shinka-evolve

This method root runs the released shinka-evolve workflow.

The released experiments here were run against a forked `ShinkaEvolve` codebase rather than the upstream project:

- <https://github.com/kibrq/ShinkaEvolve>

Because of that, interface details and empirical results may differ if you point these sweeps at upstream `ShinkaEvolve`.

Key files:

- [`run-sweep.sh`](./run-sweep.sh): regenerate a sweep and execute it through `run_grid.py`
- [`run_evo.sh`](./run_evo.sh): shell entrypoint used by generated sweeps
- [`run_evo.py`](./run_evo.py): Python driver for the method
- `sweeps/`: released sweep definitions for this method

This method generally materializes prompt text as `PROMPT.md` and carries runtime configuration through sweep-local `base.env` files.
