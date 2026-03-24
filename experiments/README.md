# Experiments

This folder contains the released method runners and sweep definitions.

Structure:

- [`best-of-n/`](./best-of-n/README.md)
- [`loong-flow/`](./loong-flow/README.md)
- [`shinka-evolve/`](./shinka-evolve/README.md)
- [`run_grid.py`](./run_grid.py): shared sweep executor

In this repository, a sweep is a named released experiment configuration: a small folder that pins a task composition, a seed program, method-specific config, and a generated `sweep.sh` launcher. Running a sweep means executing one of those packaged experiment definitions through the corresponding method runner.

The key design choice is that method runners are separate, but the underlying task definitions are shared through the `tasks/` tree. Each method root regenerates its sweep-local assets from those shared sources before launching.
