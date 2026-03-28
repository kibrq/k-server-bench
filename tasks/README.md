# Tasks

This folder contains the shared task components used to build released sweeps.

Subfolders:

- [`goals/`](./goals/README.md): high-level problem statements
- [`implementation/`](./implementation/README.md): candidate code contracts and seed files
- [`hints/`](./hints/README.md): optional structural or metric-specific guidance
- [`k3-stress-test/`](./k3-stress-test/README.md): predefined non-legacy + canonical + `k3_stress_test` preset
- [`k4-general-task/`](./k4-general-task/README.md): predefined non-legacy + canonical + `k4_general_task` preset
- [`build_preset.py`](./build_preset.py): regenerates the packaged preset `README.md` and `.env` files from shared task components

The important design principle is composition. A sweep does not usually invent a task from scratch; it assembles these shared pieces into a method-specific prompt or task file.
