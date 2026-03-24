# k-server-bench

`k-server-bench` is an executable benchmark and research workspace for automated potential discovery around the `k`-server conjecture.

The repository is built around a code-based challenge derived from a central open problem in competitive analysis. The task is to discover a potential function satisfying a large graph-structured system of inequalities associated with the Work Function Algorithm. Evaluation is sound but incomplete: any violated inequality definitively refutes a candidate, while satisfying all checked inequalities is strong evidence rather than a full proof of the corresponding infinite-case statement.

This makes the repository useful in two ways:

1. as tooling for researchers exploring new potential-function hypotheses for the `k`-server problem
2. as a benchmark for agentic code-based mathematical discovery, where progress can be tracked through automatically computed violations

Operationally, the repository has three main jobs:

1. define reusable task components
2. provide evaluators and metric instances
3. provide recipes for running discovery agents

## Repository Map

- [`docs/`](./docs/README.md): in-depth concepts and workflow documentation
- [`k-servers/`](./k-servers/README.md): the `kserver` Python package used by the evaluators and tools
- [`tasks/`](./tasks/README.md): reusable goals, implementation seeds, and hints that are assembled into tasks
- [`experiments/`](./experiments/README.md): released method runners and sweep directories
- [`tools/`](./tools/README.md): evaluators, metric-building utilities
- [`metrics/`](./metrics/README.md): serialized benchmark instances used for scoring
- [`examples/`](./examples/README.md): concrete evaluator invocations with fixed kwargs
- [`tests/`](./tests/README.md): smoke tests for evaluator paths and benchmark wiring
- [`agents/`](./agents/README.md): agent-specific setup and evaluator-facing docs
- [`docker/`](./docker/README.md): container image used for reproducible runs

## Core Ideas

A candidate is usually a Python file that exposes:

- a `Potential` class, which assigns a value to a normalized work function
- optionally a search procedure, usually `main(args)`, which proposes hyperparameters for that `Potential`

Evaluation is performed on precomputed metric instances built from finite work-function graphs. The main benchmark target is minimizing `violations_k`, with `1.0` as the perfect product-style score across metrics.

Two evaluator styles are supported:

- non-legacy evaluation, where the candidate program returns `potential_kwargs` and the evaluator instantiates `Potential`
- legacy evaluation, where candidate components are split across `PotentialFamily`, `Potential`, and `SearchEvaluator`

The benchmark is intentionally aimed at iterative discovery rather than full formal proof production:

- dense automatic feedback comes from checking many explicit inequalities
- solved `k=3` regimes serve as calibration targets
- open `k=4` regimes remain hard enough that improved candidates are already mathematically interesting

For new work, prefer the non-legacy evaluator. It is less restrictive and more robust operationally, while leaving parallelization decisions to the agent or candidate code instead of relying on brittle Ray-cluster orchestration.

## Quick Installation

```bash
pip install -r requirements.txt
```

## Quickstart

### Coding Agents

To prepare a target workspace for a coding agent, run:

```bash
bash agents/setup.sh <agent-name> <target-dir>
```

At the moment, only `codex` is supported as `<agent-name>`.

Then move into the target directory, start the coding agent, and give it a trivial first task such as implementing a potential that always returns zero:

```bash
cd <target-dir>
<coding-agent>
```

For example, with Codex:

```bash
bash agents/setup.sh codex /tmp/k-server-codex
cd /tmp/k-server-codex
codex
```

and then ask it to develop a very simple potential that always returns zero.

### Discovery Agents

If you are building or testing your own discovery agent, the main non-legacy evaluator entrypoint is:

```bash
PYTHONPATH="${PWD}/k-servers/src" python tools/evaluator/evaluate.py --help
```

The evaluator accepts these core inputs:

- `--program_path`: candidate program to evaluate
- `--results_dir`: directory where `correct.json` and `metrics.json` are written
- `--metrics_names`: comma-separated metric filenames

Those options can also be supplied through the corresponding `K_SERVER_EVALUATE_*` environment variables.

## Common Workflows

### Run a released sweep

```bash
cd experiments/best-of-n
./run-sweep.sh canonical-potential-just-potential-family-k4-general-task
```

Each method-level `run-sweep.sh` script:

1. resolves the requested sweep directory
2. regenerates local task assets with `generate_task.py`
3. launches the derived `sweep.sh` through [`experiments/run_grid.py`](./experiments/run_grid.py)

### Run the evaluator directly

```bash
cd k-server-bench
PYTHONPATH="${PWD}/k-servers/src" python tools/evaluator/evaluate.py --help
```

### Run a concrete example

```bash
bash examples/unifying_potential/evaluate_all_k4.sh
```

## Documentation Guide

Start here for detail:

- [`docs/concepts.md`](./docs/concepts.md): work functions, potentials, metrics, and scoring
- [`docs/task-model.md`](./docs/task-model.md): how task pieces compose into released sweeps
- [`docs/evaluators.md`](./docs/evaluators.md): legacy vs non-legacy evaluator contracts
- [`docs/experiments.md`](./docs/experiments.md): how released sweeps are structured and run

## Status

This repository contains released benchmark infrastructure. Sweep-local assets are generated from shared sources under `tasks/`, while the evaluators and the `kserver` package remain the stable execution layer.
