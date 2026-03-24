# k-server-bench

`k-server-bench` is a benchmark workspace for searching, evaluating, and comparing candidate potentials for the k-server problem.

The repository has three main jobs:

1. define reusable task components
2. provide evaluators and metric instances
3. run method-specific sweeps against those shared assets

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

Evaluation is performed on precomputed metric instances. The main benchmark target is minimizing `violations_k`, with `1.0` as the perfect product-style score across metrics.

Two evaluator styles are supported:

- non-legacy evaluation, where the candidate program returns `potential_kwargs` and the evaluator instantiates `Potential`
- legacy evaluation, where candidate components are split across `PotentialFamily`, `Potential`, and `SearchEvaluator`

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
