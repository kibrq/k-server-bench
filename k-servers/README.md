# k-servers

This folder contains the reusable Python package that powers the benchmark runtime.

The import name is `kserver`.

## What It Contains

The package under [`src/kserver/`](./src/README.md) provides:

- work-function contexts
- metric definitions
- potential implementations
- evaluation helpers
- graph exploration utilities

The rest of `k-server-bench` uses this package as its mathematical execution layer. Evaluators, tools, and tests import from `kserver`.

## Installation

Install it from inside this folder:

```bash
cd k-servers
pip install -e .
```

Or import it in-place from the repository root:

```bash
cd ..
PYTHONPATH="${PWD}/k-servers/src" python -c "import kserver"
```

## Layout

- [`src/`](./src/README.md): package source
- [`tests/`](./tests/README.md): package-level tests
- [`setup.py`](./setup.py): editable-install packaging entrypoint
- [`requirements.txt`](./requirements.txt): package dependencies

## Known Issue

`parallel_bfs_exploration` can show periodic throughput drops during long runs due to Python GC pauses in the main process.

Disabling GC may reduce pause spikes but usually increases memory growth, so the safer default is to keep GC enabled unless you are explicitly tuning for throughput.
