# k-server-bench

Top-level workspace for k-server benchmarking and evaluation.

## Layout

- `k-servers/`: library code for WF contexts, potential evaluation, graph exploration, metrics, and tests
- `scripts/`: top-level utility scripts used around the benchmark/library workflow

## Usage

Install the library first:

```bash
cd k-servers
pip install -e .
```

Then run the top-level scripts from this repository root, for example:

```bash
python scripts/build_circle_legacy_instance.py --help
```
