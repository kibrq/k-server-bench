# k-server-bench

Top-level workspace for k-server benchmarking and evaluation.

## Layout

- `k-servers/`: library code for WF contexts, potential evaluation, graph exploration, metrics, and tests
- `tasks/`: shared task components such as `goals/`, `implementations/`, and `canonical-potential/`
- `experiments/`: draft-v2-style method runners and sweep folders for `best-of-n`, `shinka-evolve`, and `loong-flow`
- `tools/`: evaluator scripts and utility scripts used around the benchmark workflow

## Usage

Install the library first:

```bash
cd k-servers
pip install -e .
```

Then run the top-level scripts from this repository root, for example:

```bash
python tools/build_circle_legacy_instance.py --help
```

## Experiments

Released experiment folders live under:

- `experiments/best-of-n/`
- `experiments/shinka-evolve/`
- `experiments/loong-flow/`

Each method root provides a `run-sweep.sh` entrypoint that mirrors the draft-v2 workflow:

```bash
cd experiments/best-of-n
./run-sweep.sh canonical-potential-just-potential-family
```

Each sweep regenerates its local prompt and initial files from the shared task components under `tasks/` before launching.
