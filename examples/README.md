# Examples

This folder contains fixed evaluator examples and staged search procedures.

Each subfolder typically contains:

- `kwargs.json`
- `evaluate_all_k4.sh`

Search-oriented examples may instead contain a Python driver plus a local `README.md`.

The example scripts run the released evaluator on all four `k=4` benchmark metrics and print the resulting `metrics.json`.

Use these examples to:

- sanity-check the evaluator wiring
- compare known constructions on the same metrics
- inspect concrete canonical-potential parameterizations

Current examples include:

- [`unifying_potential/`](./unifying_potential/)
- [`unifying_potential_kplus1_competitiveness/`](./unifying_potential_kplus1_competitiveness/)
- [`unifying_potential_optimized_coefs/`](./unifying_potential_optimized_coefs/)
- [`k3_circle_k4_generalization/`](./k3_circle_k4_generalization/)
- [`search_n7_async_pipeline/`](./search_n7_async_pipeline/): Codex-developed staged search for a fixed `n=7` index matrix that optimizes coefficient vectors and can reach a 3-violation candidate in roughly 20-30 minutes.
