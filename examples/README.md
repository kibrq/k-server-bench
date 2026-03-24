# Examples

This folder contains fixed evaluator examples with explicit `kwargs.json` files.

Each subfolder typically contains:

- `kwargs.json`
- `evaluate_all_k4.sh`

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
