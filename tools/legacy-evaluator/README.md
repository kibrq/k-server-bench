# Legacy Evaluator

This folder contains the released legacy evaluator and its support code.

Files include:

- [`evaluate.py`](./evaluate.py): main legacy evaluator entrypoint
- [`evaluate_mcp_server.py`](./evaluate_mcp_server.py): MCP wrapper
- [`naive_potential_family.py`](./naive_potential_family.py): baseline family helper
- [`naive_search_evaluator.py`](./naive_search_evaluator.py): baseline search evaluator helper
- [`ray_kserver_instance.py`](./ray_kserver_instance.py): Ray-backed instance support
- [`ray_utils.py`](./ray_utils.py): Ray helper utilities
- [`restart_ray.sh`](./restart_ray.sh): Ray restart helper for containerized runs

Use this evaluator when a task or sweep targets the legacy three-component interface rather than the simpler non-legacy `Potential` plus `main(args)` contract.
