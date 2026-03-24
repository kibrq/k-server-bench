# Evaluator

This folder contains the released non-legacy evaluator.

Files:

- [`evaluate.py`](./evaluate.py): main CLI entrypoint
- [`evaluate_mcp_server.py`](./evaluate_mcp_server.py): MCP wrapper around the evaluator

This evaluator expects a candidate module that exports a `Potential` class and usually a `main(args)` routine that returns `potential_kwargs`.

See also:

- [`../../docs/evaluators.md`](../../docs/evaluators.md)
- [`../../agents/docs/evaluator/EVALUATE.md`](../../agents/docs/evaluator/EVALUATE.md)
