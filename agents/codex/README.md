# Codex Agent

This folder contains the Codex-specific runtime wiring for `k-server-bench`.

Important files:

- [`AGENTS.md`](./AGENTS.md): instructions exposed to the Codex runtime
- [`entrypoint.sh`](./entrypoint.sh): container/runtime entrypoint
- [`setup.sh`](./setup.sh): Codex-specific bootstrap
- [`add-evaluate-mcp.py`](./add-evaluate-mcp.py): helper for registering evaluator MCP tools
- [`Dockerfile`](./Dockerfile): Codex image customization

Use this folder when you need to run the benchmark inside the Codex agent environment rather than directly from the host shell.
