# Codex Agent

This folder contains the Codex-specific runtime wiring for `k-server-bench`.

Important files:

- [`AGENTS.md`](./AGENTS.md): instructions exposed to the Codex runtime
- [`entrypoint.sh`](./entrypoint.sh): container/runtime entrypoint
- [`setup.sh`](./setup.sh): Codex-specific bootstrap, including MCP registration
- [`add-evaluate-mcp.py`](./add-evaluate-mcp.py): helper for registering evaluator MCP tools
- [`Dockerfile`](./Dockerfile): Codex image customization

Use this folder when you need to run the benchmark inside the Codex agent environment rather than directly from the host shell.

The setup split is:

- [`../setup.sh`](../setup.sh): generic workspace preparation only
- [`setup.sh`](./setup.sh): generic preparation plus Codex-specific MCP wiring

The Docker image now chains through the base `k-server-env` entrypoint first, then runs [`entrypoint.sh`](./entrypoint.sh), and finally launches Codex by default.

## Layering Autoresearch

You can add the autoresearch workflow on top of an existing Codex workspace:

```bash
bash agents/autoresearch/setup.sh <target-dir>
```

That keeps the Codex MCP setup in place and adds `program.md`, initializes git if needed, and seeds `main.py` only when it does not already exist.

## Environment Split Scenarios

### MCP Runs In Another Environment Than The Agent

If the agent itself runs in one environment, but you want the evaluator MCP server to run in another one, use the Codex-specific setup script with MCP registration overrides.

Conda example:

```bash
bash agents/codex/setup.sh /tmp/kserver-codex \
  --mcp-server-command conda \
  --args run -n my-mcp-env python /abs/path/to/k-server-bench/tools/evaluator/evaluate_mcp_server.py
```

When overriding `--args`, make sure the final argument list still launches `evaluate_mcp_server.py`.

### k-server-bench Runs In Another Environment Than The MCP

Even if the MCP server itself is already running, you may still want the evaluator subprocess to run in a different environment, for example because `kserver` and the benchmark dependencies are installed elsewhere.

You can do that either per MCP call, or persistently when registering the MCP.

Per MCP call:

- `cmd_prefix="conda run -n my-kserver-env python"`

Persistent registration:

```bash
bash agents/codex/setup.sh /tmp/kserver-codex \
  --env K_SERVER_MCP_CMD_PREFIX="conda run -n my-kserver-env python"
```

This leaves the MCP server where it is, but changes the environment used to launch the actual evaluator process.
