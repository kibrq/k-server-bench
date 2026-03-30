# Autoresearch Agent Setup

This folder provides an `autoresearch`-style agent setup for `k-server-bench`.

Use it when you want the same human-maintained `program.md` workflow as the Codex integration, but without Codex-specific MCP wiring or runtime assumptions.
It can also be layered onto an existing Codex-prepared workspace.

Important files:

- [`AGENTS.md`](./AGENTS.md): bootstrap instructions exposed to the agent runtime
- [`program.md`](./program.md): human-maintained research loop and iteration policy
- [`setup.sh`](./setup.sh): copies the autoresearch agent files into a target workspace, scaffolds an initial solution, and initializes git
- [`entrypoint.sh`](./entrypoint.sh): container entrypoint that bootstraps the target workspace

After running setup, the target workspace contains:

- `AGENTS.md`
- `program.md`
- `docs/`
- `main.py`
- `.git/`

This is intended for agents that can:

- read an agent context file such as `AGENTS.md`
- run shell commands directly
- invoke `tools/evaluator/evaluate.py` without MCP

Usage:

```bash
bash agents/autoresearch/setup.sh <target-dir>
```

Then start your preferred agent in `<target-dir>` and prompt it to read `program.md` before beginning work.

The initial solution is copied from the released non-legacy evaluator seed implementation so the agent has a runnable starting point immediately.

The setup is conservative by default:

- existing `AGENTS.md` is preserved
- existing `main.py` is preserved
- `program.md` is added or refreshed
- git is initialized if needed
- docs are only copied if they are not already present

## Docker

You can also invoke the entrypoint directly in an existing image, for example:

```bash
docker run --rm -it codex-k-server:latest /home/kserver/k-server-bench/agents/autoresearch/entrypoint.sh
```

The entrypoint infers the repository root from its own path, runs `agents/autoresearch/setup.sh` into `$HOME/workspace` by default, and then executes `bash`. Set `TARGET_DIR` if you want a different destination.
In the current container layout this default is `/home/kserver/workspace`.
