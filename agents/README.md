# Agents

This folder contains agent-specific integration files.

Main contents:

- [`Dockerfile`](./Dockerfile): generic container wrapper for agent runtimes on top of `k-server-env`
- [`entrypoint.sh`](./entrypoint.sh): generic bootstrap entrypoint that can run a chosen setup script before the agent command
- [`autoresearch/`](./autoresearch/AGENTS.md): agent-neutral setup for a direct-evaluator `autoresearch` workflow
- [`codex/`](./codex/AGENTS.md): Codex runner setup, entrypoint logic, and evaluator registration helpers
- [`docs/`](./docs/README.md): agent-facing evaluator documentation
- [`setup.sh`](./setup.sh): shared bootstrap that copies evaluator docs and papers into a target workspace

The purpose of this subtree is operational. It does not define benchmark semantics; it wires agent runtimes to the released evaluator interfaces.

The split is:

- [`setup.sh`](./setup.sh): generic workspace preparation only
- agent-specific setup such as MCP registration or agent context bootstrapping: handled in the corresponding agent subfolder, for example [`codex/setup.sh`](./codex/setup.sh) or [`autoresearch/setup.sh`](./autoresearch/setup.sh)

## Generic Container

The root [`Dockerfile`](./Dockerfile) and [`entrypoint.sh`](./entrypoint.sh) provide a generic image/entrypoint pair for agents that are not Codex-specific.

The entrypoint always runs [`setup.sh`](./setup.sh) into `$HOME/workspace` and then executes the provided command.

Build it with:

```bash
docker build -t generic-k-server:latest -f agents/Dockerfile .
```

```bash
docker run --rm -it generic-k-server:latest bash
```

This is meant to create a prepared workspace first, then hand off to whatever agent CLI or shell command the image provides.
