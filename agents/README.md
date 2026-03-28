# Agents

This folder contains agent-specific integration files.

Main contents:

- [`codex/`](./codex/AGENTS.md): Codex runner setup, entrypoint logic, and evaluator registration helpers
- [`docs/`](./docs/README.md): agent-facing evaluator documentation
- [`setup.sh`](./setup.sh): shared bootstrap that copies evaluator docs and papers into a target workspace

The purpose of this subtree is operational. It does not define benchmark semantics; it wires agent runtimes to the released evaluator interfaces.

The split is:

- [`setup.sh`](./setup.sh): generic workspace preparation only
- agent-specific setup such as MCP registration: handled in the corresponding agent subfolder, for example [`codex/setup.sh`](./codex/setup.sh)
