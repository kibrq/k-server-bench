# Agents

This folder contains agent-specific integration files.

Main contents:

- [`codex/`](./codex/AGENTS.md): Codex runner setup, entrypoint logic, and evaluator registration helpers
- [`docs/`](./docs/README.md): agent-facing evaluator documentation
- [`setup.sh`](./setup.sh): shared bootstrap used by agent environments

The purpose of this subtree is operational. It does not define benchmark semantics; it wires agent runtimes to the released evaluator interfaces.
