#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage:
  agents/codex/setup.sh <target-dir> [add-evaluate-mcp-args...]
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

TARGET_DIR="$(mkdir -p "$1" && cd "$1" && pwd)"
shift
CODEX_HOME_DIR="${TARGET_DIR}/.codex"

python "${SCRIPT_DIR}/add-evaluate-mcp.py" \
  --k-server-bench-home "${BENCH_ROOT}" \
  --codex-home "${CODEX_HOME_DIR}" \
  --create-config \
  --override-mcp-config \
  "$@"

cat <<EOF
Codex MCP setup complete.
Use:
  CODEX_HOME=${CODEX_HOME_DIR} codex
to run Codex with this MCP configuration.
Use that form when you want this Codex setup to stay independent from your general Codex setup.
Otherwise, running codex from:
  ${TARGET_DIR}
will also see the added MCP because it uses ${CODEX_HOME_DIR}.
EOF
