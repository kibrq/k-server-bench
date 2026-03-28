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
ADD_EVALUATE_MCP_ARGS=("$@")
HAS_EVALUATOR_HOME=false
SHARED_SETUP="${SCRIPT_DIR}/../setup.sh"

if [[ ! -x "${SHARED_SETUP}" ]]; then
  chmod +x "${SHARED_SETUP}"
fi

"${SHARED_SETUP}" "${TARGET_DIR}"

if [[ -f "${SCRIPT_DIR}/AGENTS.md" ]]; then
  cp "${SCRIPT_DIR}/AGENTS.md" "${TARGET_DIR}/AGENTS.md"
fi

for ((i = 0; i < ${#ADD_EVALUATE_MCP_ARGS[@]}; i++)); do
  case "${ADD_EVALUATE_MCP_ARGS[$i]}" in
    --evaluator-home|--evaluator-home=*)
      HAS_EVALUATOR_HOME=true
      break
      ;;
  esac
done

if [[ "${K_SERVER_BENCH_USE_LEGACY_EVALUATOR:-false}" == "true" && "${HAS_EVALUATOR_HOME}" != "true" ]]; then
  ADD_EVALUATE_MCP_ARGS+=("--evaluator-home" "tools/legacy-evaluator")
fi

python "${SCRIPT_DIR}/add-evaluate-mcp.py" \
  --k-server-bench-home "${BENCH_ROOT}" \
  --codex-home "${CODEX_HOME_DIR}" \
  --create-config \
  --override-mcp-config \
  "${ADD_EVALUATE_MCP_ARGS[@]}"

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
