#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage:
  agents/setup.sh <agent-name> <target-dir> [agent-setup-args...]

Currently supported agents:
  codex
EOF
}

download_arxiv_source() {
  local label="$1"
  local arxiv_id="$2"
  local dest_dir="$3"
  local archive_path="${dest_dir}/${label}.src"
  local unpack_dir="${dest_dir}/${label}"

  mkdir -p "${unpack_dir}"
  curl -fsSL "https://arxiv.org/e-print/${arxiv_id}" -o "${archive_path}"

  if tar -xf "${archive_path}" -C "${unpack_dir}" 2>/dev/null; then
    :
  elif gzip -dc "${archive_path}" > "${unpack_dir}/main.tex" 2>/dev/null; then
    :
  else
    echo "Error: failed to unpack arXiv source for ${label} (${arxiv_id})" >&2
    exit 1
  fi

  rm -f "${archive_path}"
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

AGENT_NAME="$1"
TARGET_DIR_INPUT="$2"
shift 2

case "${AGENT_NAME}" in
  codex)
    ;;
  *)
    echo "Error: unsupported agent '${AGENT_NAME}'" >&2
    usage
    exit 2
    ;;
esac

TARGET_DIR="$(mkdir -p "${TARGET_DIR_INPUT}" && cd "${TARGET_DIR_INPUT}" && pwd)"
DOCS_DIR="${TARGET_DIR}/docs"
AGENT_SETUP="${SCRIPT_DIR}/${AGENT_NAME}/setup.sh"
AGENT_INSTRUCTIONS="${SCRIPT_DIR}/${AGENT_NAME}/AGENTS.md"
K_SERVER_BENCH_USE_LEGACY_EVALUATOR="${K_SERVER_BENCH_USE_LEGACY_EVALUATOR:-false}"
EVALUATOR_DOCS_DIR="${SCRIPT_DIR}/docs/evaluator"
if [[ "${K_SERVER_BENCH_USE_LEGACY_EVALUATOR}" == "true" ]]; then
  EVALUATOR_DOCS_DIR="${SCRIPT_DIR}/docs/legacy-evaluator"
fi

if [[ ! -x "${AGENT_SETUP}" ]]; then
  chmod +x "${AGENT_SETUP}"
fi

mkdir -p "${DOCS_DIR}"
cp "${SCRIPT_DIR}/docs/QUICKSTART.md" "${DOCS_DIR}/"
cp -a "${EVALUATOR_DOCS_DIR}/." "${DOCS_DIR}/"

if [[ -f "${AGENT_INSTRUCTIONS}" ]]; then
  cp "${AGENT_INSTRUCTIONS}" "${TARGET_DIR}/AGENTS.md"
fi

download_arxiv_source "coester_21" "2102.10474" "${DOCS_DIR}"
download_arxiv_source "huang_22" "2205.08103" "${DOCS_DIR}"

"${AGENT_SETUP}" "${TARGET_DIR}" "$@"
