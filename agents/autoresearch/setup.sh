#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_SETUP="${SCRIPT_DIR}/../setup.sh"
BENCH_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SEED_PROGRAM="${BENCH_ROOT}/tasks/implementation/non-legacy-evaluator/initial.py"

usage() {
  cat >&2 <<'EOF'
Usage:
  agents/autoresearch/setup.sh <target-dir>
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

TARGET_DIR="$(mkdir -p "$1" && cd "$1" && pwd)"

if [[ ! -x "${SHARED_SETUP}" ]]; then
  chmod +x "${SHARED_SETUP}"
fi

if [[ ! -f "${TARGET_DIR}/docs/EVALUATE.md" ]]; then
  "${SHARED_SETUP}" "${TARGET_DIR}"
fi

cp "${SCRIPT_DIR}/program.md" "${TARGET_DIR}/program.md"
if [[ ! -f "${TARGET_DIR}/AGENTS.md" ]]; then
  cp "${SCRIPT_DIR}/AGENTS.md" "${TARGET_DIR}/AGENTS.md"
fi

if [[ ! -f "${TARGET_DIR}/main.py" ]]; then
  cp "${SEED_PROGRAM}" "${TARGET_DIR}/main.py"
fi

if [[ ! -d "${TARGET_DIR}/.git" ]]; then
  git -C "${TARGET_DIR}" init -q
  git -C "${TARGET_DIR}" config user.name "autoresearch"
  git -C "${TARGET_DIR}" config user.email "autoresearch@local"
  git -C "${TARGET_DIR}" add AGENTS.md program.md main.py docs .gitignore 2>/dev/null || true
  git -C "${TARGET_DIR}" add .
  git -C "${TARGET_DIR}" commit -q -m "Initial autoresearch setup"
fi

cat <<EOF
Autoresearch agent setup complete.
Start your preferred agent in:
  ${TARGET_DIR}
and instruct it to read program.md first.
Autoresearch files were added without touching existing Codex config.
If no candidate existed yet, an initial seed was created at:
  ${TARGET_DIR}/main.py
EOF
