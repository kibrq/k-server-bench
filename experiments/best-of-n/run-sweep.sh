#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_ROOT="${SCRIPT_DIR}/sweeps"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <experiment-name> [run_grid args...]" >&2
  exit 1
fi

EXPERIMENT_NAME="$1"
shift
SWEEP_DIR="${EXPERIMENTS_ROOT}/${EXPERIMENT_NAME}"
SWEEP_FILE="${SWEEP_DIR}/sweep.sh"
GENERATE_TASK="${SWEEP_DIR}/generate_task.py"

if [[ ! -d "${SWEEP_DIR}" ]]; then
  echo "Error: sweep directory not found: ${SWEEP_DIR}" >&2
  exit 1
fi

if [[ -f "${GENERATE_TASK}" ]]; then
  python "${GENERATE_TASK}"
fi

if [[ ! -f "${SWEEP_FILE}" ]]; then
  echo "Error: sweep.sh not found: ${SWEEP_FILE}" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"
exec python "${SCRIPT_DIR}/../run_grid.py" "${SWEEP_FILE}" "$@"
