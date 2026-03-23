#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SHINKA_ROOT="${REPO_ROOT}/../ShinkaEvolve"

ENV_FILES=()
FORWARD_ARGS=()

load_env_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Error: env file not found: $path" >&2
    exit 2
  fi
  set -a
  # shellcheck disable=SC1090
  source "$path"
  set +a
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      if [[ $# -lt 2 ]]; then
        echo "Error: --env requires a file path" >&2
        exit 2
      fi
      ENV_FILES+=("$2")
      shift 2
      ;;
    --env=*)
      ENV_FILES+=("${1#*=}")
      shift
      ;;
    --)
      shift
      FORWARD_ARGS+=("$@")
      break
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

for env_path in "${ENV_FILES[@]}"; do
  load_env_file "${env_path}"
done

if [[ ! -d "${SHINKA_ROOT}" ]]; then
  echo "Error: ShinkaEvolve root not found: ${SHINKA_ROOT}" >&2
  exit 1
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${SHINKA_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${SHINKA_ROOT}"
fi

if [[ -z "${SHINKA_WFA_EXPERIMENT_OUTPUTS_DIR:-}" && -n "${RUN_GRID_OUTPUTS_DIR:-}" ]]; then
  export SHINKA_WFA_EXPERIMENT_OUTPUTS_DIR="${RUN_GRID_OUTPUTS_DIR}"
fi

exec python "${SCRIPT_DIR}/run_evo.py" "${FORWARD_ARGS[@]}"
