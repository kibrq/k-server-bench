#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <task_name> [--env FILE] [--env=FILE] [--no-default-env] [-- <extra args>]" >&2
  exit 1
fi

TASK_NAME="$1"
shift

TASK_DIR="${SCRIPT_DIR}/${TASK_NAME}"
DEFAULT_ENV_FILE="${TASK_DIR}/.env"

if [[ ! -d "${TASK_DIR}" ]]; then
  echo "Error: task directory not found: ${TASK_DIR}" >&2
  exit 1
fi

USE_DEFAULT_ENV=1
EXTRA_ENV_FILES=()
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
      EXTRA_ENV_FILES+=("$2")
      shift 2
      ;;
    --env=*)
      EXTRA_ENV_FILES+=("${1#*=}")
      shift
      ;;
    --no-default-env)
      USE_DEFAULT_ENV=0
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

if [[ "${USE_DEFAULT_ENV}" -eq 1 && -f "${DEFAULT_ENV_FILE}" ]]; then
  load_env_file "${DEFAULT_ENV_FILE}"
fi
for env_path in "${EXTRA_ENV_FILES[@]}"; do
  load_env_file "${env_path}"
done

exec python "${SCRIPT_DIR}/run_best_of_n.py" "${TASK_NAME}" "${FORWARD_ARGS[@]}"
