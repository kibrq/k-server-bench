#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_SCRIPT="${SCRIPT_DIR}/setup.sh"
TARGET_DIR="${TARGET_DIR:-${HOME:-/home/kserver}/workspace}"
K_SERVER_ENV_NAME="${K_SERVER_ENV_NAME:-k-server}"

run_in_env() {
  if command -v micromamba >/dev/null 2>&1; then
    micromamba run -n "${K_SERVER_ENV_NAME}" "$@"
  else
    "$@"
  fi
}

exec_in_env() {
  if command -v micromamba >/dev/null 2>&1; then
    exec micromamba run -n "${K_SERVER_ENV_NAME}" "$@"
  else
    exec "$@"
  fi
}

mkdir -p "${TARGET_DIR}"

if [[ -n "${SETUP_SCRIPT}" ]]; then
  run_in_env "${SETUP_SCRIPT}" "${TARGET_DIR}"
fi

if [[ $# -eq 0 ]]; then
  exec_in_env bash
fi

exec_in_env "$@"
