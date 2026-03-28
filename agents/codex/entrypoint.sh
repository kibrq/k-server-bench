#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SETUP_SCRIPT="${REPO_ROOT}/agents/codex/setup.sh"
RUN_AS_USER="kserver"
TARGET_DIR="${TARGET_DIR:-$(pwd)}"
USER_HOME="$(getent passwd "${RUN_AS_USER}" | cut -d: -f6)"
CODEX_HOME_DIR="${TARGET_DIR}/.codex"
GLOBAL_CODEX_HOME_DIR="${GLOBAL_CODEX_HOME_DIR:-${USER_HOME}/.codex}"
CODEX_AUTH_MOUNT_PATH="${CODEX_AUTH_MOUNT_PATH:-/run/secrets/codex-auth.json}"
CODEX_AUTH_PATH="${GLOBAL_CODEX_HOME_DIR}/auth.json"
K_SERVER_ENV_NAME="${K_SERVER_ENV_NAME:-k-server}"

run_as_kserver() {
  runuser -u "${RUN_AS_USER}" -- "$@"
}

mkdir -p "${CODEX_HOME_DIR}"
mkdir -p "${GLOBAL_CODEX_HOME_DIR}"
chown -R "${RUN_AS_USER}:${RUN_AS_USER}" "${TARGET_DIR}"
chown -R "${RUN_AS_USER}:${RUN_AS_USER}" "${GLOBAL_CODEX_HOME_DIR}"

if [[ -f "${CODEX_AUTH_MOUNT_PATH}" ]]; then
  cp "${CODEX_AUTH_MOUNT_PATH}" "${CODEX_AUTH_PATH}"
  chmod 600 "${CODEX_AUTH_PATH}"
  chown "${RUN_AS_USER}:${RUN_AS_USER}" "${CODEX_AUTH_PATH}"
else
  echo "Warning: Codex auth file not found at ${CODEX_AUTH_MOUNT_PATH}; continuing without mounted auth." >&2
fi

run_as_kserver micromamba run -n "${K_SERVER_ENV_NAME}" "${SETUP_SCRIPT}" "${TARGET_DIR}"

if [[ $# -eq 0 ]]; then
  exec runuser -u "${RUN_AS_USER}" -- micromamba run -n "${K_SERVER_ENV_NAME}" bash
fi

exec runuser -u "${RUN_AS_USER}" -- micromamba run -n "${K_SERVER_ENV_NAME}" "$@"
