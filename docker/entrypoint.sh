#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/kserver/k-server-bench"
RESTART_RAY_SCRIPT="${REPO_ROOT}/tools/legacy-evaluator/restart_ray.sh"
K_SERVER_ENV_NAME="${K_SERVER_ENV_NAME:-k-server}"

if [[ "${K_SERVER_BENCH_USE_LEGACY_EVALUATOR:-false}" == "true" ]]; then
  if [[ ! -x "${RESTART_RAY_SCRIPT}" ]]; then
    chmod +x "${RESTART_RAY_SCRIPT}"
  fi
  runuser -u kserver -- micromamba run -n "${K_SERVER_ENV_NAME}" "${RESTART_RAY_SCRIPT}"
fi

if [[ $# -eq 0 ]]; then
  exec bash
fi

exec "$@"
