#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
POTENTIAL_FILE="${BENCH_ROOT}/k-servers/src/kserver/potential/canonical_potential.py"
KWARGS_FILE="${SCRIPT_DIR}/kwargs.json"
RESULTS_DIR="/tmp"

python3 "${BENCH_ROOT}/tools/evaluator/evaluate.py" \
  --program_path "${POTENTIAL_FILE}" \
  --evaluate_home "${BENCH_ROOT}" \
  --metrics_names "circle_k4_m6.pickle,circle_k4_m8.pickle,circle_taxi_k4_m6.pickle,circle_taxi_k4_m8.pickle" \
  --potential_kwargs_json "${KWARGS_FILE}" \
  --results_dir "${RESULTS_DIR}" \
  --final_evaluation_num_processes 5 \
  --keep_only_violations_k \
  --rho 5

cat "${RESULTS_DIR}/metrics.json"
