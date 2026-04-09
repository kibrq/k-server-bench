#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KWARGS_FILE="${ROOT_DIR}/examples/evader_parametrized_circle_taxi_k4_m6/kwargs.json"

python "$ROOT_DIR/tools/evaluator/evaluate.py" \
  --program_path "$ROOT_DIR/examples/evader_parametrized_circle_taxi_k4_m6/evader_parametrized_potential.py" \
  --evaluate_home "$ROOT_DIR" \
  --potential_kwargs_json "$KWARGS_FILE" \
  --results_dir /tmp/eval_evader_parametrized_circle_taxi_k4_m6 \
  --metrics_names circle_taxi_k4_m6.pickle \
  --timeout 1200 \
  --final_evaluation_timeout 1200 \
  --final_evaluation_num_processes 10 \
  --keep_only_violations_k \
  "$@"
