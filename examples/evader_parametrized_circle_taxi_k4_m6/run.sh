#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python "$ROOT_DIR/tools/evaluator/evaluate.py" \
  --program_path "$ROOT_DIR/examples/evader_parametrized_circle_taxi_k4_m6/evader_parametrized_potential.py" \
  --results_dir /tmp/eval_evader_parametrized_circle_taxi_k4_m6 \
  --metrics_path "$ROOT_DIR/metrics" \
  --metrics_names circle_taxi_k4_m6.pickle \
  --timeout 1200 \
  --final_evaluation_timeout 1200 \
  --final_evaluation_num_processes 10 \
  "$@"
