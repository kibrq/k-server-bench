#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

RESULTS_DIR="${TMP_DIR}/canonical-k4"
METRICS_NAMES="circle_k4_m6.pickle,circle_k4_m8.pickle,circle_taxi_k4_m6.pickle,circle_taxi_k4_m8.pickle"

PYTHONPATH="${ROOT}/k-servers/src:${PYTHONPATH:-}" \
python "${ROOT}/tools/legacy-evaluator/evaluate.py" \
  --program_path "${ROOT}/k-servers/src/kserver/potential/canonical_potential.py" \
  --home "${ROOT}/tools/legacy-evaluator" \
  --ray_address local \
  --metrics_path "${ROOT}/metrics" \
  --metrics_names "${METRICS_NAMES}" \
  --results_dir "${RESULTS_DIR}" \
  --use_default_potential_family \
  --use_default_search_evaluator \
  --potential_path "${ROOT}/k-servers/src/kserver/potential/canonical_potential.py" \
  --potential_family_kwargs_path "${ROOT}/tests/legacy_evaluator_canonical_k4_kwargs.json" \
  --search_max_concurrent 1 \
  --search_min_worker_timeout 5 \
  --search_max_worker_timeout 5 \
  --final_evaluation_max_concurrent 5

RESULTS_DIR="${RESULTS_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

results_dir = Path(os.environ["RESULTS_DIR"])
metrics = json.load((results_dir / "metrics.json").open())
correct = json.load((results_dir / "correct.json").open())

assert correct["correct"] is True, correct
assert metrics["num_valid_runs"] == 1, metrics
assert metrics["num_invalid_runs"] == 0, metrics

public = metrics["public"]
assert int(public["0/violations_k"]) == 0, public
assert int(public["1/violations_k"]) == 0, public
assert int(public["2/violations_k"]) == 17, public
assert int(public["3/violations_k"]) == 20000, public

print("circle_k4_m6.pickle", public["0/violations_k"])
print("circle_k4_m8.pickle", public["1/violations_k"])
print("circle_taxi_k4_m6.pickle", public["2/violations_k"])
print("circle_taxi_k4_m8.pickle", public["3/violations_k"])
PY
