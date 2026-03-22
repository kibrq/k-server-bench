#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

LOOKUP_JSON="${TMP_DIR}/bellman_lookup.json"
POTENTIAL_MODULE="${TMP_DIR}/bellman_cheat.py"
POTENTIAL_FAMILY_KWARGS="${TMP_DIR}/bellman_cheat_kwargs.json"
RESULTS_DIR="${TMP_DIR}/results"

ROOT="${ROOT}" LOOKUP_JSON="${LOOKUP_JSON}" python - <<'PY'
import json
import os
import pickle
from pathlib import Path

root = Path(os.environ["ROOT"])
lookup_path = Path(os.environ["LOOKUP_JSON"])
metric_path = root / "metrics" / "circle_k4_m6.pickle"

with metric_path.open("rb") as f:
    payload = pickle.load(f)

lookup = {
    ",".join(str(float(x)) for x in node["wf_norm"]): float(payload["bellman"][i])
    for i, node in enumerate(payload["nodes"])
}

with lookup_path.open("w", encoding="utf-8") as f:
    json.dump(lookup, f)
PY

cat > "${POTENTIAL_MODULE}" <<'PY'
import json


class Potential:
    def __init__(self, context, lookup_path, **kwargs):
        del context, kwargs
        with open(lookup_path, "r", encoding="utf-8") as f:
            self.lookup = json.load(f)

    def __call__(self, wf):
        key = ",".join(str(float(x)) for x in wf)
        return self.lookup.get(key, 0.0)
PY

cat > "${POTENTIAL_FAMILY_KWARGS}" <<EOF
{
  "potential_kwargs": {
    "lookup_path": "${LOOKUP_JSON}"
  }
}
EOF

PYTHONPATH="${ROOT}/k-servers/src:${PYTHONPATH:-}" \
python "${ROOT}/tools/legacy-evaluator/evaluate.py" \
  --program_path "${POTENTIAL_MODULE}" \
  --home "${ROOT}/tools/legacy-evaluator" \
  --ray_address local \
  --metrics_path "${ROOT}/metrics" \
  --metrics_names "circle_k4_m6.pickle" \
  --results_dir "${RESULTS_DIR}" \
  --use_default_potential_family \
  --use_default_search_evaluator \
  --potential_path "${POTENTIAL_MODULE}" \
  --potential_family_kwargs_path "${POTENTIAL_FAMILY_KWARGS}" \
  --search_max_concurrent 1 \
  --search_min_worker_timeout 5 \
  --search_max_worker_timeout 5 \
  --final_evaluation_max_concurrent 5 \
  --robustness_check

RESULTS_DIR="${RESULTS_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

results_dir = Path(os.environ["RESULTS_DIR"])
metrics = json.load((results_dir / "metrics.json").open())
correct = json.load((results_dir / "correct.json").open())

assert correct["correct"] is True, correct
assert metrics["num_valid_runs"] == 1, metrics
assert metrics["public"]["0/robustness"] is False, metrics
assert int(metrics["public"]["0/violations_k"]) == 0, metrics
assert metrics["combined_score"] == 0.0, metrics
PY
