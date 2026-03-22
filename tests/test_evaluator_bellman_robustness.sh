#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

LOOKUP_JSON="${TMP_DIR}/bellman_lookup.json"
CANDIDATE="${TMP_DIR}/bellman_candidate.py"
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

cat > "${CANDIDATE}" <<PY
from argparse import ArgumentParser
import json
import traceback


class Potential:
    def __init__(self, context, lookup_path, **kwargs):
        del context, kwargs
        with open(lookup_path, "r", encoding="utf-8") as f:
            self.lookup = json.load(f)

    def __call__(self, wf):
        key = ",".join(str(float(x)) for x in wf)
        return self.lookup.get(key, 0.0)


def main():
    return {"potential_kwargs": {"lookup_path": r"${LOOKUP_JSON}"}}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--metrics", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--n_cpus", type=int, default=None)
    args = parser.parse_args()

    try:
        result = main()
    except Exception:
        result = {"failure": "Search Failed", "reason": traceback.format_exc()}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f)
PY

PYTHONPATH="${ROOT}/k-servers/src:${PYTHONPATH:-}" \
python "${ROOT}/tools/evaluator/evaluate.py" \
  --program_path "${CANDIDATE}" \
  --evaluate_home "${ROOT}" \
  --metrics_names "circle_k4_m6.pickle" \
  --results_dir "${RESULTS_DIR}" \
  --robustness_check

RESULTS_DIR="${RESULTS_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

results_dir = Path(os.environ["RESULTS_DIR"])
metrics = json.load((results_dir / "metrics.json").open())
correct = json.load((results_dir / "correct.json").open())

assert correct["correct"] is True, correct
assert metrics["public"]["num_valid_runs"] == 1, metrics
assert metrics["public"]["num_invalid_runs"] == 0, metrics
assert int(metrics["public"]["0/violations_k"]) == 0, metrics
assert metrics["public"]["0/robustness"] is False, metrics
assert metrics["combined_score"] == 0.0, metrics
PY
