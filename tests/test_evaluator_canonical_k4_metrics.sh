#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

CANDIDATE="${TMP_DIR}/canonical_candidate.py"
RESULTS_DIR="${TMP_DIR}/results"

cat > "${CANDIDATE}" <<'PY'
from argparse import ArgumentParser
import json
import traceback

from kserver.potential.canonical_potential import Potential


POTENTIAL_KWARGS = {
    "n": 4,
    "index_matrix": [
        [1, 2, 3, 4],
        [-1, 2, 3, 4],
        [-2, -2, 3, 4],
        [-3, -3, -3, 4],
        [-4, -4, -4, -4],
    ],
    "coefs": [0, 0, 0, 0, 0, 0],
}


def main():
    return {"potential_kwargs": POTENTIAL_KWARGS}


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
  --metrics_names "circle_k4_m6.pickle,circle_k4_m8.pickle,circle_taxi_k4_m6.pickle,circle_taxi_k4_m8.pickle" \
  --results_dir "${RESULTS_DIR}" \
  --final_evaluation_num_processes 5

RESULTS_DIR="${RESULTS_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

results_dir = Path(os.environ["RESULTS_DIR"])
metrics = json.load((results_dir / "metrics.json").open())
correct = json.load((results_dir / "correct.json").open())

assert correct["correct"] is True, correct
public = metrics["public"]
assert public["num_valid_runs"] == 4, metrics
assert public["num_invalid_runs"] == 0, metrics
assert int(public["0/violations_k"]) == 0, public
assert int(public["1/violations_k"]) == 0, public
assert int(public["2/violations_k"]) == 17, public
assert int(public["3/violations_k"]) == 20000, public

print("circle_k4_m6.pickle", public["0/violations_k"])
print("circle_k4_m8.pickle", public["1/violations_k"])
print("circle_taxi_k4_m6.pickle", public["2/violations_k"])
print("circle_taxi_k4_m8.pickle", public["3/violations_k"])
PY
