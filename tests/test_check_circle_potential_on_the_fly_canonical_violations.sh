#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_OUTPUT="$(mktemp)"
N_WORKERS="${N_WORKERS:-10}"
trap 'rm -f "$TMP_OUTPUT"' EXIT

PYTHONPATH="${ROOT}/k-servers/src:${PYTHONPATH:-}" \
python "${ROOT}/tools/check_circle_potential_on_the_fly.py" \
  --k 4 \
  --m 6 \
  --n-workers "${N_WORKERS}" \
  --include-k-taxi \
  --disable-gc-during-run \
  --potential-file "${ROOT}/k-servers/src/kserver/potential/canonical_potential.py" \
  --potential-kwargs-file "${ROOT}/tests/canonical_circle_taxi_k4_m6_kwargs.json" \
  >"$TMP_OUTPUT"

grep -q '^violations=17$' "$TMP_OUTPUT"
