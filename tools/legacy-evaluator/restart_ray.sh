USAGE_THRESHOLD=${K_SERVER_EVALUATE_RAY_USAGE_THRESHOLD:-0.5}
OBJECT_STORE_MEMORY=${K_SERVER_EVALUATE_RAY_OBJECT_STORE_MEMORY:-32}
RAY_PORT=${K_SERVER_EVALUATE_RAY_PORT:-}

TEMP_DIR=${K_SERVER_EVALUATE_RAY_TEMP_DIR:-$HOME/.cache/ray}
SPILL_DIR=${K_SERVER_EVALUATE_RAY_SPILL_DIR:-$HOME/.cache/ray/spill}


mkdir -p $TEMP_DIR
mkdir -p $SPILL_DIR

ray stop --force

RAY_START_ARGS=(
  --head
  --object-store-memory=$((OBJECT_STORE_MEMORY * 1024**3))
  --enable-resource-isolation
  --temp-dir=$TEMP_DIR
  --object-spilling-directory=$SPILL_DIR
)

if [[ -n "${RAY_PORT}" ]]; then
  RAY_START_ARGS+=(--port="${RAY_PORT}")
fi

RAY_memory_usage_threshold=$USAGE_THRESHOLD \
ray start "${RAY_START_ARGS[@]}"
