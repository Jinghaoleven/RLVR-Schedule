#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

MODEL_NAME=${MODEL_NAME:-}
MERGE_PATH=${MERGE_PATH:-}

if [[ -z "$MODEL_NAME" || -z "$MERGE_PATH" ]]; then
    echo "Usage: MODEL_NAME=/path/to/global_step_x/actor MERGE_PATH=/path/to/output/hf bash scripts/model_merge.sh" >&2
    exit 1
fi

cd "$PROJECT_ROOT"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$MODEL_NAME" \
    --target_dir "$MERGE_PATH"
