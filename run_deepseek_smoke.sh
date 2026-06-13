#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET="${DATASET:-eval_data/example/standardized_data.jsonl}"
export OUTPUT_PATH="${OUTPUT_PATH:-./results}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-deepseek-v4-flash-smoke}"
export ROLLOUT_COUNT="${ROLLOUT_COUNT:-1}"
export MAX_WORKERS="${MAX_WORKERS:-1}"

ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}" bash "$SCRIPT_DIR/run_react_infer.sh"
