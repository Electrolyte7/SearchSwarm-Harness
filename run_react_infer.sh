#!/bin/bash
# ==============================================================================
# Run inference. Reads every setting from .env, then launches run_multi_react.py.
# If MODEL_MODE=local (or SUB_AGENT_MODE=local), start the vLLM servers first
# with `bash deploy_model.sh`.
#
# Override the env file with `ENV_FILE=/path/to/other.env bash run_react_infer.sh`.
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: env file not found at $ENV_FILE. Edit .env first."
    exit 1
fi

# Preserve explicit run-specific overrides supplied by wrapper scripts. Sourcing
# .env normally overwrites exported variables, which would make variants such
# as ENABLE_SUB_AGENT=0/1 ineffective.
OVERRIDE_NAMES=(
    DATASET OUTPUT_PATH EXPERIMENT_NAME ROLLOUT_COUNT MAX_WORKERS
    TEMPERATURE TOP_P PRESENCE_PENALTY ENABLE_SUB_AGENT TOOL_TYPE SEARCH_MODE
    RUN_TIMEOUT_MINUTES SUB_AGENT_TIMEOUT_MINUTES REQUIRE_SUB_AGENT_CALL
    SUB_AGENT_MAX_LLM_CALLS PARENT_FINAL_RESERVE_MINUTES
    SUB_AGENT_MIN_TIMEOUT_SECONDS MAX_TOOL_FORMAT_RETRIES
)
declare -A RUN_OVERRIDES=()
for name in "${OVERRIDE_NAMES[@]}"; do
    if [[ -v "$name" ]]; then
        RUN_OVERRIDES["$name"]="${!name}"
    fi
done

echo "Loading environment from $ENV_FILE ..."
set -a            # export everything sourced below
source "$ENV_FILE"
set +a

for name in "${!RUN_OVERRIDES[@]}"; do
    printf -v "$name" '%s' "${RUN_OVERRIDES[$name]}"
    export "$name"
done

# Allow run-specific values to live in wrapper scripts instead of .env.
: "${DATASET:=eval_data/example/standardized_data.jsonl}"
: "${OUTPUT_PATH:=./results}"
: "${EXPERIMENT_NAME:=}"
: "${ROLLOUT_COUNT:=1}"
: "${MAX_WORKERS:=1}"
: "${TEMPERATURE:=0.7}"
: "${TOP_P:=0.95}"
: "${PRESENCE_PENALTY:=1.1}"

cd "$SCRIPT_DIR"

python -u run_multi_react.py \
    --model "$MODEL_PATH" \
    --dataset "$DATASET" \
    --output "$OUTPUT_PATH" \
    --max_workers "$MAX_WORKERS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --presence_penalty "$PRESENCE_PENALTY" \
    --roll_out_count "$ROLLOUT_COUNT" \
    --total_splits "${WORLD_SIZE:-1}" \
    --worker_split "$(( ${RANK:-0} + 1 ))"
