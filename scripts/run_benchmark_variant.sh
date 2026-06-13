#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <single|swarm> <dataset.jsonl>" >&2
    exit 2
fi

SETTING="$1"
DATASET_INPUT="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"

if [[ "$SETTING" != "single" && "$SETTING" != "swarm" ]]; then
    echo "Error: setting must be 'single' or 'swarm'." >&2
    exit 2
fi
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: env file not found: $ENV_FILE" >&2
    exit 1
fi

cd "$SCRIPT_DIR"
if [[ ! -f "$DATASET_INPUT" ]]; then
    echo "Error: dataset not found: $DATASET_INPUT" >&2
    exit 1
fi

DATASET_PATH="$(realpath "$DATASET_INPUT")"
DATASET_FILE="$(basename "$DATASET_PATH")"
DATASET_NAME="${DATASET_FILE%.jsonl}"
DATASET_NAME="${DATASET_NAME%.json}"
DATASET_NAME="$(printf '%s' "$DATASET_NAME" | sed -E 's/_smoke_[0-9]+$//')"
MAX_SAMPLES="$(awk 'NF {count++} END {print count+0}' "$DATASET_PATH")"

# Load API credentials and the shared inference configuration without printing
# secret values. Explicit variant settings below are exported afterwards.
set -a
source "$ENV_FILE"
set +a

if [[ "${MODEL_MODE:-}" != "api" ]]; then
    echo "Error: smoke benchmark requires MODEL_MODE=api in $ENV_FILE." >&2
    exit 1
fi
for required in MODEL_PATH API_BASE_URL API_KEY SERPER_API_KEY JINA_API_KEY; do
    if [[ -z "${!required:-}" ]]; then
        echo "Error: required setting $required is empty in $ENV_FILE." >&2
        exit 1
    fi
done

if [[ "$SETTING" == "single" ]]; then
    ENABLE_SUB_AGENT_VALUE=0
else
    ENABLE_SUB_AGENT_VALUE=1
    if [[ -z "${SUB_AGENT_MODEL:-}" ]]; then
        export SUB_AGENT_MODEL="$MODEL_PATH"
    fi
fi

RUN_ID="${BENCHMARK_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUN_ROOT="$SCRIPT_DIR/results/benchmark/$DATASET_NAME/$SETTING/$RUN_ID"
EXPERIMENT_NAME_VALUE="smoke"
MODEL_NAME="$(basename "${MODEL_PATH%/}")"
RESULT_DIR="$RUN_ROOT/${MODEL_NAME}_${EXPERIMENT_NAME_VALUE}"

if [[ -e "$RUN_ROOT" ]]; then
    echo "Error: run directory already exists: $RUN_ROOT" >&2
    exit 1
fi
mkdir -p "$RUN_ROOT"

export DATASET="$DATASET_PATH"
export OUTPUT_PATH="$RUN_ROOT"
export EXPERIMENT_NAME="$EXPERIMENT_NAME_VALUE"
export ENABLE_SUB_AGENT="$ENABLE_SUB_AGENT_VALUE"
export TOOL_TYPE="${BENCHMARK_TOOL_TYPE:-four}"
export SEARCH_MODE="${BENCHMARK_SEARCH_MODE:-multi}"
export ROLLOUT_COUNT=1
export MAX_WORKERS=1
export RUN_TIMEOUT_MINUTES="${BENCHMARK_TIMEOUT_MINUTES:-10}"
export SUB_AGENT_TIMEOUT_MINUTES="${BENCHMARK_TIMEOUT_MINUTES:-10}"

JUDGE_ENABLED="${BENCHMARK_JUDGE_ENABLED:-0}"
WALL_TIMEOUT_MINUTES="$((RUN_TIMEOUT_MINUTES + 2))"

cat <<EOF
Benchmark smoke configuration
=============================
dataset: $DATASET_PATH
dataset_name: $DATASET_NAME
setting: $SETTING
sub_agent_enabled: $ENABLE_SUB_AGENT
model: $MODEL_PATH
sub_agent_model: ${SUB_AGENT_MODEL:-n/a}
tools: $TOOL_TYPE ($SEARCH_MODE search)
output_root: $RUN_ROOT
result_dir: $RESULT_DIR
max_samples: $MAX_SAMPLES
rollouts: $ROLLOUT_COUNT
run_timeout_minutes: $RUN_TIMEOUT_MINUTES
judge_enabled: $JUDGE_ENABLED
judge_model: ${JUDGE_MODEL_NAME:-unset}
EOF

python - "$RUN_ROOT/run_config.json" <<'PY'
import json
import os
import sys

keys = [
    "DATASET", "OUTPUT_PATH", "EXPERIMENT_NAME", "ENABLE_SUB_AGENT",
    "MODEL_MODE", "MODEL_PATH", "SUB_AGENT_MODE", "SUB_AGENT_MODEL",
    "JUDGE_MODEL_MODE", "JUDGE_MODEL_NAME", "TOOL_TYPE", "SEARCH_MODE",
    "SEARCH_NUM_RESULTS", "ROLLOUT_COUNT", "MAX_WORKERS",
    "MAX_LLM_CALL_PER_RUN", "MAX_CONTEXT_TOKENS", "MAX_GENERATION_TOKENS",
    "RUN_TIMEOUT_MINUTES", "SUB_AGENT_MAX_LLM_CALLS",
    "SUB_AGENT_TIMEOUT_MINUTES", "TEMPERATURE", "TOP_P",
    "PRESENCE_PENALTY",
]
config = {key: os.environ.get(key) for key in keys}
config["judge_enabled"] = os.environ.get("BENCHMARK_JUDGE_ENABLED", "0") == "1"
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
PY

START_EPOCH="$(date +%s)"
set +e
timeout --signal=TERM --kill-after=30s "${WALL_TIMEOUT_MINUTES}m" \
    bash "$SCRIPT_DIR/run_react_infer.sh" 2>&1 | tee "$RUN_ROOT/run.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e
END_EPOCH="$(date +%s)"
ELAPSED_SECONDS="$((END_EPOCH - START_EPOCH))"

python - "$RUN_ROOT/run_status.json" "$RUN_STATUS" "$ELAPSED_SECONDS" "$RESULT_DIR" <<'PY'
import json
import os
import sys

status_code = int(sys.argv[2])
result_dir = sys.argv[4]
payload = {
    "exit_code": status_code,
    "elapsed_seconds": int(sys.argv[3]),
    "timed_out": status_code in (124, 137),
    "result_dir": result_dir,
    "result_file": os.path.join(result_dir, "iter1.jsonl"),
    "subagent_trajectory_file": os.path.join(
        result_dir, "subagent_trajectories.jsonl"
    ),
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
PY

echo "elapsed_seconds: $ELAPSED_SECONDS"
echo "exit_code: $RUN_STATUS"
echo "result_file: $RESULT_DIR/iter1.jsonl"
echo "subagent_trajectory_file: $RESULT_DIR/subagent_trajectories.jsonl"
exit "$RUN_STATUS"
