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
    REQUIRE_SUB_AGENT_CALL_VALUE=0
else
    ENABLE_SUB_AGENT_VALUE=1
    if [[ -n "${BENCHMARK_REQUIRE_SUB_AGENT_CALL+x}" ]]; then
        REQUIRE_SUB_AGENT_CALL_VALUE="$BENCHMARK_REQUIRE_SUB_AGENT_CALL"
    elif [[ "$MAX_SAMPLES" -eq 1 ]]; then
        REQUIRE_SUB_AGENT_CALL_VALUE=1
    else
        REQUIRE_SUB_AGENT_CALL_VALUE=0
    fi
    if [[ -z "${SUB_AGENT_MODEL:-}" ]]; then
        export SUB_AGENT_MODEL="$MODEL_PATH"
    fi
fi
if [[ "$REQUIRE_SUB_AGENT_CALL_VALUE" != "0" \
      && "$REQUIRE_SUB_AGENT_CALL_VALUE" != "1" ]]; then
    echo "Error: BENCHMARK_REQUIRE_SUB_AGENT_CALL must be 0 or 1." >&2
    exit 2
fi

RUN_ID="${BENCHMARK_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUN_ROOT="$SCRIPT_DIR/results/benchmark/$DATASET_NAME/$SETTING/$RUN_ID"
if [[ -n "${BENCHMARK_EXPERIMENT_NAME:-}" ]]; then
    EXPERIMENT_NAME_VALUE="$BENCHMARK_EXPERIMENT_NAME"
elif [[ "$MAX_SAMPLES" -eq 1 ]]; then
    EXPERIMENT_NAME_VALUE="smoke"
else
    EXPERIMENT_NAME_VALUE="benchmark"
fi
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
export REQUIRE_SUB_AGENT_CALL="$REQUIRE_SUB_AGENT_CALL_VALUE"
export TOOL_TYPE="${BENCHMARK_TOOL_TYPE:-four}"
export SEARCH_MODE="${BENCHMARK_SEARCH_MODE:-multi}"
export ROLLOUT_COUNT=1
export MAX_WORKERS=1
export RUN_TIMEOUT_MINUTES="${BENCHMARK_TIMEOUT_MINUTES:-10}"
export SUB_AGENT_TIMEOUT_MINUTES="${BENCHMARK_SUB_AGENT_TIMEOUT_MINUTES:-2}"
export SUB_AGENT_MAX_LLM_CALLS="${BENCHMARK_SUB_AGENT_MAX_LLM_CALLS:-3}"
export PARENT_FINAL_RESERVE_MINUTES="${BENCHMARK_PARENT_FINAL_RESERVE_MINUTES:-1.5}"
export SUB_AGENT_MIN_TIMEOUT_SECONDS="${BENCHMARK_SUB_AGENT_MIN_TIMEOUT_SECONDS:-30}"
export MAX_TOOL_FORMAT_RETRIES="${BENCHMARK_MAX_TOOL_FORMAT_RETRIES:-3}"

JUDGE_ENABLED="${BENCHMARK_JUDGE_ENABLED:-0}"
WALL_TIMEOUT_MINUTES="${BENCHMARK_WALL_TIMEOUT_MINUTES:-$((RUN_TIMEOUT_MINUTES * MAX_SAMPLES + 10))}"

cat <<EOF
Benchmark run configuration
===========================
dataset: $DATASET_PATH
dataset_name: $DATASET_NAME
setting: $SETTING
sub_agent_enabled: $ENABLE_SUB_AGENT
require_sub_agent_call: $REQUIRE_SUB_AGENT_CALL
model: $MODEL_PATH
sub_agent_model: ${SUB_AGENT_MODEL:-n/a}
tools: $TOOL_TYPE ($SEARCH_MODE search)
output_root: $RUN_ROOT
result_dir: $RESULT_DIR
max_samples: $MAX_SAMPLES
rollouts: $ROLLOUT_COUNT
run_timeout_minutes: $RUN_TIMEOUT_MINUTES
sub_agent_timeout_minutes: ${SUB_AGENT_TIMEOUT_MINUTES:-n/a}
sub_agent_max_llm_calls: ${SUB_AGENT_MAX_LLM_CALLS:-n/a}
parent_final_reserve_minutes: ${PARENT_FINAL_RESERVE_MINUTES:-n/a}
judge_enabled: $JUDGE_ENABLED
judge_model: ${JUDGE_MODEL_NAME:-unset}
EOF

python - "$RUN_ROOT/run_config.json" <<'PY'
import json
import os
import sys

keys = [
    "DATASET", "OUTPUT_PATH", "EXPERIMENT_NAME", "ENABLE_SUB_AGENT",
    "REQUIRE_SUB_AGENT_CALL",
    "MODEL_MODE", "MODEL_PATH", "SUB_AGENT_MODE", "SUB_AGENT_MODEL",
    "JUDGE_MODEL_MODE", "JUDGE_MODEL_NAME", "TOOL_TYPE", "SEARCH_MODE",
    "SEARCH_NUM_RESULTS", "ROLLOUT_COUNT", "MAX_WORKERS",
    "MAX_LLM_CALL_PER_RUN", "MAX_CONTEXT_TOKENS", "MAX_GENERATION_TOKENS",
    "RUN_TIMEOUT_MINUTES", "SUB_AGENT_MAX_LLM_CALLS",
    "SUB_AGENT_TIMEOUT_MINUTES", "SUB_AGENT_FORCE_ANSWER_ATTEMPTS",
    "PARENT_FINAL_RESERVE_MINUTES", "SUB_AGENT_MIN_TIMEOUT_SECONDS",
    "TEMPERATURE", "TOP_P",
    "PRESENCE_PENALTY", "MAX_TOOL_FORMAT_RETRIES",
    "SEARCHSWARM_PATCH_V1", "SEARCHSWARM_PATCH_BUDGET_AWARE",
    "SEARCHSWARM_PATCH_DUPLICATE_FILTER",
    "SEARCHSWARM_PATCH_REPORT_QUALITY",
    "SEARCHSWARM_PATCH_EARLY_STOP_RATIO",
    "SEARCHSWARM_PATCH_DUPLICATE_THRESHOLD",
    "SEARCHSWARM_PATCH_V2", "SEARCHSWARM_PATCH_FINAL_VERIFY",
    "SEARCHSWARM_PATCH_CANDIDATE_LEDGER",
    "SEARCHSWARM_PATCH_ADAPTIVE_ROUTER",
    "SEARCHSWARM_PATCH_MAIN_EARLY_FINALIZE",
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

RESULT_FILE="$RESULT_DIR/iter1.jsonl"
TRAJECTORY_FILE="$RESULT_DIR/subagent_trajectories.jsonl"
VALIDATION_STATUS=""
VALIDATION_STATE="skipped"
VALIDATION_EXECUTED=0
VALIDATION_SUMMARY_FILE="$RUN_ROOT/validation_summary.json"
if [[ "$RUN_STATUS" -eq 0 ]]; then
    set +e
    VALIDATION_ARGS=(
        --result "$RESULT_FILE"
        --trajectory "$TRAJECTORY_FILE"
        --setting "$SETTING"
        --run-exit-code "$RUN_STATUS"
        --summary-json "$VALIDATION_SUMMARY_FILE"
    )
    if [[ "$REQUIRE_SUB_AGENT_CALL" == "1" ]]; then
        VALIDATION_ARGS+=(--require-sub-agent)
    fi
    VALIDATION_ARGS+=(--expected-count "$MAX_SAMPLES")
    python "$SCRIPT_DIR/scripts/validate_smoke_run.py" \
        "${VALIDATION_ARGS[@]}"
    VALIDATION_STATUS=$?
    VALIDATION_EXECUTED=1
    set -e
    if [[ "$VALIDATION_STATUS" -ne 0 ]]; then
        VALIDATION_STATE="failed"
        RUN_STATUS=65
    else
        VALIDATION_STATE="success"
    fi
else
    VALIDATION_STATUS="null"
    python - "$VALIDATION_SUMMARY_FILE" "$RUN_STATUS" "$RESULT_FILE" "$TRAJECTORY_FILE" <<'PY'
import json
import sys
from pathlib import Path

from final_safety import (
    contains_pseudo_tool_call,
    is_failed_placeholder_prediction,
    is_suppressed_prediction,
    is_usable_prediction,
)


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


records = load_jsonl(sys.argv[3])
trajectories = load_jsonl(sys.argv[4])
statuses = [str(row.get("status") or "").lower() for row in trajectories]
steps = [float(row.get("steps", row.get("llm_calls", 0)) or 0) for row in trajectories]
tool_calls = [float(row.get("tool_calls", 0) or 0) for row in trajectories]
duplicate_pairs = [
    {
        "prompt": row.get("prompt", ""),
        "matched_prompt": row.get("duplicate_matched_prompt", ""),
        "similarity": row.get("duplicate_similarity"),
    }
    for row in trajectories
    if row.get("duplicate_subagent_skipped")
]
def sum_int(rows, key):
    return sum(int(row.get(key) or 0) for row in rows)

payload = {
    "run_success": False,
    "run_exit_code": int(sys.argv[2]),
    "validation_status": "skipped",
    "validation_executed": False,
    "result_count": len(records),
    "prediction_empty_count": sum(
        1 for row in records
        if not str(row.get("prediction") or "").strip()
    ),
    "prediction_dsml_count": sum(
        1 for row in records
        if contains_pseudo_tool_call(row.get("prediction"))
    ),
    "prediction_no_answer_count": sum(
        1 for row in records
        if str(row.get("prediction") or "").strip().lower().startswith("no answer found")
    ),
    "prediction_suppressed_count": sum(
        1 for row in records
        if is_suppressed_prediction(row.get("prediction"))
    ),
    "prediction_failed_placeholder_count": sum(
        1 for row in records
        if is_failed_placeholder_prediction(row.get("prediction"))
    ),
    "usable_prediction_count": sum(
        1 for row in records
        if is_usable_prediction(row.get("prediction"))
    ),
    "subagent_total": len(trajectories),
    "subagent_completed": statuses.count("completed"),
    "subagent_fallback": sum(1 for status in statuses if status.endswith("_fallback")),
    "subagent_max_calls": sum(1 for status in statuses if status.startswith("max_calls")),
    "patch_enabled": any(bool(row.get("patch_enabled")) for row in trajectories),
    "subagent_early_stop_count": sum(1 for row in trajectories if row.get("early_stop_triggered")),
    "avg_subagent_steps": round(sum(steps) / len(steps), 2) if steps else 0,
    "avg_subagent_tool_calls": round(sum(tool_calls) / len(tool_calls), 2) if tool_calls else 0,
    "subagent_completed_count": statuses.count("completed"),
    "subagent_fallback_count": sum(1 for status in statuses if status.endswith("_fallback")),
    "subagent_max_calls_count": sum(1 for status in statuses if status.startswith("max_calls")),
    "duplicate_subagent_skipped_count": sum(1 for row in trajectories if row.get("duplicate_subagent_skipped")),
    "duplicate_subagent_brief_pairs": duplicate_pairs,
    "subagent_brief_count_before_filter": sum(
        int(row.get("brief_count_before_filter") or 0)
        for row in trajectories
        if row.get("duplicate_subagent_skipped")
    ),
    "subagent_brief_count_after_filter": sum(
        int(row.get("brief_count_after_filter") or 0)
        for row in trajectories
        if row.get("duplicate_subagent_skipped")
    ),
    "low_quality_report_count": sum(1 for row in trajectories if row.get("low_quality_report") is True),
    "high_quality_report_count": sum(1 for row in trajectories if row.get("low_quality_report") is False),
    "report_with_candidate_count": sum(1 for row in trajectories if row.get("report_has_candidate")),
    "report_with_evidence_count": sum(1 for row in trajectories if row.get("report_has_evidence")),
    "patch_v2_enabled": any(bool(row.get("patch_v2_enabled")) for row in records),
    "candidate_ledger_enabled": any(bool(row.get("candidate_ledger_enabled")) for row in records),
    "candidate_count": sum_int(records, "candidate_count"),
    "candidate_from_main_count": sum_int(records, "candidate_from_main_count"),
    "candidate_from_subagent_count": sum_int(records, "candidate_from_subagent_count"),
    "candidate_from_low_quality_report_count": sum_int(records, "candidate_from_low_quality_report_count"),
    "candidate_deduplicated_count": sum_int(records, "candidate_deduplicated_count"),
    "final_verifier_used_count": sum_int(records, "final_verifier_used_count"),
    "final_verifier_changed_answer_count": sum_int(records, "final_verifier_changed_answer_count"),
    "final_verifier_kept_answer_count": sum_int(records, "final_verifier_kept_answer_count"),
    "final_verifier_rejected_candidate_count": sum_int(records, "final_verifier_rejected_candidate_count"),
    "final_verifier_low_confidence_count": sum_int(records, "final_verifier_low_confidence_count"),
    "final_verifier_empty_or_failed_count": sum_int(records, "final_verifier_empty_or_failed_count"),
    "adaptive_router_enabled": any(bool(row.get("adaptive_router_enabled")) for row in records),
    "router_decision_count": sum_int(records, "router_decision_count"),
    "router_skip_delegation_count": sum_int(records, "router_skip_delegation_count"),
    "router_allow_delegation_count": sum_int(records, "router_allow_delegation_count"),
    "router_force_diverse_brief_count": sum_int(records, "router_force_diverse_brief_count"),
    "router_stop_delegation_count": sum_int(records, "router_stop_delegation_count"),
    "diverse_brief_generated_count": sum_int(records, "diverse_brief_generated_count"),
    "duplicate_brief_rewritten_count": sum_int(records, "duplicate_brief_rewritten_count"),
    "main_agent_early_finalize_count": sum_int(records, "main_agent_early_finalize_count"),
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
PY
fi

python - "$RUN_ROOT/run_status.json" "$RUN_STATUS" "$ELAPSED_SECONDS" \
    "$RESULT_DIR" "$VALIDATION_STATUS" "$VALIDATION_STATE" \
    "$VALIDATION_EXECUTED" "$VALIDATION_SUMMARY_FILE" <<'PY'
import json
import os
import sys

status_code = int(sys.argv[2])
result_dir = sys.argv[4]
validation_raw = sys.argv[5]
validation_exit_code = None if validation_raw == "null" else int(validation_raw)
summary = {}
try:
    with open(sys.argv[8], "r", encoding="utf-8") as handle:
        summary = json.load(handle)
except Exception:
    summary = {}
payload = {
    "exit_code": status_code,
    "elapsed_seconds": int(sys.argv[3]),
    "timed_out": status_code in (124, 137),
    "validation_exit_code": validation_exit_code,
    "validation_status": sys.argv[6],
    "validation_executed": sys.argv[7] == "1",
    "validation_summary_file": sys.argv[8],
    "result_dir": result_dir,
    "result_file": os.path.join(result_dir, "iter1.jsonl"),
    "subagent_trajectory_file": os.path.join(
        result_dir, "subagent_trajectories.jsonl"
    ),
}
payload.update(summary)
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
PY

echo "elapsed_seconds: $ELAPSED_SECONDS"
echo "exit_code: $RUN_STATUS"
echo "validation_status: $VALIDATION_STATE"
echo "validation_exit_code: $VALIDATION_STATUS"
echo "validation_summary_file: $VALIDATION_SUMMARY_FILE"
echo "result_file: $RESULT_FILE"
echo "subagent_trajectory_file: $TRAJECTORY_FILE"
exit "$RUN_STATUS"
