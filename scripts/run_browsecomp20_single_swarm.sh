#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

DATASET="${DATASET:-$SCRIPT_DIR/eval_data/benchmark/browsecomp_subset_20.jsonl}"
ENV_PREFIX="${ENV_PREFIX:-/home/electrolyte/miniconda3/envs/searchswarm-harness}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-browsecomp20-single-swarm}"

if [[ ! -f "$DATASET" ]]; then
    echo "Error: dataset not found: $DATASET" >&2
    exit 1
fi
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    echo "Error: conda env python not found: $ENV_PREFIX/bin/python" >&2
    exit 1
fi

export PATH="$ENV_PREFIX/bin:$PATH"
export BENCHMARK_RUN_ID="$RUN_ID"
export BENCHMARK_TIMEOUT_MINUTES="${BENCHMARK_TIMEOUT_MINUTES:-10}"
export BENCHMARK_WALL_TIMEOUT_MINUTES="${BENCHMARK_WALL_TIMEOUT_MINUTES:-210}"
export BENCHMARK_TOOL_TYPE="${BENCHMARK_TOOL_TYPE:-four}"
export BENCHMARK_SEARCH_MODE="${BENCHMARK_SEARCH_MODE:-multi}"
export BENCHMARK_SUB_AGENT_TIMEOUT_MINUTES="${BENCHMARK_SUB_AGENT_TIMEOUT_MINUTES:-2}"
export BENCHMARK_SUB_AGENT_MAX_LLM_CALLS="${BENCHMARK_SUB_AGENT_MAX_LLM_CALLS:-3}"
export BENCHMARK_PARENT_FINAL_RESERVE_MINUTES="${BENCHMARK_PARENT_FINAL_RESERVE_MINUTES:-1.5}"
export BENCHMARK_SUB_AGENT_MIN_TIMEOUT_SECONDS="${BENCHMARK_SUB_AGENT_MIN_TIMEOUT_SECONDS:-30}"
export BENCHMARK_MAX_TOOL_FORMAT_RETRIES="${BENCHMARK_MAX_TOOL_FORMAT_RETRIES:-3}"

DATASET_NAME="$(basename "$DATASET")"
DATASET_NAME="${DATASET_NAME%.jsonl}"
DATASET_NAME="${DATASET_NAME%.json}"
RUN_BASE="$SCRIPT_DIR/results/benchmark/$DATASET_NAME"
SINGLE_ROOT="$RUN_BASE/single/$RUN_ID"
SWARM_ROOT="$RUN_BASE/swarm/$RUN_ID"
COMBINED_DIR="$RUN_BASE/combined/$RUN_ID"

mkdir -p "$COMBINED_DIR"

cat <<EOF
BrowseComp 20 Single + Swarm run
================================
dataset: $DATASET
run_id: $RUN_ID
single_root: $SINGLE_ROOT
swarm_root: $SWARM_ROOT
combined_dir: $COMBINED_DIR
timeout_per_item_minutes: $BENCHMARK_TIMEOUT_MINUTES
wall_timeout_per_variant_minutes: $BENCHMARK_WALL_TIMEOUT_MINUTES
max_workers: 1
EOF

echo
echo "==> Running Single..."
set +e
BENCHMARK_EXPERIMENT_NAME=browsecomp20-single \
BENCHMARK_REQUIRE_SUB_AGENT_CALL=0 \
    bash "$SCRIPT_DIR/scripts/run_benchmark_variant.sh" single "$DATASET"
SINGLE_STATUS=$?
set -e
echo "$SINGLE_STATUS" > "$COMBINED_DIR/single_wrapper_exit_code.txt"

if [[ "$SINGLE_STATUS" -ne 0 ]]; then
    echo
    echo "Single finished with non-zero status: $SINGLE_STATUS"
    echo "Not starting Swarm. Inspect: $SINGLE_ROOT"
    "$ENV_PREFIX/bin/python" - "$COMBINED_DIR/combined_summary.json" \
        "$SINGLE_ROOT/run_status.json" "" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
single_path = Path(sys.argv[2])
payload = {"single": None, "swarm": None}
if single_path.exists():
    payload["single"] = json.loads(single_path.read_text(encoding="utf-8"))
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
    exit "$SINGLE_STATUS"
fi

echo
echo "==> Running Swarm..."
set +e
BENCHMARK_EXPERIMENT_NAME=browsecomp20-swarm \
BENCHMARK_REQUIRE_SUB_AGENT_CALL=1 \
    bash "$SCRIPT_DIR/scripts/run_benchmark_variant.sh" swarm "$DATASET"
SWARM_STATUS=$?
set -e
echo "$SWARM_STATUS" > "$COMBINED_DIR/swarm_wrapper_exit_code.txt"

"$ENV_PREFIX/bin/python" - "$COMBINED_DIR/combined_summary.json" \
    "$SINGLE_ROOT/run_status.json" "$SWARM_ROOT/run_status.json" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
single_path = Path(sys.argv[2])
swarm_path = Path(sys.argv[3])

def load(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

single = load(single_path)
swarm = load(swarm_path)

def compact(status):
    if not status:
        return None
    keys = [
        "exit_code",
        "elapsed_seconds",
        "validation_status",
        "validation_executed",
        "validation_exit_code",
        "result_count",
        "prediction_empty_count",
        "prediction_dsml_count",
        "prediction_no_answer_count",
        "prediction_suppressed_count",
        "prediction_failed_placeholder_count",
        "usable_prediction_count",
        "subagent_total",
        "subagent_completed",
        "subagent_fallback",
        "subagent_max_calls",
        "result_file",
        "subagent_trajectory_file",
        "validation_summary_file",
    ]
    return {key: status.get(key) for key in keys}

payload = {
    "single": compact(single),
    "swarm": compact(swarm),
    "raw_single_status": single,
    "raw_swarm_status": swarm,
}
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY

cat <<EOF

Done.
====
single_status: $SINGLE_STATUS
swarm_status: $SWARM_STATUS
single_root: $SINGLE_ROOT
swarm_root: $SWARM_ROOT
combined_summary: $COMBINED_DIR/combined_summary.json

Quick view:
EOF
cat "$COMBINED_DIR/combined_summary.json"

exit "$SWARM_STATUS"
