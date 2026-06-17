#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

DATASET="${DATASET:-$SCRIPT_DIR/eval_data/benchmark/browsecomp_subset_20.jsonl}"
ENV_PREFIX="${ENV_PREFIX:-/home/electrolyte/miniconda3/envs/searchswarm-harness}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-browsecomp20-swarm}"

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
export BENCHMARK_EXPERIMENT_NAME="${BENCHMARK_EXPERIMENT_NAME:-browsecomp20-swarm}"
export BENCHMARK_REQUIRE_SUB_AGENT_CALL="${BENCHMARK_REQUIRE_SUB_AGENT_CALL:-1}"
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
RUN_ROOT="$SCRIPT_DIR/results/benchmark/$DATASET_NAME/swarm/$RUN_ID"

cat <<EOF
BrowseComp 20 Swarm run
=======================
dataset: $DATASET
run_id: $RUN_ID
swarm_root: $RUN_ROOT
require_sub_agent_call: $BENCHMARK_REQUIRE_SUB_AGENT_CALL
timeout_per_item_minutes: $BENCHMARK_TIMEOUT_MINUTES
wall_timeout_minutes: $BENCHMARK_WALL_TIMEOUT_MINUTES
sub_agent_timeout_minutes: $BENCHMARK_SUB_AGENT_TIMEOUT_MINUTES
sub_agent_max_llm_calls: $BENCHMARK_SUB_AGENT_MAX_LLM_CALLS
parent_final_reserve_minutes: $BENCHMARK_PARENT_FINAL_RESERVE_MINUTES
max_workers: 1
EOF

set +e
bash "$SCRIPT_DIR/scripts/run_benchmark_variant.sh" swarm "$DATASET"
STATUS=$?
set -e

cat <<EOF

Done.
====
swarm_status: $STATUS
swarm_root: $RUN_ROOT
run_status: $RUN_ROOT/run_status.json
validation_summary: $RUN_ROOT/validation_summary.json
EOF

if [[ -f "$RUN_ROOT/run_status.json" ]]; then
    echo
    echo "Quick view:"
    cat "$RUN_ROOT/run_status.json"
fi

exit "$STATUS"
