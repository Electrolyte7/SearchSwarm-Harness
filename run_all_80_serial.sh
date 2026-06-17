#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASETS=(
    "eval_data/benchmark/browsecomp_subset_20.jsonl"
    "eval_data/benchmark/gaia_subset_20.jsonl"
    "eval_data/benchmark/xbench_deepsearch_subset_20.jsonl"
    "eval_data/benchmark/self_deepsearch_qa_20.jsonl"
)

SUITE_ID="${BENCHMARK_SUITE_ID:-$(date +%Y%m%d-%H%M%S)-$$}"
CONTINUE_ON_ERROR="${BENCHMARK_CONTINUE_ON_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"
SUITE_DIR="results/benchmark/suites/$SUITE_ID"
STATUS_FILE="$SUITE_DIR/status.tsv"

if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$SUITE_DIR"
    printf 'dataset\tsetting\texit_code\n' > "$STATUS_FILE"
fi

cat <<EOF
Full benchmark suite
====================
suite_id: $SUITE_ID
datasets: ${#DATASETS[@]} x 20 rows
settings: single, swarm
total_model_tasks: 160 (80 per setting)
force_delegation: disabled
execution: serial
dry_run: $DRY_RUN
EOF

run_one() {
    local dataset="$1"
    local setting="$2"
    local runner

    if [[ "$setting" == "single" ]]; then
        runner="run_single_agent_benchmark.sh"
    else
        runner="run_searchswarm_benchmark.sh"
    fi

    echo
    echo "[$dataset][$setting] starting"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "BENCHMARK_RUN_ID=$SUITE_ID BENCHMARK_EXPERIMENT_NAME=benchmark BENCHMARK_REQUIRE_SUB_AGENT_CALL=0 bash $runner $dataset"
        return 0
    fi

    BENCHMARK_RUN_ID="$SUITE_ID" \
    BENCHMARK_EXPERIMENT_NAME="benchmark" \
    BENCHMARK_REQUIRE_SUB_AGENT_CALL="0" \
        bash "$runner" "$dataset"
    local exit_code=$?

    printf '%s\t%s\t%s\n' \
        "$dataset" "$setting" "$exit_code" >> "$STATUS_FILE"
    echo "[$dataset][$setting] exit_code=$exit_code"
    return "$exit_code"
}

failures=0
for dataset in "${DATASETS[@]}"; do
    if [[ ! -f "$dataset" ]]; then
        echo "Dataset not found: $dataset" >&2
        exit 2
    fi

    for setting in single swarm; do
        if ! run_one "$dataset" "$setting"; then
            failures=$((failures + 1))
            if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
                echo "Suite stopped after a failed run. Set BENCHMARK_CONTINUE_ON_ERROR=1 to continue remaining runs." >&2
                exit 1
            fi
        fi
    done
done

echo
if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run complete; no API calls were made."
elif [[ "$failures" -eq 0 ]]; then
    echo "Suite complete. Status: $STATUS_FILE"
else
    echo "Suite complete with $failures failed run(s). Status: $STATUS_FILE" >&2
    exit 1
fi
