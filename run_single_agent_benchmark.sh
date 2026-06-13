#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -ne 1 ]]; then
    echo "Usage: bash run_single_agent_benchmark.sh <dataset.jsonl>" >&2
    exit 2
fi

exec bash "$SCRIPT_DIR/scripts/run_benchmark_variant.sh" single "$1"
