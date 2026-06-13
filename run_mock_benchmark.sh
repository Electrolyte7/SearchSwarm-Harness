#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${ENV_PREFIX:-$SCRIPT_DIR/.conda-env}"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.mock}"
MOCK_HOST="${MOCK_OPENAI_HOST:-127.0.0.1}"
MOCK_PORT="${MOCK_OPENAI_PORT:-18080}"
PYTHON_BIN="$ENV_PREFIX/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python not found in conda env: $PYTHON_BIN"
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "Env file not found: $ENV_FILE"
    exit 1
fi

cleanup() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

cd "$SCRIPT_DIR"

echo "Starting mock OpenAI server on ${MOCK_HOST}:${MOCK_PORT}"
MOCK_OPENAI_HOST="$MOCK_HOST" MOCK_OPENAI_PORT="$MOCK_PORT" \
    "$PYTHON_BIN" mock_openai_server.py &
SERVER_PID=$!

for _ in $(seq 1 30); do
    if "$PYTHON_BIN" - <<PY
import json
import urllib.request

url = "http://${MOCK_HOST}:${MOCK_PORT}/health"
with urllib.request.urlopen(url, timeout=2) as resp:
    data = json.load(resp)
assert data["status"] == "ok"
PY
    then
        break
    fi
    sleep 1
done

echo "Running harness benchmark with $ENV_FILE"
PATH="$ENV_PREFIX/bin:$PATH" ENV_FILE="$ENV_FILE" bash "$SCRIPT_DIR/run_react_infer.sh"
