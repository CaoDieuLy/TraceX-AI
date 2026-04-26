#!/bin/sh
set -e
export PYTHONPATH="${PYTHONPATH:-/app:/workspace/a20-root}"
export A20_ROOT="${A20_ROOT:-/workspace/a20-root}"

# Metadata API (internal to this container)
export METADATA_SERVICE_URL="${METADATA_SERVICE_URL:-http://127.0.0.1:8001}"

# Start metadata service in background
uvicorn metadata_app.main:app --host 127.0.0.1 --port 8001 &
METADATA_PID=$!

# Brief wait for metadata to bind (health is optional)
sleep 3

cleanup() {
  if [ -n "$METADATA_PID" ]; then
    kill "$METADATA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

exec uvicorn gateway_app.main:app --host 0.0.0.0 --port 8000
