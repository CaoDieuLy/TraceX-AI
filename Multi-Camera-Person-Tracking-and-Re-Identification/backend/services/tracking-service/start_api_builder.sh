#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="$SCRIPT_DIR"
REPO_ROOT="$(cd "$SERVICE_DIR/../../../.." && pwd)"
SECRETS_ROOT="${MCPT_SECRETS_ROOT:-$REPO_ROOT/secrets}"
SHARED_ENV_FILE="${MCPT_SHARED_ENV_FILE:-$SECRETS_ROOT/shared.env}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PIP_DISABLE_PIP_VERSION_CHECK="${PIP_DISABLE_PIP_VERSION_CHECK:-1}"
export MCPT_SECRETS_ROOT="$SECRETS_ROOT"
export MCPT_SHARED_ENV_FILE="$SHARED_ENV_FILE"

echo "[start_api_builder] repo_root=$REPO_ROOT"
echo "[start_api_builder] service_dir=$SERVICE_DIR"
echo "[start_api_builder] secrets_root=$MCPT_SECRETS_ROOT"
echo "[start_api_builder] shared_env=$MCPT_SHARED_ENV_FILE"

if [[ "${SKIP_PIP_INSTALL:-0}" != "1" ]]; then
  python -m pip install -r "$SERVICE_DIR/requirements.txt"
fi

exec uvicorn \
  --app-dir "$SERVICE_DIR" \
  app.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --log-level "${UVICORN_LOG_LEVEL:-info}"
