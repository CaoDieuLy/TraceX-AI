#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVICE_SCRIPT="$REPO_ROOT/Multi-Camera-Person-Tracking-and-Re-Identification/backend/services/tracking-service/start_api_builder.sh"

if [[ ! -f "$SERVICE_SCRIPT" ]]; then
  echo "[start_tracking_service_api_builder] missing script: $SERVICE_SCRIPT" >&2
  exit 1
fi

exec bash "$SERVICE_SCRIPT"
