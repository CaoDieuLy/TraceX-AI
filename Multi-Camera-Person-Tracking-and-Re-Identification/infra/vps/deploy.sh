#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${1:-$(pwd)}"
SECRETS_ENV="${2:-$APP_DIR/../secrets/shared.env}"

cd "$APP_DIR"
docker compose --env-file "$SECRETS_ENV" down --remove-orphans || true
docker compose --env-file "$SECRETS_ENV" build metadata-service api-gateway frontend
docker compose --env-file "$SECRETS_ENV" up -d postgres metadata-service queue-worker api-gateway frontend
