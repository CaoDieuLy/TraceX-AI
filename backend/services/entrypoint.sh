#!/bin/bash
# Entry point for TraceX-AI services on LightningAI via Coolify
# This script runs inside each service container

set -e

echo "[entrypoint] Starting $SERVICE_NAME..."
echo "[entrypoint] POSTGRES_HOST=$POSTGRES_HOST"
echo "[entrypoint] DATABASE_URL=${DATABASE_URL:0:30}..."

# Wait for postgres to be ready
echo "[entrypoint] Waiting for postgres..."
until pg_isready -h "$POSTGRES_HOST" -p "${POSTGRES_PORT:-5432}" -U "${POSTGRES_USER:-mcpt_user}"; do
  echo "[entrypoint] postgres not ready, sleeping..."
  sleep 2
done
echo "[entrypoint] postgres is ready!"

# Execute the CMD passed by docker
exec "$@"
