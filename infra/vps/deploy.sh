#!/usr/bin/env bash
set -euo pipefail

# G?c repo tr�n VPS (ch?a backend/, frontend/, ai_service/, infra/, shared_secret_runtime.py)
ROOT="${1:-$(pwd)}"
cd "$ROOT"

bash infra/vps/check-secrets.sh "$ROOT"

docker compose -f infra/docker-compose.yml --env-file infra/env/backend.env up -d --build

echo "[ok] Stack started"
echo "[next] docker compose -f infra/docker-compose.yml --env-file infra/env/backend.env ps"
