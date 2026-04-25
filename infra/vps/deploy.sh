#!/usr/bin/env bash
set -euo pipefail
# Gốc repo trên VPS (chứa backend/, frontend/, ai_service/, infra/, shared_secret_runtime.py)
ROOT="${1:-$(pwd)}"
cd "$ROOT"
docker compose -f infra/docker-compose.yml --env-file infra/env/backend.env up -d --build
