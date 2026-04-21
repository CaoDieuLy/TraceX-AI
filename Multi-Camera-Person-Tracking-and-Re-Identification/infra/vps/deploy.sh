#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${1:-$(pwd)}"

cd "$APP_DIR"
docker compose down --remove-orphans || true
docker compose up -d --build
