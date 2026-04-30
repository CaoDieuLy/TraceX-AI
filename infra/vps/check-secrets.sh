#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
cd "$ROOT"

echo "[check] Root: $ROOT"

if [[ ! -f "infra/docker-compose.yml" ]]; then
  echo "[error] Missing infra/docker-compose.yml"
  exit 1
fi

if [[ ! -f "infra/env/backend.env" ]]; then
  echo "[error] Missing infra/env/backend.env — run: bash scripts/sync_secrets.sh"
  exit 1
fi

required_paths=(
  "secrets/shared.env"
  "secrets/oauth/oauth2_credentials.json"
  "secrets/oauth/oauth2_token.pickle"
)

for p in "${required_paths[@]}"; do
  if [[ ! -f "$p" ]]; then
    echo "[error] Missing required file: $p"
    echo "        Copy secrets/ folder to this machine, then run: bash scripts/sync_secrets.sh"
    exit 1
  fi
  echo "[ok] $p"
done

echo "[ok] Preflight check passed"
