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
  echo "[error] Missing infra/env/backend.env"
  exit 1
fi

required_paths=(
  "secrets/shared.env"
  "secret/shared.env"
)

optional_paths=(
  "secrets/oauth/oauth2_credentials.json"
  "secrets/oauth/oauth2_token.pickle"
  "secret/oauth/oauth2_credentials.json"
  "secret/oauth/oauth2_token.pickle"
)

for p in "${required_paths[@]}"; do
  if [[ ! -f "$p" ]]; then
    echo "[error] Missing required file: $p"
    exit 1
  fi
  echo "[ok] $p"
done

missing_optional=0
for p in "${optional_paths[@]}"; do
  if [[ -f "$p" ]]; then
    echo "[ok] optional present: $p"
  else
    echo "[warn] optional missing: $p"
    missing_optional=1
  fi
done

if [[ $missing_optional -eq 1 ]]; then
  echo "[warn] N?u dùng Google Drive, c?n d? credentials + token trong secrets/oauth ho?c secret/oauth."
fi

echo "[ok] preflight secret/secrets check passed"
