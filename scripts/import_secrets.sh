#!/usr/bin/env bash
# import_secrets.sh — Restore secrets/ từ file export
# Dùng: bash scripts/import_secrets.sh /path/to/mcpt_secrets_20260430.tar.gz
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="${1:-}"

if [[ -z "$ARCHIVE" || ! -f "$ARCHIVE" ]]; then
  echo "[error] Thiếu file: bash scripts/import_secrets.sh /path/to/mcpt_secrets.tar.gz"
  exit 1
fi

tar -xzf "$ARCHIVE" -C "$REPO_ROOT"

echo "✅ secrets/ restored:"
find "$REPO_ROOT/secrets" -type f | grep -v ".cache" | sort | sed "s|$REPO_ROOT/||"
echo ""
echo "Tiếp theo:"
echo "  bash scripts/sync_secrets.sh        # generate env files"
echo "  bash scripts/sync_secrets.sh --vps  # sync + restart VPS"
