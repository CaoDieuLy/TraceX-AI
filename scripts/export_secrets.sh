#!/usr/bin/env bash
# export_secrets.sh — Đóng gói secrets/ thành 1 file để chuyển máy
# Dùng: bash scripts/export_secrets.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$REPO_ROOT/mcpt_secrets_$(date +%Y%m%d).tar.gz}"

tar --exclude="secrets/master.env.example" \
    --exclude="secrets/.cache" \
    -czf "$OUT" -C "$REPO_ROOT" secrets/

echo "✅ $(du -sh "$OUT" | cut -f1)  $OUT"
echo ""
echo "Máy mới: bash scripts/import_secrets.sh $OUT"
