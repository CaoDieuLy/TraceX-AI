#!/usr/bin/env bash
# =============================================================================
# import_secrets.sh — Restore secrets/ từ file export sang máy mới
# =============================================================================
# Cách dùng:
#   bash scripts/import_secrets.sh /path/to/mcpt_secrets_20260430.tar.gz.enc
#
# Sau đó:
#   bash scripts/sync_secrets.sh        # generate các file env
#   bash scripts/sync_secrets.sh --vps  # sync + restart VPS
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="${1:-}"

if [[ -z "$ARCHIVE" || ! -f "$ARCHIVE" ]]; then
  echo "[error] File không tồn tại: ${ARCHIVE:-<chưa truyền>}"
  echo "Dùng: bash scripts/import_secrets.sh /path/to/mcpt_secrets.tar.gz.enc"
  exit 1
fi

echo "[import] Giải nén: $ARCHIVE"
echo "Nhập passphrase:"

TMP_TAR="/tmp/mcpt_secrets_$$.tar.gz"
openssl enc -d -aes-256-cbc -pbkdf2 -iter 100000 \
  -in "$ARCHIVE" -out "$TMP_TAR"

# Backup nếu đã có secrets/
if [[ -d "$REPO_ROOT/secrets" ]]; then
  BACKUP="$REPO_ROOT/secrets.bak.$(date +%Y%m%d_%H%M%S)"
  mv "$REPO_ROOT/secrets" "$BACKUP"
  echo "[import] Backup cũ → $BACKUP"
fi

tar -xzf "$TMP_TAR" -C "$REPO_ROOT"
rm -f "$TMP_TAR"

echo ""
echo "✅ secrets/ đã restore:"
find "$REPO_ROOT/secrets" -type f | grep -v ".cache" | sort | sed "s|$REPO_ROOT/||"
echo ""
echo "Bước tiếp theo:"
echo "  bash scripts/sync_secrets.sh           # generate infra/env/*.env"
echo "  bash scripts/sync_secrets.sh --vps     # sync + restart VPS luôn"
