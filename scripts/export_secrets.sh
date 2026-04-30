#!/usr/bin/env bash
# =============================================================================
# export_secrets.sh — Đóng gói secrets/ thành file mã hoá để chuyển máy
# =============================================================================
# Cách dùng:
#   bash scripts/export_secrets.sh
#   # Sẽ hỏi passphrase → tạo file mcpt_secrets_YYYYMMDD.tar.gz.enc
#
# Ở máy mới:
#   bash scripts/import_secrets.sh mcpt_secrets_YYYYMMDD.tar.gz.enc
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SECRETS_DIR="$REPO_ROOT/secrets"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${1:-$REPO_ROOT/mcpt_secrets_$TIMESTAMP.tar.gz.enc}"

if [[ ! -d "$SECRETS_DIR" ]]; then
  echo "[error] Không tìm thấy secrets/ tại $SECRETS_DIR" >&2
  exit 1
fi

echo "[export] Đóng gói secrets/ → $OUT"
echo "         (Bỏ qua: master.env.example, .cache/)"
echo ""

TMP_TAR="/tmp/mcpt_secrets_$$.tar.gz"

# Pack
cd "$REPO_ROOT"
tar --exclude="secrets/master.env.example" \
    --exclude="secrets/.cache" \
    -czf "$TMP_TAR" secrets/

# Encrypt với passphrase
echo "Nhập passphrase để mã hoá:"
openssl enc -aes-256-cbc -pbkdf2 -iter 100000 \
  -in "$TMP_TAR" -out "$OUT"
rm -f "$TMP_TAR"

SIZE=$(du -sh "$OUT" 2>/dev/null | cut -f1)
echo ""
echo "✅ Đã tạo: $OUT ($SIZE)"
echo ""
echo "Chuyển file này sang máy mới, rồi chạy:"
echo "  bash scripts/import_secrets.sh $OUT"
