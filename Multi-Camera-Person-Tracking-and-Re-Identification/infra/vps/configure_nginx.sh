#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <domain> <frontend_port> <api_gateway_port> [template_path]"
  exit 1
fi

DOMAIN="$1"
FRONTEND_PORT="$2"
API_GATEWAY_PORT="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_PATH="${4:-$SCRIPT_DIR/../nginx/search-engine.conf.template}"
TARGET_PATH="/etc/nginx/sites-available/$DOMAIN"

if [[ ! -f "$TEMPLATE_PATH" ]]; then
  echo "Missing nginx template: $TEMPLATE_PATH"
  exit 1
fi

sed \
  -e "s/__DOMAIN__/$DOMAIN/g" \
  -e "s/__FRONTEND_PORT__/$FRONTEND_PORT/g" \
  -e "s/__API_GATEWAY_PORT__/$API_GATEWAY_PORT/g" \
  "$TEMPLATE_PATH" > "$TARGET_PATH"

ln -sfn "$TARGET_PATH" "/etc/nginx/sites-enabled/$DOMAIN"
rm -f /etc/nginx/sites-enabled/default

nginx -t
systemctl reload nginx

echo "Nginx configured for $DOMAIN"
