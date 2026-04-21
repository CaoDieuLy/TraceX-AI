#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <domain> <email>"
  exit 1
fi

DOMAIN="$1"
EMAIL="$2"

certbot --nginx \
  -d "$DOMAIN" \
  -m "$EMAIL" \
  --agree-tos \
  --redirect \
  --non-interactive

nginx -t
systemctl reload nginx

echo "HTTPS enabled for $DOMAIN"
