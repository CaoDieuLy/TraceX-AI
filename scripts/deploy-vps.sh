#!/bin/bash
# TraceX-AI Deployment Script — VPS (Coolify) Frontend Only
# Run this ON VPS after LightningAI backend is deployed
#
# This script deploys ONLY the frontend container.
# Backend services + postgres run on LightningAI.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${TRACE_VPS_ENV_FILE:-$HOME/.tracex-vps.env}"

echo "=== TraceX-AI Frontend Deployment (VPS/Coolify) ==="
echo "Env: $ENV_FILE"

# 1. Generate env file if missing
if [ ! -f "$ENV_FILE" ]; then
    echo "[1/4] Creating $ENV_FILE..."
    cat > "$ENV_FILE" << 'EOF'
# IMPORTANT: Set to your LightningAI public URL + /api/v1
# e.g. https://8000-xxxxxxxx.cloudspaces.litng.ai/api/v1
NEXT_PUBLIC_API_BASE_URL=https://YOUR-LIGHTNINGAI-URL/api/v1
EOF
    echo "  Created $ENV_FILE — EDIT IT to set NEXT_PUBLIC_API_BASE_URL!"
fi

# 2. Pull latest code
echo "[2/4] Pulling latest code..."
cd "$SCRIPT_DIR"
git checkout main
git pull origin main

# 3. Build frontend image
echo "[3/4] Building frontend image..."
docker compose build frontend

# 4. Start frontend
echo "[4/4] Starting frontend..."
docker compose --env-file "$ENV_FILE" up -d frontend

echo ""
echo "=== Deployment complete ==="
echo "Frontend: docker compose ps"
echo "Logs:    docker compose logs -f frontend"
echo ""
echo "Next step: Open your frontend URL and verify it connects to backend."
echo "  Check browser console for any API errors."
