#!/bin/bash
# TraceX-AI Deployment Script — LightningAI A100 80GB
# Run this ON LightningAI after code is pushed to git
#
# This script deploys ONLY backend services + postgres.
# Frontend runs on Coolify VPS and must be deployed separately.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Internal compose: no Traefik, no frontend — just postgres + 3 GPU services
COMPOSE_FILE="$SCRIPT_DIR/backend/services/lightningai-internal-compose.yml"
ENV_FILE="${TRACE_ENV_FILE:-$HOME/.tracex.env}"

echo "=== TraceX-AI Backend Deployment (LightningAI) ==="
echo "Compose: $COMPOSE_FILE"
echo "Env:     $ENV_FILE"

# 1. Create bind mount directories
echo "[1/5] Creating bind mount directories..."
sudo mkdir -p /workspace/models \
    /workspace/datasets/MTMC_Tracking_2024/train \
    /workspace/a20-root \
    /workspace/storage/videos \
    /workspace/storage/tracking-output \
    /workspace/storage/traces \
    /workspace/storage/candidate-previews \
    /workspace/storage/cache \
    /workspace/secrets
sudo chmod -R 777 /workspace

# 2. Generate env file if missing
if [ ! -f "$ENV_FILE" ]; then
    echo "[2/5] Creating $ENV_FILE..."
    JWT_SECRET=$(openssl rand -base64 64 | tr -d '\n')
    cat > "$ENV_FILE" << EOF
# ─── Required ───────────────────────────────────────────────
# LightningAI public URL (get from LightningAI workspace settings)
# e.g. https://8000-xxxxxxxx.cloudspaces.litng.ai
COOLIFY_PUBLIC_URL=https://YOUR-LIGHTNINGAI-URL

# ─── Database ─────────────────────────────────────────────────
POSTGRES_DB=video_tracking
POSTGRES_USER=mcpt_user
POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -d '\n')
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# ─── Auth ────────────────────────────────────────────────────
JWT_SECRET_KEY=$JWT_SECRET

# ─── Storage ─────────────────────────────────────────────────
A20_ROOT=/workspace/a20-root
STORAGE_BASE_URL=https://tracex-ai.smartnovi.tech/storage

# ─── AI Models ───────────────────────────────────────────────
SEAMLESS_M4T_MODEL=facebook/seamless-m4t-v2-large
HF_HOME=/workspace/models/huggingface
TORCH_HOME=/workspace/models/torch
TRANSFORMERS_OFFLINE=0

# ─── Bootstrap admin ──────────────────────────────────────────
BOOTSTRAP_ADMIN_EMAIL=admin@mcpt.local
BOOTSTRAP_ADMIN_PASSWORD=Admin@123456
BOOTSTRAP_ADMIN_FULL_NAME=Administrator
EOF
    echo "  Created $ENV_FILE — EDIT IT to set COOLIFY_PUBLIC_URL and other secrets!"
fi

# 3. Pull latest code
echo "[3/5] Pulling latest code..."
cd "$SCRIPT_DIR"
git checkout main
git pull origin main

# 4. Build images
echo "[4/5] Building backend images..."
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" build

# 5. Start services
echo "[5/5] Starting services..."
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d

echo ""
echo "=== Deployment complete ==="
echo "Services: docker compose -f $COMPOSE_FILE ps"
echo "Logs:    docker compose -f $COMPOSE_FILE logs -f"
echo ""
echo "Next step: Deploy frontend on Coolify VPS."
echo "  1. Update NEXT_PUBLIC_API_BASE_URL in VPS .env to your COOLIFY_PUBLIC_URL + /api/v1"
echo "  2. On VPS: docker compose build frontend && docker compose up -d"
