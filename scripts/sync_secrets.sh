#!/usr/bin/env bash
# =============================================================================
# sync_secrets.sh — Phân phối secrets/master.env ra tất cả các nơi cần thiết
# =============================================================================
# Chạy sau mỗi lần chỉnh secrets/master.env:
#   bash scripts/sync_secrets.sh
#
# Với flag --vps: sync + restart services trên VPS luôn:
#   bash scripts/sync_secrets.sh --vps
#
# Script này cập nhật:
#   1. infra/env/backend.env   ← Docker Compose --env-file
#   2. infra/env/ai.env        ← Docker Compose --env-file
#   3. infra/env/frontend.env  ← Docker Compose --env-file
#   4. secrets/shared.env      ← Docker volume mount
#   5. VPS (nếu --vps)         ← SSH copy + restart containers
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MASTER="$REPO_ROOT/secrets/master.env"

if [[ ! -f "$MASTER" ]]; then
  echo "[error] secrets/master.env not found." >&2
  echo "        Copy secrets/ folder vào máy này, rồi chạy lại." >&2
  exit 1
fi

set -a; source "$MASTER"; set +a
echo "[sync] Loaded secrets/master.env"

# =============================================================================
# 1. infra/env/backend.env
# =============================================================================
cat > "$REPO_ROOT/infra/env/backend.env" << EOF
# AUTO-GENERATED từ secrets/master.env — không edit trực tiếp
# Regenerate: bash scripts/sync_secrets.sh

POSTGRES_HOST=$POSTGRES_HOST
POSTGRES_PORT=$POSTGRES_PORT
POSTGRES_USER=$POSTGRES_USER
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
POSTGRES_DATABASE=$POSTGRES_DATABASE
POSTGRES_DB=$POSTGRES_DB

METADATA_SERVICE_URL=$METADATA_SERVICE_URL
AI_SERVICE_URL=$AI_SERVICE_URL
TRACKING_SERVICE_URL=$TRACKING_SERVICE_URL
LIGHTNING_API_TOKEN=$LIGHTNING_API_TOKEN
LIGHTNING_API_AUTH_HEADER=$LIGHTNING_API_AUTH_HEADER
LIGHTNING_API_AUTH_PREFIX=$LIGHTNING_API_AUTH_PREFIX

CORS_ALLOWED_ORIGINS=$CORS_ALLOWED_ORIGINS
NEXT_PUBLIC_API_GATEWAY_URL=$NEXT_PUBLIC_API_GATEWAY_URL
JWT_SECRET_KEY=$JWT_SECRET_KEY

MCPT_SECRETS_ROOT=$MCPT_SECRETS_ROOT
MCPT_SHARED_ENV_FILE=$MCPT_SHARED_ENV_FILE

GOOGLE_DRIVE_ENABLED=$GOOGLE_DRIVE_ENABLED
QUEUE_LOCAL_ROOT=$QUEUE_LOCAL_ROOT
QUEUE_VIDEO_FOLDER_NAME=$QUEUE_VIDEO_FOLDER_NAME
STORAGE_INGEST_ENABLED=$STORAGE_INGEST_ENABLED
STORAGE_INGEST_ROOT=$STORAGE_INGEST_ROOT
STORAGE_INGEST_BATCH_SIZE=$STORAGE_INGEST_BATCH_SIZE
STORAGE_INGEST_SAMPLE_FPS=$STORAGE_INGEST_SAMPLE_FPS
QUEUE_PARALLEL_JOBS=$QUEUE_PARALLEL_JOBS
QUEUE_POLL_INTERVAL_SECONDS=$QUEUE_POLL_INTERVAL_SECONDS
QUEUE_MAX_SIZE=$QUEUE_MAX_SIZE
STORAGE_INGEST_MIN_FILE_AGE_SECONDS=$STORAGE_INGEST_MIN_FILE_AGE_SECONDS
VIDEO_STORAGE_ROOT=$VIDEO_STORAGE_ROOT
STORAGE_INGEST_SOURCE_BACKEND=$STORAGE_INGEST_SOURCE_BACKEND
GOOGLE_DRIVE_SOURCE_STORAGE_FOLDER_ID=$GOOGLE_DRIVE_SOURCE_STORAGE_FOLDER_ID
EOF
echo "[sync] Updated infra/env/backend.env"

# =============================================================================
# 2. infra/env/ai.env
# =============================================================================
cat > "$REPO_ROOT/infra/env/ai.env" << EOF
# AUTO-GENERATED từ secrets/master.env — không edit trực tiếp

POSTGRES_HOST=$POSTGRES_HOST
POSTGRES_PORT=$POSTGRES_PORT
POSTGRES_USER=$POSTGRES_USER
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
POSTGRES_DATABASE=$POSTGRES_DATABASE

TRACKING_SERVICE_URL=$TRACKING_SERVICE_URL
LIGHTNING_API_TOKEN=$LIGHTNING_API_TOKEN
LIGHTNING_API_AUTH_HEADER=$LIGHTNING_API_AUTH_HEADER
LIGHTNING_API_AUTH_PREFIX=$LIGHTNING_API_AUTH_PREFIX

MCPT_SECRETS_ROOT=$MCPT_SECRETS_ROOT
MCPT_SHARED_ENV_FILE=$MCPT_SHARED_ENV_FILE

GOOGLE_DRIVE_ENABLED=$GOOGLE_DRIVE_ENABLED
QUEUE_LOCAL_ROOT=$QUEUE_LOCAL_ROOT
VIDEO_STORAGE_ROOT=$VIDEO_STORAGE_ROOT
EOF
echo "[sync] Updated infra/env/ai.env"

# =============================================================================
# 3. infra/env/frontend.env
# =============================================================================
cat > "$REPO_ROOT/infra/env/frontend.env" << EOF
# AUTO-GENERATED từ secrets/master.env — không edit trực tiếp

NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_GATEWAY_URL
INTERNAL_API_GATEWAY_URL=http://backend:8000
EOF
echo "[sync] Updated infra/env/frontend.env"

# =============================================================================
# 4. secrets/shared.env — Docker volume mount + shared_secret_runtime.py
# =============================================================================
cat > "$REPO_ROOT/secrets/shared.env" << EOF
# AUTO-GENERATED từ secrets/master.env — không edit trực tiếp

MCPT_SECRETS_ROOT=$MCPT_SECRETS_ROOT
MCPT_SHARED_ENV_FILE=$MCPT_SHARED_ENV_FILE

POSTGRES_DATABASE=$POSTGRES_DATABASE
POSTGRES_DB=$POSTGRES_DB
POSTGRES_USER=$POSTGRES_USER
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
POSTGRES_PORT=$POSTGRES_PORT

METADATA_SERVICE_URL=$METADATA_SERVICE_URL
TRACKING_SERVICE_URL=$TRACKING_SERVICE_URL
NEXT_PUBLIC_API_GATEWAY_URL=$NEXT_PUBLIC_API_GATEWAY_URL
CORS_ALLOWED_ORIGINS=$CORS_ALLOWED_ORIGINS

JWT_SECRET_KEY=$JWT_SECRET_KEY
VIDEO_STORAGE_ROOT=$VIDEO_STORAGE_ROOT
QUEUE_LOCAL_ROOT=$QUEUE_LOCAL_ROOT
QUEUE_MAX_SIZE=$QUEUE_MAX_SIZE
QUEUE_PARALLEL_JOBS=$QUEUE_PARALLEL_JOBS
QUEUE_POLL_INTERVAL_SECONDS=$QUEUE_POLL_INTERVAL_SECONDS

GOOGLE_DRIVE_ENABLED=$GOOGLE_DRIVE_ENABLED
GOOGLE_DRIVE_MAKE_PUBLIC=$GOOGLE_DRIVE_MAKE_PUBLIC
GOOGLE_DRIVE_ROOT_FOLDER_ID=$GOOGLE_DRIVE_ROOT_FOLDER_ID
GOOGLE_DRIVE_VINUNI_FOLDER_ID=$GOOGLE_DRIVE_VINUNI_FOLDER_ID
MCPT_OAUTH2_CREDENTIALS_FILE=$MCPT_OAUTH2_CREDENTIALS_FILE
MCPT_OAUTH2_TOKEN_FILE=$MCPT_OAUTH2_TOKEN_FILE

LIGHTNING_API_BASE_URL=$LIGHTNING_API_BASE_URL
LIGHTNING_API_ENDPOINT=$LIGHTNING_API_ENDPOINT
LIGHTNING_API_TOKEN=$LIGHTNING_API_TOKEN
TRACKING_SERVICE_PREFER_LOCAL=$TRACKING_SERVICE_PREFER_LOCAL
EOF
echo "[sync] Updated secrets/shared.env"

# =============================================================================
# 5. VPS (optional, flag --vps)
# =============================================================================
if [[ "${1:-}" == "--vps" ]]; then
  if ! command -v sshpass &>/dev/null; then
    echo "[warn] sshpass not found — bỏ qua sync VPS" >&2
  else
    echo "[sync] Syncing to VPS $VPS_HOST ..."
    sshpass -p "$VPS_PASSWORD" scp -o StrictHostKeyChecking=no \
      "$REPO_ROOT/infra/env/backend.env" \
      "$VPS_USER@$VPS_HOST:$VPS_REPO_PATH/infra/env/backend.env"
    sshpass -p "$VPS_PASSWORD" scp -o StrictHostKeyChecking=no \
      "$REPO_ROOT/infra/env/ai.env" \
      "$VPS_USER@$VPS_HOST:$VPS_REPO_PATH/infra/env/ai.env"
    sshpass -p "$VPS_PASSWORD" scp -o StrictHostKeyChecking=no \
      "$REPO_ROOT/secrets/shared.env" \
      "$VPS_USER@$VPS_HOST:$VPS_REPO_PATH/secrets/shared.env"
    sshpass -p "$VPS_PASSWORD" ssh -o StrictHostKeyChecking=no \
      "$VPS_USER@$VPS_HOST" \
      "cd $VPS_REPO_PATH && docker compose -f infra/docker-compose.yml --env-file infra/env/backend.env restart backend ai_service 2>&1 | tail -4"
    echo "[sync] VPS updated and services restarted"
  fi
fi

echo ""
echo "[done] Sync complete."
echo "  Sync local only : bash scripts/sync_secrets.sh"
echo "  Sync + VPS      : bash scripts/sync_secrets.sh --vps"
