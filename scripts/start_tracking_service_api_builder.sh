#!/usr/bin/env bash
# =============================================================================
# MCPT Tracking Service — LightningAI API Builder Startup Script
# =============================================================================
# Chạy trên LightningAI (NVIDIA A100 80GB) — được đặt làm "On start command"
# trong LightningAI Studio → API Builder → tracking-service → Settings.
#
# Flow:
#   1. Resolve repo root (repo nằm ở /teamspace/studios/this_studio/TraceX-AI)
#   2. Load secrets từ secrets/master.env (nếu có)
#   3. Set defaults cho LightningAI environment
#   4. Start uvicorn FastAPI server trên port 8000
# =============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT="/teamspace/studios/this_studio/TraceX-AI"
SERVICE_DIR="$REPO_ROOT/backend/services/tracking-service"
STORAGE_ROOT="/teamspace/studios/storage"

# ── Load secrets (nếu tồn tại) ───────────────────────────────────────────────
MASTER_ENV="$REPO_ROOT/secrets/master.env"
if [[ -f "$MASTER_ENV" ]]; then
    # shellcheck disable=SC1090
    set -a
    source "$MASTER_ENV"
    set +a
    echo "[start_tracking_service] Loaded secrets from $MASTER_ENV"
else
    echo "[start_tracking_service] WARNING: $MASTER_ENV not found, using environment variables"
fi

# ── Defaults cho LightningAI ───────────────────────────────────────────────────
# Storage
export A20_STORAGE_ROOT="${A20_STORAGE_ROOT:-/teamspace/studios/storage}"

# Model cache
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$STORAGE_ROOT/transformers_cache}"
export HF_HOME="${HF_HOME:-$STORAGE_ROOT/hf_home}"

# Model overrides (A100: can use larger batches than L4)
export MCPT_DINOV2_MODEL_ID="${MCPT_DINOV2_MODEL_ID:-facebook/dinov2-large}"
export MCPT_DETECTOR_BATCH_SIZE="${MCPT_DETECTOR_BATCH_SIZE:-16}"
export MCPT_STREAM_BATCH_SIZE="${MCPT_STREAM_BATCH_SIZE:-300}"
export MCPT_RF_DETR_WEIGHTS="${MCPT_RF_DETR_WEIGHTS:-}"
export MCPT_REID_BATCH_SIZE="${MCPT_REID_BATCH_SIZE:-128}"

# Server
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

# Feature flags
export STORAGE_INGEST_ENABLED="${STORAGE_INGEST_ENABLED:-true}"

# ── Storage paths ──────────────────────────────────────────────────────────────
STORAGE_ROOT="${A20_STORAGE_ROOT:-/teamspace/studios/storage}"
mkdir -p "$STORAGE_ROOT/model_cache"
mkdir -p "$STORAGE_ROOT/video_cache"
mkdir -p "$STORAGE_ROOT/tracking-ingestion"
mkdir -p "$STORAGE_ROOT/tracking-artifacts"

# Model cache
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$STORAGE_ROOT/transformers_cache}"
HF_HOME="${HF_HOME:-$STORAGE_ROOT/hf_home}"
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$HF_HOME"

export A20_STORAGE_ROOT STORAGE_ROOT TRANSFORMERS_CACHE HF_HOME
echo "[start_tracking_service] Storage root: $STORAGE_ROOT"
echo "[start_tracking_service] Transformers cache: $TRANSFORMERS_CACHE"

# ── Change to service directory ───────────────────────────────────────────────
cd "$SERVICE_DIR"
echo "[start_tracking_service] Working directory: $(pwd)"

# ── Ensure dependencies installed ────────────────────────────────────────────
if [[ -f requirements.txt ]]; then
    echo "[start_tracking_service] Installing dependencies..."
    pip install --no-cache-dir -q -r requirements.txt
fi

# ── Health check before starting ───────────────────────────────────────────────
echo "[start_tracking_service] Pre-flight check..."
python3 -c "
import sys
try:
    import fastapi, uvicorn, torch, cv2, numpy, transformers
    print('[start_tracking_service] All critical dependencies OK')
    print(f'  PyTorch: {torch.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'  GPU: {gpu_name}')
        print(f'  VRAM: {gpu_mem:.1f} GB')
        if 'A100' not in gpu_name and 'a100' not in gpu_name.lower():
            print('[start_tracking_service] NOTE: Expected A100, but got: ' + gpu_name)
except ImportError as e:
    print(f'[start_tracking_service] FATAL: Missing dependency: {e}')
    sys.exit(1)
"

# ── Start FastAPI server ───────────────────────────────────────────────────────
echo "[start_tracking_service] Starting uvicorn on $HOST:$PORT..."
echo "[start_tracking_service] Health check: http://$HOST:$PORT/health"

exec uvicorn app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1 \
    --log-level info \
    --access-log
