#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="$SCRIPT_DIR"
REPO_ROOT="$(cd "$SERVICE_DIR/../../../.." && pwd)"
SECRETS_ROOT="${MCPT_SECRETS_ROOT:-$REPO_ROOT/secrets}"
SHARED_ENV_FILE="${MCPT_SHARED_ENV_FILE:-$SECRETS_ROOT/shared.env}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PIP_DISABLE_PIP_VERSION_CHECK="${PIP_DISABLE_PIP_VERSION_CHECK:-1}"
export MCPT_SECRETS_ROOT="$SECRETS_ROOT"
export MCPT_SHARED_ENV_FILE="$SHARED_ENV_FILE"

echo "[start_api_builder] repo_root=$REPO_ROOT"
echo "[start_api_builder] service_dir=$SERVICE_DIR"
echo "[start_api_builder] secrets_root=$MCPT_SECRETS_ROOT"
echo "[start_api_builder] shared_env=$MCPT_SHARED_ENV_FILE"
echo "[start_api_builder] python=$(command -v python)"
echo "[start_api_builder] python_version=$(python --version 2>&1)"
echo "[start_api_builder] port=${PORT:-8000}"
echo "[start_api_builder] require_cuda=${REQUIRE_CUDA:-0}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[start_api_builder] nvidia_smi=$(command -v nvidia-smi)"
  if ! nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; then
    echo "[start_api_builder] WARNING: nvidia-smi exists but could not query GPU state" >&2
  fi
else
  echo "[start_api_builder] WARNING: nvidia-smi not found in PATH"
fi

python - <<'PY'
try:
    import torch
    print(f"[start_api_builder] torch_version={getattr(torch, '__version__', 'unknown')}")
    print(f"[start_api_builder] torch_cuda_available={torch.cuda.is_available()}")
    print(f"[start_api_builder] torch_cuda_device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print(f"[start_api_builder] torch_cuda_device_name={torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"[start_api_builder] WARNING: torch CUDA probe failed: {exc}")
PY

if [[ "${REQUIRE_CUDA:-0}" == "1" ]]; then
  python - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print(f"[start_api_builder] ERROR: REQUIRE_CUDA=1 but torch import failed: {exc}", file=sys.stderr)
    raise SystemExit(1)

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print(
        "[start_api_builder] ERROR: REQUIRE_CUDA=1 but no CUDA device is visible to the tracking-service process",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
fi

missing_modules="$(
python - <<'PY'
import importlib

checks = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pydantic_settings", "pydantic-settings"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python-headless"),
    ("PIL", "pillow"),
    ("httpx", "httpx"),
    ("multipart", "python-multipart"),
    ("googleapiclient", "google-api-python-client"),
    ("google.auth", "google-auth"),
    ("psycopg2", "psycopg2-binary"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("ultralytics", "ultralytics"),
    ("sentence_transformers", "sentence-transformers"),
    ("open_clip", "open-clip-torch"),
    ("chromadb", "chromadb"),
    ("transformers", "transformers"),
    ("timm", "timm"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("scipy", "scipy"),
    ("tqdm", "tqdm"),
]

missing = []
for module_name, package_name in checks:
    try:
        importlib.import_module(module_name)
    except Exception:
        missing.append(package_name)

print(" ".join(missing))
PY
)"

if [[ "${SKIP_PIP_INSTALL:-0}" == "1" ]]; then
  echo "[start_api_builder] skipping pip install because SKIP_PIP_INSTALL=1"
elif [[ -n "$missing_modules" ]]; then
  echo "[start_api_builder] missing Python packages detected: $missing_modules"
  python -m pip install -r "$SERVICE_DIR/requirements.txt"
else
  echo "[start_api_builder] required Python packages already available; skipping pip install"
fi

exec uvicorn \
  --app-dir "$SERVICE_DIR" \
  app.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --log-level "${UVICORN_LOG_LEVEL:-info}"
