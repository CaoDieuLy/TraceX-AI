#!/usr/bin/env python3
"""
exchange.py - Unit test / Integration test cho luồng hoàn chỉnh:

1. Nhận input video từ `videos/` hoặc đường dẫn truyền vào
2. Nếu là `.mp4` thì convert sang `.h265`, nếu đã là `.h265/.hevc` thì dùng trực tiếp
3. Upload .h265 lên Google Drive: VinUni/Queue/.h265
4. Tạo metadata (VLM + tracking) với code hiện tại
5. Upload metadata lên Google Drive: VinUni/Queue/Metadata
6. Lưu vào PostgreSQL (queue_video_assets)
7. Gọi LightningAI GPU để xử lý (tracking + re-identification)
8. Trả về link video trong DB

Usage:
    python exchange.py --video-file ./videos/Camera_01.h265
    python exchange.py --input-dir ./videos --batch
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

# Add project to path
PROJECT_ROOT = Path(__file__).parent
A20_ROOT = PROJECT_ROOT.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "videos"
SUPPORTED_VIDEO_EXTENSIONS = (".h265", ".hevc", ".mp4")
A20_ROOT_STR = str(A20_ROOT)
if A20_ROOT_STR not in sys.path:
    sys.path.insert(0, A20_ROOT_STR)
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "services" / "tracking-service"))

from app.execution_plan import build_execution_plan
from app.hardware_profiles import detect_gpu_inventory, resolve_hardware_profile
from shared_secret_runtime import (  # noqa: E402
    ensure_canonical_secret_dirs,
    load_runtime_env,
    resolve_oauth2_credentials_path,
    resolve_oauth2_token_path,
)

load_runtime_env()
ensure_canonical_secret_dirs()

print("=" * 80)
print("EXCHANGE.PY - FULL PIPELINE UNIT TEST")
print("=" * 80)

_RUNTIME_PROFILE: dict[str, Any] | None = None


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return max(1, int(raw_value))
    except ValueError:
        return default


def _probe_lightning_health(
    api_base_url: str,
    api_token: str,
    *,
    connect_timeout: int,
    read_timeout: int,
) -> None:
    import requests

    health_url = api_base_url.rstrip("/") + "/health"
    headers = {"Authorization": f"Bearer {api_token}"}
    try:
        response = requests.get(
            health_url,
            headers=headers,
            timeout=(connect_timeout, min(read_timeout, 20)),
        )
        print(f"  Health probe: {response.status_code}")
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            "LightningAI authenticated health probe failed before job submission. "
            f"Remote service may be sleeping, unhealthy, or blocked: {exc}"
        ) from exc


def detect_runtime_profile() -> dict[str, Any]:
    global _RUNTIME_PROFILE
    if _RUNTIME_PROFILE is not None:
        return _RUNTIME_PROFILE

    cpu_cores = multiprocessing.cpu_count()
    host_ram_gb = detect_host_ram_gb()
    inventory = detect_gpu_inventory()
    gpu_name = inventory["gpu_names"][0] if inventory["gpu_names"] else ""
    gpu_count = inventory["gpu_count"]
    has_gpu = gpu_count > 0
    nvenc_available = False

    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        )
        nvenc_available = "hevc_nvenc" in result.stdout
    except Exception:
        nvenc_available = False

    hardware_profile = resolve_hardware_profile(
        "auto",
        gpu_count=gpu_count,
        host_cpu_count=cpu_cores,
        host_ram_gb=host_ram_gb,
    )
    execution_plan = build_execution_plan(
        pipeline_profile={"profile": "accuracy_first"},
        hardware_profile=hardware_profile,
        gpu_count=gpu_count,
        host_cpu_count=cpu_cores,
        host_ram_gb=host_ram_gb,
    )
    use_gpu = has_gpu and nvenc_available
    max_workers = int(execution_plan["parallelism"]["parallel_video_jobs"])
    ffmpeg_preset = str(hardware_profile.get("ffmpeg_hevc_preset", "p3" if use_gpu else "medium"))

    _RUNTIME_PROFILE = {
        "has_gpu": has_gpu,
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "nvenc_available": nvenc_available,
        "use_gpu": use_gpu,
        "cpu_cores": cpu_cores,
        "host_ram_gb": host_ram_gb,
        "max_workers": max_workers,
        "ffmpeg_preset": ffmpeg_preset,
        "hardware_profile": hardware_profile,
        "execution_plan": execution_plan,
    }
    return _RUNTIME_PROFILE


def detect_host_ram_gb() -> int:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        return max(1, int((page_size * phys_pages) / (1024 ** 3)))
    except Exception:
        return 64


def configure_host_performance() -> dict[str, Any]:
    runtime_profile = detect_runtime_profile()
    hardware_profile = runtime_profile["hardware_profile"]
    cpu_threads = max(
        1,
        min(
            runtime_profile["cpu_cores"],
            int(hardware_profile.get("torch_thread_cap", runtime_profile["cpu_cores"])),
        ),
    )
    env_updates = {
        "OMP_NUM_THREADS": str(cpu_threads),
        "MKL_NUM_THREADS": str(cpu_threads),
        "OPENBLAS_NUM_THREADS": str(cpu_threads),
        "NUMEXPR_NUM_THREADS": str(cpu_threads),
        "TOKENIZERS_PARALLELISM": "true",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_DEVICE_MAX_CONNECTIONS": str(runtime_profile["execution_plan"]["parallelism"]["gpu_streams"] or 1),
    }
    for key, value in env_updates.items():
        os.environ[key] = value
    return {
        "cpu_threads": cpu_threads,
        "host_ram_gb": detect_host_ram_gb(),
        "runtime_profile": runtime_profile,
    }

# ==================== STEP 1: PREPARE INPUT VIDEO ====================

def prepare_input_video(input_path: Path) -> Path:
    """
    Normalize input video for the rest of the pipeline.

    - `.mp4` -> convert to `.h265`
    - `.h265` / `.hevc` -> use directly
    """
    suffix = input_path.suffix.lower()
    if suffix == ".mp4":
        return convert_mp4_to_h265(input_path)
    if suffix in {".h265", ".hevc"}:
        print(f"\n[STEP 1] Using pre-encoded input")
        print(f"  Input: {input_path}")
        return input_path
    raise ValueError(f"Unsupported input format: {input_path.name}")

def convert_mp4_to_h265(
    input_path: Path,
    output_dir: Path | None = None,
    crf: int = 28,
    preset: str = "medium"
) -> Path:
    """
    Convert .mp4 video → .h265 (HEVC) using ffmpeg.

    Args:
        input_path: Path to input .mp4 file
        output_dir: Directory for output (default: same as input)
        crf: Quality (lower = better, 28 is default)
        preset: Encoding speed/quality tradeoff

    Returns:
        Path to .h265 file
    """
    print(f"\n[STEP 1] Converting .mp4 → .h265")
    print(f"  Input: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_dir = output_dir or input_path.parent
    output_path = output_dir / f"{input_path.stem}.h265"

    print(f"  Output: {output_path}")

    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError("ffmpeg not installed. Install: apt-get install ffmpeg")

    runtime_profile = detect_runtime_profile()
    print(
        "  Runtime:"
        f" {'GPU NVENC' if runtime_profile['use_gpu'] else 'CPU'}"
        f" | GPU={runtime_profile['gpu_name'] or 'N/A'}"
        f" | workers={runtime_profile['max_workers']}"
    )

    # Convert command
    if runtime_profile["use_gpu"]:
        cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "cuda",
            "-i", str(input_path),
            "-c:v", "hevc_nvenc",
            "-preset", runtime_profile["ffmpeg_preset"],
            "-cq", str(crf),
            "-an",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",  # Overwrite output
            "-i", str(input_path),
            "-c:v", "libx265",
            "-crf", str(crf),
            "-preset", preset,
            "-an",
            str(output_path)
        ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ❌ ffmpeg error:\n{result.stderr[-500:]}")
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-200:]}")

    output_size = output_path.stat().st_size / (1024*1024)
    print(f"  ✅ Converted! Size: {output_size:.1f} MB")
    return output_path


# ==================== STEP 2: UPLOAD TO GOOGLE DRIVE ====================

def upload_to_drive_folder(
    local_path: Path,
    parent_folder_id: str,
    mime_type: str
) -> dict[str, str]:
    """
    Upload file to Google Drive folder using OAuth2 user credentials.
    Supports My Drive (not Shared Drives).

    Returns:
        {"file_id": ..., "view_link": ..., "download_link": ...}
    """
    print(f"\n[STEP 2] Uploading to Google Drive (OAuth2)")
    print(f"  File: {local_path.name}")
    print(f"  Parent folder ID: {parent_folder_id}")

    import pickle
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError

    # Load OAuth2 token from A20-App-119 folder
    token_path = resolve_oauth2_token_path()
    creds_path = resolve_oauth2_credentials_path()
    
    print(f"  Token path: {token_path}")
    print(f"  Creds path: {creds_path}")
    
    if not token_path.exists():
        raise FileNotFoundError(
            f"OAuth2 token not found: {token_path}\n"
            f"Please run upload_oauth2.py first to authenticate."
        )
    
    # Load credentials
    with open(token_path, "rb") as f:
        creds = pickle.load(f)
    
    # Check if token needs refresh
    if creds.expired:
        print(f"  Token expired, refreshing...")
        import json
        with open(creds_path, "r") as f:
            creds_data = json.load(f)
        creds.refresh(
            google.auth.transport.requests.Request()
        )
        # Save refreshed token
        with open(token_path, "wb") as f:
            pickle.dump(creds, f)
        print(f"  ✅ Token refreshed and saved")
    
    # Build drive service
    drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)
    
    # Check parent folder info
    try:
        parent_meta = drive_service.files().get(
            fileId=parent_folder_id,
            fields="driveId,mimeType,name"
        ).execute()
        parent_name = parent_meta.get("name")
        print(f"  Parent folder: '{parent_name}'")
    except Exception as e:
        print(f"  ⚠️  Could not get parent folder info: {e}")

    # Upload with OAuth2 (My Drive only)
    media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=False)
    file_metadata = {"name": local_path.name, "parents": [parent_folder_id]}

    try:
        created = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, webViewLink, webContentLink",
        ).execute()
        file_id = str(created["id"])

        # Set public permission
        perm_body = {"type": "anyone", "role": "reader"}
        try:
            drive_service.permissions().create(
                fileId=file_id,
                body=perm_body,
                fields="id",
            ).execute()
            print(f"  🔓 Set public read permission")
        except Exception as e:
            print(f"  ⚠️  Could not set permission: {e}")

        result = {
            "file_id": file_id,
            "view_link": str(created.get("webViewLink") or f"https://drive.google.com/file/d/{file_id}/view"),
            "download_link": str(created.get("webContentLink") or f"https://drive.google.com/uc?id={file_id}&export=download"),
        }

        print(f"  ✅ Uploaded!")
        print(f"     File ID: {result['file_id']}")
        print(f"     View: {result['view_link']}")
        return result

    except HttpError as e:
        raise


# ==================== STEP 3: GENERATE METADATA ====================

def generate_metadata(
    h265_path: Path,
    camera_id: str | None = None,
    recorded_start: datetime | None = None,
    work_dir: Path | None = None,
) -> tuple[Path, dict, list]:
    """
    Generate metadata using existing VLM + tracking pipeline.

    Returns:
        (metadata_path, video_payload, people_list)
    """
    print(f"\n[STEP 3] Generating metadata (VLM + tracking)")
    print(f"  Video: {h265_path}")

    from app.ingestion_runtime import VideoIngestionRuntime
    from app.config import settings
    import shutil

    perf = configure_host_performance()
    runtime_profile = perf["runtime_profile"]
    resolved_hardware_profile = runtime_profile["hardware_profile"]

    # Use temp directory to avoid permission issues
    work_dir = work_dir or Path(tempfile.mkdtemp(prefix="mcpt_exchange_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    video_dir = work_dir / "videos"
    metadata_dir = work_dir / "metadata"
    video_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Copy .h265 to video_dir
    local_video_copy = video_dir / h265_path.name
    shutil.copy2(h265_path, local_video_copy)

    # PATCH settings: override ONLY output paths (keep legacy_root as real path)
    original_values = {
        "ingestion_work_root": settings.ingestion_work_root,
        "video_conversion_output_dir": settings.video_conversion_output_dir,
        "video_download_output_dir": settings.video_download_output_dir,
        "camera_calibration_path": settings.camera_calibration_path,
        "pipeline_profile": settings.pipeline_profile,
        "gpu_hardware_profile": settings.gpu_hardware_profile,
        "gpu_count": settings.gpu_count,
        "host_cpu_count": settings.host_cpu_count,
        "host_ram_gb": settings.host_ram_gb,
    }
    # Use temp work_dir for outputs, but keep legacy_root pointing to real legacy-engine
    settings.ingestion_work_root = str(work_dir)
    settings.video_conversion_output_dir = str(work_dir / "video-conversion")
    settings.video_download_output_dir = str(work_dir / "tracking-outputs")
    # Use real camera calibration
    settings.camera_calibration_path = str(PROJECT_ROOT / "backend" / "config" / "camera_calibration.json")
    settings.pipeline_profile = "accuracy_first"
    settings.gpu_hardware_profile = str(resolved_hardware_profile.get("resolved_profile_name") or settings.gpu_hardware_profile)
    settings.gpu_count = int(runtime_profile.get("gpu_count") or 0)
    settings.host_cpu_count = perf["cpu_threads"]
    settings.host_ram_gb = perf["host_ram_gb"]

    print(
        "  Performance profile:"
        f" cpu_threads={settings.host_cpu_count}"
        f" host_ram_gb={settings.host_ram_gb}"
        f" gpu_profile={settings.gpu_hardware_profile}"
        f" gpu_count={settings.gpu_count}"
        f" gpu_streams={runtime_profile['execution_plan']['parallelism']['gpu_streams']}"
        f" batch_jobs={runtime_profile['execution_plan']['parallelism']['parallel_video_jobs']}"
    )

    try:
        runtime = VideoIngestionRuntime()
        # Override dirs explicitly
        runtime.work_root = work_dir
        runtime.default_video_dir = video_dir
        runtime.default_metadata_dir = metadata_dir
        runtime.default_source_dir = work_dir / "sources"
        runtime.default_source_dir.mkdir(parents=True, exist_ok=True)

        print("  Running pipeline...")
        # Use recorded_start if provided, else use current time
        if recorded_start is None:
            recorded_start = datetime.now(timezone.utc)

        result = runtime.process_video(
            source_path=str(local_video_copy),
            camera_id=camera_id,
            recorded_start=recorded_start,
            output_video_dir=str(video_dir),
            output_metadata_dir=str(metadata_dir),
            upload_outputs_to_drive=False,
            metadata={},
        )

        metadata_path = Path(result["metadata_path"])
        video_payload = result["video"]
        people = result["people"]

        print(f"  ✅ Metadata generated!")
        print(f"     Metadata file: {metadata_path}")
        print(f"     People detected: {len(people)}")
        print(f"     Video ID: {video_payload.get('video_id', 'N/A')}")

        if people:
            print(f"     First person: {people[0].get('human_key', 'unknown')}")
            print(f"     Track ID: {people[0].get('track_id', 'N/A')}")

        return metadata_path, video_payload, people

    finally:
        # Restore original settings
        for key, value in original_values.items():
            setattr(settings, key, value)


# ==================== STEP 4: SAVE TO POSTGRESQL ====================

def save_to_postgresql(
    video_payload: dict,
    people: list[dict],
    drive_video_file_id: str,
    drive_metadata_file_id: str,
    video_view_link: str,
    metadata_view_link: str,
) -> int:
    """
    Save video + people metadata to PostgreSQL queue_video_assets.

    Returns:
        queue_asset_id (int)
    """
    print(f"\n[STEP 4] Saving to PostgreSQL")

    import os
    import socket
    import subprocess
    from sqlalchemy import create_engine
    from sqlalchemy.engine import make_url
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.exc import OperationalError

    from app.models import Base, QueueVideoAsset

    load_runtime_env()

    def _build_db_url() -> str:
        direct_url = os.getenv("DATABASE_URL", "").strip()
        if direct_url:
            return direct_url
        db_user = os.getenv("POSTGRES_USER", "postgres")
        db_pass = quote_plus(os.getenv("POSTGRES_PASSWORD", ""))
        db_host = os.getenv("POSTGRES_HOST", "localhost")
        db_port = os.getenv("POSTGRES_PORT", "5432")
        db_name = os.getenv("POSTGRES_DATABASE", os.getenv("POSTGRES_DB", "postgres"))
        return f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    def _port_open(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except Exception:
            return False

    def _ensure_local_postgres(url: str) -> None:
        parsed = make_url(url)
        host = str(parsed.host or "localhost")
        port = int(parsed.port or 5432)
        if host not in {"localhost", "127.0.0.1"}:
            return
        if _port_open(host, port):
            return
        starter = PROJECT_ROOT / "start_postgres.py"
        if not starter.exists():
            raise RuntimeError(
                f"PostgreSQL is not reachable at {host}:{port} and start_postgres.py was not found."
            )
        print(f"  PostgreSQL not reachable at {host}:{port}. Starting local PostgreSQL helper...")
        subprocess.run([sys.executable, str(starter)], check=True)

    db_url = _build_db_url()
    parsed_url = make_url(db_url)
    print(
        "  Connecting to:"
        f" {parsed_url.username or 'unknown'}@{parsed_url.host or 'localhost'}:{parsed_url.port or 5432}/{parsed_url.database or ''}"
    )
    _ensure_local_postgres(db_url)

    engine = create_engine(db_url, pool_pre_ping=True)
    try:
        Base.metadata.create_all(bind=engine)
    except OperationalError:
        _ensure_local_postgres(db_url)
        engine = create_engine(db_url, pool_pre_ping=True)
        Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)

    db = SessionLocal()

    try:
        # Generate queue video ID
        queue_video_id = video_payload.get("video_id", f"queue_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")

        # Create QueueVideoAsset record
        queue_asset = QueueVideoAsset(
            video_id=queue_video_id,
            camera_id=video_payload.get("camera_id"),
            title=video_payload.get("title", f"Video {queue_video_id}"),
            source_filename=video_payload.get("source_filename"),
            source_mode="exchange_import",
            queue_position=0,  # Will be updated by queue manager
            storage_backend="google_drive",
            available_link_video=video_view_link,
            available_link_metadata=metadata_view_link,
            drive_video_file_id=drive_video_file_id,
            drive_metadata_file_id=drive_metadata_file_id,
            local_video_path=str(video_payload.get("compressed_path", "")),
            local_metadata_path=str(video_payload.get("metadata_path", "")),
            raw_video_metadata=video_payload,
        )

        db.add(queue_asset)
        db.commit()
        db.refresh(queue_asset)

        print(f"  ✅ Saved to PostgreSQL!")
        print(f"     QueueAsset ID: {queue_asset.id}")
        print(f"     Video ID: {queue_asset.video_id}")

        return queue_asset.id

    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()


# ==================== STEP 5: CALL LIGHTNING AI GPU ====================

def call_lightning_ai_gpu(
    video_id: str,
    drive_video_file_id: str,
    source_filename: str | None = None,
    camera_id: str | None = None,
    api_base_url: str | None = None,
    api_token: str | None = None,
) -> dict[str, Any]:
    """
    Call LightningAI GPU endpoint to run tracking + re-identification.

    Returns:
        AI response dict
    """
    print(f"\n[STEP 5] Calling LightningAI GPU")
    print(f"  Video ID: {video_id}")

    import os
    load_runtime_env()

    api_base_url = api_base_url or os.getenv("LIGHTNING_API_BASE_URL")
    api_token = api_token or os.getenv("LIGHTNING_API_TOKEN")
    api_endpoint = os.getenv("LIGHTNING_API_ENDPOINT", "/api/v1/ingestion/process")

    if not api_base_url or not api_token:
        raise ValueError("LIGHTNING_API_BASE_URL and LIGHTNING_API_TOKEN required")

    import requests

    normalized_endpoint = "/" + api_endpoint.strip().lstrip("/")
    if normalized_endpoint in {"/api/v1/ai/process", "/api/v1/ai/worker", "/predict"}:
        print(
            "  Remapping configured Lightning endpoint"
            f" {normalized_endpoint} -> /api/v1/ingestion/process for exchange ingest flow"
        )
        normalized_endpoint = "/api/v1/ingestion/process"

    url = api_base_url.rstrip("/") + normalized_endpoint
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "source_drive_file_id": drive_video_file_id,
        "source_filename": source_filename or video_id,
        "camera_id": camera_id or Path(source_filename or video_id).stem,
        "output_basename": source_filename or video_id,
        "upload_outputs_to_drive": False,
        "metadata": {
            "pipeline_profile": "accuracy_first",
            "source_mode": "exchange_remote_ingestion",
        },
    }

    print(f"  POST {url}")
    print(f"  Payload: {json.dumps(payload, indent=2)}")

    connect_timeout = _env_int("LIGHTNING_CONNECT_TIMEOUT_SECONDS", 30)
    read_timeout = _env_int("LIGHTNING_READ_TIMEOUT_SECONDS", 1800)
    max_attempts = _env_int("LIGHTNING_MAX_ATTEMPTS", 1)
    preflight_enabled = os.getenv("LIGHTNING_PREFLIGHT_HEALTHCHECK", "true").strip().lower() not in {"0", "false", "no"}

    print(f"  Timeouts: connect={connect_timeout}s, read={read_timeout}s")
    print(f"  Attempts: {max_attempts}")
    print(f"  Preflight healthcheck: {'on' if preflight_enabled else 'off'}")

    if preflight_enabled:
        _probe_lightning_health(
            api_base_url=api_base_url,
            api_token=api_token,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
        )

    last_error: Exception | None = None
    response = None
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            print(f"  Retry attempt {attempt}/{max_attempts}...")
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=(connect_timeout, read_timeout),
            )
            break
        except requests.exceptions.ReadTimeout as exc:
            last_error = exc
            print(f"  Read timeout on attempt {attempt}: {exc}")
        except requests.exceptions.RequestException as exc:
            last_error = exc
            print(f"  Request error on attempt {attempt}: {exc}")
            break

    if response is None:
        raise RuntimeError(
            f"LightningAI request failed after {max_attempts} attempt(s): {last_error}"
        ) from last_error

    print(f"  Status: {response.status_code}")

    if response.status_code == 200:
        body = response.json()
        normalized = {
            "status": body.get("status", "completed"),
            "job_id": body.get("job_id") or body.get("video", {}).get("video_id") or video_id,
            "provider": "lightningai",
            "mode": "remote_ingestion",
            "summary": body.get("summary")
            or (
                f"Remote ingestion completed with {body.get('person_count', 0)} detected people"
                if body.get("person_count") is not None
                else "Remote ingestion completed"
            ),
            "raw_response": body,
        }
        if "person_count" in body:
            normalized["person_count"] = body.get("person_count")
        if body.get("compressed_path"):
            normalized["compressed_path"] = body.get("compressed_path")
        if body.get("metadata_path"):
            normalized["metadata_path"] = body.get("metadata_path")
        print(f"  ✅ LightningAI response received")
        print(f"     Job ID: {normalized.get('job_id', 'N/A')}")
        print(f"     Status: {normalized.get('status', 'N/A')}")
        return normalized
    else:
        print(f"  ❌ Error: {response.text[:500]}")
        response.raise_for_status()


# ==================== MAIN PIPELINE ====================

def run_full_pipeline(
    input_video_path: Path,
    camera_id: str | None = None,
    upload_to_drive: bool = True,
    call_lightning: bool = True,
) -> dict[str, Any]:
    """
    Run full pipeline: Input video -> H265 -> Drive -> Metadata -> PostgreSQL -> LightningAI.

    Args:
        input_video_path: Input `.mp4`, `.h265`, or `.hevc` video file
        camera_id: Camera identifier (optional)
        upload_to_drive: Upload outputs to Google Drive
        call_lightning: Call LightningAI GPU endpoint

    Returns:
        Dict with all results
    """
    print(f"\n{'='*80}")
    print(f"FULL PIPELINE START")
    print(f"  Input: {input_video_path}")
    print(f"  Camera: {camera_id or 'auto'}")
    print(f"  Upload to Drive: {upload_to_drive}")
    print(f"  Call LightningAI: {call_lightning}")
    print(f"{'='*80}")

    results = {
        "input_video": str(input_video_path),
        "camera_id": camera_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "steps": {},
    }

    try:
        # ── STEP 1: Prepare input ────────────────────────────────
        h265_path = prepare_input_video(input_video_path)
        results["steps"]["input_prepare"] = {
            "input_path": str(input_video_path),
            "prepared_path": str(h265_path),
            "size_mb": h265_path.stat().st_size / (1024 * 1024),
        }

        # ── STEP 2: Upload .h265 to Drive ───────────────────────
        if upload_to_drive:
            import os
            from app.config import Settings

            load_runtime_env()
            settings = Settings()

            video_folder_id = os.getenv("GOOGLE_DRIVE_VINUNI_FOLDER_ID") or os.getenv("GOOGLE_DRIVE_ROOT_FOLDER_ID") or settings.google_drive_vinuni_folder_id or settings.google_drive_root_folder_id
            if not video_folder_id:
                # Print env for debug
                print(f"DEBUG ENV:")
                print(f"  VINUNI_FOLDER_ID: {os.getenv('GOOGLE_DRIVE_VINUNI_FOLDER_ID')}")
                print(f"  ROOT_FOLDER_ID: {os.getenv('GOOGLE_DRIVE_ROOT_FOLDER_ID')}")
                raise ValueError("GOOGLE_DRIVE_VINUNI_FOLDER_ID or GOOGLE_DRIVE_ROOT_FOLDER_ID required")

            print(f"  Using folder ID: {video_folder_id}")

            # Upload .h265 to VinUni/Queue/.h265
            drive_video = upload_to_drive_folder(
                h265_path,
                parent_folder_id=video_folder_id,
                mime_type="video/h265"
            )
            results["steps"]["upload_video"] = drive_video

        # ── STEP 3: Generate metadata ───────────────────────────
        metadata_path, video_payload, people = generate_metadata(
            h265_path=h265_path,
            camera_id=camera_id,
        )
        results["steps"]["metadata"] = {
            "metadata_path": str(metadata_path),
            "person_count": len(people),
            "video_id": video_payload.get("video_id"),
        }

        # ── STEP 4: Upload metadata to Drive ────────────────────
        if upload_to_drive:
            # Use metadata folder (nếu có) hoặc cùng folder với video
            metadata_folder_id = os.getenv("GOOGLE_DRIVE_METADATA_FOLDER_ID") or video_folder_id
            print(f"  Uploading metadata to folder ID: {metadata_folder_id}")
            drive_metadata = upload_to_drive_folder(
                metadata_path,
                parent_folder_id=metadata_folder_id,
                mime_type="application/json"
            )
            results["steps"]["upload_metadata"] = drive_metadata

        # ── STEP 5: Save to PostgreSQL ──────────────────────────
        # Always save to DB (independent of Drive upload)
        drive_video_file_id = drive_video.get("file_id", "local_only") if upload_to_drive else "local_only"
        drive_metadata_file_id = drive_metadata.get("file_id", "local_only") if upload_to_drive else "local_only"
        video_view_link = drive_video.get("view_link", "") if upload_to_drive else ""
        metadata_view_link = drive_metadata.get("view_link", "") if upload_to_drive else ""

        queue_asset_id = save_to_postgresql(
            video_payload=video_payload,
            people=people,
            drive_video_file_id=drive_video_file_id,
            drive_metadata_file_id=drive_metadata_file_id,
            video_view_link=video_view_link,
            metadata_view_link=metadata_view_link,
        )
        results["steps"]["postgres"] = {
            "queue_asset_id": queue_asset_id,
            "video_id": video_payload.get("video_id"),
        }
        print(f"  ✅ Saved to PostgreSQL (ID: {queue_asset_id})")

        # ── STEP 6: Call LightningAI GPU ────────────────────────
        if call_lightning and upload_to_drive:
            try:
                lightning_result = call_lightning_ai_gpu(
                    video_id=video_payload.get("video_id"),
                    drive_video_file_id=drive_video["file_id"],
                    source_filename=Path(h265_path).name,
                    camera_id=video_payload.get("camera_id"),
                )
                results["steps"]["lightning_ai"] = lightning_result
            except Exception as exc:
                results["steps"]["lightning_ai_error"] = str(exc)
                results["status"] = "completed_with_lightning_error"
                print(f"  ⚠️ LightningAI step failed: {exc}")

        # ── SUCCESS ──────────────────────────────────────────────
        if results.get("status") != "completed_with_lightning_error":
            results["status"] = "completed"
        results["video_id"] = video_payload.get("video_id")
        results["drive_video_link"] = drive_video["view_link"] if upload_to_drive else None
        results["drive_metadata_link"] = drive_metadata["view_link"] if upload_to_drive else None

        print(f"\n{'='*80}")
        if results["status"] == "completed_with_lightning_error":
            print("⚠️ PIPELINE COMPLETED WITH LIGHTNING ERROR")
        else:
            print(f"✅ PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"  Video ID: {results['video_id']}")
        if upload_to_drive:
            print(f"  Drive video: {results['drive_video_link']}")
            print(f"  Drive metadata: {results['drive_metadata_link']}")
        else:
            print(f"  Drive upload: skipped")
        if "postgres" in results["steps"]:
            print(f"  PostgreSQL ID: {results['steps']['postgres']['queue_asset_id']}")
        else:
            print(f"  PostgreSQL: {results['steps'].get('postgres_error', 'not saved')}")
        if "lightning_ai" in results["steps"]:
            print(f"  LightningAI Job: {results['steps']['lightning_ai'].get('job_id', 'N/A')}")
        elif "lightning_ai_error" in results["steps"]:
            print(f"  LightningAI: {results['steps']['lightning_ai_error']}")

        return results

    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        print(f"\n{'='*80}")
        print(f"❌ PIPELINE FAILED")
        print(f"{'='*80}")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return results


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="Full pipeline unit test: input video -> H265 -> Drive -> Metadata -> LightningAI")
    parser.add_argument("--video-file", type=Path, help="Single video file to process (.h265/.hevc/.mp4)")
    parser.add_argument("--mp4-file", type=Path, help="Backward-compatible alias for --video-file")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help=f"Directory containing input videos (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--batch", action="store_true", help="Process all matching videos in --input-dir")
    parser.add_argument("--max-workers", type=int, help="Override max parallel workers for batch mode")
    parser.add_argument("--camera-id", type=str, help="Camera ID (default: auto from filename)")
    parser.add_argument("--no-drive", action="store_true", help="Skip Google Drive upload")
    parser.add_argument("--no-lightning", action="store_true", help="Skip LightningAI call")
    parser.add_argument("--output", type=Path, help="Save results to JSON file")

    args = parser.parse_args()

    # Find videos
    videos = []
    explicit_video = args.video_file or args.mp4_file
    if explicit_video:
        if not explicit_video.exists():
            print(f"❌ File not found: {explicit_video}")
            sys.exit(1)
        videos = [explicit_video]
    else:
        if not args.input_dir.exists():
            print(f"❌ Directory not found: {args.input_dir}")
            sys.exit(1)
        h265_videos = sorted(
            [path for ext in (".h265", ".hevc") for path in args.input_dir.glob(f"*{ext}")]
        )
        mp4_videos = sorted(args.input_dir.glob("*.mp4"))
        videos = h265_videos or mp4_videos
        if not videos:
            print(f"❌ No supported videos in {args.input_dir}")
            print(f"   Supported: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}")
            sys.exit(1)
        if not args.batch:
            videos = videos[:1]
        print(f"Found {len(videos)} video(s) in {args.input_dir}")

    runtime_profile = detect_runtime_profile()
    max_workers = args.max_workers or runtime_profile["max_workers"]

    # Process each video
    all_results = []
    if args.batch and len(videos) > 1:
        worker_count = max(1, min(max_workers, len(videos)))
        print(
            f"Running batch in parallel with {worker_count} workers"
            f" ({'GPU NVENC' if runtime_profile['use_gpu'] else 'CPU'})"
        )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_video = {
                executor.submit(
                    run_full_pipeline,
                    input_video_path=video_path,
                    camera_id=args.camera_id,
                    upload_to_drive=not args.no_drive,
                    call_lightning=not args.no_lightning,
                ): video_path
                for video_path in videos
            }
            for future in as_completed(future_to_video):
                all_results.append(future.result())
    else:
        for video_path in videos:
            print(f"\n{'='*80}")
            print(f"Processing: {video_path.name}")
            print(f"{'='*80}")

            result = run_full_pipeline(
                input_video_path=video_path,
                camera_id=args.camera_id,
                upload_to_drive=not args.no_drive,
                call_lightning=not args.no_lightning,
            )
            all_results.append(result)

    # Save output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n✅ Results saved to: {args.output}")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: Processed {len(videos)} video(s)")
    success = sum(1 for r in all_results if r.get("status") == "completed")
    partial = sum(1 for r in all_results if r.get("status") == "completed_with_lightning_error")
    failed = len(videos) - success - partial
    print(f"  ✅ Success: {success}")
    print(f"  ⚠️ Partial: {partial}")
    print(f"  ❌ Failed: {failed}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
