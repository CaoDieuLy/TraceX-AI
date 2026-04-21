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
import shutil
import subprocess
import sys
import tempfile
import threading
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")
os.environ.setdefault("OPENCV_VIDEOCAPTURE_DEBUG", "0")
os.environ.setdefault("OPENCV_FFMPEG_DEBUG", "0")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "16")

# Add project to path
PROJECT_ROOT = Path(__file__).parent
A20_ROOT = PROJECT_ROOT.parent
DEFAULT_INPUT_DIR = (PROJECT_ROOT / "videos").resolve()
SUPPORTED_VIDEO_EXTENSIONS = (".h265", ".hevc", ".mp4")
A20_ROOT_STR = str(A20_ROOT)
if A20_ROOT_STR not in sys.path:
    sys.path.insert(0, A20_ROOT_STR)
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "services" / "tracking-service"))

from app.execution_plan import build_execution_plan
from app.hardware_profiles import detect_gpu_inventory, resolve_hardware_profile
from shared_secret_runtime import (  # noqa: E402
    build_google_drive_oauth_service,
    ensure_canonical_secret_dirs,
    load_runtime_env,
    resolve_oauth2_credentials_path,
    resolve_oauth2_token_path,
)

load_runtime_env(override=False)
ensure_canonical_secret_dirs()

print("=" * 80)
print("EXCHANGE.PY - FULL PIPELINE UNIT TEST")
print("=" * 80)

_RUNTIME_PROFILE: dict[str, Any] | None = None
_DRIVE_THREAD_LOCAL = threading.local()
_DRIVE_LAYOUT_CACHE: dict[str, str] | None = None
_DRIVE_LAYOUT_LOCK = threading.Lock()
_METADATA_PIPELINE_SEMAPHORE: threading.BoundedSemaphore | None = None
_METADATA_PIPELINE_SEMAPHORE_LOCK = threading.Lock()


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return max(1, int(raw_value))
    except ValueError:
        return default


def _metadata_pipeline_slots(runtime_profile: dict[str, Any]) -> int:
    execution_parallelism = runtime_profile["execution_plan"]["parallelism"]
    default_slots = int(
        execution_parallelism.get("parallel_video_jobs")
        or execution_parallelism.get("gpu_streams")
        or runtime_profile["max_workers"]
        or 1
    )
    requested_slots = _env_int("MCPT_METADATA_PIPELINE_SLOTS", default_slots)
    return max(1, min(requested_slots, int(runtime_profile["max_workers"]), int(runtime_profile["cpu_cores"])))


def _metadata_pipeline_semaphore(runtime_profile: dict[str, Any]) -> threading.BoundedSemaphore:
    global _METADATA_PIPELINE_SEMAPHORE

    capacity = _metadata_pipeline_slots(runtime_profile)
    with _METADATA_PIPELINE_SEMAPHORE_LOCK:
        current = _METADATA_PIPELINE_SEMAPHORE
        if current is None or getattr(current, "_initial_value", None) != capacity:
            current = threading.BoundedSemaphore(capacity)
            setattr(current, "_initial_value", capacity)
            _METADATA_PIPELINE_SEMAPHORE = current
        return current


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


def _build_drive_service():
    drive_service = getattr(_DRIVE_THREAD_LOCAL, "drive_service", None)
    if drive_service is None:
        drive_service = build_google_drive_oauth_service()
        _DRIVE_THREAD_LOCAL.drive_service = drive_service
    return drive_service


def _query_drive_children(parent_id: str, *, name: str | None = None, mime_type: str | None = None) -> list[dict[str, Any]]:
    drive_service = _build_drive_service()
    query_parts = [f"'{parent_id}' in parents", "trashed = false"]
    if name:
        escaped_name = name.replace("'", "\\'")
        query_parts.append(f"name = '{escaped_name}'")
    if mime_type:
        query_parts.append(f"mimeType = '{mime_type}'")
    query = " and ".join(query_parts)
    response = drive_service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name, mimeType, webViewLink, webContentLink, createdTime)",
        pageSize=200,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    return list(response.get("files") or [])


def _get_drive_file(file_id: str) -> dict[str, Any]:
    drive_service = _build_drive_service()
    return drive_service.files().get(
        fileId=file_id,
        fields="id, name, mimeType, webViewLink, webContentLink, parents, createdTime",
        supportsAllDrives=True,
    ).execute()


def _ensure_drive_folder(parent_id: str, name: str) -> str:
    folder_mime = "application/vnd.google-apps.folder"
    existing = _query_drive_children(parent_id, name=name, mime_type=folder_mime)
    if existing:
        return str(existing[0]["id"])
    drive_service = _build_drive_service()
    created = drive_service.files().create(
        body={"name": name, "mimeType": folder_mime, "parents": [parent_id]},
        fields="id",
        supportsAllDrives=True,
    ).execute()
    return str(created["id"])


def resolve_drive_layout() -> dict[str, str]:
    global _DRIVE_LAYOUT_CACHE
    if _DRIVE_LAYOUT_CACHE is not None:
        return dict(_DRIVE_LAYOUT_CACHE)

    from app.config import Settings

    with _DRIVE_LAYOUT_LOCK:
        if _DRIVE_LAYOUT_CACHE is not None:
            return dict(_DRIVE_LAYOUT_CACHE)

        load_runtime_env(override=False)
        settings = Settings()
        vinuni_folder_id = (
            os.getenv("GOOGLE_DRIVE_VINUNI_FOLDER_ID")
            or settings.google_drive_vinuni_folder_id
            or ""
        ).strip()
        root_folder_id = (
            os.getenv("GOOGLE_DRIVE_ROOT_FOLDER_ID")
            or settings.google_drive_root_folder_id
            or ""
        ).strip()
        if not vinuni_folder_id and not root_folder_id:
            raise ValueError("GOOGLE_DRIVE_VINUNI_FOLDER_ID or GOOGLE_DRIVE_ROOT_FOLDER_ID required")

        vinuni_id = vinuni_folder_id or _ensure_drive_folder(root_folder_id, settings.google_drive_vinuni_folder_name)
        queue_id = _ensure_drive_folder(vinuni_id, settings.google_drive_queue_folder_name)
        import_id = _ensure_drive_folder(vinuni_id, settings.google_drive_import_folder_name)
        queue_h265_id = _ensure_drive_folder(queue_id, settings.google_drive_h265_folder_name)
        queue_metadata_id = _ensure_drive_folder(queue_id, settings.google_drive_metadata_folder_name)
        _DRIVE_LAYOUT_CACHE = {
            "vinuni_id": vinuni_id,
            "queue_id": queue_id,
            "import_id": import_id,
            "queue_h265_id": queue_h265_id,
            "queue_metadata_id": queue_metadata_id,
        }
        return dict(_DRIVE_LAYOUT_CACHE)


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
    input_path = input_path.resolve()
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
            "-bf", "0",
            "-g", "25",
            "-forced-idr", "1",
            "-aud", "1",
            "-pix_fmt", "yuv420p",
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
            "-x265-params", "repeat-headers=1:aud=1:no-open-gop=1:keyint=25:min-keyint=25",
            "-pix_fmt", "yuv420p",
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

    Returns:
        {"file_id": ..., "view_link": ..., "download_link": ...}
    """
    print(f"\n[STEP 2] Uploading to Google Drive (OAuth2)")
    print(f"  File: {local_path.name}")
    print(f"  Parent folder ID: {parent_folder_id}")

    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError

    token_path = resolve_oauth2_token_path()
    credentials_path = resolve_oauth2_credentials_path()
    print(f"  Token path: {token_path}")
    print(f"  Creds path: {credentials_path}")

    drive_service = _build_drive_service()
    
    # Check parent folder info
    try:
        parent_meta = drive_service.files().get(
            fileId=parent_folder_id,
            fields="driveId,mimeType,name",
            supportsAllDrives=True,
        ).execute()
        parent_name = parent_meta.get("name")
        print(f"  Parent folder: '{parent_name}'")
    except Exception as e:
        print(f"  ⚠️  Could not get parent folder info: {e}")

    existing = _query_drive_children(parent_folder_id, name=local_path.name)
    if existing:
        current = existing[0]
        file_id = str(current["id"])
        print("  ♻️ Reusing existing file in target folder")
        print(f"     File ID: {file_id}")
        return {
            "file_id": file_id,
            "view_link": str(current.get("webViewLink") or f"https://drive.google.com/file/d/{file_id}/view"),
            "download_link": str(current.get("webContentLink") or f"https://drive.google.com/uc?id={file_id}&export=download"),
        }

    media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)
    file_metadata = {"name": local_path.name, "parents": [parent_folder_id]}

    try:
        last_error: Exception | None = None
        for attempt in range(1, 3):
            created = drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id, webViewLink, webContentLink",
                supportsAllDrives=True,
            ).execute()
            file_id = str(created["id"])

            # Set public permission
            perm_body = {"type": "anyone", "role": "reader"}
            try:
                drive_service.permissions().create(
                    fileId=file_id,
                    body=perm_body,
                    fields="id",
                    supportsAllDrives=True,
                ).execute()
                print(f"  🔓 Set public read permission")
            except Exception as e:
                print(f"  ⚠️  Could not set permission: {e}")

            try:
                verified = _get_drive_file(file_id)
            except Exception as exc:
                last_error = exc
                by_name = _query_drive_children(parent_folder_id, name=local_path.name)
                if by_name:
                    verified = by_name[0]
                    file_id = str(verified["id"])
                elif attempt < 2:
                    print(f"  ⚠️  Uploaded file not immediately readable; retrying upload ({attempt}/2)...")
                    continue
                else:
                    raise RuntimeError(
                        f"Uploaded file ID {file_id} could not be verified in Google Drive"
                    ) from exc

            result = {
                "file_id": file_id,
                "view_link": str(verified.get("webViewLink") or created.get("webViewLink") or f"https://drive.google.com/file/d/{file_id}/view"),
                "download_link": str(verified.get("webContentLink") or created.get("webContentLink") or f"https://drive.google.com/uc?id={file_id}&export=download"),
            }

            print(f"  ✅ Uploaded!")
            print(f"     File ID: {result['file_id']}")
            print(f"     View: {result['view_link']}")
            return result

        if last_error is not None:
            raise last_error

    except HttpError as e:
        raise


# ==================== STEP 3: GENERATE METADATA ====================

def generate_metadata(
    h265_path: Path,
    camera_id: str | None = None,
    recorded_start: datetime | None = None,
    work_dir: Path | None = None,
    drive_video_file_id: str | None = None,
) -> tuple[Path, dict, list]:
    """
    Generate metadata using existing VLM + tracking pipeline.

    Returns:
        (metadata_path, video_payload, people_list)
    """
    print(f"\n[STEP 3] Generating metadata (tracking-service ingestion)")
    print(f"  Video: {h265_path}")

    perf = configure_host_performance()
    runtime_profile = perf["runtime_profile"]
    resolved_hardware_profile = runtime_profile["hardware_profile"]

    # Persist bootstrap artifacts locally so DB can be rebuilt from disk later.
    queue_root = PROJECT_ROOT / "storage" / "queue" / "local" / "Queue"
    source_dir = PROJECT_ROOT / "storage" / "queue" / "local" / "ExchangeSource"
    video_dir = queue_root / ".h265"
    metadata_dir = queue_root / "Metadata"
    work_dir = work_dir or (PROJECT_ROOT / "storage" / "exchange-work" / h265_path.stem)
    work_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    local_source_copy = source_dir / h265_path.name
    local_queue_video_copy = video_dir / h265_path.name
    if h265_path.resolve() != local_source_copy.resolve():
        shutil.copy2(h265_path, local_source_copy)
    if h265_path.resolve() != local_queue_video_copy.resolve():
        shutil.copy2(h265_path, local_queue_video_copy)

    print(
        "  Performance profile:"
        f" cpu_threads={perf['cpu_threads']}"
        f" host_ram_gb={perf['host_ram_gb']}"
        f" gpu_profile={resolved_hardware_profile.get('resolved_profile_name') or 'auto'}"
        f" gpu_count={runtime_profile['gpu_count']}"
        f" gpu_streams={runtime_profile['execution_plan']['parallelism']['gpu_streams']}"
        f" batch_jobs={runtime_profile['execution_plan']['parallelism']['parallel_video_jobs']}"
        f" metadata_slots={_metadata_pipeline_slots(runtime_profile)}"
    )

    metadata_runtime = {
        "pipeline_profile": "accuracy_first",
        "gpu_hardware_profile": str(
            resolved_hardware_profile.get("resolved_profile_name")
            or runtime_profile["hardware_profile"].get("resolved_profile_name")
            or "auto"
        ),
        "gpu_count": int(runtime_profile.get("gpu_count") or 0),
        "host_cpu_count": perf["cpu_threads"],
        "host_ram_gb": perf["host_ram_gb"],
        "source_mode": "exchange_remote_metadata_generation",
    }

    def _tracking_base_url() -> str:
        load_runtime_env(override=False)
        api_base_url = (
            os.getenv("TRACKING_SERVICE_URL")
            or os.getenv("LIGHTNING_API_BASE_URL")
            or ""
        ).strip()
        local_tracking_url = os.getenv("TRACKING_SERVICE_LOCAL_URL", "http://127.0.0.1:8000").strip()
        prefer_local = os.getenv("TRACKING_SERVICE_PREFER_LOCAL", "true").strip().lower() not in {"0", "false", "no"}
        if prefer_local and local_tracking_url:
            try:
                parsed_host = local_tracking_url.split("://", 1)[-1].split("/", 1)[0]
                host, _, raw_port = parsed_host.partition(":")
                port = int(raw_port or 80)
                with socket.create_connection((host, port), timeout=1.5):
                    return local_tracking_url.rstrip("/")
            except Exception:
                pass
        if not api_base_url:
            raise ValueError("TRACKING_SERVICE_URL or LIGHTNING_API_BASE_URL is required for remote metadata generation")
        return api_base_url.rstrip("/")

    def _tracking_headers(include_json: bool = False) -> dict[str, str]:
        load_runtime_env(override=False)
        token = os.getenv("LIGHTNING_API_TOKEN", "").strip()
        if not token:
            raise ValueError("LIGHTNING_API_TOKEN is required for remote metadata generation")
        header_name = os.getenv("LIGHTNING_API_AUTH_HEADER", "Authorization").strip() or "Authorization"
        prefix = os.getenv("LIGHTNING_API_AUTH_PREFIX", "Bearer ")
        if prefix and not prefix.endswith(" "):
            prefix = f"{prefix} "
        headers = {header_name: f"{prefix}{token}".strip()}
        if include_json:
            headers["Content-Type"] = "application/json"
        return headers

    def _write_local_metadata_file(target_path: Path, payload_video: dict[str, Any], payload_people: list[dict[str, Any]]) -> None:
        payload = {
            "schema_version": "hospital_person_metadata_v3",
            "video": payload_video,
            "people": payload_people,
        }
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    import requests

    endpoint_root = _tracking_base_url()
    connect_timeout = _env_int("LIGHTNING_CONNECT_TIMEOUT_SECONDS", 30)
    read_timeout = _env_int("LIGHTNING_READ_TIMEOUT_SECONDS", 1800)
    preflight_enabled = os.getenv("LIGHTNING_PREFLIGHT_HEALTHCHECK", "true").strip().lower() not in {"0", "false", "no"}

    print(f"  Metadata backend: remote tracking-service")
    print(f"  Endpoint root: {endpoint_root}")
    print("  Waiting for metadata pipeline slot...")
    with _metadata_pipeline_semaphore(runtime_profile):
        if preflight_enabled:
            _probe_lightning_health(
                api_base_url=endpoint_root,
                api_token=os.getenv("LIGHTNING_API_TOKEN", ""),
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )

        if recorded_start is None:
            recorded_start = datetime.now(timezone.utc)

        if drive_video_file_id:
            endpoint = f"{endpoint_root}/api/v1/ingestion/process"
            request_payload = {
                "source_drive_file_id": drive_video_file_id,
                "source_filename": h265_path.name,
                "camera_id": camera_id,
                "recorded_start": recorded_start.isoformat(),
                "output_basename": h265_path.name,
                "upload_outputs_to_drive": False,
                "metadata": metadata_runtime,
            }
            print(f"  POST {endpoint} using Drive file id {drive_video_file_id}")
            response = requests.post(
                endpoint,
                json=request_payload,
                headers=_tracking_headers(include_json=True),
                timeout=(connect_timeout, read_timeout),
            )
        else:
            endpoint = f"{endpoint_root}/api/v1/ingestion/upload"
            request_payload = {
                "source_filename": h265_path.name,
                "camera_id": camera_id or h265_path.stem,
                "recorded_start": recorded_start.isoformat(),
                "output_basename": h265_path.name,
                "metadata": json.dumps(metadata_runtime),
            }
            print(f"  POST {endpoint} using multipart upload")
            with local_source_copy.open("rb") as handle:
                response = requests.post(
                    endpoint,
                    data=request_payload,
                    files={"file": (h265_path.name, handle, "video/h265")},
                    headers=_tracking_headers(include_json=False),
                    timeout=(connect_timeout, read_timeout),
                )

        response.raise_for_status()
        result = response.json()

    remote_video_payload = result.get("video") or {}
    remote_people = result.get("people") or []
    remote_metadata_name = Path(str(result.get("metadata_path") or "")).name or f"{h265_path.stem}.json"
    metadata_path = metadata_dir / remote_metadata_name
    video_payload = dict(remote_video_payload)
    people = [dict(person) for person in remote_people if isinstance(person, dict)]

    video_payload["compressed_path"] = str(local_queue_video_copy)
    video_payload["metadata_path"] = str(metadata_path)
    video_payload["processing_backend"] = "tracking_service_remote"
    video_payload["source_filename"] = video_payload.get("source_filename") or h265_path.name
    if camera_id and not video_payload.get("camera_id"):
        video_payload["camera_id"] = camera_id

    for person in people:
        person["metadata_path"] = str(metadata_path)
        person.setdefault("processing_backend", "tracking_service_remote")
        if video_payload.get("video_id") and not person.get("video_id"):
            person["video_id"] = video_payload["video_id"]
        if video_payload.get("camera_id") and not person.get("camera_id"):
            person["camera_id"] = video_payload["camera_id"]

    _write_local_metadata_file(metadata_path, video_payload, people)

    print(f"  ✅ Metadata generated!")
    print(f"     Metadata file: {metadata_path}")
    print(f"     People detected: {len(people)}")
    print(f"     Video ID: {video_payload.get('video_id', 'N/A')}")

    if people:
        print(f"     First person: {people[0].get('human_key', 'unknown')}")
        print(f"     Track ID: {people[0].get('track_id', 'N/A')}")

    return metadata_path, video_payload, people


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
    from sqlalchemy import create_engine, func, select, text
    from sqlalchemy.engine import make_url
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.exc import OperationalError

    from app.models import Base, QueueVideoAsset

    load_runtime_env(override=False)

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
        queue_video_id = video_payload.get("video_id", f"queue_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
        source_filename = video_payload.get("source_filename")

        existing = None
        if drive_video_file_id and drive_video_file_id != "local_only":
            existing = db.execute(
                select(QueueVideoAsset).where(QueueVideoAsset.drive_video_file_id == drive_video_file_id)
            ).scalar_one_or_none()
        if existing is None and source_filename:
            existing = db.execute(
                select(QueueVideoAsset).where(QueueVideoAsset.source_filename == source_filename)
            ).scalar_one_or_none()

        current_queue_size = int(db.execute(select(func.count()).select_from(QueueVideoAsset)).scalar() or 0)
        next_queue_position = current_queue_size + 1

        if existing is None:
            queue_asset = QueueVideoAsset(
                video_id=queue_video_id,
                camera_id=video_payload.get("camera_id"),
                title=video_payload.get("title", f"Video {queue_video_id}"),
                source_filename=source_filename,
                source_mode="exchange_import",
                queue_position=next_queue_position,
                storage_backend="google_drive" if video_view_link else "local_bootstrap",
                available_link_video=video_view_link or f"/api/v1/queue/videos/{queue_video_id}/file",
                available_link_metadata=metadata_view_link or f"/api/v1/queue/videos/{queue_video_id}/metadata",
                drive_video_file_id=drive_video_file_id,
                drive_metadata_file_id=drive_metadata_file_id,
                local_video_path=str(video_payload.get("compressed_path", "")),
                local_metadata_path=str(video_payload.get("metadata_path", "")),
                raw_video_metadata=video_payload,
            )
            db.add(queue_asset)
        else:
            queue_asset = existing
            queue_asset.video_id = queue_video_id
            queue_asset.camera_id = video_payload.get("camera_id")
            queue_asset.title = video_payload.get("title", f"Video {queue_video_id}")
            queue_asset.source_filename = source_filename
            queue_asset.source_mode = "exchange_import"
            queue_asset.storage_backend = "google_drive" if video_view_link else "local_bootstrap"
            queue_asset.available_link_video = video_view_link or f"/api/v1/queue/videos/{queue_video_id}/file"
            queue_asset.available_link_metadata = metadata_view_link or f"/api/v1/queue/videos/{queue_video_id}/metadata"
            queue_asset.drive_video_file_id = drive_video_file_id
            queue_asset.drive_metadata_file_id = drive_metadata_file_id
            queue_asset.local_video_path = str(video_payload.get("compressed_path", ""))
            queue_asset.local_metadata_path = str(video_payload.get("metadata_path", ""))
            queue_asset.raw_video_metadata = video_payload

        db.commit()
        db.refresh(queue_asset)

        imported_people = 0
        updated_people = 0
        metadata_path = str(video_payload.get("metadata_path", "") or "")

        def _candidate_search_document(person: dict[str, Any]) -> str:
            parts: list[str] = []
            for key in ("search_text", "appearance_summary", "person_caption", "caption"):
                text_value = str(person.get(key) or "").strip()
                if text_value:
                    parts.append(text_value)
            attributes = person.get("semantic_attributes")
            if isinstance(attributes, list):
                cleaned = [str(item).strip() for item in attributes if str(item).strip()]
                if cleaned:
                    parts.append("attributes: " + ", ".join(cleaned))
            timeline = person.get("timeline")
            if isinstance(timeline, list):
                actions = [
                    str(item.get("action_summary") or "").strip()
                    for item in timeline
                    if isinstance(item, dict) and str(item.get("action_summary") or "").strip()
                ]
                if actions:
                    parts.append("timeline: " + " ".join(actions))
            return " ".join(parts).strip()

        for person in people:
            if not isinstance(person, dict):
                continue
            candidate_id = str(person.get("candidate_id") or "").strip()
            if not candidate_id:
                continue
            person_params = {
                "candidate_id": candidate_id,
                "camera_id": person.get("camera_id"),
                "video_id": person.get("video_id") or queue_video_id,
                "track_id": str(person.get("track_id")) if person.get("track_id") is not None else None,
                "human_key": person.get("human_key"),
                "frame_idx": int(person.get("frame_idx") or 0),
                "search_text": _candidate_search_document(person),
                "metadata_path": metadata_path,
                "raw_metadata": json.dumps(person),
            }
            existing_candidate = db.execute(
                text("SELECT id FROM person_candidates WHERE candidate_id = :candidate_id"),
                {"candidate_id": candidate_id},
            ).first()
            if existing_candidate:
                db.execute(
                    text(
                        """
                        UPDATE person_candidates
                        SET camera_id = :camera_id,
                            video_id = :video_id,
                            track_id = :track_id,
                            human_key = :human_key,
                            frame_idx = :frame_idx,
                            search_text = :search_text,
                            metadata_path = :metadata_path,
                            raw_metadata = CAST(:raw_metadata AS jsonb),
                            updated_at = NOW()
                        WHERE candidate_id = :candidate_id
                        """
                    ),
                    person_params,
                )
                updated_people += 1
            else:
                db.execute(
                    text(
                        """
                        INSERT INTO person_candidates
                            (candidate_id, camera_id, video_id, track_id, human_key, frame_idx, search_text, metadata_path, raw_metadata)
                        VALUES
                            (:candidate_id, :camera_id, :video_id, :track_id, :human_key, :frame_idx, :search_text, :metadata_path, CAST(:raw_metadata AS jsonb))
                        """
                    ),
                    person_params,
                )
                imported_people += 1

        db.commit()

        print(f"  ✅ Saved to PostgreSQL!")
        print(f"     QueueAsset ID: {queue_asset.id}")
        print(f"     Video ID: {queue_asset.video_id}")
        print(f"     Person candidates imported: {imported_people}")
        print(f"     Person candidates updated: {updated_people}")

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
    load_runtime_env(override=False)

    api_base_url = api_base_url or os.getenv("TRACKING_SERVICE_URL") or os.getenv("LIGHTNING_API_BASE_URL")
    local_tracking_url = os.getenv("TRACKING_SERVICE_LOCAL_URL", "http://127.0.0.1:8000").strip()
    prefer_local = os.getenv("TRACKING_SERVICE_PREFER_LOCAL", "true").strip().lower() not in {"0", "false", "no"}
    api_token = api_token or os.getenv("LIGHTNING_API_TOKEN")
    api_endpoint = os.getenv("LIGHTNING_API_ENDPOINT", "/api/v1/ingestion/process")

    if not api_base_url or not api_token:
        raise ValueError("TRACKING_SERVICE_URL or LIGHTNING_API_BASE_URL, and LIGHTNING_API_TOKEN are required")

    if prefer_local and local_tracking_url:
        try:
            parsed_host = local_tracking_url.split("://", 1)[-1].split("/", 1)[0]
            host, _, raw_port = parsed_host.partition(":")
            port = int(raw_port or 80)
            with socket.create_connection((host, port), timeout=1.5):
                api_base_url = local_tracking_url
                print(f"  Using local tracking-service endpoint: {api_base_url}")
        except Exception:
            pass

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
            drive_layout = resolve_drive_layout()
            video_folder_id = drive_layout["queue_h265_id"]
            metadata_folder_id = drive_layout["queue_metadata_id"]
            print(f"  Using Queue/.h265 folder ID: {video_folder_id}")
            print(f"  Using Queue/Metadata folder ID: {metadata_folder_id}")

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
            drive_video_file_id=drive_video.get("file_id") if upload_to_drive else None,
        )
        results["steps"]["metadata"] = {
            "metadata_path": str(metadata_path),
            "person_count": len(people),
            "video_id": video_payload.get("video_id"),
            "processing_backend": video_payload.get("processing_backend"),
        }

        # ── STEP 4: Upload metadata to Drive ────────────────────
        if upload_to_drive:
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
            if str(video_payload.get("processing_backend") or "").strip() == "tracking_service_remote":
                results["steps"]["lightning_ai"] = {
                    "status": "skipped",
                    "reason": "metadata already generated by remote tracking-service ingestion",
                    "job_id": video_payload.get("video_id"),
                }
                print("  ℹ️ Skipping legacy LightningAI step because metadata already came from remote tracking-service")
            else:
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
    parser.add_argument("--mp4-only", action="store_true", help="Batch mode: process only .mp4 files from --input-dir")
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
        explicit_video = explicit_video.resolve()
        if not explicit_video.exists():
            print(f"❌ File not found: {explicit_video}")
            sys.exit(1)
        videos = [explicit_video]
    else:
        args.input_dir = args.input_dir.resolve()
        if not args.input_dir.exists():
            print(f"❌ Directory not found: {args.input_dir}")
            sys.exit(1)
        if args.mp4_only:
            videos = [path.resolve() for path in sorted(args.input_dir.glob("*.mp4"))]
        else:
            h265_videos = sorted(
                [path.resolve() for ext in (".h265", ".hevc") for path in args.input_dir.glob(f"*{ext}")]
            )
            mp4_videos = [path.resolve() for path in sorted(args.input_dir.glob("*.mp4"))]
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
        if worker_count == 1:
            for video_path in videos:
                result = run_full_pipeline(
                    input_video_path=video_path,
                    camera_id=args.camera_id,
                    upload_to_drive=not args.no_drive,
                    call_lightning=not args.no_lightning,
                )
                all_results.append(result)
        else:
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
