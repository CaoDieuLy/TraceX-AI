import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def download_drive_video(
    drive_file_id: str,
    source_filename: str,
    cache_root: Path,
) -> Optional[Path]:
    """
    Download video from Google Drive to local cache.
    Returns local path to downloaded video.
    """
    suffix = Path(source_filename).suffix or ".mp4"
    cache_dir = cache_root / "drive-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_path = cache_dir / f"{drive_file_id}{suffix}"

    if target_path.exists() and target_path.stat().st_size > 0:
        logger.info("Using cached video: %s", target_path)
        return target_path

    try:
        _ensure_shared_secret()
        from shared_secret_runtime import build_google_drive_oauth_service
        from googleapiclient.http import MediaIoBaseDownload

        drive_service = build_google_drive_oauth_service()
        request = drive_service.files().get_media(
            fileId=drive_file_id,
            supportsAllDrives=True
        )
        with target_path.open("wb") as handle:
            downloader = MediaIoBaseDownload(handle, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        if target_path.exists():
            logger.info("Downloaded video from Drive: %s", target_path)
            return target_path
    except Exception as e:
        logger.error("Failed to download video from Drive: %s", e)

    return None


def download_from_url(url: str, filename: str, cache_root: Path) -> Optional[Path]:
    """
    Download video from a URL (e.g., Google Drive public URL) to local cache.
    Used by ingest_local.py flow.
    """
    suffix = Path(filename).suffix or ".mp4"
    import hashlib
    url_hash = hashlib.sha1(url.encode()).hexdigest()[:16]
    cache_dir = cache_root / "url-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_path = cache_dir / f"{url_hash}{suffix}"

    if target_path.exists() and target_path.stat().st_size > 0:
        logger.info("Using cached video from URL: %s", target_path)
        return target_path

    try:
        with httpx.Client(timeout=300) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                with target_path.open("wb") as f:
                    downloaded = 0
                    for chunk in response.iter_bytes(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

        if target_path.exists() and target_path.stat().st_size > 0:
            logger.info("Downloaded video from URL: %s (%s bytes)", target_path, target_path.stat().st_size)
            return target_path
    except Exception as e:
        logger.error("Failed to download video from URL: %s", e)

    return None


def _ensure_shared_secret() -> None:
    import sys
    env_root = os.getenv("A20_ROOT", "").strip()
    candidates = [Path(env_root)] if env_root else []
    candidates.extend([
        Path("/workspace/a20-root"),
        Path("/workspace"),
        Path.cwd(),
    ])
    for candidate in candidates:
        secret_path = candidate / "shared_secret_runtime.py"
        if secret_path.exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return
