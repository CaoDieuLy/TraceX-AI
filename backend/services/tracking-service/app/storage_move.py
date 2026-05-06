"""
move.py — Google Drive storage organizer.

API endpoint để move video từ Drive Temp/ sang Storage/.
Chạy trên LightningAI (A100).

Flow:
1. User bấm "Move" trên frontend
2. Frontend gọi POST /api/v1/storage/move
3. Endpoint này:
   - List all .mp4 in Drive Temp/
   - Parse filename pattern: cam_XX_yyyy-mm-dd_hh-mm.mp4
   - Create folder structure: Storage/camera_id/date/
   - Move each file
"""
from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Pattern & Constants ───────────────────────────────────────────────────────
CAMERA_VIDEO_PATTERN = re.compile(
    r"^(?P<camera_id>cam_\d{2,})_"
    r"(?P<recorded_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<hour>\d{2})-(?P<minute>\d{2})"
    r"(?:-(?P<second>\d{2}))?"
    r"(?P<suffix>\.mp4)$",
    re.IGNORECASE,
)

# Default Drive folder IDs (override via env)
TEMP_FOLDER_ID = "1Px379D5sjK95lMOUGZ4oUAgco7wBCk4I"
STORAGE_FOLDER_ID = "1G6L1d8l2YupSI0HIgB9NkBel04RqX48G"
FOLDER_MIME = "application/vnd.google-apps.folder"


def _get_token_path() -> Path:
    """Resolve OAuth token path - looks in multiple locations."""
    import os
    candidates = [
        Path(os.getenv("A20_ROOT", "")),
        Path("/workspace/TraceX-AI"),
        Path("/workspace"),
        Path(__file__).parent.parent.parent,
    ]
    for candidate in candidates:
        token_path = candidate / "secrets" / "oauth" / "oauth2_token.pickle"
        if token_path.exists():
            return token_path
    raise FileNotFoundError("oauth2_token.pickle not found in any candidate path")


def build_drive_service():
    """Build Google Drive service using OAuth token."""
    from googleapiclient.discovery import build

    token_path = _get_token_path()
    with open(token_path, "rb") as f:
        creds = pickle.load(f)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ── Drive Operations ──────────────────────────────────────────────────────────
def list_mp4s_in_temp(drive) -> list[dict]:
    """List all .mp4 files in Drive Temp/ folder."""
    temp_folder = _get_temp_folder_id()
    res = drive.files().list(
        q=f"'{temp_folder}' in parents and trashed=false",
        fields="files(id,name,mimeType,size)",
        pageSize=200,
    ).execute()
    return [f for f in res.get("files", []) if f["name"].lower().endswith(".mp4")]


def find_or_create_folder(drive, parent_id: str, name: str) -> str:
    """Find existing folder or create new one."""
    res = drive.files().list(
        q=f"'{parent_id}' in parents and trashed=false and mimeType='{FOLDER_MIME}' and name='{name}'",
        fields="files(id)",
        pageSize=5,
    ).execute()
    items = res.get("files", [])
    if items:
        return items[0]["id"]

    created = drive.files().create(
        body={"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]},
        fields="id",
        supportsAllDrives=True,
    ).execute()
    return created["id"]


def move_file_to_storage(drive, file_id: str, from_parent: str, to_parent: str) -> None:
    """Move file from one folder to another."""
    drive.files().update(
        fileId=file_id,
        addParents=to_parent,
        removeParents=from_parent,
        fields="id,parents",
        supportsAllDrives=True,
    ).execute()


def _get_temp_folder_id() -> str:
    """Get Temp folder ID from env or default."""
    import os
    return os.getenv("GOOGLE_DRIVE_TEMP_FOLDER_ID", TEMP_FOLDER_ID)


def _get_storage_folder_id() -> str:
    """Get Storage folder ID from env or default."""
    import os
    return os.getenv("GOOGLE_DRIVE_STORAGE_FOLDER_ID", STORAGE_FOLDER_ID)


# ── API Router ───────────────────────────────────────────────────────────────
router = APIRouter(prefix="/api/v1/storage", tags=["storage"])


class MoveResult(BaseModel):
    """Response for move operation."""
    moved: int
    skipped: int
    files: list[dict]


class MoveRequest(BaseModel):
    """Optional request body for move."""
    dry_run: bool = False


@router.post("/move", response_model=MoveResult)
async def move_videos(request: MoveRequest = None):
    """
    Move videos from Drive Temp/ to Storage/.

    Parses filename pattern to determine target folder:
    - Pattern: cam_XX_yyyy-mm-dd_hh-mm.mp4
    - Target: Storage/camera_id/date/filename.mp4

    Example:
        cam_01_2026-04-28_10-00.mp4 → Storage/cam_01/2026-04-28/cam_01_2026-04-28_10-00.mp4
    """
    if request is None:
        request = MoveRequest()

    try:
        drive = build_drive_service()
    except Exception as e:
        logger.error("Failed to build Drive service: %s", e)
        raise HTTPException(status_code=500, detail=f"Google Drive auth failed: {e}")

    temp_folder_id = _get_temp_folder_id()
    storage_folder_id = _get_storage_folder_id()

    try:
        files = list_mp4s_in_temp(drive)
        logger.info("Found %d .mp4 files in Temp/", len(files))
    except Exception as e:
        logger.error("Failed to list files in Temp/: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to list files: {e}")

    moved = 0
    skipped = 0
    results = []

    for f in files:
        name = f["name"]
        m = CAMERA_VIDEO_PATTERN.fullmatch(name)

        if not m:
            logger.debug("SKIP (bad name): %s", name)
            results.append({"name": name, "status": "skipped", "reason": "bad_filename_pattern"})
            skipped += 1
            continue

        camera_id = m.group("camera_id")
        recorded_date = m.group("recorded_date")
        target_path = f"{camera_id}/{recorded_date}/{name}"

        results.append({"name": name, "status": "queued", "target": target_path})

        if not request.dry_run:
            try:
                # Ensure folder structure exists
                cam_folder_id = find_or_create_folder(drive, storage_folder_id, camera_id)
                date_folder_id = find_or_create_folder(drive, cam_folder_id, recorded_date)
                # Move file
                move_file_to_storage(drive, f["id"], temp_folder_id, date_folder_id)
                results[-1]["status"] = "moved"
                moved += 1
                logger.info("Moved: %s → Storage/%s", name, target_path)
            except Exception as e:
                results[-1]["status"] = "error"
                results[-1]["error"] = str(e)
                logger.error("Failed to move %s: %s", name, e)
                skipped += 1
        else:
            results[-1]["status"] = "dry-run"
            moved += 1

    logger.info("Move complete: %d moved, %d skipped (dry_run=%s)", moved, skipped, request.dry_run)

    return MoveResult(moved=moved, skipped=skipped, files=results)


@router.get("/status")
async def get_storage_status():
    """Get current storage status from Google Drive."""
    try:
        drive = build_drive_service()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google Drive auth failed: {e}")

    temp_folder_id = _get_temp_folder_id()
    storage_folder_id = _get_storage_folder_id()

    try:
        # Count files in Temp
        temp_res = drive.files().list(
            q=f"'{temp_folder_id}' in parents and trashed=false",
            fields="files(id,name)",
            pageSize=1000,
        ).execute()
        temp_files = [f for f in temp_res.get("files", []) if f["name"].lower().endswith(".mp4")]

        # Count files in Storage
        storage_res = drive.files().list(
            q=f"'{storage_folder_id}' in parents and trashed=false",
            fields="files(id,name)",
            pageSize=1000,
        ).execute()
        storage_files = [f for f in storage_res.get("files", []) if f["name"].lower().endswith(".mp4")]

        return {
            "temp_pending": len(temp_files),
            "storage_processed": len(storage_files),
            "temp_folder_id": temp_folder_id,
            "storage_folder_id": storage_folder_id,
        }
    except Exception as e:
        logger.error("Failed to get storage status: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── CLI compatibility (for direct python move.py) ──────────────────────────────
def main():
    """CLI entrypoint for direct execution."""
    import argparse
    from app.main import app

    parser = argparse.ArgumentParser(description="Move videos from Drive Temp to Storage")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be moved without moving")
    args = parser.parse_args()

    # Run sync (blocking)
    import asyncio
    result = asyncio.run(move_videos(MoveRequest(dry_run=args.dry_run)))
    print(f"Moved: {result.moved}, Skipped: {result.skipped}")
    for f in result.files:
        print(f"  [{f['status']}] {f['name']}")


if __name__ == "__main__":
    main()
