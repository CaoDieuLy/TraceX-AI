"""
move.py — Google Drive video mover + pipeline trigger.

Reads .mp4 files from Drive Temp/ folder, parses filename pattern
cam_xx_yyyy-mm-dd_hh-mm.mp4, creates cam_xx/date/ folders under
Storage/, and moves (addParents/removeParents) each file.

After successful move, triggers metadata-service to queue videos for processing.

Usage:
    python move.py [--dry-run]
    python move.py --trigger-metadata [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CAMERA_VIDEO_PATTERN = re.compile(
    r"^(?P<camera_id>cam_\d{2,})_"
    r"(?P<recorded_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<hour>\d{2})-(?P<minute>\d{2})"
    r"(?:-(?P<second>\d{2}))?"
    r"(?P<suffix>\.mp4)$",
    re.IGNORECASE,
)

TEMP_FOLDER_ID    = "1Px379D5sjK95lMOUGZ4oUAgco7wBCk4I"
STORAGE_FOLDER_ID = "1G6L1d8l2YupSI0HIgB9NkBel04RqX48G"
FOLDER_MIME = "application/vnd.google-apps.folder"

TOKEN_PATH = Path(__file__).parent / "secrets" / "oauth" / "oauth2_token.pickle"

# Metadata service configuration
METADATA_SERVICE_URL = os.getenv("METADATA_SERVICE_URL", "http://localhost:8002")


def build_drive():
    from googleapiclient.discovery import build
    with open(TOKEN_PATH, "rb") as f:
        creds = pickle.load(f)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_mp4s(drive, folder_id: str) -> list[dict]:
    res = drive.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id,name,mimeType,size)",
        pageSize=200,
    ).execute()
    return [f for f in res.get("files", []) if f["name"].lower().endswith(".mp4")]


def find_or_create_folder(drive, parent_id: str, name: str) -> str:
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


def move_file(drive, file_id: str, from_parent: str, to_parent: str) -> None:
    drive.files().update(
        fileId=file_id,
        addParents=to_parent,
        removeParents=from_parent,
        fields="id,parents",
        supportsAllDrives=True,
    ).execute()


def trigger_metadata_service(moved_videos: list[dict[str, Any]]) -> bool:
    """
    Trigger metadata-service to queue moved videos for processing.
    
    Args:
        moved_videos: List of dicts with video info (camera_id, date, name, file_id)
    
    Returns:
        True if trigger was successful, False otherwise
    """
    if not moved_videos:
        logger.info("No videos to trigger metadata-service for")
        return True
    
    import httpx
    
    url = f"{METADATA_SERVICE_URL.rstrip('/')}/api/v1/queue/videos/trigger"
    payload = {
        "source": "google_drive_move",
        "videos": moved_videos,
    }
    
    try:
        logger.info(f"Triggering metadata-service at {url} with {len(moved_videos)} videos")
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload)
        
        if response.is_success:
            logger.info(f"Metadata-service trigger successful: {response.json()}")
            return True
        else:
            logger.warning(f"Metadata-service trigger failed: {response.status_code} - {response.text}")
            return False
    except httpx.TransportError as e:
        logger.warning(f"Could not reach metadata-service: {e}")
        logger.info("Videos will be picked up on next queue sync")
        return False
    except Exception as e:
        logger.error(f"Error triggering metadata-service: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move videos from Google Drive Temp/ to Storage/, optionally trigger metadata-service"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview moves without executing")
    parser.add_argument(
        "--trigger-metadata", 
        action="store_true", 
        default=True,
        help="Trigger metadata-service after move (default: True)"
    )
    parser.add_argument(
        "--no-trigger",
        action="store_true",
        help="Skip triggering metadata-service"
    )
    args = parser.parse_args()

    drive = build_drive()
    files = list_mp4s(drive, TEMP_FOLDER_ID)
    logger.info(f"Found {len(files)} .mp4 files in Temp/")

    moved_videos: list[dict[str, Any]] = []
    
    for f in files:
        name = f["name"]
        m = CAMERA_VIDEO_PATTERN.fullmatch(name)
        if not m:
            logger.info(f"  SKIP (bad name): {name}")
            continue

        camera_id = m.group("camera_id")
        recorded_date = m.group("recorded_date")
        target_path = f"{camera_id}/{recorded_date}/{name}"

        logger.info(f"  {'[dry-run] ' if args.dry_run else ''}move → Storage/{target_path}")

        if not args.dry_run:
            cam_folder_id = find_or_create_folder(drive, STORAGE_FOLDER_ID, camera_id)
            date_folder_id = find_or_create_folder(drive, cam_folder_id, recorded_date)
            move_file(drive, f["id"], TEMP_FOLDER_ID, date_folder_id)
            
            # Track moved video for metadata-service trigger
            moved_videos.append({
                "camera_id": camera_id,
                "recorded_date": recorded_date,
                "filename": name,
                "drive_file_id": f["id"],
            })
    
    # Trigger metadata-service if requested and files were moved
    if not args.dry_run and moved_videos and not args.no_trigger:
        trigger_metadata_service(moved_videos)
    
    logger.info(f"Result: {len(moved_videos)} moved, {len(files) - len(moved_videos)} skipped. dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
