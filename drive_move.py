"""
drive_move.py — Google Drive equivalent of move.py.

Reads .mp4 files from Drive Temp/ folder, parses filename pattern
cam_xx_yyyy-mm-dd_hh-mm.mp4, creates cam_xx/date/ folders under
Storage/, and moves (addParents/removeParents) each file.

Usage:
    python drive_move.py [--dry-run]
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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

TOKEN_PATH = Path(r"D:\workspace\project\secrets\oauth\oauth2_token.pickle")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    drive = build_drive()
    files = list_mp4s(drive, TEMP_FOLDER_ID)
    print(f"Found {len(files)} .mp4 files in Temp/")

    moved = 0
    skipped = 0
    for f in files:
        name = f["name"]
        m = CAMERA_VIDEO_PATTERN.fullmatch(name)
        if not m:
            print(f"  SKIP (bad name): {name}")
            skipped += 1
            continue

        camera_id = m.group("camera_id")
        recorded_date = m.group("recorded_date")
        target_path = f"{camera_id}/{recorded_date}/{name}"

        print(f"  {'[dry-run] ' if args.dry_run else ''}move → Storage/{target_path}")

        if not args.dry_run:
            cam_folder_id = find_or_create_folder(drive, STORAGE_FOLDER_ID, camera_id)
            date_folder_id = find_or_create_folder(drive, cam_folder_id, recorded_date)
            move_file(drive, f["id"], TEMP_FOLDER_ID, date_folder_id)
        moved += 1

    print(f"\nResult: {moved} moved, {skipped} skipped. dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
