"""
ingest_service.py — Drive Temp→Storage move + LOCAL GPU ingest into v3.3 schema.

Flow:
  1. Move videos from Drive Temp/ folder to Storage/ folder
  2. Scan Storage/ for all .mp4 files
  3. For each video NOT yet in `videos` table:
     a. Auto-register camera in `cameras` if new
     b. Insert video metadata → `videos`
     c. Stream bytes from Drive → GPU service (multipart upload, no disk cache)
     d. GPU service writes temp file, processes, deletes it
     e. Save people → `tracklets` + `tracklets_embeddings` + `tracklets_actions`
  4. Return summary

Tables written (v3.3 schema):
  cameras, videos, tracklets, tracklets_embeddings, tracklets_actions
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Drive folder IDs (from move.py / .env)
# ---------------------------------------------------------------------------
DRIVE_TEMP_FOLDER_ID = "1Px379D5sjK95lMOUGZ4oUAgco7wBCk4I"
DRIVE_STORAGE_FOLDER_ID = "1G6L1d8l2YupSI0HIgB9NkBel04RqX48G"
DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"

CAMERA_VIDEO_PATTERN = re.compile(
    r"^(?P<camera_id>cam_\d{2,})_"
    r"(?P<recorded_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<hour>\d{2})-(?P<minute>\d{2})"
    r"(?:-(?P<second>\d{2}))?"
    r"(?P<suffix>\.mp4)$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Used as metadata path in DB records only — no actual files written here
_STORAGE_PATH_METADATA = Path("/workspace/storage/storage")

# ---------------------------------------------------------------------------
# Drive helpers
# ---------------------------------------------------------------------------

DRIVE_TEMP_FOLDER_ID = "1Px379D5sjK95lMOUGZ4oUAgco7wBCk4I"
DRIVE_STORAGE_FOLDER_ID = "1G6L1d8l2YupSI0HIgB9NkBel04RqX48G"
DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"

CAMERA_VIDEO_PATTERN = re.compile(
    r"^(?P<camera_id>cam_\d{2,})_"
    r"(?P<recorded_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<hour>\d{2})-(?P<minute>\d{2})"
    r"(?:-(?P<second>\d{2}))?"
    r"(?P<suffix>\.mp4)$",
    re.IGNORECASE,
)


def _ensure_shared_secret() -> None:
    import sys
    if "/workspace/secrets" not in sys.path:
        sys.path.insert(0, "/workspace/secrets")


def _build_drive_service():
    _ensure_shared_secret()
    import importlib
    mod = importlib.import_module("shared_secret_runtime")
    try:
        return mod.build_google_drive_sa_service()
    except FileNotFoundError:
        return mod.build_google_drive_service()


def _list_drive_mp4s(service, folder_id: str) -> list[dict]:
    results: list[dict] = []
    _walk_folder(service, folder_id, results)
    return results


def _walk_folder(service, folder_id: str, out: list[dict]):
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, webViewLink, webContentLink)",
            pageSize=200,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageToken=page_token,
        ).execute()
        for f in resp.get("files", []):
            if f["mimeType"] == DRIVE_FOLDER_MIME:
                _walk_folder(service, f["id"], out)
            elif f["name"].lower().endswith(".mp4"):
                out.append(f)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def _list_drive_folders(service, parent_id: str) -> list[dict]:
    resp = service.files().list(
        q=f"'{parent_id}' in parents and trashed=false and mimeType='{DRIVE_FOLDER_MIME}'",
        fields="files(id, name)",
        pageSize=200,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    return resp.get("files", [])


def _find_or_create_folder(service, parent_id: str, name: str) -> str:
    escaped = name.replace("'", "\\'")
    resp = service.files().list(
        q=f"'{parent_id}' in parents and trashed=false and mimeType='{DRIVE_FOLDER_MIME}' and name='{escaped}'",
        fields="files(id)",
        pageSize=5,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    items = resp.get("files", [])
    if items:
        return items[0]["id"]
    created = service.files().create(
        body={"name": name, "mimeType": DRIVE_FOLDER_MIME, "parents": [parent_id]},
        fields="id",
        supportsAllDrives=True,
    ).execute()
    return created["id"]


def _move_file(service, file_id: str, from_parent: str, to_parent: str) -> None:
    service.files().update(
        fileId=file_id,
        addParents=to_parent,
        removeParents=from_parent,
        fields="id,parents",
        supportsAllDrives=True,
    ).execute()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_camera_id(filename: str) -> str | None:
    m = CAMERA_VIDEO_PATTERN.match(filename)
    return m.group("camera_id").lower() if m else None


def _parse_recorded_at(filename: str) -> datetime | None:
    m = CAMERA_VIDEO_PATTERN.search(filename)
    if not m:
        return None
    try:
        return datetime(
            int(m.group("recorded_date")[:4]),
            int(m.group("recorded_date")[5:7]),
            int(m.group("recorded_date")[8:10]),
            int(m.group("hour")),
            int(m.group("minute")),
            int(m.group("second") or "0"),
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# GPU processing (local via metadata-service)
# ---------------------------------------------------------------------------

_METADATA_SERVICE_URL = "http://localhost:8002"


def _process_video_stream(drive_file_id: str, filename: str, video_id: str, camera_id: str) -> dict:
    """
    Stream video bytes from Google Drive directly to the GPU service via multipart upload.
    The GPU service writes a temp file, processes, then deletes it.
    No video is cached to disk permanently.
    """
    import sys
    sys.path.insert(0, "/workspace/secrets")
    from shared_secret_runtime import build_google_drive_sa_service

    logger.info("[gpu] Streaming %s from Drive (id=%s) to metadata-service", filename, drive_file_id)

    drive_service = build_google_drive_sa_service()

    from googleapiclient.http import MediaIoBaseDownload, HttpRequest
    from io import BytesIO

    request: HttpRequest = drive_service.files().get_media(fileId=drive_file_id)
    buffer = BytesIO()
    downloader = MediaIoBaseDownload(buffer, request, chunksize=1024 * 1024 * 50)

    done = False
    while not done:
        _, done = downloader.next_chunk()
        logger.info("[gpu] Downloaded chunk for %s", filename)

    buffer.seek(0)
    video_bytes = buffer.getvalue()
    logger.info("[gpu] Downloaded %s: %d bytes, sending to GPU service", filename, len(video_bytes))

    # Send to GPU service via multipart upload
    import mimetypes
    content_type = mimetypes.guess_type(filename)[0] or "video/mp4"

    with httpx.Client(timeout=600) as client:
        files = {
            "video": (filename, video_bytes, content_type),
        }
        data = {
            "video_id": video_id,
            "camera_id": camera_id,
            "source_filename": filename,
            "sample_interval": 15,
            "bev_max_dist": 1.5,
        }
        response = client.post(
            f"{_METADATA_SERVICE_URL}/api/v1/video/process/stream",
            files=files,
            data=data,
            timeout=600,
        )

    response.raise_for_status()
    result = response.json()

    tracklets = result.get("tracklets", [])
    logger.info("[gpu] Got %d tracklets for %s", len(tracklets), video_id)
    return {
        "tracklets": tracklets,
        "person_count": len(tracklets),
        "total_detections": result.get("total_detections", 0),
        "processing_time_s": result.get("processing_time_s", 0),
    }


# ---------------------------------------------------------------------------
# DB helpers (v3.3)
# ---------------------------------------------------------------------------

def _ensure_camera(session: Session, camera_id: str) -> None:
    from shared.models import Camera
    existing = session.scalar(select(Camera).where(Camera.camera_id == camera_id))
    if existing is None:
        session.add(Camera(
            camera_id=camera_id,
            name=f"Camera {camera_id.replace('cam_', 'Camera ').title()}",
            location=None,
            fps=30.0,
            resolution_width=1920,
            resolution_height=1080,
            is_active=True,
        ))
        logger.info("[db] Auto-registered camera: %s", camera_id)


def _video_exists_in_db(session: Session, video_id: str) -> bool:
    from shared.models import Video
    return session.scalar(select(Video).where(Video.video_id == video_id)) is not None


def _upsert_video(
    session: Session,
    video_id: str,
    camera_id: str,
    title: str,
    source_filename: str,
    drive_file_id: str,
    recorded_at: datetime | None,
) -> None:
    from shared.models import Video
    existing = session.scalar(select(Video).where(Video.video_id == video_id))
    if existing is not None:
        return

    view_link = f"https://drive.google.com/file/d/{drive_file_id}/view"
    session.add(Video(
        video_id=video_id,
        camera_id=camera_id,
        title=title,
        storage_path=str(_STORAGE_PATH_METADATA / source_filename),
        storage_backend="google_drive",
        source_filename=source_filename,
        processed=False,
    ))
    session.flush()


def _save_tracklets_from_gpu_result(
    session: Session,
    gpu_result: dict,
    video_id: str,
    camera_id: str,
) -> int:
    """
    Parse GPU result and save to v3.3 tables.
    Returns number of tracklets saved.
    """
    from shared.models import Tracklet, TrackletEmbedding, TrackletAction

    tracklets = gpu_result.get("tracklets", [])
    saved = 0

    for t in tracklets:
        tracklet_id = str(t.get("tracklet_id") or "")
        if not tracklet_id:
            continue

        existing = session.scalar(select(Tracklet).where(Tracklet.tracklet_id == tracklet_id))
        if existing is not None:
            continue

        tracklet = Tracklet(
            tracklet_id=tracklet_id,
            video_id=video_id,
            camera_id=camera_id,
            track_id=str(t.get("track_id") or "0"),
            start_time=float(t.get("start_time") or 0.0),
            end_time=float(t.get("end_time") or 0.0),
            quality_score=float(t.get("quality_score") or 0.0),
            occlusion_score=float(t.get("occlusion_score") or 0.0),
            gender=str(t.get("gender") or "unknown"),
            age_range=str(t.get("age_range") or "unknown"),
            top_color=str(t.get("top_color") or "unknown"),
            bottom_color=str(t.get("bottom_color") or "unknown"),
            shoes_color=str(t.get("shoes_color") or "unknown"),
            appearance_summary=str(t.get("appearance_summary") or ""),
            bev_x=float(t.get("bev_x") or 0.0),
            bev_y=float(t.get("bev_y") or 0.0),
            crop_url=str(t.get("crop_url") or ""),
            representative_bbox=t.get("representative_bbox") or [],
            contributing_cameras=t.get("contributing_cameras") or [],
            contributing_video_ids=t.get("contributing_video_ids") or [],
        )
        session.add(tracklet)

        # Embedding (EVA-02 1024-dim)
        embedding_vec = t.get("embedding_vector") or []
        if embedding_vec and len(embedding_vec) > 0:
            session.add(TrackletEmbedding(
                tracklet_id=tracklet_id,
                embedding_vector=embedding_vec,
                model_version="eva02_l14",
            ))

        # Action (VideoMAE V2)
        action_label = str(t.get("action") or t.get("action_label") or "")
        if action_label and action_label != "unknown":
            session.add(TrackletAction(
                tracklet_id=tracklet_id,
                action_label=action_label,
                kinetics_label=str(t.get("kinetics_label") or ""),
                confidence=float(t.get("action_confidence") or 0.0),
            ))

        saved += 1

    return saved


# ---------------------------------------------------------------------------
# Main ingest function
# ---------------------------------------------------------------------------

def ingest_move_and_process(
    session: Session,
    dry_run: bool = False,
    max_videos: int | None = None,
) -> dict:
    """
    Full pipeline: move Temp→Storage, download, process with LOCAL GPU, save to v3.3 DB.
    """
    from shared.models import Video
    from sqlalchemy import func

    total_existing = int(session.scalar(select(func.count()).select_from(Video)) or 0)
    logger.info("[ingest] Starting. Existing videos in DB: %d", total_existing)

    if dry_run:
        logger.info("[ingest] DRY RUN")

    # --- Step 1: Move Temp → Storage (best-effort, skip if SA has read-only permissions) ---
    drive = _build_drive_service()
    temp_files = _list_drive_mp4s(drive, DRIVE_TEMP_FOLDER_ID)
    logger.info("[drive] Found %d .mp4 files in Temp/", len(temp_files))

    moved_count = 0
    for f in temp_files:
        filename = f["name"]
        if not _parse_camera_id(filename):
            logger.info("[drive] SKIP (bad name): %s", filename)
            continue

        m = CAMERA_VIDEO_PATTERN.match(filename)
        if not m:
            logger.info("[drive] SKIP (no match): %s", filename)
            continue

        recorded_date = m.group("recorded_date")

        if not dry_run:
            try:
                cam_folder_id = _find_or_create_folder(drive, DRIVE_STORAGE_FOLDER_ID, m.group("camera_id"))
                date_folder_id = _find_or_create_folder(drive, cam_folder_id, recorded_date)
                _move_file(drive, f["id"], DRIVE_TEMP_FOLDER_ID, date_folder_id)
                logger.info("[drive] Moved: %s → Storage/%s/%s/", filename, m.group("camera_id"), recorded_date)
            except Exception as move_err:
                logger.warning("[drive] Move failed for %s (SA may be read-only): %s — files already in Storage, continuing.", filename, move_err)
                break  # stop trying moves; process from Storage directly

        moved_count += 1

    logger.info("[drive] Moved %d files", moved_count)

    # --- Step 2: Scan Storage/ ---
    storage_files = _list_drive_mp4s(drive, DRIVE_STORAGE_FOLDER_ID)
    logger.info("[drive] Found %d .mp4 files in Storage/", len(storage_files))

    pending: list[dict] = []
    for f in storage_files:
        video_id = f["name"]
        if not _video_exists_in_db(session, video_id):
            pending.append(f)
        else:
            logger.debug("[ingest] Skip already ingested: %s", video_id)

    if max_videos:
        pending = pending[:max_videos]

    skipped_count = len(storage_files) - len(pending)
    logger.info("[ingest] Pending: %d videos (skipped %d already in DB)", len(pending), skipped_count)

    if not pending:
        return {
            "status": "completed",
            "moved_count": moved_count,
            "ingested_count": 0,
            "skipped_count": skipped_count,
            "total_videos_in_db": total_existing,
            "errors": [],
            "message": f"Moved {moved_count} videos, 0 pending (all already in DB)",
        }

    # --- Step 3: Process each video ---
    ingested_count = 0
    tracklet_count = 0
    errors: list[str] = []
    processed_videos = 0

    for i, f in enumerate(pending, 1):
        filename = f["name"]
        file_id = f["id"]
        camera_id = _parse_camera_id(filename) or "unknown"
        video_id = filename

        logger.info("[%d/%d] Processing: %s (camera=%s)", i, len(pending), filename, camera_id)

        if dry_run:
            logger.info("[dry-run] Would ingest: %s", filename)
            continue

        try:
            # Ensure camera and upsert video record FIRST (commit separately)
            _ensure_camera(session, camera_id)
            recorded_at = _parse_recorded_at(filename)
            _upsert_video(
                session,
                video_id=video_id,
                camera_id=camera_id,
                title=filename,
                source_filename=filename,
                drive_file_id=file_id,
                recorded_at=recorded_at,
            )
            session.commit()  # Commit video record first
            logger.info("[%d/%d] Video record saved: %s", i, len(pending), filename)

            # Stream from Drive → GPU service directly (no disk cache)
            gpu_result = _process_video_stream(file_id, filename, video_id, camera_id)

            # Save to v3.3 tables
            saved = _save_tracklets_from_gpu_result(session, gpu_result, video_id, camera_id)
            tracklet_count += saved

            # Mark video as processed (even if no tracklets detected)
            video = session.scalar(select(Video).where(Video.video_id == video_id))
            if video:
                video.processed = True

            session.commit()
            processed_videos += 1
            ingested_count += saved
            logger.info("[%d/%d] ✓ %s: %d tracklets saved", i, len(pending), filename, saved)

        except Exception as exc:
            logger.error("[%d/%d] ✗ FAILED %s: %s", i, len(pending), filename, exc)
            errors.append(f"{filename}: {exc}")
            try:
                session.rollback()
            except Exception:
                pass

    final_total = int(session.scalar(select(func.count()).select_from(Video)) or 0)

    return {
        "status": "completed" if not errors else "completed_with_errors",
        "moved_count": moved_count,
        "ingested_count": ingested_count,
        "skipped_count": skipped_count,
        "total_videos_in_db": final_total,
        "total_tracklets_saved": tracklet_count,
        "processed_videos": processed_videos,
        "errors": errors[:10],
        "message": (
            f"Moved {moved_count} videos, "
            f"processed {processed_videos} videos with {ingested_count} tracklets, "
            f"skipped {skipped_count} already in DB"
            + (f", {len(errors)} errors" if errors else "")
        ),
    }
