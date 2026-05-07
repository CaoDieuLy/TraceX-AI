from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING
from urllib.parse import urlparse

import cv2
import httpx
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..config import A20_ROOT, PROJECT_ROOT, settings

if TYPE_CHECKING:
    from shared.models import User

logger = logging.getLogger(__name__)

STORAGE_INGEST_SOURCE_MODE = "storage_ingest"
DRIVE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"

_STORAGE_VIDEO_PATTERN = re.compile(
    r"^(?P<camera_id>cam_\d{2,})_"
    r"(?P<recorded_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<hour>\d{2})-(?P<minute>\d{2})"
    r"(?:-(?P<second>\d{2}))?"
    r"(?P<suffix>\.mp4)$",
    re.IGNORECASE,
)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = str(os.getenv(name, "") or "").strip()
    if not raw_value:
        return default
    try:
        return max(int(raw_value), minimum)
    except ValueError:
        return default


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._") or "video"


def _coerce_string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text:
            result.append(text)
    return result


def _coerce_mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower()) if len(token) >= 2}


def _semantic_overlap(query_text: str, candidate: dict) -> float:
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return 0.0
    candidate_tokens = _tokenize(_candidate_search_document(candidate))
    if not candidate_tokens:
        return 0.0
    return float(len(query_tokens & candidate_tokens) / max(len(query_tokens), 1))


def _candidate_search_document(person: dict) -> str:
    parts: list[str] = []
    for key in ("search_text", "appearance_summary"):
        text = str(person.get(key) or "").strip()
        if text:
            parts.append(text)

    attributes = _coerce_string_list(person.get("semantic_attributes"))
    if attributes:
        parts.append("attributes: " + ", ".join(attributes))

    timeline = person.get("timeline")
    if isinstance(timeline, list):
        actions = [str(item.get("action_summary") or "").strip() for item in timeline if isinstance(item, dict)]
        actions = [item for item in actions if item]
        if actions:
            parts.append("timeline: " + " ".join(actions))

    world_position = person.get("world_position") or person.get("top_point_projection")
    if isinstance(world_position, dict) and world_position:
        axis_tokens = []
        for axis in ("x", "y", "z"):
            if axis in world_position:
                axis_tokens.append(f"{axis}={world_position[axis]}")
        if axis_tokens:
            parts.append("world: " + ", ".join(axis_tokens))

    return " ".join(parts).strip()


def _candidate_bbox(raw_metadata: dict[str, Any]) -> list[int]:
    for key in ("bbox", "representative_bbox"):
        bbox = raw_metadata.get(key)
        if isinstance(bbox, list) and len(bbox) >= 4:
            cleaned: list[int] = []
            for value in bbox[:4]:
                try:
                    cleaned.append(int(float(value)))
                except (TypeError, ValueError):
                    cleaned.append(0)
            return cleaned
    return []


def _candidate_raw_metadata_subset(raw_metadata: object) -> dict[str, Any]:
    payload = raw_metadata if isinstance(raw_metadata, dict) else {}
    reduced: dict[str, Any] = {}

    for key in ("attribute_summary", "appearance_summary", "score", "tracklet_feature_pipeline"):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            reduced[key] = value

    action_payload = payload.get("action_semantic_embedding")
    if isinstance(action_payload, dict) and action_payload:
        reduced_action_payload = dict(action_payload)
        reduced_action_payload["segments"] = _trim_tracking_segments(action_payload.get("segments"))
        reduced["action_semantic_embedding"] = reduced_action_payload

    for key in ("attribute_embedding_vector", "appearance_embedding_vector", "embedding_vector"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            reduced[key] = value

    for key in ("bbox", "representative_bbox"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            reduced[key] = value

    frame_idx = payload.get("frame_idx")
    if isinstance(frame_idx, (int, float)):
        reduced["frame_idx"] = int(frame_idx)

    for key in ("tracklet_frame_count_full", "tracklet_frame_start_idx", "tracklet_frame_end_idx"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            reduced[key] = int(value)

    tracklet_quality = _coerce_mapping(payload.get("tracklet_quality"))
    if tracklet_quality:
        reduced["tracklet_quality"] = {
            "accepted": bool(tracklet_quality.get("accepted", False)),
            "average_confidence": tracklet_quality.get("average_confidence"),
            "average_laplacian": tracklet_quality.get("average_laplacian"),
            "frame_count": tracklet_quality.get("frame_count"),
            "frame_density": tracklet_quality.get("frame_density"),
            "duration_seconds": tracklet_quality.get("duration_seconds"),
        }

    semantic_attributes = _coerce_string_list(payload.get("semantic_attributes"))
    if semantic_attributes:
        reduced["semantic_attributes"] = semantic_attributes

    visibility_scores = _coerce_mapping(payload.get("visibility_scores"))
    if visibility_scores:
        reduced["visibility_scores"] = visibility_scores

    world_position = payload.get("world_position") or payload.get("top_point_projection")
    if isinstance(world_position, dict) and world_position:
        reduced["world_position"] = world_position

    matched_segments = _trim_tracking_segments(payload.get("matched_segments"))
    if matched_segments:
        reduced["matched_segments"] = matched_segments

    timeline = _trim_tracking_segments(payload.get("timeline"))
    if timeline:
        reduced["timeline"] = timeline

    tracklet_frames = _trim_tracklet_frames(payload.get("tracklet_frames"))
    if tracklet_frames:
        reduced["tracklet_frames"] = tracklet_frames

    recorded_start = payload.get("recorded_start")
    if recorded_start:
        reduced["recorded_start"] = str(recorded_start)

    clip_drive_urls = payload.get("clip_drive_urls")
    if isinstance(clip_drive_urls, list) and clip_drive_urls:
        reduced["clip_drive_urls"] = clip_drive_urls

    return reduced


def _trim_tracking_segments(segments: object, *, limit: int = 5) -> list[dict[str, Any]]:
    if not isinstance(segments, list):
        return []
    trimmed: list[dict[str, Any]] = []
    for segment in segments[:limit]:
        if not isinstance(segment, dict):
            continue
        trimmed.append(
            {
                "start_second": segment.get("start_second"),
                "end_second": segment.get("end_second"),
                "action_summary": str(segment.get("action_summary") or "").strip(),
            }
        )
    return trimmed


def _trim_tracklet_frames(frames: object, *, limit: int = 5) -> list[dict[str, Any]]:
    if not isinstance(frames, list):
        return []
    trimmed: list[dict[str, Any]] = []
    for frame in frames[:limit]:
        if not isinstance(frame, dict):
            continue
        trimmed.append(
            {
                "frame_idx": frame.get("frame_idx"),
                "timestamp_second": frame.get("timestamp_second"),
                "bbox": frame.get("bbox"),
                "confidence": frame.get("confidence"),
            }
        )
    return trimmed


# ============================================================================
# Storage Video Items
# ============================================================================

@dataclass(frozen=True)
class StorageVideoIdentity:
    camera_id: str
    recorded_date: date
    recorded_at: datetime
    source_filename: str

    @classmethod
    def parse(cls, source_path: Path) -> "StorageVideoIdentity | None":
        match = _STORAGE_VIDEO_PATTERN.fullmatch(source_path.name)
        if not match:
            return None
        try:
            recorded_date = date.fromisoformat(match.group("recorded_date"))
            recorded_at = datetime(
                recorded_date.year,
                recorded_date.month,
                recorded_date.day,
                int(match.group("hour")),
                int(match.group("minute")),
                int(match.group("second") or "0"),
            )
        except ValueError:
            return None
        return cls(
            camera_id=match.group("camera_id").lower(),
            recorded_date=recorded_date,
            recorded_at=recorded_at,
            source_filename=source_path.name,
        )


@dataclass(frozen=True)
class StorageVideoItem:
    source_path: Path | None
    source_drive_file_id: str | None
    relative_path: str
    source_filename: str
    camera_id: str
    recorded_at: datetime
    size_bytes: int
    modified_ns: int
    fingerprint: str

    @property
    def output_basename(self) -> str:
        return self.source_filename

    def marker_payload(self) -> dict:
        payload = asdict(self)
        payload["source_path"] = str(self.source_path) if self.source_path is not None else None
        payload["recorded_at"] = self.recorded_at.isoformat()
        return payload


class StorageIngestRegistry:
    def __init__(self, marker_dir: Path) -> None:
        self.marker_dir = marker_dir

    def is_processed(self, item: StorageVideoItem) -> bool:
        marker_path = self._marker_path(item)
        if not marker_path.exists():
            return False
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return (
            str(marker.get("relative_path") or "") == item.relative_path
            and int(marker.get("size_bytes") or -1) == item.size_bytes
            and int(marker.get("modified_ns") or -1) == item.modified_ns
        )

    def mark_processed(self, item: StorageVideoItem, result: dict) -> Path:
        self.marker_dir.mkdir(parents=True, exist_ok=True)
        marker_path = self._marker_path(item)
        payload = item.marker_payload()
        payload["processed_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        payload["result_video_id"] = str((result.get("video") or {}).get("video_id") or "")
        payload["person_count"] = int(result.get("person_count") or 0)
        marker_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return marker_path

    def _marker_path(self, item: StorageVideoItem) -> Path:
        return self.marker_dir / f"{item.fingerprint}.json"


# ============================================================================
# Queue Sync Service
# ============================================================================

class QueueSyncService:
    def __init__(self) -> None:
        self.local_root = Path(settings.queue_local_root)
        self.local_queue_dir = self.local_root / "local" / settings.google_drive_queue_folder_name
        self.local_queue_video_dir = self.local_queue_dir / settings.queue_video_folder_name
        self.local_queue_metadata_dir = self.local_queue_dir / settings.google_drive_metadata_folder_name
        self.storage_ingest_root = Path(settings.storage_ingest_root)
        self.storage_processed_dir = self.local_queue_dir / settings.storage_ingest_processed_dir_name
        self._drive_service = None
        self._drive_layout: dict[str, str] | None = None

    @staticmethod
    def _parallel_jobs(count: int, default: int) -> int:
        return max(1, min(count, int(default)))

    def ensure_local_layout(self) -> None:
        self.local_queue_video_dir.mkdir(parents=True, exist_ok=True)
        self.local_queue_metadata_dir.mkdir(parents=True, exist_ok=True)
        self.storage_processed_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _video_media_type(path_or_name: str | Path) -> str:
        suffix = Path(str(path_or_name)).suffix.lower()
        if suffix == ".mp4":
            return "video/mp4"
        return "application/octet-stream"

    def _build_drive_service(self):
        if not settings.google_drive_enabled:
            raise RuntimeError("Google Drive sync is disabled.")
        if self._drive_service is not None:
            return self._drive_service
        _ensure_shared_secret()
        from shared_secret_runtime import build_google_drive_oauth_service
        self._drive_service = build_google_drive_oauth_service()
        return self._drive_service

    def process_storage_queue(self, session: Session) -> dict:
        self.ensure_local_layout()
        if not settings.storage_ingest_enabled:
            return {
                "processed_videos": 0,
                "imported_source_files": [],
                "evicted_video_ids": [],
            }

        registry = StorageIngestRegistry(self.storage_processed_dir)
        if str(settings.storage_ingest_source_backend or "filesystem").strip().lower() == "google_drive":
            _ensure_shared_secret()
            from shared_secret_runtime import build_google_drive_oauth_service
            drive_service = build_google_drive_oauth_service()
            source_root_id = self._resolve_drive_source_storage_folder_id()
            storage_items = self._scan_drive_pending(drive_service, source_root_id, registry)
        else:
            storage_items = self._scan_storage_pending(registry)

        processed_videos = 0
        imported_source_files: list[str] = []
        evicted_video_ids: list[str] = []

        for item in storage_items:
            try:
                result = self._process_storage_item(session, item)
                evicted_video_ids.extend(result.get("evicted_video_ids", []))
                imported_source_files.append(item.relative_path)
                processed_videos += 1
                registry.mark_processed(item, result)
            except Exception as exc:
                logger.exception("Failed to process storage item: %s", item.source_filename)

        return {
            "processed_videos": processed_videos,
            "imported_source_files": imported_source_files,
            "evicted_video_ids": evicted_video_ids,
        }

    def _scan_storage_pending(self, registry: StorageIngestRegistry) -> list[StorageVideoItem]:
        pending: list[StorageVideoItem] = []
        if not self.storage_ingest_root.exists():
            return pending

        candidates = sorted(self.storage_ingest_root.glob("cam_*/*/*.mp4"))
        for source_path in candidates:
            if not source_path.is_file() or source_path.name.startswith("."):
                continue
            identity = StorageVideoIdentity.parse(source_path)
            if identity is None:
                continue
            try:
                relative_path = source_path.relative_to(self.storage_ingest_root).as_posix()
                stat = source_path.stat()
            except OSError:
                continue

            item = StorageVideoItem(
                source_path=source_path,
                source_drive_file_id=None,
                relative_path=relative_path,
                source_filename=source_path.name,
                camera_id=identity.camera_id,
                recorded_at=identity.recorded_at,
                size_bytes=int(stat.st_size),
                modified_ns=int(stat.st_mtime_ns),
                fingerprint=_fingerprint_storage(relative_path),
            )
            if not self._is_too_new(item) and not registry.is_processed(item):
                pending.append(item)
        return pending

    def _scan_drive_pending(self, drive_service, source_root_id: str, registry: StorageIngestRegistry) -> list[StorageVideoItem]:
        pending: list[StorageVideoItem] = []
        items = self._list_drive_storage_items(drive_service, source_root_id)
        for item in items:
            if not self._is_too_new(item) and not registry.is_processed(item):
                pending.append(item)
        return pending

    def _list_drive_storage_items(self, drive_service, source_root_id: str) -> list[StorageVideoItem]:
        import os
        min_date = (os.environ.get("STORAGE_INGEST_MIN_DATE") or "").strip()
        items: list[StorageVideoItem] = []

        camera_folders = self._list_drive_folders(drive_service, source_root_id)
        for camera_folder in camera_folders:
            if not camera_folder.name.lower().startswith("cam_"):
                continue
            date_folders = self._list_drive_folders(drive_service, camera_folder.id)
            for date_folder in date_folders:
                date_name = date_folder.name
                if min_date and date_name < min_date:
                    continue
                for file_row in self._list_drive_mp4_files(drive_service, date_folder.id):
                    identity = StorageVideoIdentity.parse(Path(str(file_row.get("name") or "")))
                    if identity is None:
                        continue
                    if identity.camera_id != camera_folder.name.lower():
                        continue
                    if date_name != identity.recorded_date.isoformat():
                        continue
                    relative_path = f"{camera_folder.name}/{date_name}/{identity.source_filename}"
                    modified_at = self._parse_drive_time(str(file_row.get("modifiedTime") or ""))
                    items.append(StorageVideoItem(
                        source_path=None,
                        source_drive_file_id=str(file_row["id"]),
                        relative_path=relative_path,
                        source_filename=identity.source_filename,
                        camera_id=identity.camera_id,
                        recorded_at=identity.recorded_at,
                        size_bytes=int(file_row.get("size") or 0),
                        modified_ns=int(modified_at.timestamp() * 1_000_000_000),
                        fingerprint=_fingerprint_drive(relative_path, str(file_row["id"])),
                    ))

        seen: dict[str, StorageVideoItem] = {}
        for item in items:
            existing = seen.get(item.relative_path)
            if existing is None or item.modified_ns > existing.modified_ns:
                seen[item.relative_path] = item
        return sorted(seen.values(), key=lambda item: item.relative_path)

    def _list_drive_folders(self, drive_service, parent_id: str) -> list[Any]:
        query = f"'{parent_id}' in parents and trashed = false and mimeType = '{DRIVE_FOLDER_MIME_TYPE}'"
        response = drive_service.files().list(
            q=query, spaces="drive", fields="files(id, name)",
            pageSize=1000, supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        rows = response.get("files") or []
        return [type("F", (), {"id": str(r["id"]), "name": str(r["name"])})() for r in rows]

    def _list_drive_mp4_files(self, drive_service, parent_id: str) -> list[dict]:
        query = f"'{parent_id}' in parents and trashed = false"
        response = drive_service.files().list(
            q=query, spaces="drive", fields="files(id, name, size, modifiedTime, mimeType)",
            pageSize=1000, supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        return [r for r in (response.get("files") or []) if str(r.get("name") or "").lower().endswith(".mp4")]

    def _parse_drive_time(self, raw: str) -> datetime:
        value = raw.strip()
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).astimezone(timezone.utc)

    @staticmethod
    def _is_too_new(item: StorageVideoItem) -> bool:
        if item.modified_ns <= 0:
            return False
        modified_seconds = item.modified_ns / 1_000_000_000
        return (time.time() - modified_seconds) < settings.storage_ingest_min_file_age_seconds

    def _resolve_drive_source_storage_folder_id(self) -> str:
        configured_id = str(settings.google_drive_source_storage_folder_id or "").strip()
        if configured_id:
            return configured_id
        vinuni_id = str(settings.google_drive_vinuni_folder_id or "").strip()
        root_id = str(settings.google_drive_root_folder_id or "").strip()
        if not vinuni_id and not root_id:
            raise RuntimeError("Set GOOGLE_DRIVE_SOURCE_STORAGE_FOLDER_ID or VINUNI/ROOT folder IDs")
        if not vinuni_id:
            vinuni_id = self._ensure_drive_folder(root_id, settings.google_drive_vinuni_folder_name)
        folder = self._find_drive_child(vinuni_id, settings.google_drive_source_storage_folder_name)
        if folder is None:
            raise RuntimeError(f"Storage folder not found: {settings.google_drive_source_storage_folder_name}")
        return folder.id

    def _ensure_drive_folder(self, parent_id: str, name: str) -> str:
        existing = self._find_drive_child(parent_id, name)
        if existing:
            return existing.id
        service = self._build_drive_service()
        created = service.files().create(
            body={"name": name, "mimeType": DRIVE_FOLDER_MIME_TYPE, "parents": [parent_id]},
            fields="id", supportsAllDrives=True,
        ).execute()
        return str(created["id"])

    def _find_drive_child(self, parent_id: str, name: str) -> Any | None:
        escaped = name.replace("'", "\\'")
        query = f"'{parent_id}' in parents and trashed = false and mimeType = '{DRIVE_FOLDER_MIME_TYPE}' and name = '{escaped}'"
        service = self._build_drive_service()
        response = service.files().list(
            q=query, spaces="drive", fields="files(id, name)",
            pageSize=10, supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        rows = response.get("files") or []
        if not rows:
            return None
        r = rows[0]
        return type("F", (), {"id": str(r["id"]), "name": str(r["name"])})()

    def _process_storage_item(self, session: Session, item: StorageVideoItem) -> dict:
        from shared.models import PersonCandidate, QueueVideoAsset, User

        video_id = str(item.recorded_at.strftime("%Y%m%d_%H%M%S")) + "_" + item.camera_id
        queue_position = self._get_next_queue_position(session) + 1
        available_link_video = f"/api/v1/queue/videos/{video_id}/file"
        available_link_metadata = f"/api/v1/queue/videos/{video_id}/metadata"

        existing = session.scalar(select(QueueVideoAsset).where(QueueVideoAsset.video_id == video_id))
        if existing:
            return {"evicted_video_ids": []}

        row = QueueVideoAsset(
            video_id=video_id,
            camera_id=item.camera_id,
            title=item.source_filename,
            queue_position=queue_position,
            available_link_video=available_link_video,
            available_link_metadata=available_link_metadata,
            storage_backend="local_queue_storage",
            source_filename=item.source_filename,
            source_mode=STORAGE_INGEST_SOURCE_MODE,
            local_video_path=str(item.source_path) if item.source_path else None,
            raw_video_metadata={"source": "storage_ingest"},
        )
        session.add(row)
        session.commit()
        return {"evicted_video_ids": []}

    def _get_next_queue_position(self, session: Session) -> int:
        from shared.models import QueueVideoAsset
        result = session.query(QueueVideoAsset).order_by(QueueVideoAsset.queue_position.desc()).first()
        return result.queue_position if result else 0


def _ensure_shared_secret() -> None:
    import sys
    if A20_ROOT and str(A20_ROOT) not in sys.path:
        sys.path.insert(0, str(A20_ROOT))


def _fingerprint_storage(relative_path: str) -> str:
    import hashlib
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()


def _fingerprint_drive(relative_path: str, file_id: str) -> str:
    import hashlib
    return hashlib.sha1(f"{relative_path}|{file_id}".encode("utf-8")).hexdigest()


def queue_video_to_payload(video) -> dict:
    return {
        "video_id": video.video_id,
        "camera_id": video.camera_id,
        "title": video.title,
        "queue_position": video.queue_position,
        "storage_backend": video.storage_backend,
        "available_link_video": video.available_link_video,
        "available_link_metadata": video.available_link_metadata,
        "source_filename": video.source_filename,
        "source_mode": video.source_mode,
        "created_at": video.created_at,
        "updated_at": video.updated_at,
        "local_video_path": video.local_video_path,
        "local_metadata_path": video.local_metadata_path,
        "raw_video_metadata": video.raw_video_metadata or {},
    }


def list_queue_videos(session: Session) -> list[dict]:
    from shared.models import QueueVideoAsset
    statement = select(QueueVideoAsset).order_by(QueueVideoAsset.queue_position.asc(), QueueVideoAsset.id.asc())
    return [queue_video_to_payload(video) for video in session.scalars(statement).all()]


def load_queue_video_metadata(session: Session, video_id: str) -> dict[str, Any]:
    from shared.models import QueueVideoAsset, PersonCandidate
    row = session.scalar(select(QueueVideoAsset).where(QueueVideoAsset.video_id == video_id))
    if row is None:
        raise FileNotFoundError(f"Queue video not found: {video_id}")
    metadata_path = Path(str(row.local_metadata_path or "")).expanduser()
    if metadata_path.exists() and metadata_path.is_file():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    people = (
        session.scalars(
            select(PersonCandidate)
            .where(PersonCandidate.video_id == video_id)
            .order_by(PersonCandidate.frame_idx.asc(), PersonCandidate.id.asc())
        ).all()
    )
    return {
        "video": row.raw_video_metadata or {},
        "people": [candidate.raw_metadata or {} for candidate in people],
        "source": "database_fallback",
    }


def load_queue_video_file_path(session: Session, video_id: str) -> Path:
    from shared.models import QueueVideoAsset
    row = session.scalar(select(QueueVideoAsset).where(QueueVideoAsset.video_id == video_id))
    if row is None:
        raise FileNotFoundError(f"Queue video not found: {video_id}")
    path = Path(str(row.local_video_path or "")).expanduser()
    if path.exists() and path.is_file():
        return path
    raise FileNotFoundError(f"Queue video file not found: {video_id}")


def sync_local_queue_state(session: Session, *, only_if_empty: bool = False) -> dict[str, int]:
    from shared.models import PersonCandidate, QueueVideoAsset, User
    from sqlalchemy import func

    existing_candidates = int(session.scalar(select(func.count()).select_from(PersonCandidate)) or 0)
    existing_queue_videos = int(session.scalar(select(func.count()).select_from(QueueVideoAsset)) or 0)
    if only_if_empty and (existing_candidates > 0 or existing_queue_videos > 0):
        return {
            "imported_count": 0,
            "updated_count": 0,
            "queue_videos_count": existing_queue_videos,
            "file_count": 0,
        }

    local_root = Path(settings.queue_local_root)
    queue_root = local_root / "local" / settings.google_drive_queue_folder_name
    metadata_root = queue_root / settings.google_drive_metadata_folder_name
    video_root = queue_root / settings.queue_video_folder_name
    metadata_root.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)

    metadata_files = sorted(metadata_root.glob("*.json"))
    imported_count = 0
    updated_count = 0
    queue_videos_count = 0

    for queue_position, metadata_path in enumerate(metadata_files, start=1):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        video_payload = payload.get("video") or {}
        if not isinstance(video_payload, dict):
            video_payload = {}
        people_payload = payload.get("people") or []
        if not isinstance(people_payload, list):
            people_payload = []

        video_id = str(video_payload.get("video_id") or metadata_path.stem).strip()
        if not video_id:
            continue
        local_video_path = Path(str(video_payload.get("compressed_path") or "")).expanduser()
        # Use absolute path directly - no PROJECT_ROOT fallback for VPS deployment
        if not local_video_path.is_absolute():
            local_video_path = Path("/workspace/storage/videos") / local_video_path.name
        if not local_video_path.exists():
            fallback_video_path = video_root / f"{metadata_path.stem}.mp4"
            if fallback_video_path.exists():
                local_video_path = fallback_video_path

        title = str(
            video_payload.get("source_path")
            or video_payload.get("camera_id")
            or metadata_path.stem
        ).strip()
        source_filename = Path(title).name if title else f"{metadata_path.stem}.mp4"
        available_link_video = f"/api/v1/queue/videos/{video_id}/file"
        available_link_metadata = f"/api/v1/queue/videos/{video_id}/metadata"

        _upsert_queue_video_asset(
            session,
            video_id=video_id,
            camera_id=str(video_payload.get("camera_id") or "").strip() or None,
            title=source_filename,
            queue_position=queue_position,
            available_link_video=available_link_video,
            available_link_metadata=available_link_metadata,
            storage_backend="local_queue_storage",
            source_filename=source_filename,
            source_mode=str(video_payload.get("source_mode") or "").strip() or None,
            drive_video_file_id=None,
            drive_metadata_file_id=None,
            local_video_path=str(local_video_path) if local_video_path.exists() else None,
            local_metadata_path=str(metadata_path),
            raw_video_metadata=video_payload,
        )
        counts = _upsert_person_candidates(session, people_payload, str(metadata_path))
        imported_count += int(counts.get("imported_count") or 0)
        updated_count += int(counts.get("updated_count") or 0)
        queue_videos_count += 1

    session.commit()
    return {
        "imported_count": imported_count,
        "updated_count": updated_count,
        "queue_videos_count": queue_videos_count,
        "file_count": len(metadata_files),
    }


def _upsert_queue_video_asset(
    session: Session,
    *,
    video_id: str,
    camera_id: str | None,
    title: str,
    queue_position: int,
    available_link_video: str,
    available_link_metadata: str | None,
    storage_backend: str,
    source_filename: str | None,
    source_mode: str | None,
    drive_video_file_id: str | None,
    drive_metadata_file_id: str | None,
    local_video_path: str | None,
    local_metadata_path: str | None,
    raw_video_metadata: dict,
):
    from shared.models import QueueVideoAsset
    row = session.scalar(select(QueueVideoAsset).where(QueueVideoAsset.video_id == video_id))
    values = {
        "video_id": video_id,
        "camera_id": camera_id,
        "title": title,
        "queue_position": queue_position,
        "available_link_video": available_link_video,
        "available_link_metadata": available_link_metadata,
        "storage_backend": storage_backend,
        "source_filename": source_filename,
        "source_mode": source_mode,
        "drive_video_file_id": drive_video_file_id,
        "drive_metadata_file_id": drive_metadata_file_id,
        "local_video_path": local_video_path,
        "local_metadata_path": local_metadata_path,
        "raw_video_metadata": raw_video_metadata,
    }
    if row is None:
        row = QueueVideoAsset(**values)
        session.add(row)
    else:
        for key, value in values.items():
            setattr(row, key, value)
    session.flush()
    return row


def _upsert_person_candidates(session: Session, people: list[dict], metadata_path: str | None = None) -> dict[str, int]:
    from shared.models import PersonCandidate
    imported_count = 0
    updated_count = 0
    normalized_people: list[dict[str, Any]] = []

    for person in people:
        if not isinstance(person, dict):
            continue
        candidate_id = str(person.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        normalized_people.append({
            "candidate_id": candidate_id,
            "camera_id": person.get("camera_id"),
            "video_id": person.get("video_id"),
            "track_id": str(person.get("track_id")) if person.get("track_id") is not None else None,
            "human_key": person.get("human_key"),
            "frame_idx": int(person.get("frame_idx") or 0),
            "search_text": _candidate_search_document(person),
            "metadata_path": metadata_path,
            "raw_metadata": _candidate_raw_metadata_subset(person),
        })

    if not normalized_people:
        return {"imported_count": 0, "updated_count": 0}

    existing_rows = session.scalars(
        select(PersonCandidate).where(
            PersonCandidate.candidate_id.in_([item["candidate_id"] for item in normalized_people])
        )
    ).all()
    existing_by_candidate_id = {row.candidate_id: row for row in existing_rows}
    insert_rows: list[dict[str, Any]] = []

    for values in normalized_people:
        existing = existing_by_candidate_id.get(str(values["candidate_id"]))
        if existing:
            for key, value in values.items():
                setattr(existing, key, value)
            updated_count += 1
        else:
            insert_rows.append(values)
            imported_count += 1

    session.flush()
    for start in range(0, len(insert_rows), 50):
        chunk = insert_rows[start:start + 50]
        if chunk:
            session.bulk_insert_mappings(PersonCandidate, chunk)
            session.flush()

    return {"imported_count": imported_count, "updated_count": updated_count}


def delete_queue_video_asset(session: Session, video_id: str) -> None:
        from shared.models import PersonCandidate, QueueVideoAsset

        session.execute(delete(PersonCandidate).where(PersonCandidate.video_id == video_id))
        session.execute(delete(QueueVideoAsset).where(QueueVideoAsset.video_id == video_id))
        session.flush()
