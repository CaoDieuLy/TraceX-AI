from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from googleapiclient.http import MediaIoBaseDownload
from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session, joinedload

from .auth import hash_password, verify_password
from .config import PROJECT_ROOT, settings
from .models import PersonCandidate, QueueVideoAsset, User, VideoAsset, VideoQuery


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


def _score_candidate(query_text: str, candidate: dict) -> float:
    score = _semantic_overlap(query_text, candidate)
    attributes = _coerce_string_list(candidate.get("semantic_attributes"))
    if attributes and any(token in " ".join(attributes).lower() for token in _tokenize(query_text)):
        score += 0.1
    timeline = candidate.get("timeline")
    if isinstance(timeline, list) and timeline:
        score += min(len(timeline), 5) * 0.01
    return round(score, 6)


def _tracking_service_headers() -> dict[str, str]:
    token = settings.lightning_api_token.strip()
    if not token:
        return {"Content-Type": "application/json"}
    prefix = settings.lightning_api_auth_prefix
    if prefix and not prefix.endswith(" "):
        prefix = f"{prefix} "
    return {
        "Content-Type": "application/json",
        settings.lightning_api_auth_header: f"{prefix}{token}".strip(),
    }


def _tracking_service_url(path: str) -> str:
    return settings.tracking_service_url.rstrip("/") + "/" + path.lstrip("/")


def _post_tracking_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    with httpx.Client(timeout=float(settings.tracking_request_timeout_seconds)) as client:
        response = client.post(
            _tracking_service_url(path),
            json=payload,
            headers=_tracking_service_headers(),
        )
        response.raise_for_status()
        return response.json()


def _tracking_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._") or "tracking"


def _candidate_search_document(person: dict) -> str:
    parts: list[str] = []
    for key in (
        "search_text",
        "appearance_summary",
        "person_caption",
        "caption",
    ):
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


def candidate_to_payload(candidate: PersonCandidate, queue_video: QueueVideoAsset | None = None) -> dict:
    raw_metadata = candidate.raw_metadata or {}
    storage_path = (
        queue_video.local_video_path
        if queue_video and (queue_video.local_video_path or "").strip()
        else queue_video.available_link_video
        if queue_video
        else None
    )
    return {
        "candidate_id": candidate.candidate_id,
        "camera_id": candidate.camera_id,
        "video_id": candidate.video_id,
        "track_id": candidate.track_id,
        "human_key": candidate.human_key,
        "frame_idx": candidate.frame_idx,
        "search_text": candidate.search_text,
        "metadata_path": candidate.metadata_path,
        "appearance_summary": raw_metadata.get("appearance_summary") or raw_metadata.get("person_caption"),
        "semantic_attributes": _coerce_string_list(raw_metadata.get("semantic_attributes")),
        "visibility_scores": _coerce_mapping(raw_metadata.get("visibility_scores")),
        "world_position": raw_metadata.get("world_position") or raw_metadata.get("top_point_projection"),
        "reid_profile": raw_metadata.get("reid_profile"),
        "pipeline_profile": raw_metadata.get("pipeline_profile"),
        "score": raw_metadata.get("score"),
        "matched_segments": raw_metadata.get("matched_segments") or [],
        "available_link_video": queue_video.available_link_video if queue_video else None,
        "available_link_metadata": queue_video.available_link_metadata if queue_video else None,
        "drive_video_file_id": queue_video.drive_video_file_id if queue_video else None,
        "drive_metadata_file_id": queue_video.drive_metadata_file_id if queue_video else None,
        "local_video_path": queue_video.local_video_path if queue_video else None,
        "local_metadata_path": queue_video.local_metadata_path if queue_video else None,
        "storage_path": storage_path,
        "video_title": queue_video.title if queue_video else None,
        "raw_metadata": raw_metadata,
    }


def video_to_payload(video: VideoAsset) -> dict:
    return {
        "video_id": video.video_id,
        "title": video.title,
        "description": video.description,
        "storage_path": video.storage_path,
        "storage_backend": video.storage_backend,
        "source_filename": video.source_filename,
        "content_type": video.content_type,
        "created_at": video.created_at,
    }


def query_to_payload(query: VideoQuery) -> dict:
    return {
        "query_id": query.query_id,
        "video_id": query.video.video_id,
        "video_title": query.video.title,
        "storage_path": query.video.storage_path,
        "query_text": query.query_text,
        "status": query.status,
        "ai_job_id": query.ai_job_id,
        "ai_response": query.ai_response,
        "created_at": query.created_at,
        "updated_at": query.updated_at,
    }


def queue_video_to_payload(video: QueueVideoAsset) -> dict:
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
        "raw_video_metadata": video.raw_video_metadata,
    }


def get_user_by_email(session: Session, email: str) -> User | None:
    return session.scalar(select(User).where(User.email == email.lower().strip()))


def get_user_by_identifier(session: Session, identifier: str) -> User | None:
    normalized = str(identifier or "").strip().lower()
    if not normalized:
        return None
    if "@" in normalized:
        return get_user_by_email(session, normalized)
    statement = select(User).where(
        or_(
            func.lower(User.full_name) == normalized,
            func.lower(func.split_part(User.email, "@", 1)) == normalized,
        )
    )
    return session.scalar(statement)


def create_user(session: Session, email: str, full_name: str, password: str) -> User:
    existing = get_user_by_email(session, email)
    if existing is not None:
        raise ValueError("Email already registered")

    user = User(
        email=email.lower().strip(),
        full_name=full_name.strip(),
        hashed_password=hash_password(password),
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def authenticate_user(session: Session, identifier: str, password: str) -> User | None:
    user = get_user_by_identifier(session, identifier)
    if user is None or not verify_password(password, user.hashed_password):
        return None
    return user


def create_video_asset(
    session: Session,
    user: User,
    title: str,
    description: str | None,
    storage_path: str,
    storage_backend: str,
    source_filename: str | None,
    content_type: str | None,
) -> VideoAsset:
    video = VideoAsset(
        user_id=user.id,
        title=title.strip(),
        description=(description or "").strip() or None,
        storage_path=storage_path,
        storage_backend=storage_backend,
        source_filename=source_filename,
        content_type=content_type,
    )
    session.add(video)
    session.commit()
    session.refresh(video)
    return video


def save_uploaded_video_bytes(filename: str, content: bytes) -> str:
    storage_root = Path(settings.video_storage_root)
    storage_root.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename or "").suffix
    stem = _slugify(Path(filename or "video").stem)
    target_path = storage_root / f"{uuid.uuid4()}-{stem}{suffix}"
    target_path.write_bytes(content)
    return str(target_path)


def list_videos(session: Session, user: User) -> list[dict]:
    statement = select(VideoAsset).where(VideoAsset.user_id == user.id).order_by(VideoAsset.created_at.desc(), VideoAsset.id.desc())
    return [video_to_payload(video) for video in session.scalars(statement).all()]


def get_video_by_public_id(session: Session, user: User, video_id: str) -> VideoAsset | None:
    return session.scalar(select(VideoAsset).where(VideoAsset.video_id == video_id, VideoAsset.user_id == user.id))


def create_video_query(session: Session, user: User, video: VideoAsset, query_text: str) -> VideoQuery:
    query = VideoQuery(
        user_id=user.id,
        video_id=video.id,
        query_text=query_text.strip(),
        status="queued",
    )
    session.add(query)
    session.commit()
    session.refresh(query)
    return session.scalar(
        select(VideoQuery).options(joinedload(VideoQuery.video)).where(VideoQuery.id == query.id)
    )


def list_video_queries(session: Session, user: User) -> list[dict]:
    statement = (
        select(VideoQuery)
        .options(joinedload(VideoQuery.video))
        .where(VideoQuery.user_id == user.id)
        .order_by(VideoQuery.updated_at.desc(), VideoQuery.id.desc())
    )
    return [query_to_payload(query) for query in session.scalars(statement).all()]


def get_video_query(session: Session, user: User, query_id: str) -> VideoQuery | None:
    return session.scalar(
        select(VideoQuery)
        .options(joinedload(VideoQuery.video))
        .where(VideoQuery.query_id == query_id, VideoQuery.user_id == user.id)
    )


def update_video_query(
    session: Session,
    query: VideoQuery,
    status: str | None = None,
    ai_job_id: str | None = None,
    ai_response: dict | None = None,
) -> VideoQuery:
    if status is not None:
        query.status = status
    if ai_job_id is not None:
        query.ai_job_id = ai_job_id
    if ai_response is not None:
        query.ai_response = ai_response
    session.add(query)
    session.commit()
    session.refresh(query)
    return session.scalar(select(VideoQuery).options(joinedload(VideoQuery.video)).where(VideoQuery.id == query.id))


def _queue_video_map(session: Session, video_ids: list[str]) -> dict[str, QueueVideoAsset]:
    cleaned = [video_id for video_id in video_ids if video_id]
    if not cleaned:
        return {}
    rows = session.scalars(select(QueueVideoAsset).where(QueueVideoAsset.video_id.in_(cleaned))).all()
    return {row.video_id: row for row in rows}


def search_candidates(session: Session, query: str | None = None, limit: int = 20) -> list[dict]:
    statement = select(PersonCandidate).order_by(PersonCandidate.updated_at.desc(), PersonCandidate.id.desc())
    cleaned_query = (query or "").strip()
    if cleaned_query:
        pattern = f"%{cleaned_query}%"
        statement = statement.where(
            or_(
                PersonCandidate.search_text.ilike(pattern),
                PersonCandidate.camera_id.ilike(pattern),
                PersonCandidate.video_id.ilike(pattern),
                PersonCandidate.human_key.ilike(pattern),
                PersonCandidate.track_id.ilike(pattern),
            )
        )
    rows = session.scalars(statement.limit(max(1, min(limit, 100)))).all()
    queue_map = _queue_video_map(session, [str(row.video_id or "") for row in rows])
    return [candidate_to_payload(row, queue_map.get(str(row.video_id or ""))) for row in rows]


def get_candidate(session: Session, candidate_id: str) -> dict | None:
    row = session.scalar(select(PersonCandidate).where(PersonCandidate.candidate_id == candidate_id))
    if row is None:
        return None
    queue_map = _queue_video_map(session, [str(row.video_id or "")])
    return candidate_to_payload(row, queue_map.get(str(row.video_id or "")))


def rank_candidates(session: Session, query_text: str, limit: int = 5) -> list[dict]:
    cleaned_query = query_text.strip()
    if not cleaned_query:
        return []
    statement = select(PersonCandidate).order_by(PersonCandidate.updated_at.desc(), PersonCandidate.id.desc())
    rows = session.scalars(statement).all()
    queue_map = _queue_video_map(session, [str(row.video_id or "") for row in rows])
    candidates = [candidate_to_payload(row, queue_map.get(str(row.video_id or ""))) for row in rows]
    response = _post_tracking_json(
        "/api/v1/candidates/search",
        {
            "query_text": cleaned_query,
            "candidates": candidates,
            "limit": max(1, min(limit, 50)),
        },
    )
    items = response.get("items")
    return items if isinstance(items, list) else []


def get_overview(session: Session) -> dict:
    total_users = session.scalar(select(func.count()).select_from(User)) or 0
    total_managed_videos = session.scalar(select(func.count()).select_from(VideoAsset)) or 0
    total_queries = session.scalar(select(func.count()).select_from(VideoQuery)) or 0
    total_candidates = session.scalar(select(func.count()).select_from(PersonCandidate)) or 0
    total_cameras = session.scalar(select(func.count(func.distinct(PersonCandidate.camera_id))).select_from(PersonCandidate)) or 0
    total_candidate_videos = session.scalar(select(func.count(func.distinct(PersonCandidate.video_id))).select_from(PersonCandidate)) or 0
    total_queue_videos = session.scalar(select(func.count()).select_from(QueueVideoAsset)) or 0
    return {
        "metrics": {
            "total_users": int(total_users),
            "total_managed_videos": int(total_managed_videos),
            "total_queries": int(total_queries),
            "total_candidates": int(total_candidates),
            "total_cameras": int(total_cameras),
            "total_candidate_videos": int(total_candidate_videos),
            "total_queue_videos": int(total_queue_videos),
        }
    }


def import_legacy_metadata(session: Session) -> dict:
    """
    Import person candidates từ Google Drive Metadata folder (Queue) vào PostgreSQL.
    Trong production, metadata được sinh bởi ingestion pipeline và đã có trong DB qua queue worker.
    Endpoint này chỉ dùng để migrate/restore từ Drive nếu cần.
    """
    if settings.google_drive_enabled:
        # Đọc từ Google Drive Metadata folder
        from .queue_runtime import QueueSyncService

        qs = QueueSyncService()
        drive_service = qs._build_drive_service()
        layout = qs.ensure_drive_layout()
        metadata_folder_id = layout["queue_metadata_id"]

        # Query tất cả JSON files trong Metadata folder
        query = f"'{metadata_folder_id}' in parents and trashed = false and mimeType='application/json'"
        files = drive_service.files().list(q=query, fields="files(id, name)").execute().get("files", [])

        imported_count = 0
        updated_count = 0
        file_count = len(files)

        for file_meta in files:
            file_id = file_meta["id"]
            filename = file_meta["name"]
            # Download file content
            import io
            request = drive_service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            content = fh.getvalue().decode("utf-8")
            payload = json.loads(content)
            # Import persons from this metadata file
            for person in payload.get("people") or []:
                if not isinstance(person, dict):
                    continue
                candidate_id = str(person.get("candidate_id") or "").strip()
                if not candidate_id:
                    continue

                existing = session.scalar(select(PersonCandidate).where(PersonCandidate.candidate_id == candidate_id))
                values = {
                    "candidate_id": candidate_id,
                    "camera_id": person.get("camera_id"),
                    "video_id": person.get("video_id"),
                    "track_id": str(person.get("track_id")) if person.get("track_id") is not None else None,
                    "human_key": person.get("human_key"),
                    "frame_idx": int(person.get("frame_idx") or 0),
                    "search_text": _candidate_search_document(person),
                    "metadata_path": f"drive://{file_id}/{filename}",
                    "raw_metadata": person,
                }

                if existing:
                    for key, value in values.items():
                        setattr(existing, key, value)
                    updated_count += 1
                else:
                    session.add(PersonCandidate(**values))
                    imported_count += 1

        session.commit()
        return {
            "imported_count": imported_count,
            "updated_count": updated_count,
            "file_count": file_count,
        }
    else:
        # Fallback: đọc từ local LEGACY_METADATA_DIR (development only)
        metadata_root = Path(settings.legacy_metadata_dir)
        metadata_root.mkdir(parents=True, exist_ok=True)

        imported_count = 0
        updated_count = 0
        file_count = 0

        for metadata_path in sorted(metadata_root.glob("*.json")):
            file_count += 1
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            for person in payload.get("people") or []:
                if not isinstance(person, dict):
                    continue
                candidate_id = str(person.get("candidate_id") or "").strip()
                if not candidate_id:
                    continue

                existing = session.scalar(select(PersonCandidate).where(PersonCandidate.candidate_id == candidate_id))
                values = {
                    "candidate_id": candidate_id,
                    "camera_id": person.get("camera_id"),
                    "video_id": person.get("video_id"),
                    "track_id": str(person.get("track_id")) if person.get("track_id") is not None else None,
                    "human_key": person.get("human_key"),
                    "frame_idx": int(person.get("frame_idx") or 0),
                    "search_text": _candidate_search_document(person),
                    "metadata_path": str(metadata_path),
                    "raw_metadata": person,
                }

                if existing:
                    for key, value in values.items():
                        setattr(existing, key, value)
                    updated_count += 1
                else:
                    session.add(PersonCandidate(**values))
                    imported_count += 1

        session.commit()
        return {
            "imported_count": imported_count,
            "updated_count": updated_count,
            "file_count": file_count,
        }


def sync_local_queue_state(session: Session, *, only_if_empty: bool = False) -> dict[str, int]:
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
    video_root = queue_root / settings.google_drive_h265_folder_name
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
        if not local_video_path.is_absolute():
            local_video_path = (PROJECT_ROOT / local_video_path).resolve()
        if not local_video_path.exists():
            fallback_video_path = video_root / f"{metadata_path.stem}.h265"
            if fallback_video_path.exists():
                local_video_path = fallback_video_path

        title = str(
            video_payload.get("source_path")
            or video_payload.get("camera_id")
            or metadata_path.stem
        ).strip()
        source_filename = Path(title).name if title else f"{metadata_path.stem}.h265"
        available_link_video = f"/api/v1/queue/videos/{video_id}/file"
        available_link_metadata = f"/api/v1/queue/videos/{video_id}/metadata"

        upsert_queue_video_asset(
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
        counts = upsert_person_candidates(session, people_payload, str(metadata_path))
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


def list_queue_videos(session: Session) -> list[dict]:
    statement = select(QueueVideoAsset).order_by(QueueVideoAsset.queue_position.asc(), QueueVideoAsset.id.asc())
    return [queue_video_to_payload(video) for video in session.scalars(statement).all()]


def get_queue_video(session: Session, video_id: str) -> QueueVideoAsset | None:
    return session.scalar(select(QueueVideoAsset).where(QueueVideoAsset.video_id == video_id))


def load_queue_video_metadata(session: Session, video_id: str) -> dict[str, Any]:
    row = get_queue_video(session, video_id)
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
    row = get_queue_video(session, video_id)
    if row is None:
        raise FileNotFoundError(f"Queue video not found: {video_id}")
    video_path = Path(str(row.local_video_path or "")).expanduser()
    if video_path.exists() and video_path.is_file():
        return video_path
    raise FileNotFoundError(f"Queue video file not found: {video_id}")


def _resolve_candidate_source_path(candidate: dict) -> Path:
    for key in ("local_video_path", "storage_path"):
        value = str(candidate.get(key) or "").strip()
        if value:
            path = Path(value).expanduser()
            if path.exists():
                return path
    raise FileNotFoundError(f"Could not resolve source video for candidate {candidate.get('candidate_id')}")


def _candidate_segments(candidate: dict, max_segments_per_candidate: int) -> list[dict]:
    segments = candidate.get("matched_segments")
    if isinstance(segments, list) and segments:
        normalized = []
        for item in segments[:max_segments_per_candidate]:
            if not isinstance(item, dict):
                continue
            start_second = float(item.get("start_second") or 0.0)
            end_second = float(item.get("end_second") or 0.0)
            normalized.append(
                {
                    "start_second": max(0.0, start_second),
                    "end_second": max(end_second, start_second + 0.1),
                    "action_summary": str(item.get("action_summary") or "").strip(),
                }
            )
        if normalized:
            return normalized
    timeline = candidate.get("raw_metadata", {}).get("timeline")
    if isinstance(timeline, list):
        return [
            {
                "start_second": float(item.get("start_second") or 0.0),
                "end_second": max(float(item.get("end_second") or 0.0), float(item.get("start_second") or 0.0) + 0.1),
                "action_summary": str(item.get("action_summary") or "").strip(),
            }
            for item in timeline[:max_segments_per_candidate]
            if isinstance(item, dict)
        ]
    return []


def _write_tracking_manifest(manifest_path: Path, payload: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_tracking_video_local(
    session: Session,
    *,
    selected_candidate_id: str,
    candidate_ids: list[str],
    query_text: str | None = None,
    max_segments_per_candidate: int = 2,
) -> dict[str, Any]:
    import cv2

    candidate_order = [selected_candidate_id, *candidate_ids]
    unique_ids: list[str] = []
    for candidate_id in candidate_order:
        cleaned = str(candidate_id or "").strip()
        if cleaned and cleaned not in unique_ids:
            unique_ids.append(cleaned)
    if not unique_ids:
        raise ValueError("No candidate ids were provided")

    rows = session.scalars(select(PersonCandidate).where(PersonCandidate.candidate_id.in_(unique_ids))).all()
    row_map = {row.candidate_id: row for row in rows}
    queue_map = _queue_video_map(session, [str(row.video_id or "") for row in rows])
    selected_candidates: list[dict] = []
    for candidate_id in unique_ids:
        row = row_map.get(candidate_id)
        if row is None:
            continue
        payload = candidate_to_payload(row, queue_map.get(str(row.video_id or "")))
        selected_candidates.append(payload)
    if not selected_candidates:
        raise FileNotFoundError("No candidates found for tracking compilation")

    artifact_id = uuid.uuid4().hex
    output_root = Path(settings.tracking_output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{artifact_id}.mp4"
    manifest_path = output_root / f"{artifact_id}.json"

    writer = None
    written_frames = 0
    output_fps = 12.0
    output_size: tuple[int, int] | None = None
    clips_manifest: list[dict[str, Any]] = []

    for candidate in selected_candidates:
        source_path = _resolve_candidate_source_path(candidate)
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            continue
        fps = float(cap.get(cv2.CAP_PROP_FPS) or output_fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            continue
        if output_size is None:
            output_size = (width, height)
            output_fps = max(8.0, min(fps or 12.0, 24.0))
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                output_fps,
                output_size,
            )
        segments = _candidate_segments(candidate, max_segments_per_candidate=max_segments_per_candidate)
        if not segments:
            cap.release()
            continue
        for segment in segments:
            start_second = float(segment["start_second"])
            end_second = float(segment["end_second"])
            start_frame = max(0, int(start_second * fps))
            end_frame = max(start_frame, int(end_second * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame
            while frame_idx <= end_frame:
                ok, frame = cap.read()
                if not ok:
                    break
                if output_size and (frame.shape[1], frame.shape[0]) != output_size:
                    frame = cv2.resize(frame, output_size)
                overlay_1 = f"Query: {query_text or 'candidate tracking'}"
                overlay_2 = f"{candidate.get('camera_id') or 'camera'} | track {candidate.get('track_id') or '?'}"
                overlay_3 = segment.get("action_summary") or str(candidate.get("search_text") or "")[:100]
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
                cv2.putText(frame, overlay_1[:120], (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
                cv2.putText(frame, overlay_2[:120], (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, overlay_3[:120], (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
                if writer is not None:
                    writer.write(frame)
                    written_frames += 1
                frame_idx += 1
            clips_manifest.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "camera_id": candidate.get("camera_id"),
                    "track_id": candidate.get("track_id"),
                    "source_video": str(source_path),
                    "start_second": start_second,
                    "end_second": end_second,
                    "action_summary": segment.get("action_summary"),
                }
            )
        cap.release()

    if writer is not None:
        writer.release()

    if written_frames <= 0 or not output_path.exists():
        output_path.unlink(missing_ok=True)
        raise RuntimeError("Could not create tracking compilation from the selected candidates")

    manifest = {
        "artifact_id": artifact_id,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "query_text": query_text,
        "selected_candidate_id": selected_candidate_id,
        "candidate_ids": [candidate.get("candidate_id") for candidate in selected_candidates],
        "clip_count": len(clips_manifest),
        "written_frames": written_frames,
        "video_path": str(output_path),
        "clips": clips_manifest,
    }
    _write_tracking_manifest(manifest_path, manifest)
    return {
        "artifact_id": artifact_id,
        "video_url": f"/api/v1/tracking-artifacts/{artifact_id}",
        "manifest_url": f"/api/v1/tracking-artifacts/{artifact_id}/manifest",
        "manifest": manifest,
        "selected_candidate_id": selected_candidate_id,
    }


def build_tracking_video(
    session: Session,
    *,
    selected_candidate_id: str,
    candidate_ids: list[str],
    query_text: str | None = None,
    max_segments_per_candidate: int = 2,
) -> dict[str, Any]:
    candidate_order = [selected_candidate_id, *candidate_ids]
    unique_ids: list[str] = []
    for candidate_id in candidate_order:
        cleaned = str(candidate_id or "").strip()
        if cleaned and cleaned not in unique_ids:
            unique_ids.append(cleaned)
    if not unique_ids:
        raise ValueError("No candidate ids were provided")

    rows = session.scalars(select(PersonCandidate).where(PersonCandidate.candidate_id.in_(unique_ids))).all()
    row_map = {row.candidate_id: row for row in rows}
    queue_map = _queue_video_map(session, [str(row.video_id or "") for row in rows])
    selected_candidates: list[dict] = []
    for candidate_id in unique_ids:
        row = row_map.get(candidate_id)
        if row is None:
            continue
        payload = candidate_to_payload(row, queue_map.get(str(row.video_id or "")))
        selected_candidates.append(payload)
    if not selected_candidates:
        raise FileNotFoundError("No candidates found for tracking compilation")

    response = _post_tracking_json(
        "/api/v1/candidates/track",
        {
            "selected_candidate_id": selected_candidate_id,
            "candidate_ids": unique_ids,
            "candidates": selected_candidates,
            "query_text": query_text,
            "max_segments_per_candidate": max_segments_per_candidate,
        },
    )
    if not isinstance(response, dict):
        raise RuntimeError("Invalid tracking-service response for tracking build")
    return response


def resolve_tracking_artifact_paths(artifact_id: str) -> tuple[Path, Path]:
    cleaned = _tracking_slug(artifact_id)
    output_root = Path(settings.tracking_output_root)
    return output_root / f"{cleaned}.mp4", output_root / f"{cleaned}.json"


def get_queue_video_rows(session: Session) -> list[QueueVideoAsset]:
    statement = select(QueueVideoAsset).order_by(QueueVideoAsset.queue_position.asc(), QueueVideoAsset.id.asc())
    return list(session.scalars(statement).all())


def upsert_queue_video_asset(
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
) -> QueueVideoAsset:
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


def delete_queue_video_asset(session: Session, video_id: str) -> None:
    session.execute(delete(PersonCandidate).where(PersonCandidate.video_id == video_id))
    session.execute(delete(QueueVideoAsset).where(QueueVideoAsset.video_id == video_id))
    session.flush()


def upsert_person_candidates(session: Session, people: list[dict], metadata_path: str) -> dict[str, int]:
    imported_count = 0
    updated_count = 0

    for person in people:
        if not isinstance(person, dict):
            continue
        candidate_id = str(person.get("candidate_id") or "").strip()
        if not candidate_id:
            continue

        existing = session.scalar(select(PersonCandidate).where(PersonCandidate.candidate_id == candidate_id))
        values = {
            "candidate_id": candidate_id,
            "camera_id": person.get("camera_id"),
            "video_id": person.get("video_id"),
            "track_id": str(person.get("track_id")) if person.get("track_id") is not None else None,
            "human_key": person.get("human_key"),
            "frame_idx": int(person.get("frame_idx") or 0),
            "search_text": _candidate_search_document(person),
            "metadata_path": metadata_path,
            "raw_metadata": person,
        }
        if existing:
            for key, value in values.items():
                setattr(existing, key, value)
            updated_count += 1
        else:
            session.add(PersonCandidate(**values))
            imported_count += 1

    session.flush()
    return {"imported_count": imported_count, "updated_count": updated_count}
