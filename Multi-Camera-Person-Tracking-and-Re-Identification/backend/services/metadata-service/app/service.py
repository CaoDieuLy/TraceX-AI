from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session, joinedload

from .auth import hash_password, verify_password
from .config import settings
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


def candidate_to_payload(candidate: PersonCandidate) -> dict:
    raw_metadata = candidate.raw_metadata or {}
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
        "raw_video_metadata": video.raw_video_metadata,
    }


def get_user_by_email(session: Session, email: str) -> User | None:
    return session.scalar(select(User).where(User.email == email.lower().strip()))


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


def authenticate_user(session: Session, email: str, password: str) -> User | None:
    user = get_user_by_email(session, email)
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
    return [candidate_to_payload(row) for row in rows]


def get_candidate(session: Session, candidate_id: str) -> dict | None:
    row = session.scalar(select(PersonCandidate).where(PersonCandidate.candidate_id == candidate_id))
    return candidate_to_payload(row) if row else None


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


def list_queue_videos(session: Session) -> list[dict]:
    statement = select(QueueVideoAsset).order_by(QueueVideoAsset.queue_position.asc(), QueueVideoAsset.id.asc())
    return [queue_video_to_payload(video) for video in session.scalars(statement).all()]


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
