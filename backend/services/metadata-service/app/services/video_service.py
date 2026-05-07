from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from ..config import settings
from shared.models import Video, VideoQuery, User

if TYPE_CHECKING:
    from shared.models import User

logger = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._") or "video"


def video_to_payload(video: Video) -> dict:
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


def create_video_asset(
    session: Session,
    user: "User",
    title: str,
    description: str | None,
    storage_path: str,
    storage_backend: str,
    source_filename: str | None,
    content_type: str | None,
) -> Video:
    video = Video(
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


def list_videos(session: Session, user: "User") -> list[dict]:
    statement = select(Video).where(Video.user_id == user.id).order_by(Video.created_at.desc(), Video.id.desc())
    return [video_to_payload(video) for video in session.scalars(statement).all()]


def get_video_by_public_id(session: Session, user: "User", video_id: str) -> Video | None:
    return session.scalar(select(Video).where(Video.video_id == video_id, Video.user_id == user.id))


def create_video_query(session: Session, user: "User", video: Video, query_text: str) -> VideoQuery:
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


def list_video_queries(session: Session, user: "User") -> list[dict]:
    statement = (
        select(VideoQuery)
        .options(joinedload(VideoQuery.video))
        .where(VideoQuery.user_id == user.id)
        .order_by(VideoQuery.updated_at.desc(), VideoQuery.id.desc())
    )
    return [query_to_payload(query) for query in session.scalars(statement).all()]


def get_video_query(session: Session, user: "User", query_id: str) -> VideoQuery | None:
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
