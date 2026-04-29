from __future__ import annotations

import json
import math
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import httpx
from googleapiclient.http import MediaIoBaseDownload
from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session, joinedload

from .auth import hash_password, verify_password
from .config import A20_ROOT, PROJECT_ROOT, settings
from .models import PersonCandidate, QueueVideoAsset, User, VideoAsset, VideoQuery

ROLE_HIERARCHY: dict[str, int] = {
    "USER": 1,
    "ADMIN": 2,
    "SUPER_ADMIN": 3,
}


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
    max_attempts = 3
    timeout_seconds = float(settings.tracking_request_timeout_seconds)
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with httpx.Client(timeout=timeout_seconds) as client:
                response = client.post(
                    _tracking_service_url(path),
                    json=payload,
                    headers=_tracking_service_headers(),
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            # Retry transient upstream/server-side failures, otherwise fail fast.
            if exc.response.status_code < 500 or attempt >= max_attempts:
                raise
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt >= max_attempts:
                raise
        time.sleep(0.8 * attempt)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unknown tracking upstream failure")


def _tracking_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._") or "tracking"


def _public_api_url(path: str | None) -> str | None:
    raw_path = str(path or "").strip()
    if not raw_path:
        return None
    if raw_path.startswith(("http://", "https://")):
        return raw_path
    if not raw_path.startswith("/"):
        raw_path = f"/{raw_path}"
    base_url = settings.public_api_base_url.strip().rstrip("/")
    if not base_url:
        return raw_path
    return f"{base_url}{raw_path}"


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


def candidate_to_payload(candidate: PersonCandidate, queue_video: QueueVideoAsset | None = None) -> dict:
    raw_metadata = candidate.raw_metadata or {}
    bbox = _candidate_bbox(raw_metadata)
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
        "bbox": bbox,
        "search_text": candidate.search_text,
        "metadata_path": candidate.metadata_path,
        "attribute_summary": raw_metadata.get("attribute_summary"),
        "appearance_summary": raw_metadata.get("appearance_summary"),
        "attribute_embedding_vector": raw_metadata.get("attribute_embedding_vector") if isinstance(raw_metadata.get("attribute_embedding_vector"), list) else [],
        "appearance_embedding_vector": raw_metadata.get("appearance_embedding_vector") if isinstance(raw_metadata.get("appearance_embedding_vector"), list) else [],
        "semantic_attributes": _coerce_string_list(raw_metadata.get("semantic_attributes")),
        "embedding_vector": raw_metadata.get("embedding_vector") if isinstance(raw_metadata.get("embedding_vector"), list) else [],
        "visibility_scores": _coerce_mapping(raw_metadata.get("visibility_scores")),
        "world_position": raw_metadata.get("world_position") or raw_metadata.get("top_point_projection"),
        "score": raw_metadata.get("score"),
        "timeline": raw_metadata.get("timeline") if isinstance(raw_metadata.get("timeline"), list) else [],
        "matched_segments": raw_metadata.get("matched_segments") or [],
        "action_semantic_embedding": _coerce_mapping(raw_metadata.get("action_semantic_embedding")),
        "tracklet_feature_pipeline": _coerce_mapping(raw_metadata.get("tracklet_feature_pipeline")),
        "available_link_video": queue_video.available_link_video if queue_video else None,
        "available_link_metadata": queue_video.available_link_metadata if queue_video else None,
        "drive_video_file_id": queue_video.drive_video_file_id if queue_video else None,
        "drive_metadata_file_id": queue_video.drive_metadata_file_id if queue_video else None,
        "local_video_path": queue_video.local_video_path if queue_video else None,
        "local_metadata_path": queue_video.local_metadata_path if queue_video else None,
        "storage_path": storage_path,
        "video_title": queue_video.title if queue_video else None,
        "source_filename": queue_video.source_filename if queue_video else None,
        "preview_image_url": f"/api/v1/candidates/{candidate.candidate_id}/preview",
        "raw_metadata": raw_metadata,
    }


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


def _candidate_raw_metadata_subset(raw_metadata: object) -> dict[str, Any]:
    payload = raw_metadata if isinstance(raw_metadata, dict) else {}
    reduced: dict[str, Any] = {}

    for key in (
        "attribute_summary",
        "appearance_summary",
        "score",
        "action_semantic_embedding",
        "tracklet_feature_pipeline",
    ):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            reduced[key] = value

    for key in ("attribute_embedding_vector", "appearance_embedding_vector"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            reduced[key] = value

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

    return reduced


def candidate_to_ranking_payload(candidate: PersonCandidate, queue_video: QueueVideoAsset | None = None) -> dict:
    raw_metadata = candidate.raw_metadata or {}
    reduced_raw_metadata = _candidate_raw_metadata_subset(raw_metadata)
    bbox = _candidate_bbox(raw_metadata)
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
        "bbox": bbox,
        "search_text": candidate.search_text,
        "metadata_path": candidate.metadata_path,
        "attribute_summary": reduced_raw_metadata.get("attribute_summary"),
        "appearance_summary": reduced_raw_metadata.get("appearance_summary"),
        "attribute_embedding_vector": reduced_raw_metadata.get("attribute_embedding_vector") if isinstance(reduced_raw_metadata.get("attribute_embedding_vector"), list) else [],
        "appearance_embedding_vector": reduced_raw_metadata.get("appearance_embedding_vector") if isinstance(reduced_raw_metadata.get("appearance_embedding_vector"), list) else [],
        "semantic_attributes": _coerce_string_list(reduced_raw_metadata.get("semantic_attributes")),
        "visibility_scores": _coerce_mapping(reduced_raw_metadata.get("visibility_scores")),
        "world_position": reduced_raw_metadata.get("world_position"),
        "score": reduced_raw_metadata.get("score"),
        "embedding_vector": raw_metadata.get("embedding_vector") if isinstance(raw_metadata.get("embedding_vector"), list) else [],
        "timeline": reduced_raw_metadata.get("timeline") if isinstance(reduced_raw_metadata.get("timeline"), list) else [],
        "matched_segments": reduced_raw_metadata.get("matched_segments") or [],
        "action_semantic_embedding": _coerce_mapping(reduced_raw_metadata.get("action_semantic_embedding")),
        "tracklet_feature_pipeline": _coerce_mapping(reduced_raw_metadata.get("tracklet_feature_pipeline")),
        "available_link_video": queue_video.available_link_video if queue_video else None,
        "available_link_metadata": queue_video.available_link_metadata if queue_video else None,
        "drive_video_file_id": queue_video.drive_video_file_id if queue_video else None,
        "drive_metadata_file_id": queue_video.drive_metadata_file_id if queue_video else None,
        "local_video_path": queue_video.local_video_path if queue_video else None,
        "local_metadata_path": queue_video.local_metadata_path if queue_video else None,
        "storage_path": storage_path,
        "video_title": queue_video.title if queue_video else None,
        "source_filename": queue_video.source_filename if queue_video else None,
        "preview_image_url": f"/api/v1/candidates/{candidate.candidate_id}/preview",
        "raw_metadata": reduced_raw_metadata,
    }


def _remote_ranking_shortlist_limit(limit: int) -> int:
    bounded_limit = max(1, min(limit, 50))
    return min(max(200, bounded_limit * 40), 400)


def _candidate_embedding_values(candidate: dict[str, Any]) -> list[float]:
    values = candidate.get("embedding_vector")
    if not isinstance(values, list) or not values:
        return []
    vector: list[float] = []
    try:
        for value in values:
            vector.append(float(value))
    except (TypeError, ValueError):
        return []
    if vector:
        return vector
    return []


def _embedding_cosine_similarity(anchor_vector: list[float], candidate_vector: list[float]) -> float:
    if not anchor_vector or not candidate_vector or len(anchor_vector) != len(candidate_vector):
        return 0.0
    dot = 0.0
    anchor_norm = 0.0
    candidate_norm = 0.0
    for anchor_value, candidate_value in zip(anchor_vector, candidate_vector):
        dot += anchor_value * candidate_value
        anchor_norm += anchor_value * anchor_value
        candidate_norm += candidate_value * candidate_value
    if anchor_norm <= 1e-12 or candidate_norm <= 1e-12:
        return 0.0
    return float(dot / math.sqrt(anchor_norm * candidate_norm))


def _public_tracking_candidate_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    payload = dict(candidate)
    public_video_url = _public_api_url(payload.get("available_link_video"))
    if public_video_url:
        payload["available_link_video"] = public_video_url
    if not str(payload.get("source_filename") or "").strip():
        payload["source_filename"] = str(payload.get("video_id") or payload.get("video_title") or "").strip() or None
    return payload


def _global_tracking_shortlist(
    rows: list[PersonCandidate],
    queue_map: dict[str, QueueVideoAsset],
    *,
    selected_candidate_id: str,
    query_text: str | None,
    shortlist_limit: int = 180,
    per_video_limit: int = 6,
) -> list[dict[str, Any]]:
    payload_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = candidate_to_ranking_payload(row, queue_map.get(str(row.video_id or "")))
        payload_map[str(row.candidate_id)] = _public_tracking_candidate_payload(payload)

    anchor = payload_map.get(str(selected_candidate_id).strip())
    if anchor is None:
        return []

    anchor_embedding = _candidate_embedding_values(anchor)
    cleaned_query = str(query_text or "").strip()
    scored_rows: list[tuple[float, int, str, dict[str, Any]]] = []
    for row in rows:
        payload = payload_map.get(str(row.candidate_id))
        if payload is None:
            continue
        candidate_embedding = _candidate_embedding_values(payload)
        embedding_score = _embedding_cosine_similarity(anchor_embedding, candidate_embedding)
        semantic_score = _semantic_overlap(cleaned_query, payload) if cleaned_query else 0.0
        timeline_bonus = min(len(payload.get("matched_segments") or []), 3) * 0.01
        total_score = embedding_score * 0.84 + semantic_score * 0.12 + timeline_bonus
        payload["anchor_similarity"] = round(embedding_score, 6)
        payload["global_tracking_score"] = round(total_score, 6)
        scored_rows.append((total_score, int(row.id), str(payload.get("video_id") or ""), payload))

    scored_rows.sort(key=lambda item: (-float(item[0]), -int(item[1]), item[2], str(item[3].get("candidate_id") or "")))

    selected: list[dict[str, Any]] = [anchor]
    used_ids = {str(anchor.get("candidate_id") or "")}
    per_video_counts: dict[str, int] = {}
    anchor_video_id = str(anchor.get("video_id") or "")
    if anchor_video_id:
        per_video_counts[anchor_video_id] = 1

    for _score, _row_id, video_id, payload in scored_rows:
        candidate_id = str(payload.get("candidate_id") or "")
        if not candidate_id or candidate_id in used_ids:
            continue
        if video_id and per_video_counts.get(video_id, 0) >= per_video_limit:
            continue
        selected.append(payload)
        used_ids.add(candidate_id)
        if video_id:
            per_video_counts[video_id] = per_video_counts.get(video_id, 0) + 1
        if len(selected) >= max(2, shortlist_limit):
            break

    return selected


def _local_prefilter_ranked_candidates(
    rows: list[PersonCandidate],
    queue_map: dict[str, QueueVideoAsset],
    *,
    query_text: str,
    shortlist_limit: int,
) -> list[dict]:
    scored: list[tuple[float, int, dict[str, Any]]] = []

    for row in rows:
        payload = candidate_to_ranking_payload(row, queue_map.get(str(row.video_id or "")))
        score = _score_candidate(query_text, payload)

        if score <= 0:
            row_search_text = str(row.search_text or "").lower()
            query_tokens = _tokenize(query_text)
            if row_search_text and any(token in row_search_text for token in query_tokens):
                score = 0.005

        scored.append((score, row.id, payload))

    scored.sort(
        key=lambda item: (
            -float(item[0]),
            -int(item[1]),
            str(item[2].get("camera_id") or ""),
            str(item[2].get("track_id") or ""),
            str(item[2].get("candidate_id") or ""),
        )
    )

    positive = [payload for score, _row_id, payload in scored if score > 0][:shortlist_limit]
    if positive:
        return positive
    return [payload for _score, _row_id, payload in scored[:shortlist_limit]]


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


def normalize_role(role: str | None) -> str:
    normalized = str(role or "USER").strip().upper()
    if normalized not in ROLE_HIERARCHY:
        return "USER"
    return normalized


def role_rank(role: str | None) -> int:
    return ROLE_HIERARCHY.get(normalize_role(role), 0)


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


def create_user(
    session: Session,
    email: str,
    full_name: str,
    password: str,
    *,
    role: str = "USER",
    is_active: bool = True,
) -> User:
    existing = get_user_by_email(session, email)
    if existing is not None:
        raise ValueError("Email already registered")

    user = User(
        email=email.lower().strip(),
        full_name=full_name.strip(),
        hashed_password=hash_password(password),
        role=normalize_role(role),
        is_active=bool(is_active),
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def authenticate_user(session: Session, identifier: str, password: str) -> User | None:
    user = get_user_by_identifier(session, identifier)
    if user is None or not bool(user.is_active) or not verify_password(password, user.hashed_password):
        return None
    user.last_login = datetime.now(timezone.utc)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def list_users(session: Session) -> list[User]:
    statement = select(User).order_by(User.created_at.desc(), User.id.desc())
    return list(session.scalars(statement).all())


def get_user_by_id(session: Session, user_id: int) -> User | None:
    return session.scalar(select(User).where(User.id == user_id))


def update_user_access(session: Session, user: User, *, role: str | None = None, is_active: bool | None = None) -> User:
    if role is not None:
        user.role = normalize_role(role)
    if is_active is not None:
        user.is_active = bool(is_active)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def ensure_bootstrap_admin(session: Session, *, email: str, password: str, full_name: str) -> User | None:
    normalized_email = str(email or "").strip().lower()
    normalized_password = str(password or "").strip()
    if not normalized_email or not normalized_password:
        return None
    existing = get_user_by_email(session, normalized_email)
    if existing is not None:
        if str(existing.role or "").upper() != "SUPER_ADMIN":
            existing.role = "SUPER_ADMIN"
            existing.is_active = True
            session.add(existing)
            session.commit()
            session.refresh(existing)
        return existing
    admin = User(
        email=normalized_email,
        full_name=(full_name or "Administrator").strip() or "Administrator",
        hashed_password=hash_password(normalized_password),
        role="SUPER_ADMIN",
        is_active=True,
    )
    session.add(admin)
    session.commit()
    session.refresh(admin)
    return admin


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


def _bbox_xyxy(raw_bbox: object, *, frame_width: int, frame_height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(raw_bbox, list) or len(raw_bbox) < 4:
        return None
    try:
        x1 = int(float(raw_bbox[0]))
        y1 = int(float(raw_bbox[1]))
        third = int(float(raw_bbox[2]))
        fourth = int(float(raw_bbox[3]))
    except (TypeError, ValueError):
        return None

    if third > x1 and fourth > y1 and third <= frame_width and fourth <= frame_height:
        x2 = third
        y2 = fourth
    else:
        x2 = x1 + max(third, 1)
        y2 = y1 + max(fourth, 1)

    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(x1 + 1, min(x2, frame_width))
    y2 = max(y1 + 1, min(y2, frame_height))
    return x1, y1, x2, y2


def _preview_frame_index(candidate: PersonCandidate, raw_metadata: dict[str, Any], *, total_frames: int) -> int:
    if total_frames <= 0:
        return max(int(candidate.frame_idx or 0), 0)

    frame_idx = int(candidate.frame_idx or 0)
    if frame_idx > 0:
        return min(max(frame_idx, 0), total_frames - 1)

    timeline = raw_metadata.get("timeline")
    if isinstance(timeline, list):
        for item in timeline:
            if not isinstance(item, dict):
                continue
            sample = item.get("frame_idx")
            if sample is None:
                continue
            try:
                frame_idx = int(sample)
            except (TypeError, ValueError):
                continue
            return min(max(frame_idx, 0), total_frames - 1)
    return 0


def _draw_bbox_trails(frame: Any, raw_metadata: dict[str, Any], *, frame_width: int, frame_height: int) -> None:
    timeline = raw_metadata.get("timeline")
    if not isinstance(timeline, list):
        return

    trail_count = 0
    for item in timeline[:3]:
        if not isinstance(item, dict):
            continue
        samples = item.get("bbox_samples")
        if not isinstance(samples, list):
            continue
        for sample in samples[:2]:
            bbox = _bbox_xyxy(sample, frame_width=frame_width, frame_height=frame_height)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
            trail_count += 1
            if trail_count >= 6:
                return


def _build_metadata_only_candidate_preview(
    row: PersonCandidate,
    raw_metadata: dict[str, Any],
    preview_path: Path,
) -> Path:
    import numpy as np

    frame_height = 720
    frame_width = 1280
    frame = np.full((frame_height, frame_width, 3), (18, 26, 34), dtype=np.uint8)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_width, 120), (8, 18, 28), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(
        frame,
        "Candidate preview from DB raw_metadata",
        (24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"{row.camera_id or row.video_id or 'camera?'} | track {row.track_id or '?'}",
        (24, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (110, 230, 255),
        2,
    )
    cv2.putText(
        frame,
        f"{row.candidate_id}",
        (24, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (190, 255, 190),
        2,
    )

    bbox = _bbox_xyxy(_candidate_bbox(raw_metadata), frame_width=frame_width, frame_height=frame_height)
    _draw_bbox_trails(frame, raw_metadata, frame_width=frame_width, frame_height=frame_height)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 120), 3)

    success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not success:
        raise RuntimeError(f"Could not encode metadata-only candidate preview for {row.candidate_id}")
    preview_path.write_bytes(encoded.tobytes())
    return preview_path


def build_candidate_preview_image(session: Session, candidate_id: str) -> Path:
    row = session.scalar(select(PersonCandidate).where(PersonCandidate.candidate_id == candidate_id))
    if row is None:
        raise FileNotFoundError(f"Candidate not found: {candidate_id}")

    preview_path = _preview_root() / f"{_slugify(candidate_id)}.jpg"
    raw_metadata = row.raw_metadata or {}
    try:
        queue_video = get_queue_video(session, str(row.video_id or ""))
        if queue_video is None:
            raise FileNotFoundError(f"Queue video not found for candidate: {candidate_id}")

        source_path = _resolve_queue_video_file_path(queue_video)
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open source video for candidate preview: {source_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        target_frame = _preview_frame_index(row, raw_metadata, total_frames=total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read preview frame {target_frame} from {source_path}")

        frame_height = int(frame.shape[0])
        frame_width = int(frame.shape[1])
        bbox = _bbox_xyxy(_candidate_bbox(raw_metadata), frame_width=frame_width, frame_height=frame_height)

        overlay = frame.copy()
        banner_height = 72
        cv2.rectangle(overlay, (0, 0), (frame_width, banner_height), (8, 18, 28), -1)
        cv2.addWeighted(overlay, 0.48, frame, 0.52, 0, frame)
        cv2.putText(
            frame,
            f"{row.camera_id or row.video_id or 'candidate'} | frame {target_frame}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Track {row.track_id or '?'} | {row.candidate_id}",
            (16, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (110, 230, 255),
            2,
        )

        _draw_bbox_trails(frame, raw_metadata, frame_width=frame_width, frame_height=frame_height)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 120), 3)
            label = f"track {row.track_id or '?'}"
            label_origin_y = max(y1 - 14, 24)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.64, 2)
            cv2.rectangle(
                frame,
                (x1, label_origin_y - text_height - 8),
                (x1 + text_width + 14, label_origin_y + 4),
                (80, 255, 120),
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x1 + 7, label_origin_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.64,
                (8, 18, 28),
                2,
            )

        success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not success:
            raise RuntimeError(f"Could not encode candidate preview image for {candidate_id}")
        preview_path.write_bytes(encoded.tobytes())
        return preview_path
    except Exception:
        return _build_metadata_only_candidate_preview(row, raw_metadata, preview_path)


def rank_candidates(
    session: Session,
    query_text: str,
    limit: int = 5,
    camera_ids: list[str] | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
) -> list[dict]:
    """
    Rank candidate — 5-phase pipeline:
    1) Hard filter DB by camera_ids (zone) if provided
    2) Local prefilter to reduce candidate set
    3) Call tracking_service /api/v1/candidates/search with CLIP encoding + hybrid scoring
    No fallback local when upstream fails.
    """
    cleaned_query = query_text.strip()
    if not cleaned_query:
        return []

    bounded_limit = max(1, min(limit, 50))

    # Phase 1 — Hard filter at DB level: camera zone
    statement = select(PersonCandidate).order_by(PersonCandidate.updated_at.desc(), PersonCandidate.id.desc())
    if camera_ids:
        cam_lower = [c.lower().strip() for c in camera_ids if c.strip()]
        if cam_lower:
            statement = statement.where(
                func.lower(PersonCandidate.camera_id).in_(cam_lower)
            )

    rows = session.scalars(statement).all()
    if not rows:
        return []

    queue_map = _queue_video_map(session, [str(row.video_id or "") for row in rows])
    shortlist_limit = _remote_ranking_shortlist_limit(bounded_limit)
    candidates = _local_prefilter_ranked_candidates(
        rows,
        queue_map,
        query_text=cleaned_query,
        shortlist_limit=shortlist_limit,
    )

    # Phase 3-5 — Send to tracking service with full context
    payload: dict = {
        "query_text": cleaned_query,
        "candidates": candidates,
        "limit": bounded_limit,
    }
    if camera_ids:
        payload["camera_ids"] = camera_ids
    if time_from:
        payload["time_from"] = time_from
    if time_to:
        payload["time_to"] = time_to

    response = _post_tracking_json("/api/v1/candidates/search", payload)
    items = response.get("items")
    if not isinstance(items, list):
        raise RuntimeError("Tracking service did not return a valid items list.")
    return items


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
        if not local_video_path.is_absolute():
            local_video_path = (PROJECT_ROOT / local_video_path).resolve()
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


def _preview_root() -> Path:
    root = PROJECT_ROOT / "storage" / "candidate-previews"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _preview_source_cache_root() -> Path:
    root = _preview_root() / "source-cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _queue_video_path_candidates(row: QueueVideoAsset) -> list[Path]:
    candidates: list[Path] = []
    raw_value = str(row.local_video_path or "").strip()
    source_filename = Path(str(row.source_filename or row.video_id or "")).name

    def push(path: Path | None) -> None:
        if path is None:
            return
        normalized = path.expanduser()
        if normalized not in candidates:
            candidates.append(normalized)

    if raw_value:
        raw_path = Path(raw_value).expanduser()
        push(raw_path)
        if raw_path.is_absolute() and A20_ROOT:
            raw_parts = raw_path.parts
            host_root_parts = Path("/opt/mcpt/A20-App-119").parts
            if raw_parts[: len(host_root_parts)] == host_root_parts:
                try:
                    relative = raw_path.relative_to(Path("/opt/mcpt/A20-App-119"))
                    push(A20_ROOT / relative)
                except ValueError:
                    pass
            project_mount_root = A20_ROOT
            if str(raw_path).startswith(str(PROJECT_ROOT)):
                try:
                    relative = raw_path.relative_to(PROJECT_ROOT)
                    push(project_mount_root / relative)
                except ValueError:
                    pass

    if source_filename:
        push(PROJECT_ROOT / "storage" / "queue" / "local" / "Queue" / settings.queue_video_folder_name / source_filename)
        push(Path("/workspace/storage/queue/local/Queue") / settings.queue_video_folder_name / source_filename)
        if A20_ROOT:
            push(A20_ROOT / "storage" / "queue" / "local" / "Queue" / settings.queue_video_folder_name / source_filename)

    return candidates


def _download_drive_video_to_cache(row: QueueVideoAsset) -> Path | None:
    drive_file_id = str(row.drive_video_file_id or "").strip()
    if not drive_file_id:
        return None

    suffix = Path(str(row.source_filename or row.video_id or drive_file_id)).suffix or ".mp4"
    target_path = _preview_source_cache_root() / f"{_slugify(drive_file_id)}{suffix}"
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    from shared_secret_runtime import build_google_drive_oauth_service

    drive_service = build_google_drive_oauth_service()
    request = drive_service.files().get_media(fileId=drive_file_id, supportsAllDrives=True)
    with target_path.open("wb") as handle:
        downloader = MediaIoBaseDownload(handle, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return target_path if target_path.exists() else None


def _resolve_queue_video_file_path(row: QueueVideoAsset) -> Path:
    for path in _queue_video_path_candidates(row):
        if path.exists() and path.is_file():
            return path

    downloaded = _download_drive_video_to_cache(row)
    if downloaded is not None and downloaded.exists():
        return downloaded

    raise FileNotFoundError(f"Queue video file not found: {row.video_id}")


def load_queue_video_file_path(session: Session, video_id: str) -> Path:
    row = get_queue_video(session, video_id)
    if row is None:
        raise FileNotFoundError(f"Queue video not found: {video_id}")
    return _resolve_queue_video_file_path(row)


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
        public_video_url = _public_api_url(payload.get("available_link_video"))
        if public_video_url:
            payload["available_link_video"] = public_video_url
        if not str(payload.get("source_filename") or "").strip():
            payload["source_filename"] = (
                str(payload.get("video_id") or payload.get("video_title") or "").strip() or None
            )
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
    cleaned_selected_id = str(selected_candidate_id or "").strip()
    if not cleaned_selected_id:
        raise ValueError("selected_candidate_id is required")

    rows = session.scalars(select(PersonCandidate).order_by(PersonCandidate.updated_at.desc(), PersonCandidate.id.desc())).all()
    if not rows:
        raise FileNotFoundError("No candidates found for tracking compilation")
    queue_map = _queue_video_map(session, [str(row.video_id or "") for row in rows])
    shortlisted_candidates = _global_tracking_shortlist(
        rows,
        queue_map,
        selected_candidate_id=cleaned_selected_id,
        query_text=query_text,
    )
    if not shortlisted_candidates:
        raise FileNotFoundError(f"Selected candidate was not found: {cleaned_selected_id}")

    shortlisted_ids = [str(item.get("candidate_id") or "").strip() for item in shortlisted_candidates]
    shortlisted_ids = [item for item in shortlisted_ids if item]
    response = _post_tracking_json(
        "/api/v1/candidates/track",
        {
            "selected_candidate_id": cleaned_selected_id,
            "candidate_ids": shortlisted_ids,
            "candidates": shortlisted_candidates,
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


def upsert_person_candidates(session: Session, people: list[dict], metadata_path: str | None = None) -> dict[str, int]:
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

