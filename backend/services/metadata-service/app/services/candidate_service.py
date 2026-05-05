from __future__ import annotations

import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any

import cv2
import httpx
from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.orm import Session

from ..config import A20_ROOT, PROJECT_ROOT, settings
from ..core.models import PersonCandidate, QueueVideoAsset

logger = logging.getLogger(__name__)

PERSON_CANDIDATE_INSERT_CHUNK_SIZE = 50


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
        "recorded_start": raw_metadata.get("recorded_start"),
        "clip_drive_urls": raw_metadata.get("clip_drive_urls") or [],
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


def _queue_video_map(session: Session, video_ids: list[str]) -> dict[str, QueueVideoAsset]:
    cleaned = [video_id for video_id in video_ids if video_id]
    if not cleaned:
        return {}
    rows = session.scalars(select(QueueVideoAsset).where(QueueVideoAsset.video_id.in_(cleaned))).all()
    return {row.video_id: row for row in rows}


def search_candidates(
    session: Session,
    query: str | None = None,
    limit: int = 20,
    camera_ids: list[str] | None = None,
) -> list[dict]:
    statement = select(PersonCandidate).order_by(PersonCandidate.updated_at.desc(), PersonCandidate.id.desc())
    cleaned_query = (query or "").strip()
    if cleaned_query:
        tokens = [t for t in cleaned_query.lower().split() if len(t) >= 2]
        for token in tokens:
            pattern = f"%{token}%"
            statement = statement.where(
                or_(
                    PersonCandidate.search_text.ilike(pattern),
                    PersonCandidate.camera_id.ilike(pattern),
                    PersonCandidate.video_id.ilike(pattern),
                    PersonCandidate.human_key.ilike(pattern),
                )
            )
    if camera_ids:
        statement = statement.where(PersonCandidate.camera_id.in_(camera_ids))
    rows = session.scalars(statement.limit(max(1, min(limit, 500)))).all()
    queue_map = _queue_video_map(session, [str(row.video_id or "") for row in rows])
    return [candidate_to_payload(row, queue_map.get(str(row.video_id or ""))) for row in rows]


def get_candidate(session: Session, candidate_id: str) -> dict | None:
    row = session.scalar(select(PersonCandidate).where(PersonCandidate.candidate_id == candidate_id))
    if row is None:
        return None
    queue_map = _queue_video_map(session, [str(row.video_id or "")])
    return candidate_to_payload(row, queue_map.get(str(row.video_id or "")))


def get_queue_video(session: Session, video_id: str) -> QueueVideoAsset | None:
    return session.scalar(select(QueueVideoAsset).where(QueueVideoAsset.video_id == video_id))


def get_queue_video_rows(session: Session) -> list[QueueVideoAsset]:
    statement = select(QueueVideoAsset).order_by(QueueVideoAsset.queue_position.asc(), QueueVideoAsset.id.asc())
    return list(session.scalars(statement).all())


def upsert_person_candidates(session: Session, people: list[dict], metadata_path: str | None = None) -> dict[str, int]:
    imported_count = 0
    updated_count = 0
    normalized_people: list[dict[str, Any]] = []

    for person in people:
        if not isinstance(person, dict):
            continue
        candidate_id = str(person.get("candidate_id") or "").strip()
        if not candidate_id:
            continue

        normalized_people.append(
            {
            "candidate_id": candidate_id,
            "camera_id": person.get("camera_id"),
            "video_id": person.get("video_id"),
            "track_id": str(person.get("track_id")) if person.get("track_id") is not None else None,
            "human_key": person.get("human_key"),
            "frame_idx": int(person.get("frame_idx") or 0),
            "search_text": _candidate_search_document(person),
            "metadata_path": metadata_path,
            "raw_metadata": _candidate_raw_metadata_subset(person),
            }
        )

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
    for start in range(0, len(insert_rows), PERSON_CANDIDATE_INSERT_CHUNK_SIZE):
        chunk = insert_rows[start: start + PERSON_CANDIDATE_INSERT_CHUNK_SIZE]
        if chunk:
            session.bulk_insert_mappings(PersonCandidate, chunk)
            session.flush()

    return {"imported_count": imported_count, "updated_count": updated_count}


# ============================================================================
# Preview Image Generation
# ============================================================================

def _preview_root() -> Path:
    """Return absolute path for candidate preview images on VPS."""
    root = Path("/workspace/storage/candidate-previews")
    root.mkdir(parents=True, exist_ok=True)
    return root


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

    cv2.putText(frame, "Candidate preview from DB raw_metadata", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"{row.camera_id or row.video_id or 'camera?'} | track {row.track_id or '?'}", (24, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (110, 230, 255), 2)
    cv2.putText(frame, f"{row.candidate_id}", (24, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (190, 255, 190), 2)

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

    queue_video = get_queue_video(session, str(row.video_id or ""))
    if queue_video is None:
        raise FileNotFoundError(f"Queue video not found for candidate: {candidate_id}")

    source_path = _resolve_queue_video_file_path(queue_video)
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source video for candidate preview: {source_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        target_frame = _preview_frame_index(row, raw_metadata, total_frames=total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read preview frame {target_frame} from {source_path}")

        frame_height = int(frame.shape[0])
        frame_width = int(frame.shape[1])
        bbox = _bbox_xyxy(_candidate_bbox(raw_metadata), frame_width=frame_width, frame_height=frame_height)

        overlay = frame.copy()
        banner_height = 72
        cv2.rectangle(overlay, (0, 0), (frame_width, banner_height), (8, 18, 28), -1)
        cv2.addWeighted(overlay, 0.48, frame, 0.52, 0, frame)
        cv2.putText(frame, f"{row.camera_id or row.video_id or 'candidate'} | frame {target_frame}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        cv2.putText(frame, f"Track {row.track_id or '?'} | {row.candidate_id}", (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (110, 230, 255), 2)

        _draw_bbox_trails(frame, raw_metadata, frame_width=frame_width, frame_height=frame_height)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 120), 3)
            label = f"track {row.track_id or '?'}"
            label_origin_y = max(y1 - 14, 24)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.64, 2)
            cv2.rectangle(frame, (x1, label_origin_y - text_height - 8), (x1 + text_width + 14, label_origin_y + 4), (80, 255, 120), -1)
            cv2.putText(frame, label, (x1 + 7, label_origin_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (8, 18, 28), 2)

        success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not success:
            raise RuntimeError(f"Could not encode candidate preview image for {candidate_id}")
        preview_path.write_bytes(encoded.tobytes())
        return preview_path
    finally:
        cap.release()


# ============================================================================
# Ranking and Tracking
# ============================================================================

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


def _public_api_url(path: str | None) -> str | None:
    from ..config import settings
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


def _tracking_service_headers() -> dict[str, str]:
    from ..config import settings
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
    from ..config import settings
    return settings.tracking_service_url.rstrip("/") + "/" + path.lstrip("/")


def _tracking_service_is_remote() -> bool:
    from urllib.parse import urlparse
    from ..config import settings
    host = (urlparse(settings.tracking_service_url).hostname or "").strip().lower()
    return host not in {"", "127.0.0.1", "localhost", "tracking-service"}


def _is_tracking_startup_timeout_detail(detail: str) -> bool:
    text = str(detail or "").strip().lower()
    if not text:
        return False
    markers = ("api startup timed out", "startup timed out", "cold start", "warming up", "service unavailable")
    return any(marker in text for marker in markers)


def _wait_for_tracking_upstream_ready(*, context: str) -> None:
    import time
    if not _tracking_service_is_remote():
        return

    from ..config import settings
    health_url = _tracking_service_url("/health")
    deadline = time.monotonic() + max(int(settings.tracking_startup_max_wait_seconds), 1)
    poll_interval = max(int(settings.tracking_startup_poll_interval_seconds), 1)
    timeout_seconds = max(int(settings.tracking_health_timeout_seconds), 1)
    last_error: str | None = None
    attempt = 0

    while time.monotonic() < deadline:
        attempt += 1
        try:
            with httpx.Client(timeout=float(timeout_seconds)) as client:
                response = client.get(health_url, headers=_tracking_service_headers())
            if response.is_success:
                if attempt > 1:
                    logger.info("Tracking upstream ready after %s health check attempts for %s", attempt, context)
                return
            last_error = f"status={response.status_code} body={response.text[:200]}"
            logger.info("Tracking upstream not ready yet for %s: %s", context, last_error)
        except httpx.ReadTimeout:
            logger.info("Tracking upstream health read-timeout for %s — inference in progress, proceeding", context)
            return
        except httpx.HTTPError as exc:
            last_error = str(exc)
            logger.info("Tracking upstream health probe failed for %s: %s", context, exc)
        time.sleep(poll_interval)

    raise RuntimeError(
        f"Tracking upstream was not ready within {settings.tracking_startup_max_wait_seconds}s for {context}. "
        f"Last error: {last_error or 'unknown'}"
    )


def _post_tracking_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    import time
    from ..config import settings
    max_attempts = max(int(settings.tracking_startup_retry_attempts), 1)
    timeout_seconds = float(settings.tracking_request_timeout_seconds)
    last_exc: Exception | None = None
    context = f"POST {path}"
    for attempt in range(1, max_attempts + 1):
        try:
            _wait_for_tracking_upstream_ready(context=context)
            with httpx.Client(timeout=timeout_seconds) as client:
                response = client.post(
                    _tracking_service_url(path),
                    json=payload,
                    headers=_tracking_service_headers(),
                )
            if response.is_error:
                detail = response.text[:2000]
                if attempt < max_attempts and _is_tracking_startup_timeout_detail(detail):
                    logger.warning(
                        "Tracking upstream cold start during %s attempt=%s status=%s detail=%s",
                        context, attempt, response.status_code, detail,
                    )
                    time.sleep(max(2, int(settings.tracking_startup_poll_interval_seconds)))
                    continue
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            if attempt < max_attempts and _is_tracking_startup_timeout_detail(exc.response.text):
                logger.warning("Retrying tracking request after startup timeout for %s attempt=%s status=%s", context, attempt, exc.response.status_code)
                time.sleep(max(2, int(settings.tracking_startup_poll_interval_seconds)))
                continue
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


def rank_candidates(
    session: Session,
    query_text: str,
    limit: int = 5,
    camera_ids: list[str] | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
) -> list[dict]:
    cleaned_query = query_text.strip()
    if not cleaned_query:
        return []

    bounded_limit = max(1, min(limit, 50))

    statement = select(PersonCandidate).order_by(PersonCandidate.updated_at.desc(), PersonCandidate.id.desc())
    if camera_ids:
        cam_lower = [c.lower().strip() for c in camera_ids if c.strip()]
        if cam_lower:
            statement = statement.where(func.lower(PersonCandidate.camera_id).in_(cam_lower))

    rows = session.scalars(statement).all()
    if not rows:
        return []

    queue_map = _queue_video_map(session, [str(row.video_id or "") for row in rows])
    shortlist_limit = _remote_ranking_shortlist_limit(bounded_limit)
    candidates = _local_prefilter_ranked_candidates(
        rows, queue_map, query_text=cleaned_query, shortlist_limit=shortlist_limit,
    )

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

    try:
        response = _post_tracking_json("/api/v1/candidates/search", payload)
        items = response.get("items")
        if not isinstance(items, list):
            raise RuntimeError("Tracking service did not return a valid items list.")
        return items
    except Exception:
        raise


def get_overview(session: Session) -> dict:
    from sqlalchemy import func
    from ..core.models import User, VideoAsset, VideoQuery
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


def _resolve_queue_video_file_path(row: QueueVideoAsset) -> Path:
    for path in _queue_video_path_candidates(row):
        if path.exists() and path.is_file():
            return path

    downloaded = _download_drive_video_to_cache(row)
    if downloaded is not None and downloaded.exists():
        return downloaded

    raise FileNotFoundError(f"Queue video file not found: {row.video_id}")


def _queue_video_path_candidates(row: QueueVideoAsset) -> list[Path]:
    """Return list of candidate paths where video file might exist on VPS."""
    candidates: list[Path] = []
    raw_value = str(row.local_video_path or "").strip()
    source_filename = Path(str(row.source_filename or row.video_id or "")).name

    def push(path: Path | None) -> None:
        if path is None:
            return
        normalized = path.expanduser()
        if normalized not in candidates:
            candidates.append(normalized)

    # Absolute path from database
    if raw_value:
        raw_path = Path(raw_value).expanduser()
        push(raw_path)

    # VPS standard paths
    if source_filename:
        push(Path("/workspace/storage/queue/Videos") / source_filename)
        push(Path("/workspace/storage/queue/local/Queue/Videos") / source_filename)

    return candidates


def _download_drive_video_to_cache(row: QueueVideoAsset) -> Path | None:
    drive_file_id = str(row.drive_video_file_id or "").strip()
    if not drive_file_id:
        return None

    suffix = Path(str(row.source_filename or row.video_id or drive_file_id)).suffix or ".mp4"
    target_path = _preview_source_cache_root() / f"{_slugify(drive_file_id)}{suffix}"
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    _ensure_shared_secret()
    from shared_secret_runtime import build_google_drive_oauth_service

    drive_service = build_google_drive_oauth_service()
    request = drive_service.files().get_media(fileId=drive_file_id, supportsAllDrives=True)
    with target_path.open("wb") as handle:
        downloader = MediaIoBaseDownload(handle, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return target_path if target_path.exists() else None


def _preview_source_cache_root() -> Path:
    root = _preview_root() / "source-cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ensure_shared_secret() -> None:
    import sys
    from ..config import A20_ROOT
    if A20_ROOT and str(A20_ROOT) not in sys.path:
        sys.path.insert(0, str(A20_ROOT))


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
            normalized.append({
                "start_second": max(0.0, start_second),
                "end_second": max(end_second, start_second + 0.1),
                "action_summary": str(item.get("action_summary") or "").strip(),
            })
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


def _tracking_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._") or "tracking"


def resolve_tracking_artifact_paths(artifact_id: str) -> tuple[Path, Path]:
    from ..config import settings
    cleaned = _tracking_slug(artifact_id)
    output_root = Path(settings.tracking_output_root)
    return output_root / f"{cleaned}.mp4", output_root / f"{cleaned}.json"


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
