import json
from pathlib import Path

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from .config import settings
from .models import PersonCandidate


def candidate_to_payload(candidate: PersonCandidate) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "camera_id": candidate.camera_id,
        "video_id": candidate.video_id,
        "track_id": candidate.track_id,
        "human_key": candidate.human_key,
        "frame_idx": candidate.frame_idx,
        "search_text": candidate.search_text,
        "metadata_path": candidate.metadata_path,
        "raw_metadata": candidate.raw_metadata,
    }


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
    total_candidates = session.scalar(select(func.count()).select_from(PersonCandidate)) or 0
    total_cameras = session.scalar(select(func.count(func.distinct(PersonCandidate.camera_id))).select_from(PersonCandidate)) or 0
    total_videos = session.scalar(select(func.count(func.distinct(PersonCandidate.video_id))).select_from(PersonCandidate)) or 0
    return {
        "total_candidates": int(total_candidates),
        "total_cameras": int(total_cameras),
        "total_videos": int(total_videos),
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
                "search_text": person.get("search_text") or person.get("person_caption") or person.get("caption"),
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
