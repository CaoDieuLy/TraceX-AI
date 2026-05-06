"""
Queue Worker - Process videos from queue via LightningAI trace-service.

This worker runs continuously, polling the queue for new videos,
and sends them to the trace-service for AI processing.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class QueueItem:
    video_id: str
    camera_id: Optional[str]
    drive_file_id: Optional[str]
    source_filename: Optional[str]
    local_video_path: Optional[str]


def _get_trace_service_url() -> str:
    return os.getenv(
        "TRACE_SERVICE_URL",
        "https://8000-01kqhxrsmzj0gjh7fe5fqga4jm.cloudspaces.litng.ai"
    )


def _get_trace_headers() -> dict:
    token = os.getenv("LIGHTNING_API_TOKEN", "").strip()
    if not token:
        return {"Content-Type": "application/json"}
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


def _wait_for_trace_service(max_wait: int = 120) -> bool:
    url = _get_trace_service_url().rstrip("/") + "/health"
    deadline = time.time() + max_wait
    poll_interval = 5

    while time.time() < deadline:
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(url, headers=_get_trace_headers())
            if response.is_success:
                logger.info("Trace service is ready")
                return True
        except httpx.HTTPError as e:
            logger.info("Waiting for trace service: %s", e)
        time.sleep(poll_interval)

    logger.warning("Trace service not ready after %ds, proceeding anyway", max_wait)
    return True


def process_queue_item(
    session: Session,
    item: QueueItem,
    timeout: int = 600,
) -> dict:
    """
    Send a queue item to trace-service for processing.
    Returns the trace service response.
    """
    trace_url = _get_trace_service_url().rstrip("/") + "/api/v1/video/process"

    payload = {
        "video_id": item.video_id,
        "camera_id": item.camera_id,
    }

    if item.drive_file_id:
        payload["drive_file_id"] = item.drive_file_id
        payload["source_filename"] = item.source_filename
    elif item.local_video_path:
        payload["video_path"] = item.local_video_path
    else:
        raise ValueError(f"No video source for {item.video_id}")

    logger.info("Processing video: %s (drive: %s)", item.video_id, item.drive_file_id or "local")

    with httpx.Client(timeout=float(timeout)) as client:
        response = client.post(
            trace_url,
            json=payload,
            headers=_get_trace_headers(),
        )

    response.raise_for_status()
    return response.json()


def run_worker(
    session_factory,
    poll_interval: int = 30,
    max_concurrent: int = 3,
    request_timeout: int = 600,
):
    """
    Main worker loop. Polls the queue for unprocessed videos
    and sends them to trace-service.
    """
    from ..core.models import QueueVideoAsset, PersonCandidate

    _wait_for_trace_service()

    while True:
        session = session_factory()
        try:
            # Find queued videos that haven't been processed yet
            # A video is "processed" if it has PersonCandidate records
            statement = (
                select(QueueVideoAsset)
                .where(QueueVideoAsset.processed_at.is_(None))
                .order_by(QueueVideoAsset.queue_position.asc())
                .limit(max_concurrent)
            )

            queued_videos = list(session.scalars(statement).all())
            if not queued_videos:
                logger.debug("Queue empty, sleeping %ds", poll_interval)
                time.sleep(poll_interval)
                continue

            logger.info("Found %d videos in queue", len(queued_videos))

            for video in queued_videos:
                try:
                    item = QueueItem(
                        video_id=video.video_id,
                        camera_id=video.camera_id,
                        drive_file_id=video.drive_video_file_id,
                        source_filename=video.source_filename,
                        local_video_path=video.local_video_path,
                    )

                    result = process_queue_item(session, item, timeout=request_timeout)

                    # Save candidates to database
                    people = result.get("people", [])
                    _save_candidates(session, video.video_id, people)

                    # Mark video as processed
                    video.processed_at = datetime.now(timezone.utc)
                    session.commit()

                    logger.info(
                        "Video %s processed: %d people detected",
                        video.video_id,
                        len(people),
                    )

                except httpx.HTTPStatusError as e:
                    logger.error(
                        "Trace service error for %s: %s %s",
                        video.video_id,
                        e.response.status_code,
                        e.response.text[:500],
                    )
                    session.rollback()
                except Exception as e:
                    logger.exception("Error processing video %s", video.video_id)
                    session.rollback()

        except Exception as e:
            logger.exception("Worker error: %s", e)
            session.rollback()
        finally:
            session.close()

        time.sleep(poll_interval)


def _save_candidates(session: Session, video_id: str, people: list[dict]) -> int:
    """
    Save person candidates to database.
    """
    from .models import PersonCandidate

    imported = 0
    for person in people:
        candidate_id = str(person.get("candidate_id") or "").strip()
        if not candidate_id:
            continue

        raw_metadata = {
            k: v
            for k, v in person.items()
            if k not in ("candidate_id", "camera_id", "video_id", "track_id")
        }

        existing = session.scalar(
            select(PersonCandidate).where(PersonCandidate.candidate_id == candidate_id)
        )

        if existing:
            existing.raw_metadata = raw_metadata
            existing.search_text = _build_search_text(person)
        else:
            candidate = PersonCandidate(
                candidate_id=candidate_id,
                camera_id=str(person.get("camera_id") or "").strip() or None,
                video_id=video_id,
                track_id=str(person.get("track_id") or "").strip(),
                human_key=person.get("human_key"),
                frame_idx=int(person.get("frame_idx") or 0),
                search_text=_build_search_text(person),
                raw_metadata=raw_metadata,
            )
            session.add(candidate)
            imported += 1

    session.flush()
    return imported


def _build_search_text(person: dict) -> str:
    parts = []
    for key in ("search_text", "appearance_summary", "attribute_summary"):
        text = str(person.get(key) or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts)
