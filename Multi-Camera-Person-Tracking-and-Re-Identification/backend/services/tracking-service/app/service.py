from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import httpx

from .config import settings
from .legacy_runtime import LEGACY_ROOT, legacy_workdir


@lru_cache(maxsize=1)
def get_pipeline_config() -> dict:
    mode = "mock" if settings.tracking_use_mock or not settings.lightning_api_base_url else "remote"
    return {
        "provider": "lightningai",
        "mode": mode,
        "lightning_api_base_url": settings.lightning_api_base_url or None,
        "lightning_api_endpoint": settings.lightning_api_endpoint,
        "legacy_root": str(LEGACY_ROOT),
    }


@lru_cache(maxsize=1)
def get_tracker():
    with legacy_workdir():
        from src_vlm.tracker import ReID_Tracker

        return ReID_Tracker(use_mock=settings.tracking_use_mock)


def _build_lightning_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = settings.lightning_api_token.strip()
    if token:
        prefix = settings.lightning_api_auth_prefix or ""
        headers[settings.lightning_api_auth_header] = f"{prefix}{token}"
    return headers


def process_video_query(payload: dict) -> dict:
    storage_path = str(payload.get("storage_path") or "").strip()
    file_exists = bool(storage_path and Path(storage_path).exists())
    processed_at = datetime.now(timezone.utc)
    query_text = str(payload.get("query_text") or "").strip()
    video_id = str(payload.get("video_id") or "").strip()
    query_id = payload.get("query_id")

    if not settings.tracking_use_mock and settings.lightning_api_base_url.strip():
        endpoint = f"{settings.lightning_api_base_url.rstrip('/')}/{settings.lightning_api_endpoint.lstrip('/')}"
        request_payload = {
            "query_id": query_id,
            "video_id": video_id,
            "video_title": payload.get("video_title"),
            "video_path": storage_path,
            "query_text": query_text,
            "metadata": payload.get("metadata") or {},
        }
        with httpx.Client(timeout=settings.lightning_timeout_seconds) as client:
            response = client.post(endpoint, json=request_payload, headers=_build_lightning_headers())
            response.raise_for_status()
            try:
                body = response.json()
            except ValueError:
                body = {"text": response.text}

        return {
            "status": str(body.get("status") or "completed"),
            "provider": "lightningai",
            "mode": "remote",
            "query_id": query_id,
            "video_id": video_id,
            "job_id": str(body.get("job_id") or uuid4()),
            "summary": str(body.get("summary") or f"LightningAI processed query '{query_text}'."),
            "file_exists": file_exists,
            "processed_at": processed_at,
            "raw_response": body,
        }

    return {
        "status": "completed",
        "provider": "lightningai",
        "mode": "mock",
        "query_id": query_id,
        "video_id": video_id,
        "job_id": str(uuid4()),
        "summary": f"Mock LightningAI response for '{query_text}' on video '{video_id}'.",
        "file_exists": file_exists,
        "processed_at": processed_at,
        "raw_response": {
            "matched_segments": [
                {
                    "start_second": 12.5,
                    "end_second": 19.0,
                    "confidence": 0.92,
                    "note": "Demo segment generated in mock mode.",
                }
            ],
            "storage_path": storage_path,
            "metadata": payload.get("metadata") or {},
        },
    }


def run_tracking(candidate_info: dict) -> dict:
    tracker = get_tracker()
    with legacy_workdir():
        output_path = tracker.run_tracking_on_candidate(
            candidate_info,
            "data/videos/compressed",
            "data/videos/tracking_output",
        )

    resolved = Path(output_path).resolve() if output_path else None
    relative = None
    if resolved is not None:
        try:
            relative = str(resolved.relative_to(LEGACY_ROOT))
        except ValueError:
            relative = str(resolved)

    return {
        "output_path": str(resolved) if resolved else None,
        "relative_output_path": relative,
        "exists": bool(resolved and resolved.exists()),
        "tracking_use_mock": settings.tracking_use_mock,
    }
