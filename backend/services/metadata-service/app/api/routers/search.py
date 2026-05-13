"""Search router — forwards to query-service (GPU) for ranking.

Frontend calls POST /api/v1/search → metadata-service → query-service (GPU).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ...core.dependencies import get_current_user
from shared.models import User


class SearchRequest(BaseModel):
    query: str | None = None
    text: str | None = None  # alias for query (FE may send either)
    top_k: int = 20
    offset: int = 0
    camera_ids: list[str] | None = None
    time_from: str | None = None
    time_to: str | None = None
    query_image_url: str | None = None  # URL of uploaded query image (for history display)
    query_id: str | None = None  # passed back by FE during pagination so the qh row is reused

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])

_QUERY_SERVICE_URL: str | None = None


def _get_query_service_url() -> str:
    global _QUERY_SERVICE_URL
    if _QUERY_SERVICE_URL is None:
        import os
        _QUERY_SERVICE_URL = os.getenv(
            "QUERY_SERVICE_URL",
            "http://query-service:8003"
        )
    return _QUERY_SERVICE_URL


def _post_to_query_service(path: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    url = f"{_get_query_service_url().rstrip('/')}/api/v1{path}"
    max_attempts = 3
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt < max_attempts:
                import time
                time.sleep(0.5 * attempt)
        except Exception as exc:
            last_exc = exc
            break
        import time
        time.sleep(0.5)

    raise HTTPException(
        status_code=503,
        detail=f"Query service unavailable: {last_exc}",
    )


@router.post("")
def search_candidates(
    body: SearchRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Search candidates — forwarded to query-service for GPU ranking.
    Accepts JSON body with: query (or text), top_k, offset, camera_ids, time_from, time_to.
    """
    query_text = body.query or body.text or ""
    payload: dict[str, Any] = {
        "query": query_text,
        "top_k": body.top_k,
        "offset": body.offset,
        "user_id": current_user.id,
    }
    if body.camera_ids:
        payload["camera_ids"] = body.camera_ids
    if body.time_from:
        payload["time_from"] = body.time_from
    if body.time_to:
        payload["time_to"] = body.time_to
    if body.query_image_url:
        payload["query_image_url"] = body.query_image_url
    if body.query_id:
        payload["query_id"] = body.query_id

    try:
        result = _post_to_query_service("/search", payload)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Search failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")


@router.get("/overview")
def search_overview() -> dict[str, Any]:
    """Get overview statistics — forwarded to query-service."""
    try:
        return _post_to_query_service("/overview", {}, timeout=30.0)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Overview failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Overview failed: {exc}")
