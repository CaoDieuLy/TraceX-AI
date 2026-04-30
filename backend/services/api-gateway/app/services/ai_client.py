"""HTTP client cho ai_service (search + proxy tracking — gateway không gọi Lightning trực tiếp)."""
from __future__ import annotations

from typing import Any

import httpx

from ..config import settings
from ..http_client import get_http_client


def _ai_base() -> str:
    return settings.ai_service_url.rstrip("/")


def _auth_headers_from_request_headers(request_headers: dict[str, str] | None) -> dict[str, str]:
    if not request_headers:
        return {}
    for key, value in request_headers.items():
        if key.lower() == "authorization" and value:
            return {"Authorization": value}
    return {}


async def search_internal(*, query: str, top_k: int, offset: int) -> dict[str, Any]:
    payload = {"query": query, "top_k": top_k, "offset": offset}
    response = await get_http_client().post(f"{_ai_base()}/internal/search", json=payload)
    response.raise_for_status()
    return response.json()


async def tracking_runtime_config(*, request_headers: dict[str, str] | None = None) -> Any:
    response = await get_http_client().get(
        f"{_ai_base()}/internal/tracking/v1/runtime-config",
        headers=_auth_headers_from_request_headers(request_headers) or None,
    )
    response.raise_for_status()
    return response.json()


async def tracking_ai_process(payload: dict[str, Any], *, request_headers: dict[str, str] | None = None) -> Any:
    response = await get_http_client().post(
        f"{_ai_base()}/internal/tracking/v1/ai/process",
        json=payload,
        headers=_auth_headers_from_request_headers(request_headers) or None,
    )
    response.raise_for_status()
    return response.json()


async def tracking_run(payload: dict[str, Any], *, request_headers: dict[str, str] | None = None) -> Any:
    response = await get_http_client().post(
        f"{_ai_base()}/internal/tracking/v1/tracking/run",
        json=payload,
        headers=_auth_headers_from_request_headers(request_headers) or None,
    )
    response.raise_for_status()
    return response.json()


async def tracking_artifact_bytes(artifact_id: str, *, request_headers: dict[str, str] | None = None) -> tuple[bytes, str]:
    response = await get_http_client().get(
        f"{_ai_base()}/internal/tracking/v1/artifacts/{artifact_id}",
        headers=_auth_headers_from_request_headers(request_headers) or None,
    )
    response.raise_for_status()
    return response.content, response.headers.get("content-type", "application/octet-stream")


async def tracking_artifact_manifest(artifact_id: str, *, request_headers: dict[str, str] | None = None) -> Any:
    response = await get_http_client().get(
        f"{_ai_base()}/internal/tracking/v1/artifacts/{artifact_id}/manifest",
        headers=_auth_headers_from_request_headers(request_headers) or None,
    )
    response.raise_for_status()
    return response.json()
