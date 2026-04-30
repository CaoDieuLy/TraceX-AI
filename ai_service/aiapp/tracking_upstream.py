"""Proxy HTTP tới tracking / Lightning — toàn bộ URL model nằm ở đây, không dùng trong gateway."""
from __future__ import annotations

from typing import Any

import httpx
from fastapi import HTTPException, Request

from .config import settings
from .http_client import get_http_client


def _service_auth_headers() -> dict[str, str]:
    token = settings.lightning_api_token.strip()
    if not token:
        return {}
    header_name = settings.lightning_api_auth_header.strip() or "Authorization"
    auth_prefix = settings.lightning_api_auth_prefix
    if auth_prefix and not auth_prefix.endswith(" "):
        auth_prefix = f"{auth_prefix} "
    return {header_name: f"{auth_prefix}{token}".strip()}


def tracking_headers_from_request(request: Request) -> dict[str, str]:
    authorization = request.headers.get("authorization")
    if authorization:
        return {"Authorization": authorization}
    return _service_auth_headers()


def _tracking_base() -> str:
    return settings.tracking_service_url.rstrip("/")


async def proxy_get_json(path: str, request: Request) -> Any:
    url = f"{_tracking_base()}/{path.lstrip('/')}"
    headers = tracking_headers_from_request(request)
    response = await get_http_client().get(url, headers=headers or None)
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


async def proxy_post_json(path: str, payload: dict[str, Any], request: Request) -> Any:
    url = f"{_tracking_base()}/{path.lstrip('/')}"
    headers = tracking_headers_from_request(request)
    response = await get_http_client().post(url, json=payload, headers=headers or None)
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


async def proxy_get_bytes(path: str, request: Request) -> tuple[bytes, str]:
    url = f"{_tracking_base()}/{path.lstrip('/')}"
    headers = tracking_headers_from_request(request)
    response = await get_http_client().get(url, headers=headers or None)
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.content, response.headers.get("content-type", "application/octet-stream")
