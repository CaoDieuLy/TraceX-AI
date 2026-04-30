from __future__ import annotations

import httpx

from .config import settings

_client: httpx.AsyncClient | None = None


def _build_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        timeout=float(settings.tracking_proxy_timeout_seconds),
        connect=float(settings.tracking_proxy_connect_timeout_seconds),
        read=float(settings.tracking_proxy_timeout_seconds),
        write=float(settings.tracking_proxy_timeout_seconds),
        pool=float(settings.tracking_proxy_pool_timeout_seconds),
    )


def _build_limits() -> httpx.Limits:
    return httpx.Limits(
        max_connections=int(settings.tracking_proxy_max_connections),
        max_keepalive_connections=int(settings.tracking_proxy_max_keepalive_connections),
        keepalive_expiry=float(settings.tracking_proxy_keepalive_expiry_seconds),
    )


def init_http_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=_build_timeout(), limits=_build_limits())
    return _client


async def close_http_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def get_http_client() -> httpx.AsyncClient:
    if _client is None:
        return init_http_client()
    return _client
