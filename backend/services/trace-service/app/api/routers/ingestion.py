"""
ingestion.py — Forward ingestion requests to LightningAI GPU service.

POST /api/v1/ingestion/process  — submit video for GPU processing
GET  /api/v1/ingestion/status/{job_id}  — poll job status
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingestion"])

_LIGHTNING_BASE_URL = os.getenv(
    "LIGHTNING_API_BASE_URL",
    "https://8000-01kqhxrsmzj0gjh7fe5fqga4jm.cloudspaces.litng.ai",
).rstrip("/")

_LIGHTNING_TOKEN = os.getenv("LIGHTNING_API_TOKEN", "abc123")
_AUTH_HEADER = os.getenv("LIGHTNING_API_AUTH_HEADER", "Authorization")
_AUTH_PREFIX = os.getenv("LIGHTNING_API_AUTH_PREFIX", "Bearer ")


def _headers() -> dict:
    headers = {}
    if _LIGHTNING_TOKEN:
        headers[_AUTH_HEADER] = f"{_AUTH_PREFIX}{_LIGHTNING_TOKEN}"
    return headers


@router.post("/api/v1/ingestion/process")
def submit_ingestion(req: dict[str, Any]) -> dict[str, Any]:
    """Submit a video to LightningAI GPU pipeline for processing."""
    url = f"{_LIGHTNING_BASE_URL}/api/v1/ingestion/process"
    for attempt in range(1, 5):
        try:
            response = httpx.post(
                url,
                json=req,
                headers=_headers(),
                timeout=60,
            )
            if response.status_code == 503:
                wait = 15 * attempt
                logger.warning("[lightning] 503 queue full, retry %d/4 in %ds", attempt, wait)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 503:
                wait = 15 * attempt
                logger.warning("[lightning] 503, retry %d/4 in %ds", attempt, wait)
                time.sleep(wait)
                continue
            logger.error("LightningAI error: %s %s", exc.response.status_code, exc.response.text[:300])
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text[:300])
        except httpx.TimeoutException:
            logger.warning("[lightning] timeout on attempt %d", attempt)
            time.sleep(15)
            continue
        except httpx.HTTPError as exc:
            logger.warning("[lightning] connection error: %s", exc)
            time.sleep(10)
            continue
    raise HTTPException(status_code=503, detail="LightningAI queue full after 4 retries")


@router.get("/api/v1/ingestion/status/{job_id}")
def poll_ingestion(job_id: str) -> dict[str, Any]:
    """Poll LightningAI job status."""
    url = f"{_LIGHTNING_BASE_URL}/api/v1/ingestion/status/{job_id}"
    try:
        response = httpx.get(url, headers=_headers(), timeout=30)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        logger.error("LightningAI poll error: %s %s", exc.response.status_code, exc.response.text[:200])
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text[:200])
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LightningAI timeout")
    except httpx.HTTPError as exc:
        logger.warning("[lightning] poll connection error: %s", exc)
        raise HTTPException(status_code=503, detail=f"LightningAI unavailable: {exc}")
