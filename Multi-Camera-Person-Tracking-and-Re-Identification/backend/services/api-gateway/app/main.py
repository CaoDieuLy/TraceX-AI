from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import settings

app = FastAPI(title="MCPT API Gateway", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


async def _post_json(url: str, payload: dict[str, Any] | None = None) -> Any:
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=payload or {})
        response.raise_for_status()
        return response.json()


@app.get("/health")
async def healthcheck() -> dict:
    return {"status": "ok", "service": "api-gateway"}


@app.get("/api/v1/overview")
async def overview() -> dict:
    try:
        metadata = await _get_json(f"{settings.metadata_service_url}/api/v1/overview")
        pipeline = await _get_json(f"{settings.tracking_service_url}/api/v1/pipeline/config")
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Downstream service error: {exc}") from exc
    return {"metadata": metadata, "pipeline": pipeline}


@app.post("/api/v1/candidates/import-legacy")
async def import_legacy() -> dict:
    try:
        return await _post_json(f"{settings.metadata_service_url}/api/v1/candidates/import-legacy")
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc


@app.get("/api/v1/candidates")
async def candidates(
    query: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict:
    try:
        return await _get_json(f"{settings.metadata_service_url}/api/v1/candidates", params={"query": query, "limit": limit})
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc


@app.get("/api/v1/candidates/{candidate_id}")
async def candidate_detail(candidate_id: str) -> dict:
    try:
        return await _get_json(f"{settings.metadata_service_url}/api/v1/candidates/{candidate_id}")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc


@app.post("/api/v1/tracking/run")
async def tracking_run(payload: dict[str, Any]) -> dict:
    try:
        return await _post_json(f"{settings.tracking_service_url}/api/v1/tracking/run", payload)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Tracking service error: {exc}") from exc
