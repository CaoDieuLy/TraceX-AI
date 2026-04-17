from __future__ import annotations

from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import settings

app = FastAPI(title="MCPT API Gateway", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _forward_auth_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    authorization = request.headers.get("authorization")
    if authorization:
        headers["Authorization"] = authorization
    return headers


async def _get_json(url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()


async def _post_json(url: str, payload: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(url, json=payload or {}, headers=headers)
        response.raise_for_status()
        return response.json()


async def _patch_json(url: str, payload: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.patch(url, json=payload or {}, headers=headers)
        response.raise_for_status()
        return response.json()


@app.get("/health")
async def healthcheck() -> dict:
    return {"status": "ok", "service": "api-gateway"}


@app.get("/api/v1/overview")
async def overview() -> dict:
    try:
        metadata = await _get_json(f"{settings.metadata_service_url}/api/v1/overview")
        ai = await _get_json(f"{settings.tracking_service_url}/api/v1/pipeline/config")
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Downstream service error: {exc}") from exc
    return {"metadata": metadata, "ai": ai}


@app.post("/api/v1/auth/register")
async def register(payload: dict[str, Any]) -> dict:
    try:
        return await _post_json(f"{settings.metadata_service_url}/api/v1/auth/register", payload)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc


@app.post("/api/v1/auth/login")
async def login(payload: dict[str, Any]) -> dict:
    try:
        return await _post_json(f"{settings.metadata_service_url}/api/v1/auth/login", payload)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc


@app.get("/api/v1/auth/me")
async def me(request: Request) -> dict:
    try:
        return await _get_json(f"{settings.metadata_service_url}/api/v1/auth/me", headers=_forward_auth_headers(request))
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc


@app.post("/api/v1/videos")
async def create_video(
    request: Request,
    title: str = Form(...),
    description: str | None = Form(default=None),
    storage_url: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
) -> dict:
    data = {"title": title}
    if description:
        data["description"] = description
    if storage_url:
        data["storage_url"] = storage_url

    files = None
    if file is not None:
        content = await file.read()
        files = {"file": (file.filename or "video.bin", content, file.content_type or "application/octet-stream")}

    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            response = await client.post(
                f"{settings.metadata_service_url}/api/v1/videos",
                data=data,
                files=files,
                headers=_forward_auth_headers(request),
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc


@app.get("/api/v1/videos")
async def videos(request: Request) -> dict:
    try:
        return await _get_json(f"{settings.metadata_service_url}/api/v1/videos", headers=_forward_auth_headers(request))
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc


@app.get("/api/v1/videos/{video_id}")
async def video_detail(video_id: str, request: Request) -> dict:
    try:
        return await _get_json(
            f"{settings.metadata_service_url}/api/v1/videos/{video_id}",
            headers=_forward_auth_headers(request),
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc


@app.get("/api/v1/video-queries")
async def video_queries(request: Request) -> dict:
    try:
        return await _get_json(
            f"{settings.metadata_service_url}/api/v1/video-queries",
            headers=_forward_auth_headers(request),
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc


@app.post("/api/v1/video-queries/run")
async def run_video_query(payload: dict[str, Any], request: Request) -> dict:
    headers = _forward_auth_headers(request)
    try:
        created_query = await _post_json(f"{settings.metadata_service_url}/api/v1/video-queries", payload, headers=headers)
        ai_payload = {
            "query_id": created_query["query_id"],
            "video_id": created_query["video_id"],
            "video_title": created_query.get("video_title"),
            "storage_path": created_query["storage_path"],
            "query_text": created_query["query_text"],
            "metadata": {"gateway_source": "api-gateway"},
        }
        ai_result = await _post_json(f"{settings.tracking_service_url}/api/v1/ai/process", ai_payload)
        updated_query = await _patch_json(
            f"{settings.metadata_service_url}/api/v1/video-queries/{created_query['query_id']}",
            {
                "status": ai_result["status"],
                "ai_job_id": ai_result["job_id"],
                "ai_response": ai_result,
            },
            headers=headers,
        )
        return {"query": updated_query, "ai_result": ai_result}
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Service error: {exc}") from exc


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
