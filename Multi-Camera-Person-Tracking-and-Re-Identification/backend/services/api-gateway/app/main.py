from __future__ import annotations

from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

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


def _service_auth_headers() -> dict[str, str]:
    token = settings.lightning_api_token.strip()
    if not token:
        return {}
    header_name = settings.lightning_api_auth_header.strip() or "Authorization"
    auth_prefix = settings.lightning_api_auth_prefix
    if auth_prefix and not auth_prefix.endswith(" "):
        auth_prefix = f"{auth_prefix} "
    return {header_name: f"{auth_prefix}{token}".strip()}


def _tracking_service_headers(request: Request | None = None) -> dict[str, str]:
    if request is not None:
        forwarded = _forward_auth_headers(request)
        if forwarded:
            return forwarded
    return _service_auth_headers()


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


async def _get_bytes(url: str, headers: dict[str, str] | None = None) -> tuple[bytes, str]:
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.content, response.headers.get("content-type", "application/octet-stream")


@app.get("/health")
async def healthcheck() -> dict:
    return {"status": "ok", "service": "api-gateway"}


@app.get("/api/v1/overview")
async def overview() -> dict:
    try:
        metadata = await _get_json(f"{settings.metadata_service_url}/api/v1/overview")
        ai = await _get_json(
            f"{settings.tracking_service_url}/api/v1/pipeline/config",
            headers=_tracking_service_headers(),
        )
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
        ai_result = await _post_json(
            f"{settings.tracking_service_url}/api/v1/ai/process",
            ai_payload,
            headers=_tracking_service_headers(request),
        )
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


@app.post("/api/v1/ai/process")
async def ai_process(payload: dict[str, Any], request: Request) -> dict:
    try:
        return await _post_json(
            f"{settings.tracking_service_url}/api/v1/ai/process",
            payload,
            headers=_tracking_service_headers(request),
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Tracking service error: {exc}") from exc


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


@app.get("/api/v1/candidates/{candidate_id}/preview")
async def candidate_preview(candidate_id: str) -> Response:
    try:
        content, content_type = await _get_bytes(f"{settings.metadata_service_url}/api/v1/candidates/{candidate_id}/preview")
        return Response(content=content, media_type=content_type)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc


@app.post("/api/v1/candidates/search")
async def candidate_search(payload: dict[str, Any], request: Request) -> dict:
    try:
        return await _post_json(
            f"{settings.metadata_service_url}/api/v1/candidates/search",
            payload,
            headers=_forward_auth_headers(request),
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc


@app.post("/api/v1/candidates/track")
async def candidate_track(payload: dict[str, Any], request: Request) -> dict:
    try:
        return await _post_json(
            f"{settings.metadata_service_url}/api/v1/candidates/track",
            payload,
            headers=_forward_auth_headers(request),
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc


@app.get("/api/v1/queue/videos")
async def queue_videos() -> dict:
    try:
        return await _get_json(f"{settings.metadata_service_url}/api/v1/queue/videos")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc


@app.get("/api/v1/queue/videos/{video_id}/metadata")
async def queue_video_metadata(video_id: str) -> dict:
    try:
        return await _get_json(f"{settings.metadata_service_url}/api/v1/queue/videos/{video_id}/metadata")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc


@app.get("/api/v1/queue/videos/{video_id}/file")
async def queue_video_file(video_id: str) -> Response:
    try:
        content, content_type = await _get_bytes(f"{settings.metadata_service_url}/api/v1/queue/videos/{video_id}/file")
        return Response(content=content, media_type=content_type)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc


@app.post("/api/v1/queue/bootstrap")
async def queue_bootstrap(payload: dict[str, Any]) -> dict:
    try:
        return await _post_json(f"{settings.metadata_service_url}/api/v1/queue/bootstrap", payload)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc


@app.post("/api/v1/queue/process-imports")
async def queue_process_imports(payload: dict[str, Any] | None = None) -> dict:
    try:
        return await _post_json(f"{settings.metadata_service_url}/api/v1/queue/process-imports", payload or {})
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc


@app.post("/api/v1/tracking/run")
async def tracking_run(payload: dict[str, Any], request: Request) -> dict:
    try:
        return await _post_json(
            f"{settings.tracking_service_url}/api/v1/tracking/run",
            payload,
            headers=_tracking_service_headers(request),
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Tracking service error: {exc}") from exc


@app.get("/api/v1/tracking-artifacts/{artifact_id}")
async def tracking_artifact_video(artifact_id: str) -> Response:
    try:
        content, content_type = await _get_bytes(
            f"{settings.tracking_service_url}/api/v1/tracking-artifacts/{artifact_id}",
            headers=_tracking_service_headers(),
        )
        return Response(content=content, media_type=content_type)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
        content, content_type = await _get_bytes(f"{settings.metadata_service_url}/api/v1/tracking-artifacts/{artifact_id}")
        return Response(content=content, media_type=content_type)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Tracking service error: {exc}") from exc


@app.get("/api/v1/tracking-artifacts/{artifact_id}/manifest")
async def tracking_artifact_manifest(artifact_id: str) -> dict:
    try:
        return await _get_json(
            f"{settings.tracking_service_url}/api/v1/tracking-artifacts/{artifact_id}/manifest",
            headers=_tracking_service_headers(),
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
        return await _get_json(f"{settings.metadata_service_url}/api/v1/tracking-artifacts/{artifact_id}/manifest")
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Tracking service error: {exc}") from exc
