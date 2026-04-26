from __future__ import annotations

from typing import Any

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .config import settings
from .services.ai_client import (
    search_internal,
    tracking_ai_process as ai_tracking_process,
    tracking_artifact_bytes,
    tracking_artifact_manifest as ai_tracking_artifact_manifest,
    tracking_runtime_config,
    tracking_run as ai_tracking_run,
)

app = FastAPI(title="MCPT API Gateway", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=10, ge=1, le=50)
    offset: int = Field(default=0, ge=0)


class SearchResultItem(BaseModel):
    id: str
    thumbnail_url: str
    description: str


class SearchResponse(BaseModel):
    results: list[SearchResultItem]


class VideoSegmentItem(BaseModel):
    id: str
    video_url: str
    title: str
    description: str


class VideoDetailResponse(BaseModel):
    id: str
    segments: list[VideoSegmentItem]


def _forward_auth_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    authorization = request.headers.get("authorization")
    if authorization:
        headers["Authorization"] = authorization
    return headers


def require_auth_header(request: Request) -> None:
    authorization = str(request.headers.get("authorization") or "").strip()
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")


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


def _to_search_item(video: dict[str, Any]) -> SearchResultItem:
    video_id = str(video.get("video_id") or video.get("id") or "").strip()
    title = str(video.get("title") or f"Video {video_id or 'unknown'}").strip()
    summary_parts = [
        title,
        str(video.get("source_filename") or "").strip(),
        str(video.get("source_mode") or "").strip(),
        str(video.get("storage_backend") or "").strip(),
    ]
    description = " · ".join([part for part in summary_parts if part]) or "Video result"

    thumbnail_url = str(video.get("available_link_video") or "").strip()
    if not thumbnail_url and video_id:
        thumbnail_url = f"/api/v1/queue/videos/{video_id}/file"
    if not thumbnail_url:
        thumbnail_url = "https://picsum.photos/seed/mcpt-default/400/225"

    return SearchResultItem(id=video_id or "unknown", thumbnail_url=thumbnail_url, description=description)


def _to_candidate_search_item(candidate: dict[str, Any]) -> SearchResultItem:
    candidate_id = str(candidate.get("candidate_id") or candidate.get("id") or "").strip()
    video_id = str(candidate.get("video_id") or "").strip()
    result_id = candidate_id or video_id or "unknown"

    preview_url = str(candidate.get("preview_image_url") or "").strip()
    if not preview_url and candidate_id:
        preview_url = f"/api/v1/candidates/{candidate_id}/preview"
    if not preview_url:
        preview_url = "https://picsum.photos/seed/mcpt-default/400/225"

    description_parts = [
        str(candidate.get("appearance_summary") or "").strip(),
        str(candidate.get("search_text") or "").strip(),
        f"candidate={candidate_id}" if candidate_id else "",
        f"video={video_id}" if video_id else "",
        f"track={candidate.get('track_id')}" if candidate.get("track_id") else "",
    ]
    description = " · ".join([part for part in description_parts if part]) or "Candidate result"

    return SearchResultItem(id=result_id, thumbnail_url=preview_url, description=description)


def _score_video_for_query(video: dict[str, Any], query: str) -> int:
    q = query.lower().strip()
    if not q:
        return 0
    content = " ".join(
        [
            str(video.get("title") or ""),
            str(video.get("source_filename") or ""),
            str(video.get("source_mode") or ""),
            str(video.get("video_id") or ""),
        ]
    ).lower()
    return 1 if q in content else 0


def _build_segment_description(segment_payload: dict[str, Any], fallback: str) -> str:
    if not isinstance(segment_payload, dict):
        return fallback
    candidates = [
        segment_payload.get("description"),
        segment_payload.get("appearance_summary"),
        segment_payload.get("search_text"),
    ]
    for value in candidates:
        text = str(value or "").strip()
        if text:
            return text
    return fallback


@app.get("/health")
async def healthcheck() -> dict:
    return {"status": "ok", "service": "api-gateway"}


@app.post("/search", response_model=SearchResponse)
async def search(payload: SearchRequest, request: Request, _auth: None = Depends(require_auth_header)) -> SearchResponse:
    # Strict mode: search must come from ai_service pipeline only.
    try:
        ai_payload = await search_internal(
            query=payload.query,
            top_k=payload.top_k,
            offset=payload.offset,
        )
        items = ai_payload.get("results") if isinstance(ai_payload, dict) else []
        rows = [item for item in items if isinstance(item, dict)]
        if rows:
            mapped = [
                SearchResultItem(
                    id=str(item.get("id") or "").strip(),
                    thumbnail_url=str(item.get("thumbnail_url") or "").strip(),
                    description=(str(item.get("description") or "").strip() or "Search result"),
                )
                for item in rows
                if str(item.get("id") or "").strip()
            ]
            if mapped:
                return SearchResponse(results=mapped)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        try:
            payload = exc.response.json()
            if isinstance(payload, dict) and payload.get("detail") is not None:
                detail = str(payload.get("detail"))
        except Exception:
            pass
        raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"AI service error: {exc}") from exc

    return SearchResponse(results=[])


@app.get("/videos/{video_id}", response_model=VideoDetailResponse)
async def videos_by_id(video_id: str, request: Request, _auth: None = Depends(require_auth_header)) -> VideoDetailResponse:
    try:
        queue_payload = await _get_json(
            f"{settings.metadata_service_url}/api/v1/queue/videos",
            headers=_forward_auth_headers(request),
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc

    queue_items = queue_payload.get("items") if isinstance(queue_payload, dict) else []
    requested_id = str(video_id or "").strip()
    resolved_video_id = requested_id
    candidate_payload: dict[str, Any] | None = None
    queue_video = next(
        (item for item in queue_items if isinstance(item, dict) and str(item.get("video_id")) == resolved_video_id),
        None,
    )
    if queue_video is None and requested_id:
        try:
            candidate_raw = await _get_json(
                f"{settings.metadata_service_url}/api/v1/candidates/{requested_id}",
                headers=_forward_auth_headers(request),
            )
            if isinstance(candidate_raw, dict):
                candidate_payload = candidate_raw
                resolved_video_id = str(candidate_raw.get("video_id") or "").strip() or requested_id
                queue_video = next(
                    (item for item in queue_items if isinstance(item, dict) and str(item.get("video_id")) == resolved_video_id),
                    None,
                )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
        except httpx.HTTPError:
            pass
    if queue_video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    metadata_payload: dict[str, Any] = {}
    try:
        raw_metadata = await _get_json(
            f"{settings.metadata_service_url}/api/v1/queue/videos/{resolved_video_id}/metadata",
            headers=_forward_auth_headers(request),
        )
        if isinstance(raw_metadata, dict):
            metadata_payload = raw_metadata
    except httpx.HTTPError:
        metadata_payload = {}

    people = metadata_payload.get("people") if isinstance(metadata_payload, dict) else []
    people_items = [item for item in people if isinstance(item, dict)]

    video_url = str(queue_video.get("available_link_video") or "").strip()
    if not video_url:
        video_url = f"/api/v1/queue/videos/{resolved_video_id}/file"

    title_prefix = str(queue_video.get("title") or f"Video {resolved_video_id}").strip()
    segments: list[VideoSegmentItem] = []

    if candidate_payload is not None:
        segments.append(
            VideoSegmentItem(
                id=str(candidate_payload.get("candidate_id") or requested_id or resolved_video_id),
                video_url=video_url,
                title=f"{title_prefix} · Candidate",
                description=_build_segment_description(candidate_payload, fallback=f"Candidate from {title_prefix}"),
            )
        )

    if people_items:
        for index, person in enumerate(people_items[:10], start=1):
            segments.append(
                VideoSegmentItem(
                    id=str(person.get("candidate_id") or person.get("track_id") or f"{resolved_video_id}-seg-{index}"),
                    video_url=video_url,
                    title=f"{title_prefix} · Segment {index}",
                    description=_build_segment_description(person, fallback=f"Segment {index} from {title_prefix}"),
                )
            )
    else:
        for index in range(1, 6):
            segments.append(
                VideoSegmentItem(
                    id=f"{resolved_video_id}-seg-{index}",
                    video_url=video_url,
                    title=f"{title_prefix} · Segment {index}",
                    description=f"Segment {index} from {title_prefix}",
                )
            )

    return VideoDetailResponse(id=requested_id or resolved_video_id, segments=segments)


@app.get("/api/v1/overview")
async def overview() -> dict:
    try:
        metadata = await _get_json(f"{settings.metadata_service_url}/api/v1/overview")
        ai = await tracking_runtime_config(request_headers=None)
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


@app.post("/auth/login")
async def auth_login(payload: dict[str, Any]) -> dict:
    try:
        return await _post_json(f"{settings.metadata_service_url}/auth/login", payload)
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


@app.post("/users")
async def create_user(payload: dict[str, Any], request: Request, _auth: None = Depends(require_auth_header)) -> dict:
    try:
        return await _post_json(f"{settings.metadata_service_url}/users", payload, headers=_forward_auth_headers(request))
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc


@app.get("/users")
async def users(request: Request, _auth: None = Depends(require_auth_header)) -> dict:
    try:
        return await _get_json(f"{settings.metadata_service_url}/users", headers=_forward_auth_headers(request))
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc


@app.patch("/users/{user_id}")
async def patch_user(
    user_id: int,
    payload: dict[str, Any],
    request: Request,
    _auth: None = Depends(require_auth_header),
) -> dict:
    try:
        return await _patch_json(f"{settings.metadata_service_url}/users/{user_id}", payload, headers=_forward_auth_headers(request))
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
        ai_result = await ai_tracking_process(
            ai_payload,
            request_headers=_forward_auth_headers(request),
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
        return await ai_tracking_process(
            payload,
            request_headers=_forward_auth_headers(request),
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Tracking service error: {exc}") from exc


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


@app.post("/api/v1/queue/process-storage")
async def queue_process_storage(payload: dict[str, Any] | None = None) -> dict:
    try:
        return await _post_json(f"{settings.metadata_service_url}/api/v1/queue/process-storage", payload or {})
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Metadata service error: {exc}") from exc


@app.post("/api/v1/tracking/run")
async def tracking_run(payload: dict[str, Any], request: Request) -> dict:
    try:
        return await ai_tracking_run(
            payload,
            request_headers=_forward_auth_headers(request),
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Tracking service error: {exc}") from exc


@app.get("/api/v1/tracking-artifacts/{artifact_id}")
async def tracking_artifact_video(artifact_id: str) -> Response:
    try:
        content, content_type = await tracking_artifact_bytes(
            artifact_id,
            request_headers=None,
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
        return await ai_tracking_artifact_manifest(
            artifact_id,
            request_headers=None,
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
        return await _get_json(f"{settings.metadata_service_url}/api/v1/tracking-artifacts/{artifact_id}/manifest")
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Tracking service error: {exc}") from exc
