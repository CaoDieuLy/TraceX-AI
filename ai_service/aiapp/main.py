"""
AI service: tìm kiếm nội bộ + proxy tracking/Lightning cho gateway.
Gateway không gọi trực tiếp TRACKING_SERVICE_URL.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.database import SessionLocal
from app.service import rank_candidates

from .http_client import close_http_client, init_http_client
from .runtime_contract import build_tracking_runtime_contract
from .tracking_upstream import proxy_get_bytes, proxy_get_json, proxy_post_json


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_http_client()
    yield
    await close_http_client()


app = FastAPI(title="MCPT AI Service", version="1.0.0", lifespan=lifespan)


class InternalSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=10, ge=1, le=50)
    offset: int = Field(default=0, ge=0)
    # Hard constraints (Phase 1)
    camera_ids: list[str] | None = Field(default=None, description="Zone filter: list of camera IDs")
    time_from: str | None = Field(default=None, description="ISO datetime lower bound")
    time_to: str | None = Field(default=None, description="ISO datetime upper bound")


class SearchResultItem(BaseModel):
    id: str
    thumbnail_url: str
    description: str


class InternalSearchResponse(BaseModel):
    results: list[SearchResultItem]


def _to_result_item(candidate: dict[str, Any]) -> SearchResultItem:
    """Chuan hoa candidate tu metadata/tracking thanh schema response on dinh cho gateway."""
    candidate_id = str(candidate.get("candidate_id") or candidate.get("id") or "").strip()
    video_id = str(candidate.get("video_id") or "").strip()
    result_id = candidate_id or video_id or "unknown"

    preview_url = str(candidate.get("preview_image_url") or "").strip()
    if not preview_url and candidate_id:
        preview_url = f"/api/v1/candidates/{candidate_id}/preview"

    description_parts = [
        str(candidate.get("appearance_summary") or "").strip(),
        str(candidate.get("search_text") or "").strip(),
        f"candidate={candidate_id}" if candidate_id else "",
        f"video={video_id}" if video_id else "",
        f"track={candidate.get('track_id')}" if candidate.get("track_id") else "",
    ]
    description = " · ".join([p for p in description_parts if p]) or "Candidate result"

    return SearchResultItem(id=result_id, thumbnail_url=preview_url or "", description=description)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ai_service"}


@app.post("/internal/search", response_model=InternalSearchResponse)
def internal_search(payload: InternalSearchRequest) -> InternalSearchResponse:
    """Diem vao tim kiem noi bo: goi rank_candidates (metadata + tracking) va cat trang ket qua."""
    target_limit = min(max(payload.offset + payload.top_k, payload.top_k), 50)
    ranked_limit = min(target_limit, 50)

    session = SessionLocal()
    try:
        ranked = rank_candidates(
            session=session,
            query_text=payload.query.strip(),
            limit=ranked_limit,
            camera_ids=payload.camera_ids,
            time_from=payload.time_from,
            time_to=payload.time_to,
        )
    except httpx.HTTPStatusError as exc:
        upstream_detail = (exc.response.text or "").strip()
        raise HTTPException(
            status_code=502,
            detail=f"Tracking upstream returned {exc.response.status_code}: {upstream_detail}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        session.close()

    if not isinstance(ranked, list):
        return InternalSearchResponse(results=[])

    start = payload.offset
    end = payload.offset + payload.top_k
    slice_items = [item for item in ranked[start:end] if isinstance(item, dict)]
    return InternalSearchResponse(results=[_to_result_item(item) for item in slice_items])


# --- Proxy nội bộ: gateway gọi, ai_service gọi tiếp tới Lightning/tracking ---


@app.get("/internal/tracking/v1/runtime-config")
async def internal_tracking_runtime_config(request: Request) -> Any:
    """Return the fixed deployed contract without depending on upstream support."""
    return build_tracking_runtime_contract()


@app.post("/internal/tracking/v1/ai/process")
async def internal_tracking_ai_process(payload: dict[str, Any], request: Request) -> Any:
    """Gateway uy quyen ai_service proxy request ai/process toi tracking upstream."""
    return await proxy_post_json("api/v1/ai/process", payload, request)


@app.post("/internal/tracking/v1/tracking/run")
async def internal_tracking_run(payload: dict[str, Any], request: Request) -> Any:
    """Gateway uy quyen ai_service proxy request tracking/run toi tracking upstream."""
    return await proxy_post_json("api/v1/tracking/run", payload, request)


@app.get("/internal/tracking/v1/artifacts/{artifact_id}")
async def internal_tracking_artifact(artifact_id: str, request: Request) -> Response:
    """Tai artifact nhi phan (video/zip/...) tu tracking upstream va tra nguoc ve gateway."""
    content, media_type = await proxy_get_bytes(f"api/v1/tracking-artifacts/{artifact_id}", request)
    return Response(content=content, media_type=media_type)


@app.get("/internal/tracking/v1/artifacts/{artifact_id}/manifest")
async def internal_tracking_artifact_manifest(artifact_id: str, request: Request) -> Any:
    """Lay metadata manifest cua artifact tracking de frontend/backend doc thong tin bo sung."""
    return await proxy_get_json(f"api/v1/tracking-artifacts/{artifact_id}/manifest", request)
