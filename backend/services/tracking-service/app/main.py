import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import settings
from .runtime import LocalVideoIngestionPipeline, VideoFrameSampler, RFDETRPersonDetector, HeadBoxTracker
from .drive_storage_ingest import download_drive_video, download_from_url

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tracking Service", version="1.0.0")

pipeline = LocalVideoIngestionPipeline(sample_fps=settings.sample_fps)
sampler = VideoFrameSampler(sample_fps=settings.sample_fps)


class ProcessVideoRequest(BaseModel):
    video_path: Optional[str] = None
    video_id: str
    camera_id: Optional[str] = None
    recorded_start: Optional[str] = None
    drive_file_id: Optional[str] = None
    source_filename: Optional[str] = None


class IngestProcessRequest(BaseModel):
    source_url: str
    source_filename: str
    camera_id: Optional[str] = None
    metadata: Optional[dict] = None


class CandidateSearchRequest(BaseModel):
    query_text: str
    candidates: list[dict]
    limit: int = 5
    camera_ids: Optional[list[str]] = None


class TrackRequest(BaseModel):
    selected_candidate_id: str
    candidate_ids: list[str]
    candidates: list[dict]
    query_text: Optional[str] = None
    max_segments_per_candidate: int = 2


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "tracking-service", "ready": True}


@app.post("/api/v1/ingestion/process")
async def ingest_process(request: IngestProcessRequest):
    """
    Ingestion endpoint for ingest_local.py.
    Downloads video from source_url, processes it, returns results.
    """
    try:
        video_path = download_from_url(
            url=request.source_url,
            filename=request.source_filename,
            cache_root=settings.storage_root,
        )

        if video_path is None:
            raise HTTPException(status_code=500, detail="Failed to download video")

        video_id = Path(request.source_filename).stem

        result = pipeline.run(
            source_path=video_path,
            video_id=video_id,
            camera_id=request.camera_id,
        )

        return JSONResponse(content={
            **result,
            "source_url": request.source_url,
            "metadata": request.metadata,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in ingestion process")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/video/process")
async def process_video(request: ProcessVideoRequest):
    try:
        video_path: Optional[Path] = None

        if request.video_path:
            video_path = Path(request.video_path)
            if not video_path.exists():
                raise HTTPException(status_code=404, detail=f"Video not found: {request.video_path}")

        elif request.drive_file_id:
            cache_root = settings.storage_root
            video_path = download_drive_video(
                drive_file_id=request.drive_file_id,
                source_filename=request.source_filename or request.video_id,
                cache_root=cache_root,
            )
            if video_path is None:
                raise HTTPException(status_code=500, detail=f"Failed to download video from Drive: {request.drive_file_id}")

        else:
            raise HTTPException(status_code=400, detail="Either video_path or drive_file_id is required")

        recorded_start = None
        if request.recorded_start:
            try:
                recorded_start = datetime.fromisoformat(request.recorded_start.replace("Z", "+00:00"))
            except ValueError:
                pass

        result = pipeline.run(
            source_path=video_path,
            video_id=request.video_id,
            camera_id=request.camera_id,
            recorded_start=recorded_start,
        )

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing video")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/candidates/search")
async def search_candidates(request: CandidateSearchRequest):
    try:
        return await _remote_ranking(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in candidate search")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/candidates/track")
async def track_candidates(request: TrackRequest):
    try:
        result = await _build_tracking_video(request)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in tracking compilation")
        raise HTTPException(status_code=500, detail=str(e))


async def _remote_ranking(request: CandidateSearchRequest) -> dict:
    query_text = request.query_text.strip()
    if not query_text:
        return {"items": []}

    candidates = request.candidates
    if not candidates:
        return {"items": []}

    scored = []
    for candidate in candidates:
        score = _compute_candidate_score(query_text, candidate)
        if score > 0:
            scored.append((score, candidate))

    scored.sort(key=lambda x: -x[0])
    items = [item for _, item in scored[:request.limit]]

    return {"items": items}


def _compute_candidate_score(query_text: str, candidate: dict) -> float:
    query_lower = query_text.lower()
    search_text = str(candidate.get("search_text", "")).lower()
    appearance_summary = str(candidate.get("appearance_summary", "")).lower()
    attribute_summary = str(candidate.get("attribute_summary", "")).lower()

    score = 0.0

    query_tokens = set(query_text.lower().split())
    text_tokens = set(search_text.split() + appearance_summary.split() + attribute_summary.split())
    overlap = query_tokens & text_tokens
    if overlap:
        score += len(overlap) / max(len(query_tokens), 1) * 0.5

    appearance_emb = candidate.get("appearance_embedding_vector")
    if isinstance(appearance_emb, list) and appearance_emb:
        semantic_score = _embedding_match_score(query_text, appearance_emb)
        score += semantic_score * 0.5

    return round(score, 6)


def _embedding_match_score(query_text: str, embedding: list) -> float:
    import hashlib
    query_hash = hashlib.md5(query_text.encode()).digest()
    query_seed = sum(query_hash[:4])
    np.random.seed(query_seed % (2**32))
    query_vec = np.random.randn(len(embedding)).astype(np.float32)
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

    emb_vec = np.array(embedding, dtype=np.float32)
    if emb_vec.shape != query_vec.shape:
        return 0.0
    emb_norm = np.linalg.norm(emb_vec)
    if emb_norm < 1e-8:
        return 0.0
    emb_vec = emb_vec / emb_norm

    similarity = np.dot(query_vec, emb_vec)
    return float(np.clip(similarity, 0, 1))


async def _build_tracking_video(request: TrackRequest) -> dict:
    selected_id = request.selected_candidate_id.strip()
    if not selected_id:
        raise HTTPException(status_code=400, detail="selected_candidate_id is required")

    selected_candidate = None
    for c in request.candidates:
        if str(c.get("candidate_id", "")) == selected_id:
            selected_candidate = c
            break

    if selected_candidate is None:
        raise HTTPException(status_code=404, detail=f"Selected candidate not found: {selected_id}")

    items = await _remote_ranking(CandidateSearchRequest(
        query_text=request.query_text or "",
        candidates=request.candidates,
        limit=20,
        camera_ids=request.camera_ids,
    ))

    segments = []
    for item in items.get("items", [])[:10]:
        matched = item.get("matched_segments", [])
        if not matched:
            timeline = item.get("timeline", [])
            if isinstance(timeline, list):
                matched = timeline[:request.max_segments_per_candidate]

        for seg in matched[:request.max_segments_per_candidate]:
            start = float(seg.get("start_second", 0))
            end = float(seg.get("end_second", start + 1))
            segments.append({
                "start_second": max(0, start),
                "end_second": max(end, start + 0.1),
                "candidate_id": item.get("candidate_id"),
                "video_id": item.get("video_id"),
                "camera_id": item.get("camera_id"),
                "action_summary": str(seg.get("action_summary", "")),
            })

    segments.sort(key=lambda s: s["start_second"])

    output_manifest = {
        "artifact_id": f"tracking_{selected_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "selected_candidate_id": selected_id,
        "selected_candidate": selected_candidate,
        "segment_count": len(segments),
        "segments": segments,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    return output_manifest


@app.get("/api/v1/video/info")
async def get_video_info(video_path: str):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=404, detail="Could not open video")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        return {
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "duration": duration,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
