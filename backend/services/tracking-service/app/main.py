import json
import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .schemas import (
    AiProcessRequest,
    AiProcessResponse,
    CandidateSearchRequest,
    CandidateSearchResponse,
    CandidateTrackRequest,
    CandidateTrackResponse,
    TrackingRequest,
    TrackingResponse,
    VideoIngestionRequest,
    VideoIngestionResponse,
)
from .config import settings
from .service import (
    build_tracking_video_remote,
    get_runtime_config,
    process_video_ingestion,
    process_video_query,
    process_video_query_worker,
    resolve_tracking_artifact_paths,
    run_tracking,
    search_candidates_remote,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="MCPT Tracking Service", version="2.0.0")


@app.get("/")
def root() -> dict:
    config = get_runtime_config()
    return {
        "status": "ok",
        "service": "tracking-service",
        "provider": config.get("provider"),
        "mode": config.get("mode"),
    }


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok", "service": "tracking-service"}


@app.get("/api/v1/runtime-config")
def runtime_config() -> dict:
    return get_runtime_config()


@app.get("/api/v1/runtime-config/hardware")
def runtime_config_hardware() -> dict:
    config = get_runtime_config()
    return {
        "detected_hardware": config.get("detected_hardware"),
        "execution_plan": config.get("execution_plan"),
    }


@app.post("/api/v1/ai/process", response_model=AiProcessResponse)
def ai_process(payload: AiProcessRequest) -> dict:
    return process_video_query(payload.model_dump())


@app.post("/api/v1/ai/worker")
async def ai_worker(
    file: UploadFile = File(...),
    query_id: str | None = Form(default=None),
    video_id: str = Form(...),
    query_text: str = Form(default=""),
    video_title: str | None = Form(default=None),
    metadata: str | None = Form(default=None),
    detected_hardware: str | None = Form(default=None),
    execution_plan: str | None = Form(default=None),
    acceleration_state: str | None = Form(default=None),
) -> dict:
    worker_input_dir = Path(settings.ingestion_work_root) / "remote-ai-inputs"
    worker_input_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "").suffix or ".bin"
    local_input_path = worker_input_dir / f"{(video_id or 'query-video').strip() or 'query-video'}-{query_id or 'worker'}{suffix}"
    local_input_path.write_bytes(await file.read())
    return process_video_query_worker(
        {
            "query_id": query_id,
            "video_id": video_id,
            "video_title": video_title,
            "query_text": query_text,
            "source_path": str(local_input_path),
            "metadata": metadata,
            "detected_hardware": detected_hardware,
            "execution_plan": execution_plan,
            "acceleration_state": acceleration_state,
        }
    )


@app.post("/api/v1/tracking/run", response_model=TrackingResponse)
def tracking_run(payload: TrackingRequest) -> dict:
    return run_tracking(payload.candidate_info)


@app.post("/api/v1/ingestion/process", response_model=VideoIngestionResponse)
def ingestion_process(payload: VideoIngestionRequest) -> dict:
    try:
        return process_video_ingestion(payload.model_dump())
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Video ingestion failed for source_filename=%s camera_id=%s",
            payload.source_filename,
            payload.camera_id,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": type(exc).__name__,
                "message": str(exc),
                "source_filename": payload.source_filename,
                "camera_id": payload.camera_id,
            },
        ) from exc


@app.post("/api/v1/candidates/search", response_model=CandidateSearchResponse)
def candidate_search(payload: CandidateSearchRequest) -> dict:
    return search_candidates_remote(payload.query_text, payload.candidates, payload.limit)


@app.post("/api/v1/candidates/track", response_model=CandidateTrackResponse)
def candidate_track(payload: CandidateTrackRequest) -> dict:
    return build_tracking_video_remote(
        selected_candidate_id=payload.selected_candidate_id,
        candidates=payload.candidates,
        candidate_ids=payload.candidate_ids,
        query_text=payload.query_text,
        max_segments_per_candidate=payload.max_segments_per_candidate,
    )


@app.get("/api/v1/tracking-artifacts/{artifact_id}")
def tracking_artifact_video(artifact_id: str) -> FileResponse:
    video_path, _manifest_path = resolve_tracking_artifact_paths(artifact_id)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Tracking artifact not found: {artifact_id}")
    return FileResponse(video_path, media_type="video/mp4", filename=video_path.name)


@app.get("/api/v1/tracking-artifacts/{artifact_id}/manifest")
def tracking_artifact_manifest(artifact_id: str) -> JSONResponse:
    _video_path, manifest_path = resolve_tracking_artifact_paths(artifact_id)
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"Tracking artifact manifest not found: {artifact_id}")
    return JSONResponse(content=json.loads(manifest_path.read_text(encoding="utf-8")))
