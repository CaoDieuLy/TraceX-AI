import json
import logging
from contextlib import asynccontextmanager
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
    resolve_tracking_artifact_paths,
    run_tracking,
    search_candidates_remote,
)

logger = logging.getLogger(__name__)


def _warmup_models() -> None:
    """
    Pre-load tất cả AI models vào GPU memory khi service start.
    Đảm bảo latency thấp cho request đầu tiên của user.
    Models: RF-DETR 2XLarge, TransReID ViT-Base, VideoMAE Large, SigLIP2 ViT-L-16-512.
    """
    from .model_adapters import SigLIP2ModelHub, TransReIDHub, VideoMAEHub

    logger.info("[warmup] Pre-loading AI models into GPU memory...")

    # SigLIP2 — dùng cho attribute zero-shot + action embedding + query encoding
    try:
        hub = SigLIP2ModelHub()
        hub._ensure_loaded()
        logger.info("[warmup] SigLIP2 ViT-L-16-512 ready")
    except Exception as exc:
        logger.error("[warmup] SigLIP2 failed: %s", exc)

    # TransReID — dùng cho appearance Re-ID embedding
    try:
        hub = TransReIDHub()
        hub._ensure_loaded()
        logger.info("[warmup] TransReID ViT-Base ready")
    except Exception as exc:
        logger.error("[warmup] TransReID failed: %s", exc)

    # VideoMAE — dùng cho action recognition
    try:
        hub = VideoMAEHub()
        hub._ensure_loaded()
        logger.info("[warmup] VideoMAE Large ready")
    except Exception as exc:
        logger.error("[warmup] VideoMAE failed: %s", exc)

    # RF-DETR — dùng cho person detection (heaviest model, load cuối)
    try:
        from .local_ingestion_pipeline import RFDETRPersonDetector
        det = RFDETRPersonDetector()
        det._ensure_loaded()
        logger.info("[warmup] RF-DETR 2XLarge ready")
    except Exception as exc:
        logger.error("[warmup] RF-DETR failed: %s", exc)

    logger.info("[warmup] All models pre-loaded. Service ready for requests.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _warmup_models)
    yield


app = FastAPI(title="MCPT Tracking Service", version="2.0.0", lifespan=lifespan)


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
    raise HTTPException(
        status_code=501,
        detail="AI worker upload endpoint not supported in strict pipeline mode. Use /api/v1/ingestion/process with source_url instead.",
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
    return search_candidates_remote(
        payload.query_text,
        payload.candidates,
        payload.limit,
        camera_ids=payload.camera_ids,
        time_from=payload.time_from,
        time_to=payload.time_to,
    )


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
