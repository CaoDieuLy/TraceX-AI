import json
import logging
import queue as _queue_module
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import asyncio
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
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
    get_ingestion_background_status,
    process_video_ingestion_background,
    process_video_query,
    resolve_tracking_artifact_paths,
    run_tracking,
    search_candidates_remote,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Warmup state
# ---------------------------------------------------------------------------
_warmup_task: asyncio.Task | None = None
_warmup_state: dict[str, object] = {
    "enabled": bool(settings.startup_warmup_enabled),
    "status": "pending",
    "ready": False,
    "started_at": None,
    "finished_at": None,
    "last_error": None,
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _set_warmup_state(**updates: object) -> None:
    _warmup_state.update(updates)


def _warmup_snapshot() -> dict[str, object]:
    return dict(_warmup_state)


def _warmup_models() -> None:
    from .model_adapters import SigLIP2ModelHub, TransReIDHub, VideoMAEHub

    logger.info("[warmup] Pre-loading AI models into GPU memory...")

    try:
        hub = SigLIP2ModelHub()
        hub._ensure_loaded()
        logger.info("[warmup] SigLIP2 ViT-L-16-512 ready")
    except Exception as exc:
        logger.error("[warmup] SigLIP2 failed: %s", exc)

    try:
        hub = TransReIDHub()
        hub._ensure_loaded()
        logger.info("[warmup] TransReID ViT-Base ready")
    except Exception as exc:
        logger.error("[warmup] TransReID failed: %s", exc)

    try:
        hub = VideoMAEHub()
        hub._ensure_loaded()
        logger.info("[warmup] VideoMAE Large ready")
    except Exception as exc:
        logger.error("[warmup] VideoMAE failed: %s", exc)

    try:
        from .local_ingestion_pipeline import RFDETRPersonDetector
        det = RFDETRPersonDetector()
        det._ensure_loaded()
        logger.info("[warmup] RF-DETR 2XLarge ready")
    except Exception as exc:
        logger.error("[warmup] RF-DETR failed: %s", exc)

    logger.info("[warmup] All models pre-loaded. Service ready for requests.")


def _run_warmup_models() -> None:
    _set_warmup_state(status="running", ready=False, started_at=_utcnow_iso(), finished_at=None, last_error=None)
    try:
        _warmup_models()
    except Exception as exc:
        logger.exception("[warmup] Background warmup failed")
        _set_warmup_state(status="failed", ready=False, finished_at=_utcnow_iso(), last_error=str(exc))
        return
    _set_warmup_state(status="completed", ready=True, finished_at=_utcnow_iso(), last_error=None)


# ---------------------------------------------------------------------------
# Async ingestion job queue
# ---------------------------------------------------------------------------
_job_store: dict[str, dict] = {}
_job_store_lock = threading.Lock()
_ingestion_queue: _queue_module.Queue = _queue_module.Queue(maxsize=32)
_active_ingestion_jobs: dict[str, str] = {}
_worker_thread: threading.Thread | None = None


def _ingestion_source_key(*, camera_id: str | None, source_filename: str | None) -> str:
    camera = str(camera_id or "").strip().lower()
    filename = str(source_filename or "").strip().lower()
    return f"{camera}::{filename}"


def _ingestion_worker() -> None:
    """Background thread: drains _ingestion_queue one job at a time."""
    while True:
        job_id, payload_dict = _ingestion_queue.get()
        # Wait for warmup before starting inference
        while not _warmup_state.get("ready") and _warmup_state.get("status") not in ("disabled", "failed"):
            time.sleep(2)
        with _job_store_lock:
            _job_store[job_id]["status"] = "processing"
            _job_store[job_id]["started_at"] = _utcnow_iso()
        logger.info("[worker] Processing job %s source_filename=%s", job_id, payload_dict.get("source_filename"))
        try:
            result = process_video_ingestion(payload_dict)
            with _job_store_lock:
                _job_store[job_id].update({
                    "status": "done",
                    "result": result,
                    "finished_at": _utcnow_iso(),
                })
            logger.info("[worker] Job %s done people=%s", job_id, result.get("person_count"))
        except Exception as exc:
            logger.exception("[worker] Job %s failed: %s", job_id, exc)
            with _job_store_lock:
                _job_store[job_id].update({
                    "status": "failed",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "finished_at": _utcnow_iso(),
                })
        finally:
            with _job_store_lock:
                source_key = str(_job_store.get(job_id, {}).get("source_key") or "")
                if source_key and _active_ingestion_jobs.get(source_key) == job_id:
                    _active_ingestion_jobs.pop(source_key, None)
            _ingestion_queue.task_done()


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _warmup_task, _worker_thread
    _worker_thread = threading.Thread(target=_ingestion_worker, daemon=True, name="ingestion-worker")
    _worker_thread.start()
    logger.info("[startup] Ingestion worker thread started")
    if settings.startup_warmup_enabled:
        _warmup_task = asyncio.create_task(asyncio.to_thread(_run_warmup_models))
    else:
        _set_warmup_state(status="disabled", ready=False)
    yield
    if _warmup_task is not None and not _warmup_task.done():
        _warmup_task.cancel()


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
    warmup = _warmup_snapshot()
    return {
        "status": "ok",
        "service": "tracking-service",
        "ready": bool(warmup.get("ready")),
        "warmup": warmup,
    }


@app.get("/api/v1/runtime-config")
def runtime_config() -> dict:
    config = get_runtime_config()
    config["warmup"] = _warmup_snapshot()
    return config


@app.get("/api/v1/runtime-config/hardware")
def runtime_config_hardware() -> dict:
    config = get_runtime_config()
    return {
        "detected_hardware": config.get("detected_hardware"),
        "execution_plan": config.get("execution_plan"),
        "warmup": _warmup_snapshot(),
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


<<<<<<< HEAD
@app.post("/api/v1/ingestion/process")
def ingestion_process(payload: VideoIngestionRequest) -> dict:
    """Enqueue a video ingestion job. Returns immediately with job_id."""
    source_key = _ingestion_source_key(
        camera_id=payload.camera_id,
        source_filename=payload.source_filename,
    )

    with _job_store_lock:
        existing_job_id = _active_ingestion_jobs.get(source_key)
        if existing_job_id:
            existing_job = _job_store.get(existing_job_id)
            if existing_job and existing_job.get("status") in {"queued", "processing"}:
                logger.info(
                    "[ingestion] Reusing active job job_id=%s source_filename=%s camera_id=%s status=%s",
                    existing_job_id,
                    payload.source_filename,
                    payload.camera_id,
                    existing_job.get("status"),
                )
                return {
                    "job_id": existing_job_id,
                    "status": existing_job.get("status"),
                    "source_filename": payload.source_filename,
                    "deduplicated": True,
                }
            _active_ingestion_jobs.pop(source_key, None)

    job_id = str(uuid.uuid4())
    with _job_store_lock:
        _job_store[job_id] = {
            "status": "queued",
            "queued_at": _utcnow_iso(),
            "source_filename": payload.source_filename,
            "camera_id": payload.camera_id,
            "source_key": source_key,
        }
        _active_ingestion_jobs[source_key] = job_id
    try:
        _ingestion_queue.put_nowait((job_id, payload.model_dump()))
    except _queue_module.Full:
        with _job_store_lock:
            _active_ingestion_jobs.pop(source_key, None)
            del _job_store[job_id]
        raise HTTPException(status_code=503, detail="Ingestion queue full, try again later")
    logger.info("[ingestion] Job queued job_id=%s source_filename=%s", job_id, payload.source_filename)
    return {"job_id": job_id, "status": "queued", "source_filename": payload.source_filename}


@app.get("/api/v1/ingestion/status/{job_id}")
def ingestion_job_status(job_id: str) -> dict:
    """Poll ingestion job status. Returns result when status=='done'."""
    with _job_store_lock:
        job = _job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Ingestion job not found: {job_id}")
    return job
=======
@app.post("/api/v1/ingestion/process", response_model=VideoIngestionResponse)
def ingestion_process(payload: VideoIngestionRequest, background_tasks: BackgroundTasks) -> dict:
    try:
        job_id = uuid4().hex
        background_tasks.add_task(
            process_video_ingestion_background,
            job_id,
            payload.model_dump(),
        )
        return {
            "status": "accepted",
            "message": "Video is being processed in background",
            "job_id": job_id,
            "source_path": payload.source_url,
            "video": {},
            "people": [],
            "person_count": 0,
        }
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
>>>>>>> f76bfdd62a7f681d71b6f236730c4f1afa3e2504


@app.get("/api/v1/ingestion/jobs/{job_id}")
def ingestion_job_status(job_id: str) -> dict:
    return get_ingestion_background_status(job_id)


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
