import json
from datetime import datetime
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
        "gpu_hardware_profile": config.get("gpu_hardware_profile"),
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
    gpu_hardware_profile: str | None = Form(default=None),
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
            "gpu_hardware_profile": gpu_hardware_profile,
            "execution_plan": execution_plan,
            "acceleration_state": acceleration_state,
        }
    )


@app.post("/api/v1/tracking/run", response_model=TrackingResponse)
def tracking_run(payload: TrackingRequest) -> dict:
    return run_tracking(payload.candidate_info)


@app.post("/api/v1/ingestion/process", response_model=VideoIngestionResponse)
def ingestion_process(payload: VideoIngestionRequest) -> dict:
    return process_video_ingestion(payload.model_dump())


@app.post("/api/v1/ingestion/upload", response_model=VideoIngestionResponse)
async def ingestion_upload(
    file: UploadFile = File(...),
    source_filename: str | None = Form(default=None),
    camera_id: str | None = Form(default=None),
    recorded_start: str | None = Form(default=None),
    output_basename: str | None = Form(default=None),
    metadata: str | None = Form(default=None),
) -> dict:
    upload_root = Path(settings.ingestion_work_root) / "uploaded-ingestion-inputs"
    upload_root.mkdir(parents=True, exist_ok=True)
    filename = source_filename or file.filename or "upload.mp4"
    local_input_path = upload_root / filename
    local_input_path.write_bytes(await file.read())

    parsed_recorded_start = None
    if recorded_start:
        normalized = recorded_start.strip().replace("Z", "+00:00")
        parsed_recorded_start = datetime.fromisoformat(normalized)

    parsed_metadata = {}
    if metadata:
        try:
            payload = json.loads(metadata)
        except ValueError:
            payload = {}
        if isinstance(payload, dict):
            parsed_metadata = payload

    return process_video_ingestion(
        {
            "source_path": str(local_input_path),
            "source_filename": filename,
            "camera_id": camera_id,
            "recorded_start": parsed_recorded_start,
            "output_basename": output_basename,
            "metadata": parsed_metadata,
        }
    )


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
