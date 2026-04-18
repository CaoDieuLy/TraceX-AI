from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile

from .schemas import (
    AiProcessRequest,
    AiProcessResponse,
    TrackingRequest,
    TrackingResponse,
    VideoIngestionRequest,
    VideoIngestionResponse,
)
from .config import settings
from .service import get_pipeline_config, process_video_ingestion, process_video_query, process_video_query_worker, run_tracking

app = FastAPI(title="MCPT Tracking Service", version="2.0.0")


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok", "service": "tracking-service"}


@app.get("/api/v1/pipeline/config")
def pipeline_config() -> dict:
    return get_pipeline_config()


@app.get("/api/v1/pipeline/hardware")
def pipeline_hardware() -> dict:
    config = get_pipeline_config()
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
    pipeline_profile: str | None = Form(default=None),
    hyperparameters: str | None = Form(default=None),
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
            "pipeline_profile": pipeline_profile,
            "hyperparameters": hyperparameters,
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
