from fastapi import FastAPI

from .schemas import AiProcessRequest, AiProcessResponse, TrackingRequest, TrackingResponse
from .service import get_pipeline_config, process_video_query, run_tracking

app = FastAPI(title="MCPT Tracking Service", version="2.0.0")


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok", "service": "tracking-service"}


@app.get("/api/v1/pipeline/config")
def pipeline_config() -> dict:
    return get_pipeline_config()


@app.post("/api/v1/ai/process", response_model=AiProcessResponse)
def ai_process(payload: AiProcessRequest) -> dict:
    return process_video_query(payload.model_dump())


@app.post("/api/v1/tracking/run", response_model=TrackingResponse)
def tracking_run(payload: TrackingRequest) -> dict:
    return run_tracking(payload.candidate_info)
