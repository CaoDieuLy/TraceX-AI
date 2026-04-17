from fastapi import FastAPI

from .schemas import TrackingRequest, TrackingResponse
from .service import get_pipeline_config, run_tracking

app = FastAPI(title="MCPT Tracking Service", version="1.0.0")


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok", "service": "tracking-service"}


@app.get("/api/v1/pipeline/config")
def pipeline_config() -> dict:
    return get_pipeline_config()


@app.post("/api/v1/tracking/run", response_model=TrackingResponse)
def tracking_run(payload: TrackingRequest) -> dict:
    return run_tracking(payload.candidate_info)
