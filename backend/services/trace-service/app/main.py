"""FastAPI application for trace-service.

Responsibilities:
- Select candidate as the target for trace
- Build trace (24h window) from selected candidate's tracklets
- Merge video segments into complete trace video
- Handle trace feedback/verification
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routers import trace

app = FastAPI(
    title="Trace Service",
    description="Build traces from selected candidates for TraceX",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(trace.router, prefix="/api/v1/trace", tags=["trace"])


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "trace-service"}


@app.get("/")
def root():
    return {
        "service": "trace-service",
        "version": "1.0.0",
        "description": "Build traces from selected candidates",
    }
