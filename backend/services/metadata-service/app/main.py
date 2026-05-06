"""FastAPI application for metadata-service.

Note: Candidate search/ranking/trace logic has been moved to:
- query-service (search, ranking, Vietnamese translation)
- trace-service (trace building, feedback)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routers import auth, users, videos

app = FastAPI(
    title="Metadata Service",
    description="Queue management, authentication, and video metadata for TraceX",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(videos.router, prefix="/api/v1/videos", tags=["videos"])


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "metadata-service"}


@app.get("/")
def root():
    return {
        "service": "metadata-service",
        "version": "1.0.0",
        "description": "Queue management, authentication, and video metadata",
    }
