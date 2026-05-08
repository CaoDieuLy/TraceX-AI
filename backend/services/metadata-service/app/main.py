"""FastAPI application for metadata-service.

Handles queue management, authentication, video metadata, and AI video processing.
GPU models (Grounding DINO 1.6, EVA-02, SigLIP 2, VideoMAE V2) are loaded
at startup and used for the /api/v1/video/process endpoint.

Search/ranking is forwarded to query-service (GPU).
Trace building is forwarded to trace-service (Neural Video Reconstruction).
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Set HF token before any model downloads (non-fatal if missing)
os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
os.environ.pop("TRANSFORMERS_OFFLINE", None)

from .api.routers import auth, search, users, videos, ingest
from .api.routers.finetune import router as finetune_router
from .api.routers.video_process import router as video_process_router
from .config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Import here to avoid circular imports
    from .database import Base, SessionLocal, engine
    from .services.model_warmup import warmup_models
    from .services.user_service import ensure_bootstrap_admin

    # 1. Create all tables (idempotent)
    # Import shared Base which has all models registered
    from shared.models import Base as SharedBase
    logger.info("Creating database tables if they don't exist...")
    SharedBase.metadata.create_all(bind=engine)
    logger.info("Database tables ready.")

    # 2. Bootstrap admin user
    if settings.bootstrap_admin_email and settings.bootstrap_admin_password:
        logger.info("Ensuring bootstrap admin user: %s", settings.bootstrap_admin_email)
        session = SessionLocal()
        try:
            ensure_bootstrap_admin(
                session,
                email=settings.bootstrap_admin_email,
                password=settings.bootstrap_admin_password,
                full_name=settings.bootstrap_admin_full_name,
            )
            logger.info("Bootstrap admin user ready.")
        except Exception as exc:
            logger.error("Failed to bootstrap admin user: %s", exc)
        finally:
            session.close()
    else:
        logger.warning("BOOTSTRAP_ADMIN_EMAIL or BOOTSTRAP_ADMIN_PASSWORD not set — skipping admin bootstrap.")

    # 3. Warmup GPU models (Grounding DINO, EVA-02, SigLIP 2, VideoMAE V2)
    logger.info("Starting GPU model warmup...")
    await warmup_models()
    logger.info("GPU models ready.")

    yield

    logger.info("Shutting down metadata-service...")


app = FastAPI(
    title="Metadata Service",
    description="Queue management, authentication, and video metadata for TraceX",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tracex-ai.smartnovi.tech",
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(videos.router, prefix="/api/v1/videos", tags=["videos"])
app.include_router(video_process_router, prefix="/api/v1/video", tags=["video"])
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["ingest"])
app.include_router(finetune_router, prefix="/api/v1/finetune", tags=["finetune"])

@app.get("/health")
def health_check():
    from .services.model_warmup import get_loaded_models, is_warmup_done
    return {
        "status": "healthy",
        "service": "metadata-service",
        "gpu_warmup_done": is_warmup_done(),
        "loaded_models": get_loaded_models(),
    }


@app.get("/")
def root():
    return {
        "service": "metadata-service",
        "version": "2.0.0",
        "description": "Queue management, authentication, video metadata + SOTA AI processing",
        "gpu_models": ["Grounding DINO 1.6", "EVA-02 ViT-L/14", "SigLIP 2", "VideoMAE V2"],
    }


@app.get("/debug/routes")
def debug_routes():
    """List all registered routes for debugging."""
    routes = []
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            routes.append({"path": route.path, "methods": list(route.methods)})
    return {"routes": routes, "total": len(routes)}


@app.get("/debug/config")
def debug_config():
    """Show current runtime config for debugging."""
    from .config import settings
    return {
        "bootstrap_admin_email": settings.bootstrap_admin_email or "(not set)",
        "bootstrap_admin_password_set": bool(settings.bootstrap_admin_password),
        "postgres_host": settings.postgres_host or "(not set)",
        "postgres_db": settings.postgres_db or "(not set)",
    }
