"""FastAPI application for metadata-service.

Note: Candidate search/ranking/trace logic has been moved to:
- query-service (search, ranking, Vietnamese translation)
- trace-service (trace building, feedback)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routers import auth, users, videos
from .config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Import here to avoid circular imports
    from .database import Base, SessionLocal, engine
    from .services.user_service import ensure_bootstrap_admin

    # Create all tables (idempotent — only creates if missing)
    logger.info("Creating database tables if they don't exist...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready.")

    # Bootstrap admin user from environment variables
    if settings.bootstrap_admin_email and settings.bootstrap_admin_password:
        logger.info("Ensuring bootstrap admin user exists: %s", settings.bootstrap_admin_email)
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/v1/auth", tags=["auth"])
app.include_router(users.router, prefix="/v1/users", tags=["users"])
app.include_router(videos.router, prefix="/v1/videos", tags=["videos"])


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


@app.get("/debug/routes")
def debug_routes():
    """List all registered routes for debugging."""
    routes = []
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            routes.append({"path": route.path, "methods": list(route.methods)})
    return {"routes": routes, "total": len(routes)}
