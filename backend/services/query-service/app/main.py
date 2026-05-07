"""FastAPI application for query-service.

Responsibilities:
- Search candidates with hybrid ranking (text + vector)
- SeamlessM4T v2-large (Vietnamese ↔ English translation)
- Forward GPU re-ranking to trace-service (SigLIP 2 text tower)
- Forward shortlist to trace-service for EVA-02 cosine re-ranking
"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routers import candidates

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Query service starting — SeamlessM4T v2-large warmup...")

    # SeamlessM4T v2-large translation model
    try:
        from .services.translation import warmup as warmup_translation
        warmup_translation()
        logger.info("SeamlessM4T v2-large ready")
    except Exception as exc:
        logger.warning("SeamlessM4T warmup skipped (non-fatal): %s", exc)

    logger.info("Query service warmup complete")
    yield
    logger.info("Query service shutting down...")


app = FastAPI(
    title="Query Service",
    description="Candidate search, ranking, and SeamlessM4T translation for TraceX-AI",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(candidates.router, prefix="/api/v1", tags=["candidates"])


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "query-service", "version": "2.0.0"}


@app.get("/")
def root():
    return {
        "service": "query-service",
        "version": "2.0.0",
        "description": "Candidate search + SeamlessM4T v2 translation + GPU re-ranking",
        "translation_model": "SeamlessM4T v2-large",
        "ranking_service": "trace-service (SigLIP 2 text tower + EVA-02 cosine)",
    }
