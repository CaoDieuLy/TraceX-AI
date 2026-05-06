"""Database connection helpers."""

from collections.abc import Generator
import os
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Build database URL from environment variables
def _build_database_url() -> str:
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "video_tracking")
    user = os.getenv("POSTGRES_USER", "mcpt_user")
    password = os.getenv("POSTGRES_PASSWORD", "Mcpt@2026Secure")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


_database_url = os.getenv("DATABASE_URL") or _build_database_url()
engine = create_engine(
    _database_url,
    future=True,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_session() -> Generator[Session, None, None]:
    """Dependency for FastAPI to get DB session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
