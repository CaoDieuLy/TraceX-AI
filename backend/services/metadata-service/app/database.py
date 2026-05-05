from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import _build_default_database_url, settings

_database_url = settings.database_url or _build_default_database_url()
if not _database_url:
    raise RuntimeError("database_url is empty and fallback also failed")

engine = create_engine(_database_url, future=True, pool_pre_ping=True, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
