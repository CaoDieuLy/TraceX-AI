from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("email", name="uq_users_email"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    videos: Mapped[list["VideoAsset"]] = relationship(back_populates="owner", cascade="all, delete-orphan")
    queries: Mapped[list["VideoQuery"]] = relationship(back_populates="owner", cascade="all, delete-orphan")


class VideoAsset(Base):
    __tablename__ = "video_assets"
    __table_args__ = (UniqueConstraint("video_id", name="uq_video_assets_video_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    video_id: Mapped[str] = mapped_column(String(36), default=lambda: str(uuid.uuid4()), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    storage_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    storage_backend: Mapped[str] = mapped_column(String(64), nullable=False, default="local_volume")
    source_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    owner: Mapped[User] = relationship(back_populates="videos")
    queries: Mapped[list["VideoQuery"]] = relationship(back_populates="video", cascade="all, delete-orphan")


class VideoQuery(Base):
    __tablename__ = "video_queries"
    __table_args__ = (UniqueConstraint("query_id", name="uq_video_queries_query_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    query_id: Mapped[str] = mapped_column(String(36), default=lambda: str(uuid.uuid4()), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("video_assets.id", ondelete="CASCADE"), nullable=False, index=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="queued")
    ai_job_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    ai_response: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    owner: Mapped[User] = relationship(back_populates="queries")
    video: Mapped[VideoAsset] = relationship(back_populates="queries")


class PersonCandidate(Base):
    __tablename__ = "person_candidates"
    __table_args__ = (UniqueConstraint("candidate_id", name="uq_person_candidates_candidate_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    candidate_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    camera_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    video_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    track_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    human_key: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    frame_idx: Mapped[int | None] = mapped_column(Integer, nullable=True)
    search_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    raw_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
