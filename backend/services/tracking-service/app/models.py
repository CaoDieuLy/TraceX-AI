"""
SQLAlchemy models for tracking-service (queue_video_assets).
These models mirror the metadata-service schema but are used by tracking-service
to insert queue records directly.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class QueueVideoAsset(Base):
    __tablename__ = "queue_video_assets"
    __table_args__ = (UniqueConstraint("video_id", name="uq_queue_video_assets_video_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    video_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    camera_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    source_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_mode: Mapped[str | None] = mapped_column(String(64), nullable=True)
    queue_position: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    storage_backend: Mapped[str] = mapped_column(String(64), nullable=False, default="google_drive")
    available_link_video: Mapped[str] = mapped_column(String(2048), nullable=False)
    available_link_metadata: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    drive_video_file_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    drive_metadata_file_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    local_video_path: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    local_metadata_path: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    raw_video_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "video_id": self.video_id,
            "camera_id": self.camera_id,
            "title": self.title,
            "drive_video_file_id": self.drive_video_file_id,
            "drive_metadata_file_id": self.drive_metadata_file_id,
            "available_link_video": self.available_link_video,
            "available_link_metadata": self.available_link_metadata,
            "queue_position": self.queue_position,
            "storage_backend": self.storage_backend,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
