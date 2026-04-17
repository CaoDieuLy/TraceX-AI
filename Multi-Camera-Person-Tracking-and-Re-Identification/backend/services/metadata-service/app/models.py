from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


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
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[str] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
