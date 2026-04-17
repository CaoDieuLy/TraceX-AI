from typing import Any

from pydantic import BaseModel, ConfigDict


class CandidateResponse(BaseModel):
    candidate_id: str
    camera_id: str | None = None
    video_id: str | None = None
    track_id: str | None = None
    human_key: str | None = None
    frame_idx: int | None = None
    search_text: str | None = None
    metadata_path: str | None = None
    raw_metadata: dict[str, Any]

    model_config = ConfigDict(from_attributes=True)


class CandidateListResponse(BaseModel):
    count: int
    items: list[CandidateResponse]


class ImportResponse(BaseModel):
    imported_count: int
    updated_count: int
    file_count: int


class OverviewResponse(BaseModel):
    total_candidates: int
    total_cameras: int
    total_videos: int
