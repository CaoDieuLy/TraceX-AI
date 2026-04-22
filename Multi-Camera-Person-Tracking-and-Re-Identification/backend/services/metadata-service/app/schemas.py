from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


class UserRegisterRequest(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=2, max_length=255)
    password: str = Field(min_length=8, max_length=255)


class UserLoginRequest(BaseModel):
    identifier: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1, max_length=255)

    @model_validator(mode="before")
    @classmethod
    def normalize_identifier(cls, value: Any) -> Any:
        if isinstance(value, dict) and "identifier" not in value and "email" in value:
            normalized = dict(value)
            normalized["identifier"] = normalized.get("email")
            return normalized
        return value


class UserResponse(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class VideoResponse(BaseModel):
    video_id: str
    title: str
    description: str | None = None
    storage_path: str
    storage_backend: str
    source_filename: str | None = None
    content_type: str | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class VideoListResponse(BaseModel):
    count: int
    items: list[VideoResponse]


class VideoQueryCreateRequest(BaseModel):
    video_id: str
    query_text: str = Field(min_length=2, max_length=4000)


class VideoQueryUpdateRequest(BaseModel):
    status: str | None = None
    ai_job_id: str | None = None
    ai_response: dict[str, Any] | None = None


class VideoQueryResponse(BaseModel):
    query_id: str
    video_id: str
    video_title: str
    storage_path: str
    query_text: str
    status: str
    ai_job_id: str | None = None
    ai_response: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class VideoQueryListResponse(BaseModel):
    count: int
    items: list[VideoQueryResponse]


class CandidateResponse(BaseModel):
    candidate_id: str
    camera_id: str | None = None
    video_id: str | None = None
    track_id: str | None = None
    human_key: str | None = None
    frame_idx: int | None = None
    bbox: list[int] = Field(default_factory=list)
    search_text: str | None = None
    metadata_path: str | None = None
    appearance_summary: str | None = None
    semantic_attributes: list[str] = Field(default_factory=list)
    visibility_scores: dict[str, Any] = Field(default_factory=dict)
    world_position: dict[str, Any] | None = None
    reid_profile: str | None = None
    pipeline_profile: str | None = None
    score: float | None = None
    matched_segments: list[dict[str, Any]] = Field(default_factory=list)
    available_link_video: str | None = None
    available_link_metadata: str | None = None
    local_video_path: str | None = None
    local_metadata_path: str | None = None
    storage_path: str | None = None
    video_title: str | None = None
    preview_image_url: str | None = None
    raw_metadata: dict[str, Any]

    model_config = ConfigDict(from_attributes=True)


class CandidateListResponse(BaseModel):
    count: int
    items: list[CandidateResponse]


class ImportResponse(BaseModel):
    imported_count: int
    updated_count: int
    file_count: int


class QueueVideoResponse(BaseModel):
    video_id: str
    camera_id: str | None = None
    title: str
    queue_position: int
    storage_backend: str
    available_link_video: str
    available_link_metadata: str | None = None
    source_filename: str | None = None
    source_mode: str | None = None
    created_at: datetime
    updated_at: datetime
    local_video_path: str | None = None
    local_metadata_path: str | None = None
    raw_video_metadata: dict[str, Any]

    model_config = ConfigDict(from_attributes=True)


class QueueVideoListResponse(BaseModel):
    count: int
    items: list[QueueVideoResponse]


class QueueBootstrapRequest(BaseModel):
    source_dir: str = Field(min_length=1, max_length=4096)
    limit: int = Field(default=31, ge=1, le=500)
    reset_remote_queue: bool = True
    delete_source_after_import: bool = False


class QueueBootstrapResponse(BaseModel):
    processed_videos: int
    queue_size: int
    people_indexed: int
    evicted_video_ids: list[str] = Field(default_factory=list)


class QueueProcessResponse(BaseModel):
    processed_videos: int
    evicted_video_ids: list[str] = Field(default_factory=list)
    imported_source_files: list[str] = Field(default_factory=list)


class CandidateSearchRequest(BaseModel):
    query_text: str = Field(min_length=2, max_length=4000)
    limit: int = Field(default=5, ge=1, le=20)


class CandidateSearchResponse(BaseModel):
    query_text: str
    count: int
    items: list[CandidateResponse]


class CandidateTrackRequest(BaseModel):
    selected_candidate_id: str = Field(min_length=1, max_length=255)
    candidate_ids: list[str] = Field(default_factory=list)
    query_text: str | None = Field(default=None, max_length=4000)
    max_segments_per_candidate: int = Field(default=2, ge=1, le=5)


class CandidateTrackResponse(BaseModel):
    artifact_id: str
    video_url: str
    manifest_url: str
    selected_candidate_id: str
    manifest: dict[str, Any]


class OverviewMetrics(BaseModel):
    total_users: int
    total_managed_videos: int
    total_queries: int
    total_candidates: int
    total_cameras: int
    total_candidate_videos: int
    total_queue_videos: int


class OverviewResponse(BaseModel):
    metrics: OverviewMetrics
