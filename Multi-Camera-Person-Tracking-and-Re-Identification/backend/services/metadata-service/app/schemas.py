from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserRegisterRequest(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=2, max_length=255)
    password: str = Field(min_length=8, max_length=255)


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=255)


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


class OverviewMetrics(BaseModel):
    total_users: int
    total_managed_videos: int
    total_queries: int
    total_candidates: int
    total_cameras: int
    total_candidate_videos: int


class OverviewResponse(BaseModel):
    metrics: OverviewMetrics
