from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TrackingRequest(BaseModel):
    candidate_info: dict[str, Any] = Field(default_factory=dict)


class TrackingResponse(BaseModel):
    output_path: str | None = None
    relative_output_path: str | None = None
    manifest_path: str | None = None
    relative_manifest_path: str | None = None
    exists: bool
    detected_hardware: dict[str, Any] | None = None
    runtime_mode: str | None = None
    acceleration_state: dict[str, Any] | None = None


class AiProcessRequest(BaseModel):
    query_id: str | None = None
    video_id: str
    video_title: str | None = None
    storage_path: str
    query_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AiProcessResponse(BaseModel):
    status: str
    provider: str
    mode: str
    detected_hardware: dict[str, Any] | None = None
    acceleration_state: dict[str, Any] | None = None
    query_id: str | None = None
    video_id: str
    job_id: str
    summary: str
    file_exists: bool
    processed_at: datetime
    raw_response: dict[str, Any]


class VideoIngestionRequest(BaseModel):
    source_url: str
    source_filename: str | None = None
    camera_id: str | None = None
    recorded_start: datetime | None = None
    output_video_dir: str | None = None
    output_metadata_dir: str | None = None
    output_basename: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoIngestionResponse(BaseModel):
    status: str
    message: str | None = None
    job_id: str | None = None
    processing_backend: str | None = None
    detected_hardware: dict[str, Any] | None = None
    acceleration_state: dict[str, Any] | None = None
    source_path: str = ""
    compressed_path: str = ""
    metadata_path: str = ""
    drive_video_file_id: str | None = None
    drive_metadata_file_id: str | None = None
    drive_video_link: str | None = None
    drive_metadata_link: str | None = None
    video: dict[str, Any] = Field(default_factory=dict)
    people: list[dict[str, Any]] = Field(default_factory=list)
    person_count: int = 0
    processed_at: datetime | None = None


class CandidateSearchRequest(BaseModel):
    query_text: str = Field(min_length=1, max_length=4000)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=50)
    camera_ids: list[str] | None = Field(default=None)
    time_from: str | None = Field(default=None)
    time_to: str | None = Field(default=None)


class CandidateSearchResponse(BaseModel):
    query_text: str
    count: int
    items: list[dict[str, Any]] = Field(default_factory=list)


class CandidateTrackRequest(BaseModel):
    selected_candidate_id: str = Field(min_length=1, max_length=255)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)
    query_text: str | None = Field(default=None, max_length=4000)
    max_segments_per_candidate: int = Field(default=2, ge=1, le=8)


class CandidateTrackResponse(BaseModel):
    artifact_id: str
    video_url: str
    manifest_url: str
    selected_candidate_id: str
    manifest: dict[str, Any]
