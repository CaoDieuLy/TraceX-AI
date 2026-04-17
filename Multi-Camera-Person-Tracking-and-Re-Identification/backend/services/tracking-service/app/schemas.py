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
    tracking_use_mock: bool | None = None
    pipeline_profile: str | None = None
    gpu_hardware_profile: str | None = None
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
    pipeline_profile: str | None = None
    gpu_hardware_profile: dict[str, Any] | None = None
    acceleration_state: dict[str, Any] | None = None
    query_id: str | None = None
    video_id: str
    job_id: str
    summary: str
    file_exists: bool
    processed_at: datetime
    raw_response: dict[str, Any]
