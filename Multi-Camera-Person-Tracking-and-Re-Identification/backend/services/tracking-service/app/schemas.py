from typing import Any

from pydantic import BaseModel, Field


class TrackingRequest(BaseModel):
    candidate_info: dict[str, Any] = Field(default_factory=dict)


class TrackingResponse(BaseModel):
    output_path: str | None = None
    relative_output_path: str | None = None
    exists: bool
    tracking_use_mock: bool
