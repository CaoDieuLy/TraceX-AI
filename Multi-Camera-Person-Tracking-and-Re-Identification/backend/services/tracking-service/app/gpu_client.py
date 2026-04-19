"""Lightning AI GPU client for remote video processing."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .config import settings

logger = logging.getLogger(__name__)
SAFE_SELF_CALL_ENDPOINT = "/api/v1/ai/worker"


class LightningAIError(Exception):
    """Base exception for Lightning AI API errors."""

    pass


class LightningAIClient:
    """Client for communicating with Lightning AI Studio API."""

    def __init__(
        self,
        base_url: str | None = None,
        endpoint: str | None = None,
        token: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        """
        Initialize Lightning AI client.

        Args:
            base_url: Lightning AI Studio base URL (e.g., https://xxx.lightning.ai)
            endpoint: API endpoint path (e.g., /predict)
            token: API authentication token
            timeout_seconds: Request timeout
        """
        self.base_url = (base_url or settings.lightning_api_base_url).rstrip("/")
        self.endpoint = self._normalize_endpoint(endpoint or settings.lightning_api_endpoint)
        self.token = token or settings.lightning_api_token
        self.timeout = timeout_seconds or settings.lightning_timeout_seconds

        self.auth_header = settings.lightning_api_auth_header
        self.auth_prefix = settings.lightning_api_auth_prefix or ""

        if not self.base_url:
            raise ValueError("Lightning AI base URL is required")

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        normalized = f"/{str(endpoint or '').strip().lstrip('/')}" if endpoint else SAFE_SELF_CALL_ENDPOINT
        if normalized == "/api/v1/ai/process":
            logger.info("Remapping Lightning API endpoint from /api/v1/ai/process to /api/v1/ai/worker to avoid self-call recursion.")
            normalized = SAFE_SELF_CALL_ENDPOINT
        return normalized.lstrip("/")

    def _build_headers(self) -> dict[str, str]:
        """Build authentication headers."""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers[self.auth_header] = f"{self.auth_prefix}{self.token}"
        return headers

    def _get_url(self) -> str:
        """Build full endpoint URL."""
        return f"{self.base_url}/{self.endpoint}"

    def process_video(
        self,
        video_path: str,
        query_text: str = "",
        query_id: str | None = None,
        video_id: str | None = None,
        video_title: str | None = None,
        metadata: dict[str, Any] | None = None,
        pipeline_profile: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        gpu_hardware_profile: dict[str, Any] | None = None,
        execution_plan: dict[str, Any] | None = None,
        acceleration_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Send video to Lightning AI for processing.

        Args:
            video_path: Path to H.265 video file (local path)
            query_text: Text query for the video
            query_id: Optional query ID for tracking
            video_id: Optional video ID
            video_title: Optional video title
            metadata: Additional metadata
            pipeline_profile: Pipeline profile name
            hyperparameters: Pipeline hyperparameters
            gpu_hardware_profile: GPU hardware configuration
            execution_plan: Execution plan details
            acceleration_state: Acceleration state info

        Returns:
            Dictionary with processing result including:
                - status: "completed", "failed", etc.
                - job_id: Lightning AI job ID
                - summary: Processing summary
                - compressed_video_path: Path to output .h265 file (may be URL)
                - metadata: Additional metadata from AI
                - raw_response: Full raw response

        Raises:
            LightningAIError: If API request fails
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Read video file as bytes for upload
        try:
            with open(video_path, "rb") as f:
                video_bytes = f.read()
        except Exception as e:
            raise LightningAIError(f"Failed to read video file: {e}") from e

        # Prepare multipart payload
        # Lightning AI expects file upload + JSON fields
        files = {
            "file": (
                Path(video_path).name,
                video_bytes,
                "video/h265",
            )
        }

        # JSON fields (sent as form fields)
        data = {
            "query_id": query_id or str(uuid.uuid4()),
            "video_id": video_id or str(uuid.uuid4()),
            "query_text": query_text,
            "metadata": json.dumps(metadata or {}),
        }

        # Optional fields
        if video_title is not None:
            data["video_title"] = video_title
        if pipeline_profile is not None:
            data["pipeline_profile"] = pipeline_profile
        if hyperparameters:
            data["hyperparameters"] = json.dumps(hyperparameters)
        if gpu_hardware_profile:
            data["gpu_hardware_profile"] = json.dumps(gpu_hardware_profile)
        if execution_plan:
            data["execution_plan"] = json.dumps(execution_plan)
        if acceleration_state:
            data["acceleration_state"] = json.dumps(acceleration_state)

        # Send request
        url = self._get_url()
        headers = self._build_headers()
        # Remove Content-Type for multipart/form-data (httpx sets it automatically)
        headers.pop("Content-Type", None)

        logger.info(f"Sending video to Lightning AI: {video_path} → {url}")
        logger.debug(f"Payload: query_id={data['query_id']}, video_id={data['video_id']}")

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, files=files, data=data, headers=headers)
                response.raise_for_status()
                body = response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Lightning AI API error {e.response.status_code}: {e.response.text}")
            raise LightningAIError(
                f"Lightning AI returned {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.TimeoutException as e:
            raise LightningAIError(f"Lightning AI request timed out after {self.timeout}s") from e
        except Exception as e:
            raise LightningAIError(f"Unexpected error calling Lightning AI: {e}") from e

        # Parse response
        result = {
            "status": str(body.get("status", "completed")),
            "provider": "lightningai",
            "mode": "remote",
            "query_id": data["query_id"],
            "video_id": data["video_id"],
            "job_id": str(body.get("job_id", uuid.uuid4())),
            "summary": str(body.get("summary", "Processed by Lightning AI")),
            "compressed_video_path": body.get("compressed_video_path") or body.get("output_path") or "",
            "metadata": body.get("metadata", {}),
            "manifest_path": body.get("manifest_path"),
            "processed_at": body.get("processed_at") or datetime.now(timezone.utc).isoformat(),
            "raw_response": body,
        }

        logger.info(f"Lightning AI response: job_id={result['job_id']}, status={result['status']}")
        if result.get("compressed_video_path"):
            logger.info(f"Output video: {result['compressed_video_path']}")

        return result


def get_lightning_client() -> LightningAIClient:
    """
    Get configured Lightning AI client from settings.

    Returns:
        Configured LightningAIClient instance

    Raises:
        ValueError: If required config is missing
    """
    if not settings.lightning_api_base_url:
        raise ValueError("LIGHTNING_API_BASE_URL is not configured")
    if not settings.lightning_api_token:
        raise ValueError("LIGHTNING_API_TOKEN is not configured")

    return LightningAIClient(
        base_url=settings.lightning_api_base_url,
        endpoint=settings.lightning_api_endpoint,
        token=settings.lightning_api_token,
        timeout_seconds=settings.lightning_timeout_seconds,
    )
