from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import httpx

from .config import settings
from .cuda_runtime import configure_torch_runtime
from .execution_plan import resolve_execution_plan
from .ingestion_runtime import VideoIngestionRuntime
from .legacy_runtime import LEGACY_ROOT
from .pipeline_profiles import resolve_pipeline_profile
from .runtime import AccuracyFirstTrackerRuntime

logger = logging.getLogger(__name__)


def _dict_or_empty(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _load_env_profile_overrides() -> dict:
    raw = settings.tracking_hyperparameter_overrides_json.strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_env_hardware_overrides() -> dict:
    raw = settings.gpu_hardware_overrides_json.strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_profile_from_payload(default_profile: str, metadata: dict | None = None, overrides: dict | None = None) -> dict:
    metadata = _dict_or_empty(metadata)
    payload_profile = str(metadata.get("pipeline_profile") or default_profile).strip() or default_profile
    merged_overrides = _dict_or_empty(_load_env_profile_overrides())
    if overrides:
        merged_overrides.update(_dict_or_empty(overrides))
    if metadata.get("hyperparameter_overrides"):
        merged_overrides.update(_dict_or_empty(metadata.get("hyperparameter_overrides")))
    return resolve_pipeline_profile(payload_profile, merged_overrides)


@lru_cache(maxsize=1)
def get_pipeline_config() -> dict:
    mode = "mock" if settings.tracking_use_mock or not settings.lightning_api_base_url else "remote"
    profile = _resolve_profile_from_payload(settings.pipeline_profile)
    hardware_profile, execution_plan = resolve_execution_plan(
        pipeline_profile=profile,
        gpu_profile_name=settings.gpu_hardware_profile,
        gpu_profile_overrides=_load_env_hardware_overrides(),
        gpu_count=settings.gpu_count,
        host_cpu_count=settings.host_cpu_count,
        host_ram_gb=settings.host_ram_gb,
    )
    acceleration_state = configure_torch_runtime(
        allow_tf32=bool(hardware_profile.get("allow_tf32", True)),
        cudnn_benchmark=bool(hardware_profile.get("cudnn_benchmark", True)),
        host_cpu_count=settings.host_cpu_count,
    )
    return {
        "provider": "lightningai",
        "mode": mode,
        "runtime_mode": settings.tracking_runtime_mode,
        "pipeline_profile": profile["profile"],
        "pipeline_summary": profile["summary"],
        "validated_on": profile["validated_on"],
        "components": profile["components"],
        "hyperparameters": profile.get("hyperparameters", {}),
        "runtime_defaults": profile.get("runtime_defaults", {}),
        "gpu_hardware_profile": hardware_profile,
        "execution_plan": execution_plan,
        "acceleration_state": acceleration_state,
        "enable_trackeval": settings.enable_trackeval,
        "enable_geometry_gating": settings.enable_geometry_gating,
        "enable_corrective_cascade": settings.enable_corrective_cascade,
        "lightning_api_base_url": settings.lightning_api_base_url or None,
        "lightning_api_endpoint": settings.lightning_api_endpoint,
        "legacy_root": str(LEGACY_ROOT),
    }


@lru_cache(maxsize=1)
def get_runtime() -> AccuracyFirstTrackerRuntime:
    return AccuracyFirstTrackerRuntime()


@lru_cache(maxsize=1)
def get_ingestion_runtime() -> VideoIngestionRuntime:
    return VideoIngestionRuntime()


def _build_lightning_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = settings.lightning_api_token.strip()
    if token:
        prefix = settings.lightning_api_auth_prefix or ""
        headers[settings.lightning_api_auth_header] = f"{prefix}{token}"
    return headers


def _download_file(url: str, output_path: str, timeout: int = 180) -> None:
    """
    Download a file from URL to local filesystem.

    Args:
        url: Source URL (can be http/https or file path)
        output_path: Local destination path
        timeout: Download timeout in seconds

    Raises:
        Exception: If download fails
    """
    if url.startswith("http://") or url.startswith("https://"):
        logger.info(f"Downloading from {url} to {output_path}")
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)
    else:
        # Assume it's a local file path - copy if different location
        src = Path(url)
        dst = Path(output_path)
        if src.resolve() != dst.resolve():
            logger.info(f"Copying {src} to {dst}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copy2(src, dst)
        else:
            logger.debug(f"Source and destination are the same: {src}")


def process_video_query(payload: dict) -> dict:
    from pathlib import Path

    storage_path = str(payload.get("storage_path") or "").strip()
    file_exists = bool(storage_path and Path(storage_path).exists())
    processed_at = datetime.now(timezone.utc)
    query_text = str(payload.get("query_text") or "").strip()
    video_id = str(payload.get("video_id") or "").strip()
    query_id = payload.get("query_id")
    metadata = _dict_or_empty(payload.get("metadata"))
    profile = _resolve_profile_from_payload(settings.pipeline_profile, metadata=metadata)
    hardware_profile, execution_plan = resolve_execution_plan(
        pipeline_profile=profile,
        gpu_profile_name=str(metadata.get("gpu_hardware_profile") or settings.gpu_hardware_profile),
        gpu_profile_overrides=_dict_or_empty(metadata.get("gpu_hardware_overrides")) or _load_env_hardware_overrides(),
        gpu_count=settings.gpu_count,
        host_cpu_count=settings.host_cpu_count,
        host_ram_gb=settings.host_ram_gb,
    )
    acceleration_state = configure_torch_runtime(
        allow_tf32=bool(hardware_profile.get("allow_tf32", True)),
        cudnn_benchmark=bool(hardware_profile.get("cudnn_benchmark", True)),
        host_cpu_count=settings.host_cpu_count,
    )

    # Mock mode (no Lightning AI)
    if settings.tracking_use_mock or not settings.lightning_api_base_url.strip():
        logger.info("Using mock mode for video processing")
        return {
            "status": "completed",
            "provider": "lightningai",
            "mode": "mock",
            "pipeline_profile": profile["profile"],
            "query_id": query_id,
            "video_id": video_id,
            "job_id": str(uuid4()),
            "summary": (
                f"Mock accuracy-first response for '{query_text}' on video '{video_id}'. "
                f"Configured pipeline: {profile['components']['detector']['name']} + {profile['components']['tracker']['name']} + "
                f"{profile['components']['reid']['name']}."
            ),
            "file_exists": file_exists,
            "processed_at": processed_at,
            "gpu_hardware_profile": hardware_profile,
            "acceleration_state": acceleration_state,
            "raw_response": {
                "matched_segments": [
                    {
                        "start_second": 12.5,
                        "end_second": 19.0,
                        "confidence": 0.92,
                        "note": "Demo segment generated in mock mode.",
                    }
                ],
                "pipeline_profile": profile,
                "gpu_hardware_profile": hardware_profile,
                "execution_plan": execution_plan,
                "acceleration_state": acceleration_state,
                "storage_path": storage_path,
                "metadata": metadata,
            },
        }

    # Remote mode: use Lightning AI
    logger.info("Using Lightning AI remote mode")
    from .video_converter import convert_to_h265
    from .gpu_client import get_lightning_client

    try:
        # Step 1: Convert video to H.265 if needed
        input_path = Path(storage_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {storage_path}")

        # Ensure conversion output directory exists
        conversion_dir = Path(settings.video_conversion_output_dir)
        conversion_dir.mkdir(parents=True, exist_ok=True)

        # Convert to H.265 (or skip if already H.265)
        h265_path = convert_to_h265(
            input_path,
            output_path=conversion_dir / f"{input_path.stem}.h265.mp4",
            crf=settings.ffmpeg_crf,
            preset=settings.ffmpeg_preset,
            audio_codec=settings.ffmpeg_audio_codec,
            overwrite=settings.ffmpeg_overwrite_output,
        )
        logger.info(f"Video prepared for Lightning AI: {h265_path}")

        # Step 2: Call Lightning AI
        client = get_lightning_client()
        ai_response = client.process_video(
            video_path=h265_path,
            query_text=query_text,
            query_id=query_id,
            video_id=video_id,
            video_title=payload.get("video_title"),
            metadata=metadata,
            pipeline_profile=profile["profile"],
            hyperparameters=profile.get("hyperparameters", {}),
            gpu_hardware_profile=hardware_profile,
            execution_plan=execution_plan,
            acceleration_state=acceleration_state,
        )

        # Step 3: Download output video from Lightning AI (if available)
        compressed_video_path = ai_response.get("compressed_video_path", "")
        downloaded_output_path = None

        if compressed_video_path:
            try:
                # Ensure output directory exists
                output_dir = Path(settings.video_download_output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Generate output filename
                output_filename = f"{video_id or input_path.stem}_tracked.h265.mp4"
                downloaded_output_path = output_dir / output_filename

                # Download file
                logger.info(f"Downloading output video from Lightning AI: {compressed_video_path}")
                _download_file(compressed_video_path, str(downloaded_output_path))
                logger.info(f"Downloaded output video to: {downloaded_output_path}")

                # Update file_exists based on downloaded file
                file_exists = downloaded_output_path.exists()

            except Exception as e:
                logger.warning(f"Failed to download output video: {e}")
                # Continue without download error

        # Step 4: Build response
        result = {
            "status": ai_response["status"],
            "provider": "lightningai",
            "mode": "remote",
            "pipeline_profile": profile["profile"],
            "gpu_hardware_profile": hardware_profile,
            "acceleration_state": acceleration_state,
            "query_id": query_id,
            "video_id": video_id,
            "job_id": ai_response["job_id"],
            "summary": ai_response["summary"],
            "file_exists": file_exists,
            "processed_at": processed_at,
            "raw_response": ai_response,
        }

        # Add output path if downloaded
        if downloaded_output_path:
            result["output_path"] = str(downloaded_output_path)
            result["relative_output_path"] = str(downloaded_output_path.relative_to(Path.cwd()))

        return result

    except Exception as e:
        logger.error(f"Lightning AI processing failed: {e}", exc_info=True)
        # Fallback to mock on error
        return {
            "status": "failed",
            "provider": "lightningai",
            "mode": "remote_error",
            "pipeline_profile": profile["profile"],
            "query_id": query_id,
            "video_id": video_id,
            "job_id": str(uuid4()),
            "summary": f"Error processing video: {str(e)}",
            "file_exists": file_exists,
            "processed_at": processed_at,
            "gpu_hardware_profile": hardware_profile,
            "acceleration_state": acceleration_state,
            "raw_response": {"error": str(e)},
        }


def run_tracking(candidate_info: dict) -> dict:
    runtime = get_runtime()
    result = runtime.run(
        candidate_info,
        "data/videos/compressed",
        "data/videos/tracking_output",
    )
    result["tracking_use_mock"] = settings.tracking_use_mock
    return result


def process_video_ingestion(payload: dict) -> dict:
    runtime = get_ingestion_runtime()
    return runtime.process_video(
        source_path=str(payload.get("source_path") or "").strip() or None,
        source_drive_file_id=str(payload.get("source_drive_file_id") or "").strip() or None,
        source_filename=str(payload.get("source_filename") or "").strip() or None,
        camera_id=payload.get("camera_id"),
        recorded_start=payload.get("recorded_start"),
        output_video_dir=payload.get("output_video_dir"),
        output_metadata_dir=payload.get("output_metadata_dir"),
        output_basename=payload.get("output_basename"),
        destination_video_folder_id=payload.get("destination_video_folder_id"),
        destination_metadata_folder_id=payload.get("destination_metadata_folder_id"),
        upload_outputs_to_drive=bool(payload.get("upload_outputs_to_drive")),
        metadata=_dict_or_empty(payload.get("metadata")),
    )
