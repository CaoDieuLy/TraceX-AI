from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from uuid import uuid4

import httpx

from .config import settings
from .cuda_runtime import configure_torch_runtime
from .execution_plan import resolve_execution_plan
from .legacy_runtime import LEGACY_ROOT
from .pipeline_profiles import resolve_pipeline_profile
from .runtime import AccuracyFirstTrackerRuntime


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


def _build_lightning_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = settings.lightning_api_token.strip()
    if token:
        prefix = settings.lightning_api_auth_prefix or ""
        headers[settings.lightning_api_auth_header] = f"{prefix}{token}"
    return headers


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

    if not settings.tracking_use_mock and settings.lightning_api_base_url.strip():
        endpoint = f"{settings.lightning_api_base_url.rstrip('/')}/{settings.lightning_api_endpoint.lstrip('/')}"
        request_payload = {
            "query_id": query_id,
            "video_id": video_id,
            "video_title": payload.get("video_title"),
            "video_path": storage_path,
            "query_text": query_text,
            "metadata": metadata,
            "pipeline_profile": profile["profile"],
            "hyperparameters": profile.get("hyperparameters", {}),
            "gpu_hardware_profile": hardware_profile,
            "execution_plan": execution_plan,
            "acceleration_state": acceleration_state,
        }
        with httpx.Client(timeout=settings.lightning_timeout_seconds) as client:
            response = client.post(endpoint, json=request_payload, headers=_build_lightning_headers())
            response.raise_for_status()
            try:
                body = response.json()
            except ValueError:
                body = {"text": response.text}

        return {
            "status": str(body.get("status") or "completed"),
            "provider": "lightningai",
            "mode": "remote",
            "pipeline_profile": profile["profile"],
            "gpu_hardware_profile": hardware_profile,
            "acceleration_state": acceleration_state,
            "query_id": query_id,
            "video_id": video_id,
            "job_id": str(body.get("job_id") or uuid4()),
            "summary": str(body.get("summary") or f"LightningAI processed query '{query_text}'."),
            "file_exists": file_exists,
            "processed_at": processed_at,
            "raw_response": body,
        }

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


def run_tracking(candidate_info: dict) -> dict:
    runtime = get_runtime()
    result = runtime.run(
        candidate_info,
        "data/videos/compressed",
        "data/videos/tracking_output",
    )
    result["tracking_use_mock"] = settings.tracking_use_mock
    return result
