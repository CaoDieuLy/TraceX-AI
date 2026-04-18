from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import httpx
import numpy as np
import torch

from .config import settings
from .cuda_runtime import configure_torch_runtime
from .execution_plan import resolve_execution_plan
from .ingestion_runtime import VideoIngestionRuntime
from .legacy_runtime import LEGACY_ROOT
from .pipeline_profiles import resolve_pipeline_profile
from .runtime import AccuracyFirstTrackerRuntime

logger = logging.getLogger(__name__)
ALREADY_COMPRESSED_SUFFIXES = {".h265", ".hevc"}


def _dict_or_empty(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _json_dict_or_empty(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


# ═══════════════════════════════════════════════════════════════
# ITSELF-Style Ranking (RANGE Ensemble)
# Weights: embedding(0.58) + semantic(0.22) + visibility(0.12) + world(0.08)
# ═══════════════════════════════════════════════════════════════
def _embedding_similarity(query_emb: np.ndarray, candidate_emb: list[float] | None) -> float:
    """Cosine similarity between query ITSELF embedding and candidate embedding"""
    if candidate_emb is None or len(candidate_emb) == 0:
        return 0.0
    cand_vec = np.array(candidate_emb, dtype=np.float32)
    cand_norm = np.linalg.norm(cand_vec) + 1e-8
    query_norm = np.linalg.norm(query_emb) + 1e-8
    return float(np.dot(query_emb, cand_vec) / (query_norm * cand_norm))


def _semantic_overlap_itself(query_text: str, candidate: dict) -> float:
    """Token-level Jaccard similarity (same as before)"""
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return 0.0
    candidate_tokens = _tokenize(_candidate_search_document(candidate))
    if not candidate_tokens:
        return 0.0
    return float(len(query_tokens & candidate_tokens) / max(len(query_tokens), 1))


def _visibility_bonus(candidate: dict) -> float:
    scores = candidate.get("visibility_scores")
    if not isinstance(scores, dict) or not scores:
        return 0.0
    numeric_scores = [float(value) for value in scores.values() if isinstance(value, (int, float))]
    if not numeric_scores:
        return 0.0
    return float(sum(numeric_scores) / len(numeric_scores))


def _world_position_bonus(candidate: dict) -> float:
    world_position = candidate.get("world_position") or candidate.get("top_point_projection")
    if not isinstance(world_position, dict) or not world_position:
        return 0.0
    if any(world_position.get(axis) is not None for axis in ("x", "y", "z")):
        return 1.0
    return 0.0


def _rank_candidate_itself(
    query_text: str,
    query_embedding: np.ndarray | None,
    candidate: dict,
) -> float:
    """
    ITSELF RANGE-style ensemble scoring.
    Weights (accuracy_first profile):
      - Embedding similarity: 0.58
      - Semantic overlap:    0.22
      - Visibility bonus:    0.12
      - World position:      0.08
    """
    # 1. Embedding similarity (ITSELF fine-grained)
    emb_sim = _embedding_similarity(query_embedding, candidate.get("embedding_vector")) if query_embedding is not None else 0.0

    # 2. Semantic overlap (token Jaccard)
    sem_overlap = _semantic_overlap_itself(query_text, candidate)

    # 3. Visibility bonus
    vis_bonus = _visibility_bonus(candidate)

    # 4. World geometry bonus
    world_bonus = _world_position_bonus(candidate)

    # Weighted sum
    final_score = (
        emb_sim * 0.58 +
        sem_overlap * 0.22 +
        vis_bonus * 0.12 +
        world_bonus * 0.08
    )
    return round(final_score, 6)


def _rank_candidate(query_text: str, candidate: dict) -> float:
    """Legacy wrapper - now uses ITSELF ranking"""
    # In edge-first mode, we don't have pre-computed query embedding
    # Fall back to semantic-only ranking (or compute on-the-fly if needed)
    return round(_semantic_overlap_itself(query_text, candidate), 6)



def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower()) if len(token) >= 2}


def _coerce_string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value or "").strip() for value in values if str(value or "").strip()]


def _candidate_search_document(person: dict) -> str:
    parts: list[str] = []
    for key in ("search_text", "appearance_summary", "person_caption", "caption"):
        text = str(person.get(key) or "").strip()
        if text:
            parts.append(text)

    attributes = _coerce_string_list(person.get("semantic_attributes"))
    if attributes:
        parts.append("attributes: " + ", ".join(attributes))

    timeline = person.get("timeline")
    if isinstance(timeline, list):
        actions = [str(item.get("action_summary") or "").strip() for item in timeline if isinstance(item, dict)]
        actions = [item for item in actions if item]
        if actions:
            parts.append("timeline: " + " ".join(actions))

    return " ".join(parts).strip()


def _extract_candidate_segments(candidate: dict, limit: int = 3) -> list[dict]:
    timeline = candidate.get("timeline")
    if not isinstance(timeline, list):
        return []
    segments: list[dict] = []
    for item in timeline:
        if not isinstance(item, dict):
            continue
        start_second = item.get("start_second")
        end_second = item.get("end_second")
        if start_second is None or end_second is None:
            continue
        segments.append(
            {
                "start_second": float(start_second),
                "end_second": float(end_second),
                "action_summary": str(item.get("action_summary") or "").strip(),
            }
        )
        if len(segments) >= limit:
            break
    return segments


def _summarize_matches(query_text: str, video_id: str, matches: list[dict], person_count: int) -> str:
    if not matches:
        return f"Processed video '{video_id}' and generated metadata for {person_count} people, but no strong match was found for '{query_text}'."
    best = matches[0]
    camera_id = str(best.get("camera_id") or video_id or "unknown-camera")
    track_id = str(best.get("track_id") or "?")
    score = float(best.get("score") or 0.0)
    return (
        f"Processed video '{video_id}' and found {len(matches)} likely matches for '{query_text}'. "
        f"Best match: camera {camera_id}, track {track_id}, score {score:.3f}."
    )
    timeline = candidate.get("timeline")
    if not isinstance(timeline, list):
        return []
    segments: list[dict] = []
    for item in timeline:
        if not isinstance(item, dict):
            continue
        start_second = item.get("start_second")
        end_second = item.get("end_second")
        if start_second is None or end_second is None:
            continue
        segments.append(
            {
                "start_second": float(start_second),
                "end_second": float(end_second),
                "action_summary": str(item.get("action_summary") or "").strip(),
            }
        )
        if len(segments) >= limit:
            break
    return segments


def _build_worker_match(
    candidate: dict,
    query_text: str,
    query_embedding: np.ndarray | None = None,
) -> dict:
    match = dict(candidate)
    match["search_text"] = _candidate_search_document(candidate)
    match["semantic_overlap"] = round(_semantic_overlap_itself(query_text, candidate), 6)
    match["score"] = _rank_candidate_itself(query_text, query_embedding, candidate)
    match["matched_segments"] = _extract_candidate_segments(candidate)
    return match


def _summarize_matches(query_text: str, video_id: str, matches: list[dict], person_count: int) -> str:
    if not matches:
        return f"Processed video '{video_id}' and generated metadata for {person_count} people, but no strong match was found for '{query_text}'."
    best = matches[0]
    camera_id = str(best.get("camera_id") or video_id or "unknown-camera")
    track_id = str(best.get("track_id") or "?")
    score = float(best.get("score") or 0.0)
    return (
        f"Processed video '{video_id}' and found {len(matches)} likely matches for '{query_text}'. "
        f"Best match: camera {camera_id}, track {track_id}, score {score:.3f}."
    )


def _resolve_query_source_path(storage_path: str, video_id: str | None = None) -> Path:
    source = str(storage_path or "").strip()
    if not source:
        raise ValueError("storage_path is required")
    if source.startswith(("http://", "https://")):
        parsed = urlparse(source)
        suffix = Path(parsed.path).suffix or ".h265"
        target_dir = Path(settings.ingestion_work_root) / "query-inputs"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{(video_id or 'query-video').strip() or 'query-video'}-{uuid4().hex}{suffix}"
        _download_file(source, str(target_path))
        return target_path
    input_path = Path(source)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {storage_path}")
    return input_path


def _prepare_remote_query_video(source_path: Path, video_id: str | None = None) -> Path:
    if source_path.suffix.lower() in ALREADY_COMPRESSED_SUFFIXES:
        conversion_dir = Path(settings.video_conversion_output_dir)
        conversion_dir.mkdir(parents=True, exist_ok=True)
        target_path = conversion_dir / f"{(video_id or source_path.stem).strip() or source_path.stem}{source_path.suffix.lower()}"
        if source_path.resolve() != target_path.resolve():
            shutil.copy2(source_path, target_path)
        return target_path
    raise ValueError(
        f"Only pre-encoded .h265/.hevc inputs are supported for query processing. Got: {source_path.name}"
    )


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
    mode = "remote" if settings.lightning_api_base_url else "local"
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
    storage_path = str(payload.get("storage_path") or "").strip()
    processed_at = datetime.now(timezone.utc)
    query_text = str(payload.get("query_text") or "").strip()
    video_id = str(payload.get("video_id") or "").strip()
    query_id = payload.get("query_id")
    metadata = _json_dict_or_empty(payload.get("metadata"))
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
    file_exists = bool(storage_path)

    if not settings.lightning_api_base_url.strip():
        logger.info("Using local accuracy-first worker mode for video processing")
        input_path = _resolve_query_source_path(storage_path, video_id=video_id or None)
        file_exists = input_path.exists()
        local_result = process_video_query_worker(
            {
                "query_id": query_id,
                "video_id": video_id,
                "video_title": payload.get("video_title"),
                "query_text": query_text,
                "source_path": str(input_path),
                "metadata": {
                    **metadata,
                    "pipeline_profile": profile["profile"],
                    "gpu_hardware_profile": hardware_profile,
                    "execution_plan": execution_plan,
                    "acceleration_state": acceleration_state,
                },
                "pipeline_profile": profile["profile"],
                "gpu_hardware_profile": json.dumps(hardware_profile),
                "execution_plan": json.dumps(execution_plan),
                "acceleration_state": json.dumps(acceleration_state),
            }
        )
        return {
            "status": local_result["status"],
            "provider": "local",
            "mode": "local",
            "pipeline_profile": profile["profile"],
            "gpu_hardware_profile": hardware_profile,
            "acceleration_state": acceleration_state,
            "query_id": query_id,
            "video_id": video_id,
            "job_id": local_result["job_id"],
            "summary": local_result["summary"],
            "file_exists": file_exists,
            "processed_at": processed_at,
            "raw_response": local_result,
        }

    # Remote mode: use Lightning AI
    logger.info("Using Lightning AI remote mode")
    from .gpu_client import get_lightning_client

    try:
        input_path = _resolve_query_source_path(storage_path, video_id=video_id or None)
        file_exists = input_path.exists()
        h265_path = _prepare_remote_query_video(input_path, video_id=video_id or input_path.stem)
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
                output_filename = f"{video_id or input_path.stem}_tracked.h265"
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
            try:
                result["relative_output_path"] = str(downloaded_output_path.relative_to(Path.cwd()))
            except ValueError:
                result["relative_output_path"] = str(downloaded_output_path)

        return result

    except Exception as e:
        logger.error(f"Lightning AI processing failed: {e}", exc_info=True)
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


def process_video_query_worker(payload: dict) -> dict:
    processed_at = datetime.now(timezone.utc)
    query_id = str(payload.get("query_id") or uuid4())
    video_id = str(payload.get("video_id") or "").strip() or query_id
    query_text = str(payload.get("query_text") or "").strip()
    source_path = str(payload.get("source_path") or "").strip()
    metadata = _json_dict_or_empty(payload.get("metadata"))
    worker_metadata = dict(metadata)
    pipeline_profile = str(payload.get("pipeline_profile") or worker_metadata.get("pipeline_profile") or settings.pipeline_profile).strip()
    if pipeline_profile:
        worker_metadata["pipeline_profile"] = pipeline_profile

    job_root = Path(settings.ingestion_work_root) / "query-workers" / query_id
    output_video_dir = job_root / "videos"
    output_metadata_dir = job_root / "metadata"
    output_basename = f"{Path(source_path).stem or video_id}.h265"

    ingestion_result = process_video_ingestion(
        {
            "source_path": source_path,
            "camera_id": video_id,
            "output_video_dir": str(output_video_dir),
            "output_metadata_dir": str(output_metadata_dir),
            "output_basename": output_basename,
            "metadata": worker_metadata,
        }
    )

    people = ingestion_result.get("people") or []

    # Compute query embedding from query text (ITSELF-style)
    # For edge-first, we use CLIP text encoder as query embedding
    query_embedding = None
    try:
        import open_clip
        from transformers import AutoTokenizer, AutoModel
        # Use CLIP text encoder for query
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        text_tokens = tokenizer([query_text]).to(model.device)
        with torch.no_grad():
            query_emb = model.encode_text(text_tokens)
            query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
            query_embedding = query_emb.cpu().numpy()[0].astype(np.float32)
    except Exception as e:
        logger.warning(f"Failed to compute query embedding: {e}")
        query_embedding = None

    ranked_matches = [
        _build_worker_match(person, query_text, query_embedding)
        for person in people
        if isinstance(person, dict)
    ]
    ranked_matches.sort(
        key=lambda item: (
            -float(item.get("score") or 0.0),
            -float(item.get("semantic_overlap") or 0.0),
            int(item.get("frame_idx") or 0),
        )
    )
    matched_candidates = ranked_matches[:5]
    matched_segments = [segment for candidate in matched_candidates for segment in candidate.get("matched_segments") or []][:10]

    return {
        "status": "completed",
        "provider": "lightningai",
        "mode": "worker",
        "query_id": query_id,
        "video_id": video_id,
        "job_id": query_id,
        "summary": _summarize_matches(query_text, video_id, matched_candidates, int(ingestion_result.get("person_count") or 0)),
        "compressed_video_path": ingestion_result.get("compressed_path"),
        "metadata_path": ingestion_result.get("metadata_path"),
        "processed_at": processed_at.isoformat(),
        "metadata": {
            "query_text": query_text,
            "person_count": ingestion_result.get("person_count"),
            "matched_candidates": matched_candidates,
            "matched_segments": matched_segments,
            "video": ingestion_result.get("video"),
            "processing_backend": ingestion_result.get("processing_backend"),
            "pipeline_profile": ingestion_result.get("pipeline_profile"),
            "gpu_hardware_profile": ingestion_result.get("gpu_hardware_profile"),
            "acceleration_state": ingestion_result.get("acceleration_state"),
        },
    }


def run_tracking(candidate_info: dict) -> dict:
    runtime = get_runtime()
    return runtime.run(
        candidate_info,
        "data/videos/compressed",
        "data/videos/tracking_output",
    )


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
