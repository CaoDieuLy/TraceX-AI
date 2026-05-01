from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
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

from .runtime import StrictTrackerRuntime
from .strict_pipeline import get_strict_pipeline

logger = logging.getLogger(__name__)
INGESTION_VIDEO_SUFFIXES = {".mp4"}
_INGESTION_BACKGROUND_STATUS: dict[str, dict[str, object]] = {}
_INGESTION_STATUS_LOCK = threading.Lock()


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


def _tracking_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "")).strip("-._") or "tracking"


def _artifact_root() -> Path:
    root = Path(settings.tracking_artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_tracking_artifact_paths(artifact_id: str) -> tuple[Path, Path]:
    cleaned = _tracking_slug(artifact_id)
    root = _artifact_root()
    return root / f"{cleaned}.mp4", root / f"{cleaned}.json"


def _candidate_payload_view(candidate: dict) -> dict:
    payload = dict(candidate)
    raw_metadata = _json_dict_or_empty(payload.get("raw_metadata"))
    merged = dict(raw_metadata)
    for key in (
        "candidate_id",
        "camera_id",
        "video_id",
        "track_id",
        "frame_idx",
        "search_text",
        "appearance_summary",
        "semantic_attributes",
        "visibility_scores",
        "world_position",
        "timeline",
        "matched_segments",
        "action_semantic_embedding",
        "tracklet_feature_pipeline",
        "embedding_vector",
    ):
        if payload.get(key) not in (None, "", []):
            merged[key] = payload.get(key)
    return merged


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


def _action_embedding_similarity(query_emb: np.ndarray | None, action_payload: object) -> float:
    if query_emb is None or not isinstance(action_payload, dict):
        return 0.0

    query_vec = np.asarray(query_emb, dtype=np.float32)
    query_norm = float(np.linalg.norm(query_vec))
    if query_norm <= 1e-8:
        return 0.0
    query_vec = query_vec / query_norm

    candidate_vectors: list[np.ndarray] = []

    direct_vector = action_payload.get("embedding_vector")
    if isinstance(direct_vector, list) and len(direct_vector) == query_vec.shape[0]:
        candidate_vectors.append(np.asarray(direct_vector, dtype=np.float32))

    segments = action_payload.get("segments")
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            vector = segment.get("embedding_vector")
            if isinstance(vector, list) and len(vector) == query_vec.shape[0]:
                candidate_vectors.append(np.asarray(vector, dtype=np.float32))

    if not candidate_vectors:
        return 0.0

    matrix = np.stack(candidate_vectors, axis=0)
    norms = np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    matrix = matrix / norms
    return float(np.max(matrix @ query_vec))


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
    precomputed_embedding_similarity: float | None = None,
) -> float:
    """
    ITSELF RANGE-style ensemble scoring.
    Weights (strict single pipeline):
      - Embedding similarity: 0.58
      - Semantic overlap:    0.22
      - Visibility bonus:    0.12
      - World position:      0.08
    """
    # 1. Embedding similarity (ITSELF fine-grained)
    emb_sim = (
        float(precomputed_embedding_similarity)
        if precomputed_embedding_similarity is not None
        else _embedding_similarity(query_embedding, candidate.get("embedding_vector")) if query_embedding is not None else 0.0
    )

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


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower()) if len(token) >= 2}


def _coerce_string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value or "").strip() for value in values if str(value or "").strip()]


def _candidate_search_document(person: dict) -> str:
    parts: list[str] = []
    for key in ("search_text", "appearance_summary"):
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


def _normalize_tracking_segments(candidate: dict, limit: int) -> list[dict[str, float | str]]:
    raw_segments = candidate.get("matched_segments") or _extract_candidate_segments(_candidate_payload_view(candidate), limit)
    if not isinstance(raw_segments, list):
        return []

    normalized: list[dict[str, float | str]] = []
    for segment in raw_segments[:limit]:
        if not isinstance(segment, dict):
            continue
        start_second = max(0.0, float(segment.get("start_second") or 0.0))
        end_second = max(start_second + 0.1, float(segment.get("end_second") or 0.0))
        normalized.append(
            {
                "start_second": start_second,
                "end_second": end_second,
                "action_summary": str(segment.get("action_summary") or "").strip(),
            }
        )
    return normalized


def _normalized_vector_matrix(
    candidates: list[dict],
    *,
    field_name: str,
) -> tuple[list[int], np.ndarray | None]:
    row_indices: list[int] = []
    matrix_rows: list[np.ndarray] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        raw_vector = candidate.get(field_name)
        if not isinstance(raw_vector, list) or not raw_vector:
            continue
        vector = np.asarray(raw_vector, dtype=np.float32)
        if vector.ndim != 1:
            continue
        vector_norm = float(np.linalg.norm(vector))
        if vector_norm <= 1e-8:
            continue
        row_indices.append(index)
        matrix_rows.append(vector / vector_norm)
    if not matrix_rows:
        return [], None
    return row_indices, np.stack(matrix_rows, axis=0)


def _matrix_vector_similarity_scores(
    matrix: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if torch.cuda.is_available() and matrix.shape[0] >= 1024:
        device = torch.device("cuda")
        matrix_tensor = torch.as_tensor(matrix, device=device)
        vector_tensor = torch.as_tensor(vector, device=device)
        return (matrix_tensor @ vector_tensor).float().detach().cpu().numpy()
    return matrix @ vector


def _precompute_candidate_embedding_scores(query_embedding: np.ndarray | None, candidates: list[dict]) -> dict[int, float]:
    if query_embedding is None:
        return {}

    normalized_query = np.asarray(query_embedding, dtype=np.float32)
    query_norm = float(np.linalg.norm(normalized_query))
    if query_norm <= 1e-8:
        return {}
    normalized_query = normalized_query / query_norm

    row_indices, matrix = _normalized_vector_matrix(candidates, field_name="embedding_vector")
    if matrix is None:
        return {}
    compatible_rows = [idx for idx, row in zip(row_indices, matrix) if row.shape[0] == normalized_query.shape[0]]
    if not compatible_rows:
        return {}
    compatible_matrix = np.stack(
        [row for row in matrix if row.shape[0] == normalized_query.shape[0]],
        axis=0,
    )
    scores = _matrix_vector_similarity_scores(compatible_matrix, normalized_query)
    return {row_index: float(score) for row_index, score in zip(compatible_rows, scores)}


def _precompute_field_similarity_scores(
    query_embedding: np.ndarray | None,
    candidates: list[dict],
    *,
    field_name: str,
) -> dict[int, float]:
    if query_embedding is None:
        return {}

    normalized_query = np.asarray(query_embedding, dtype=np.float32)
    query_norm = float(np.linalg.norm(normalized_query))
    if query_norm <= 1e-8:
        return {}
    normalized_query = normalized_query / query_norm

    row_indices, matrix = _normalized_vector_matrix(candidates, field_name=field_name)
    if matrix is None:
        return {}
    compatible_rows: list[int] = []
    compatible_vectors: list[np.ndarray] = []
    for row_index, row in zip(row_indices, matrix):
        if row.shape[0] != normalized_query.shape[0]:
            continue
        compatible_rows.append(row_index)
        compatible_vectors.append(row)
    if not compatible_vectors:
        return {}
    compatible_matrix = np.stack(compatible_vectors, axis=0)
    scores = _matrix_vector_similarity_scores(compatible_matrix, normalized_query)
    return {row_index: float(score) for row_index, score in zip(compatible_rows, scores)}


def _candidate_embedding_values(candidate: dict) -> list[float]:
    values = candidate.get("embedding_vector")
    if not isinstance(values, list) or not values:
        return []
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError):
        return []
    if vector:
        return vector
    return []


def _precompute_anchor_similarity_scores(anchor_candidate: dict, candidates: list[dict]) -> dict[int, float]:
    anchor_embedding = _candidate_embedding_values(anchor_candidate)
    if not anchor_embedding:
        return {}

    anchor_vector = np.asarray(anchor_embedding, dtype=np.float32)
    if anchor_vector.ndim != 1:
        return {}

    anchor_norm = float(np.linalg.norm(anchor_vector))
    if anchor_norm <= 1e-8:
        return {}
    anchor_vector = anchor_vector / anchor_norm

    row_indices, matrix = _normalized_vector_matrix(candidates, field_name="embedding_vector")
    if matrix is None:
        return {}
    compatible_rows: list[int] = []
    compatible_vectors: list[np.ndarray] = []
    for row_index, row in zip(row_indices, matrix):
        if row.shape[0] != anchor_vector.shape[0]:
            continue
        compatible_rows.append(row_index)
        compatible_vectors.append(row)
    if not compatible_vectors:
        return {}
    compatible_matrix = np.stack(compatible_vectors, axis=0)
    scores = _matrix_vector_similarity_scores(compatible_matrix, anchor_vector)
    return {row_index: float(score) for row_index, score in zip(compatible_rows, scores)}


def _precompute_action_segment_similarity_scores(
    query_embedding: np.ndarray | None,
    candidates: list[dict],
) -> dict[int, float]:
    if query_embedding is None:
        return {}

    normalized_query = np.asarray(query_embedding, dtype=np.float32)
    query_norm = float(np.linalg.norm(normalized_query))
    if query_norm <= 1e-8:
        return {}
    normalized_query = normalized_query / query_norm

    row_indices: list[int] = []
    matrix_rows: list[np.ndarray] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        payload = candidate.get("action_semantic_embedding")
        if not isinstance(payload, dict):
            continue
        segments = payload.get("segments")
        if not isinstance(segments, list):
            continue
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            raw_vector = segment.get("embedding_vector")
            if not isinstance(raw_vector, list) or not raw_vector:
                continue
            vector = np.asarray(raw_vector, dtype=np.float32)
            if vector.ndim != 1 or vector.shape[0] != normalized_query.shape[0]:
                continue
            vector_norm = float(np.linalg.norm(vector))
            if vector_norm <= 1e-8:
                continue
            row_indices.append(index)
            matrix_rows.append(vector / vector_norm)
    if not matrix_rows:
        return {}

    matrix = np.stack(matrix_rows, axis=0)
    scores = _matrix_vector_similarity_scores(matrix, normalized_query)
    best_by_candidate: dict[int, float] = {}
    for row_index, score in zip(row_indices, scores):
        current = best_by_candidate.get(row_index)
        value = float(score)
        if current is None or value > current:
            best_by_candidate[row_index] = value
    return best_by_candidate


def _global_tracking_matches(
    *,
    selected_candidate_id: str,
    candidates: list[dict],
    query_text: str | None,
    max_matches: int = 24,
    per_video_limit: int = 3,
) -> list[dict]:
    normalized_candidates = [candidate for candidate in candidates if isinstance(candidate, dict)]
    if not normalized_candidates:
        return []

    anchor = next(
        (
            candidate for candidate in normalized_candidates
            if str(candidate.get("candidate_id") or "").strip() == str(selected_candidate_id or "").strip()
        ),
        None,
    )
    if anchor is None:
        return []

    cleaned_query = str(query_text or "").strip()
    anchor_scores = _precompute_anchor_similarity_scores(anchor, normalized_candidates)

    ranked: list[dict] = []
    for index, candidate in enumerate(normalized_candidates):
        candidate_view = _candidate_payload_view(candidate)
        anchor_score = float(anchor_scores.get(index) or 0.0)
        semantic_score = _semantic_overlap_itself(cleaned_query, candidate_view) if cleaned_query else 0.0
        visibility_score = _visibility_bonus(candidate_view)
        timeline_bonus = min(len(_normalize_tracking_segments(candidate_view, limit=3)), 3) * 0.01
        total_score = anchor_score * 0.78 + semantic_score * 0.12 + visibility_score * 0.06 + timeline_bonus

        enriched = dict(candidate)
        enriched["anchor_similarity"] = round(anchor_score, 6)
        enriched["score"] = round(total_score, 6)
        if not enriched.get("matched_segments"):
            enriched["matched_segments"] = _normalize_tracking_segments(candidate_view, limit=3)
        ranked.append(enriched)

    ranked.sort(
        key=lambda item: (
            -float(item.get("score") or 0.0),
            -float(item.get("anchor_similarity") or 0.0),
            str(item.get("video_id") or ""),
            str(item.get("candidate_id") or ""),
        )
    )

    selected: list[dict] = []
    used_ids: set[str] = set()
    per_video_counts: dict[str, int] = {}
    for candidate in ranked:
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if not candidate_id or candidate_id in used_ids:
            continue
        video_id = str(candidate.get("video_id") or "").strip()
        if video_id and per_video_counts.get(video_id, 0) >= per_video_limit:
            continue
        selected.append(candidate)
        used_ids.add(candidate_id)
        if video_id:
            per_video_counts[video_id] = per_video_counts.get(video_id, 0) + 1
        if len(selected) >= max(2, min(max_matches, 48)):
            break

    if str(anchor.get("candidate_id") or "").strip() not in used_ids:
        selected.insert(0, anchor)
    return selected


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
        suffix = Path(parsed.path).suffix or ".mp4"
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
    if source_path.suffix.lower() in INGESTION_VIDEO_SUFFIXES:
        return source_path
    raise ValueError(
        f"Only .mp4 inputs are supported for query processing. Got: {source_path.name}"
    )


def _cleanup_remote_query_files(*paths: Path | None) -> None:
    if not settings.cleanup_remote_query_inputs:
        return
    for path in paths:
        if path is None:
            continue
        try:
            if path.exists() and path.is_file():
                path.unlink()
        except Exception as exc:
            logger.debug("Skipped cleanup for %s: %s", path, exc)


def _strict_pipeline_spec() -> dict:
    return get_strict_pipeline()


@lru_cache(maxsize=1)
def _resolved_runtime_bundle() -> tuple[dict, dict, dict, dict]:
    pipeline = _strict_pipeline_spec()
    detected_hardware, execution_plan = resolve_execution_plan(pipeline_spec=pipeline)
    VideoIngestionRuntime._apply_execution_environment(execution_plan)
    acceleration_state = configure_torch_runtime(
        allow_tf32=bool(detected_hardware.get("allow_tf32", True)),
        cudnn_benchmark=bool(detected_hardware.get("cudnn_benchmark", True)),
        host_cpu_count=int(detected_hardware.get("host_cpu_count") or 1),
    )
    return pipeline, detected_hardware, execution_plan, acceleration_state


def prepare_runtime_for_inference() -> dict[str, object]:
    pipeline, detected_hardware, execution_plan, acceleration_state = _resolved_runtime_bundle()
    return {
        "pipeline": pipeline,
        "detected_hardware": detected_hardware,
        "execution_plan": execution_plan,
        "acceleration_state": acceleration_state,
    }


@lru_cache(maxsize=1)
def get_runtime_config() -> dict:
    mode = "remote" if settings.lightning_api_base_url else "local"
    pipeline, detected_hardware, execution_plan, acceleration_state = _resolved_runtime_bundle()
    return {
        "provider": "lightningai",
        "mode": mode,
        "runtime_mode": settings.tracking_runtime_mode,
        "pipeline_summary": pipeline["summary"],
        "validated_on": pipeline["validated_on"],
        "components": pipeline["components"],
        "hyperparameters": pipeline.get("hyperparameters", {}),
        "runtime_defaults": pipeline.get("runtime_defaults", {}),
        "detected_hardware": detected_hardware,
        "execution_plan": execution_plan,
        "acceleration_state": acceleration_state,
        "enable_trackeval": settings.enable_trackeval,
        "enable_geometry_gating": settings.enable_geometry_gating,
        "enable_corrective_cascade": settings.enable_corrective_cascade,
        "lightning_api_base_url": settings.lightning_api_base_url or None,
        "lightning_api_endpoint": settings.lightning_api_endpoint,
    }


@lru_cache(maxsize=1)
def get_runtime() -> StrictTrackerRuntime:
    return StrictTrackerRuntime()


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
            with client.stream("GET", url) as response:
                response.raise_for_status()
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    for chunk in response.iter_bytes():
                        if chunk:
                            f.write(chunk)
    else:
        # Assume it's a local file path - copy if different location
        src = Path(url)
        dst = Path(output_path)
        if src.resolve() != dst.resolve():
            logger.info(f"Copying {src} to {dst}")
            dst.parent.mkdir(parents=True, exist_ok=True)

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
    runtime = prepare_runtime_for_inference()
    detected_hardware = _dict_or_empty(runtime.get("detected_hardware"))
    execution_plan = _dict_or_empty(runtime.get("execution_plan"))
    acceleration_state = _dict_or_empty(runtime.get("acceleration_state"))
    file_exists = bool(storage_path)

    if not settings.lightning_api_base_url.strip():
        raise RuntimeError(
            "LIGHTNING_API_BASE_URL is required in strict pipeline mode. "
            "Local substitute processing is disabled."
        )

    # Remote mode: use Lightning AI
    logger.info("Using Lightning AI remote mode")
    from .gpu_client import get_lightning_client

    input_path: Path | None = None
    prepared_video_path: Path | None = None
    try:
        input_path = _resolve_query_source_path(storage_path, video_id=video_id or None)
        file_exists = input_path.exists()
        prepared_video_path = _prepare_remote_query_video(input_path, video_id=video_id or input_path.stem)
        logger.info(f"Video prepared for Lightning AI: {prepared_video_path}")

        # Step 2: Call Lightning AI
        client = get_lightning_client()
        ai_response = client.process_video(
            video_path=prepared_video_path,
            query_text=query_text,
            query_id=query_id,
            video_id=video_id,
            video_title=payload.get("video_title"),
            metadata=metadata,
            detected_hardware=detected_hardware,
            execution_plan=execution_plan,
            acceleration_state=acceleration_state,
        )

        # Step 3: Download output video from Lightning AI (if available)
        compressed_video_path = ai_response.get("compressed_video_path", "")
        downloaded_output_path = None

        if compressed_video_path and settings.download_remote_outputs:
            try:
                # Ensure output directory exists
                output_dir = Path(settings.video_download_output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Generate output filename
                output_filename = f"{video_id or input_path.stem}_tracked.mp4"
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
            "detected_hardware": detected_hardware,
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
            "query_id": query_id,
            "video_id": video_id,
            "job_id": str(uuid4()),
            "summary": f"Error processing video: {str(e)}",
            "file_exists": file_exists,
            "processed_at": processed_at,
            "detected_hardware": detected_hardware,
            "acceleration_state": acceleration_state,
            "raw_response": {"error": str(e)},
        }
    finally:
        cleanup_targets: list[Path] = []
        if prepared_video_path is not None and prepared_video_path.exists():
            try:
                if input_path is None or prepared_video_path.resolve() != input_path.resolve():
                    cleanup_targets.append(prepared_video_path)
            except Exception:
                cleanup_targets.append(prepared_video_path)
        if input_path is not None and "query-inputs" in input_path.parts:
            cleanup_targets.append(input_path)
        _cleanup_remote_query_files(*cleanup_targets)


# ═══════════════════════════════════════════════════════════════
# Phase 1-5: Full Search Pipeline
# ═══════════════════════════════════════════════════════════════

_ATTRIBUTE_KEYWORDS: dict[str, list[str]] = {
    "gender": ["male", "female", "man", "woman", "boy", "girl", "nam", "nữ"],
    "age_group": ["child", "adult", "elderly", "young", "old", "trẻ em", "người lớn", "người già"],
    "shirt": ["shirt", "áo", "jacket", "hoodie", "sweater", "top"],
    "pants": ["pants", "quần", "shorts", "skirt", "jeans"],
    "shoes": ["shoes", "giày", "sneaker", "boot", "sandal", "dép"],
    "bag": ["bag", "túi", "backpack", "balo", "luggage", "vali"],
    "hat": ["hat", "cap", "mũ", "helmet"],
}

_ACTION_KEYWORDS: list[str] = [
    "run", "walk", "stand", "sit", "fall", "jump", "carry", "push", "pull",
    "chạy", "đi", "đứng", "ngồi", "ngã", "nhảy", "mang", "đẩy",
    "throw", "fight", "loiter", "enter", "exit", "pick", "drop",
]

_COLOR_KEYWORDS: list[str] = [
    "red", "blue", "green", "black", "white", "yellow", "orange", "purple",
    "pink", "brown", "gray", "grey", "đỏ", "xanh", "đen", "trắng", "vàng",
]


def _parse_query_intent(query_text: str) -> dict:
    """Phase 2: Extract attribute tags and action intent from query."""
    lower = query_text.lower()
    tokens = set(re.findall(r"[a-zA-Zàáảãạăắặẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]+", lower))

    attributes: dict[str, list[str]] = {}
    for attr, keywords in _ATTRIBUTE_KEYWORDS.items():
        matched = [kw for kw in keywords if kw in lower]
        if matched:
            attributes[attr] = matched

    colors = [c for c in _COLOR_KEYWORDS if c in lower]
    if colors:
        attributes["colors"] = colors

    actions = [a for a in _ACTION_KEYWORDS if a in lower]

    return {
        "raw_query": query_text,
        "tokens": tokens,
        "attributes": attributes,
        "actions": actions,
        "has_action_intent": len(actions) > 0,
        "has_appearance_intent": len(attributes) > 0 or len(colors) > 0,
    }


def _encode_query_clip(query_text: str) -> np.ndarray | None:
    """
    Phase 2: Encode query text with SigLIP2 ViT-L-16-512 → 1024-dim L2-normalised vector.

    SigLIP2 text and image features share the same embedding space, enabling
    direct cosine similarity between query text and appearance embeddings.
    Returns None if encoding fails — scoring gracefully degrades to lexical only.
    """
    try:
        from .model_adapters import SigLIP2ModelHub
        hub = SigLIP2ModelHub()
        feats = hub.text_features([query_text])  # [1, 1024]
        vec = feats[0].astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-8 else None
    except Exception as exc:
        logger.warning("SigLIP2 query encoding failed: %s", exc)
        return None


def _hard_filter_candidates(
    candidates: list[dict],
    camera_ids: list[str] | None,
    time_from: str | None,
    time_to: str | None,
) -> list[dict]:
    """Phase 3 — Hard filter: remove candidates outside time window or camera zone."""
    if not camera_ids and not time_from and not time_to:
        return candidates

    result = []
    tf = datetime.fromisoformat(time_from) if time_from else None
    tt = datetime.fromisoformat(time_to) if time_to else None
    cam_set = {c.lower().strip() for c in (camera_ids or [])} if camera_ids else None

    for c in candidates:
        if cam_set:
            cam = str(c.get("camera_id") or "").lower().strip()
            if cam and cam not in cam_set:
                continue

        if tf or tt:
            timeline = c.get("timeline")
            if isinstance(timeline, list) and timeline:
                start = None
                for seg in timeline:
                    if isinstance(seg, dict) and seg.get("start_second") is not None:
                        try:
                            recorded_start = c.get("recorded_start") or ""
                            if recorded_start:
                                base = datetime.fromisoformat(str(recorded_start))
                                seg_time = base.replace(tzinfo=timezone.utc) + timedelta(seconds=float(seg["start_second"]))
                                start = seg_time
                                break
                        except Exception:
                            pass
                if start:
                    if tf and start < tf.replace(tzinfo=timezone.utc):
                        continue
                    if tt and start > tt.replace(tzinfo=timezone.utc):
                        continue
        result.append(c)
    return result


def _score_attribute_match(query_intent: dict, candidate: dict) -> float:
    """Phase 3 — Metadata match: +score for each matching attribute, -0.03 for mismatch."""
    if not query_intent["attributes"] and not query_intent["actions"]:
        return 0.0

    attrs_query = query_intent["attributes"]
    search_doc = (_candidate_search_document(candidate) + " " + str(candidate.get("search_text") or "")).lower()
    semantic_attrs = [str(a).lower() for a in _coerce_string_list(candidate.get("semantic_attributes"))]
    full_text = search_doc + " " + " ".join(semantic_attrs)

    score = 0.0
    matched = 0
    total = 0

    for attr_type, keywords in attrs_query.items():
        total += 1
        if any(kw in full_text for kw in keywords):
            score += 0.12
            matched += 1
        else:
            score -= 0.03

    for action in query_intent["actions"]:
        total += 1
        if action in full_text:
            score += 0.10
            matched += 1

    return round(score, 6)


def _score_attribute_match_precomputed(query_intent: dict, *, full_text: str) -> float:
    if not query_intent["attributes"] and not query_intent["actions"]:
        return 0.0

    score = 0.0
    for keywords in query_intent["attributes"].values():
        if any(keyword in full_text for keyword in keywords):
            score += 0.12
        else:
            score -= 0.03

    for action in query_intent["actions"]:
        if action in full_text:
            score += 0.10

    return round(score, 6)


def _rank_candidate_multimodal(
    query_text: str,
    query_clip_embed: np.ndarray | None,
    query_intent: dict,
    candidate: dict,
) -> float:
    """
    Phase 3+4: Hybrid multi-modal scoring.

    Weights:
      W_app  = 0.42  — CLIP text vs attribute_embedding_vector (768-dim)
      W_act  = 0.20  — CLIP text vs action_semantic_embedding (768-dim)
      W_meta = 0.18  — Attribute keyword match
      W_sem  = 0.12  — Token Jaccard semantic overlap
      W_vis  = 0.05  — Visibility bonus
      W_world= 0.03  — World position bonus
    """
    view = _candidate_payload_view(candidate)

    # --- Appearance similarity (CLIP text vs CLIP image attributes) ---
    app_sim = 0.0
    if query_clip_embed is not None:
        attr_embed = view.get("attribute_embedding_vector")
        if isinstance(attr_embed, list) and len(attr_embed) == query_clip_embed.shape[0]:
            app_sim = _embedding_similarity(query_clip_embed, attr_embed)

    # --- Action similarity (CLIP text vs ITSELF action embedding) ---
    act_sim = 0.0
    if query_clip_embed is not None and query_intent["has_action_intent"]:
        act_sim = _action_embedding_similarity(query_clip_embed, view.get("action_semantic_embedding"))

    # --- Attribute metadata match ---
    meta_score = _score_attribute_match(query_intent, view)

    # --- Token Jaccard ---
    sem_overlap = _semantic_overlap_itself(query_text, view)

    # --- Visibility & world bonus ---
    vis_bonus = _visibility_bonus(view)
    world_bonus = _world_position_bonus(view)

    final = (
        app_sim   * 0.42 +
        act_sim   * 0.20 +
        meta_score* 0.18 +
        sem_overlap* 0.12 +
        vis_bonus  * 0.05 +
        world_bonus* 0.03
    )
    return round(final, 6)


def _deduplicate_cross_camera(
    scored: list[tuple[float, dict]],
    similarity_threshold: float = 0.85,
    per_camera_limit: int = 2,
) -> list[dict]:
    """
    Phase 5: Cluster candidates by embedding similarity across cameras.
    Each identity cluster contributes at most 1 representative to results.
    """
    if not scored:
        return []

    candidates = [item[1] for item in scored]
    row_indices, matrix = _normalized_vector_matrix(candidates, field_name="embedding_vector")
    if matrix is None:
        row_indices = []
        matrix = np.zeros((0, 0), dtype=np.float32)
    matrix_by_scored_index = {row_index: row for row_index, row in zip(row_indices, matrix)}

    clusters: list[list[int]] = []
    assigned = [False] * len(scored)
    for i in range(len(scored)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        rep_vector = matrix_by_scored_index.get(i)
        if rep_vector is not None:
            similarities = _matrix_vector_similarity_scores(matrix, rep_vector)
            for row_index, similarity in zip(row_indices, similarities):
                if row_index <= i or assigned[row_index]:
                    continue
                if float(similarity) >= similarity_threshold:
                    cluster.append(row_index)
                    assigned[row_index] = True
        clusters.append(cluster)

    result = []
    cam_counts: dict[str, int] = {}
    for cluster in clusters:
        best_idx = cluster[0]
        enriched = dict(scored[best_idx][1])
        enriched["score"] = round(scored[best_idx][0], 6)
        if len(cluster) > 1:
            enriched["cross_camera_matches"] = [
                {"candidate_id": scored[idx][1].get("candidate_id"), "camera_id": scored[idx][1].get("camera_id")}
                for idx in cluster[1:]
            ]
        cam = str(enriched.get("camera_id") or "")
        if cam and cam_counts.get(cam, 0) >= per_camera_limit:
            continue
        cam_counts[cam] = cam_counts.get(cam, 0) + 1
        result.append(enriched)

    return result


def search_candidates_remote(
    query_text: str,
    candidates: list[dict],
    limit: int = 5,
    camera_ids: list[str] | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
    weights: dict | None = None,
) -> dict:
    """
    Full 5-phase search pipeline:
    1. Parse query intent (attributes, actions)
    2. Encode query with CLIP ViT-L/14 (768-dim)
    3. Hard filter (camera zone + time window) + Hybrid scoring
    4. Weighted fusion: app(0.42) + act(0.20) + meta(0.18) + sem(0.12) + vis(0.05) + world(0.03)
    5. Cross-camera deduplication + diversity Top-k
    """
    cleaned_query = str(query_text or "").strip()
    if not cleaned_query:
        return {"query_text": cleaned_query, "count": 0, "items": []}

    runtime = prepare_runtime_for_inference()
    pipeline = runtime["pipeline"] if isinstance(runtime, dict) else _strict_pipeline_spec()
    semantic_cfg = _dict_or_empty(_dict_or_empty(pipeline.get("hyperparameters")).get("semantic_search"))
    bounded_limit = max(1, min(int(limit or 5), 50))
    normalized_candidates = [c for c in candidates if isinstance(c, dict)]
    if not normalized_candidates:
        return {"query_text": cleaned_query, "count": 0, "items": []}

    # Phase 1+2: Parse intent & encode query
    query_intent = _parse_query_intent(cleaned_query)
    query_clip_embed = _encode_query_clip(cleaned_query)
    logger.info(
        "Query parsed: attributes=%s actions=%s clip_encoded=%s candidates=%d",
        list(query_intent["attributes"].keys()),
        query_intent["actions"],
        query_clip_embed is not None,
        len(normalized_candidates),
    )

    # Phase 3a: Hard filter
    filtered = _hard_filter_candidates(normalized_candidates, camera_ids, time_from, time_to)
    logger.info("After hard filter: %d/%d candidates", len(filtered), len(normalized_candidates))

    candidate_views = [_candidate_payload_view(candidate) for candidate in filtered]
    precomputed_app_scores = _precompute_field_similarity_scores(
        query_clip_embed,
        candidate_views,
        field_name="attribute_embedding_vector",
    )
    precomputed_action_scores = (
        _precompute_action_segment_similarity_scores(query_clip_embed, candidate_views)
        if query_clip_embed is not None and query_intent["has_action_intent"]
        else {}
    )

    score_weights = {
        "app": 0.42,
        "act": 0.20,
        "meta": 0.18,
        "sem": 0.12,
        "vis": 0.05,
        "world": 0.03,
    }
    semantic_weight_overrides = _dict_or_empty(semantic_cfg.get("score_weights"))
    for key, value in semantic_weight_overrides.items():
        if key == "semantic_overlap":
            score_weights["sem"] = float(value)
        elif key in ("embedding", "app"):
            score_weights["app"] = float(value)
        elif key == "visibility":
            score_weights["vis"] = float(value)
        elif key == "world_position":
            score_weights["world"] = float(value)
        elif key in score_weights:
            score_weights[key] = float(value)
    for key, value in _dict_or_empty(weights).items():
        if key in score_weights and isinstance(value, (int, float)):
            score_weights[key] = float(value)

    # Phase 3b+4: Hybrid scoring per candidate
    scored: list[tuple[float, dict]] = []
    for index, (candidate, candidate_view) in enumerate(zip(filtered, candidate_views)):
        search_text = _candidate_search_document(candidate_view)
        semantic_attrs = [str(item).lower() for item in _coerce_string_list(candidate_view.get("semantic_attributes"))]
        full_text = f"{search_text} {' '.join(semantic_attrs)}".strip().lower()
        semantic_overlap = 0.0
        if search_text:
            semantic_overlap = _semantic_overlap_itself(cleaned_query, {**candidate_view, "search_text": search_text})
        matched_segments = _normalize_tracking_segments(candidate_view, limit=3)
        app_sim = float(precomputed_app_scores.get(index) or 0.0)
        act_sim = float(precomputed_action_scores.get(index) or 0.0)
        meta_score = _score_attribute_match_precomputed(query_intent, full_text=full_text)
        vis_bonus = _visibility_bonus(candidate_view)
        world_bonus = _world_position_bonus(candidate_view)
        score = round(
            app_sim * score_weights["app"] +
            act_sim * score_weights["act"] +
            meta_score * score_weights["meta"] +
            semantic_overlap * score_weights["sem"] +
            vis_bonus * score_weights["vis"] +
            world_bonus * score_weights["world"],
            6,
        )
        enriched = dict(candidate)
        enriched["search_text"] = search_text
        enriched["semantic_overlap"] = round(semantic_overlap, 6)
        if not isinstance(enriched.get("matched_segments"), list) or not enriched.get("matched_segments"):
            enriched["matched_segments"] = matched_segments
        if not isinstance(enriched.get("timeline"), list) and isinstance(candidate_view.get("timeline"), list):
            enriched["timeline"] = candidate_view.get("timeline")
        scored.append((score, enriched))

    scored.sort(key=lambda x: (-x[0], str(x[1].get("video_id") or ""), str(x[1].get("candidate_id") or "")))

    # Phase 5: Cross-camera dedup + diversity Top-k
    fetch_multiplier = max(2, int(semantic_cfg.get("fetch_multiplier") or 8))
    pre_dedup_limit = min(len(scored), max(bounded_limit, bounded_limit * fetch_multiplier))
    dedup_threshold = float(semantic_cfg.get("dedup_similarity_threshold") or 0.85)
    items = _deduplicate_cross_camera(
        scored[:pre_dedup_limit],
        similarity_threshold=dedup_threshold,
        per_camera_limit=max(1, bounded_limit // 5 + 1),
    )
    items = items[:bounded_limit]

    return {"query_text": cleaned_query, "count": len(items), "items": items}


def _resolve_remote_candidate_source_path(candidate: dict, artifact_id: str) -> Path:
    source_root = _artifact_root() / artifact_id / "sources"
    source_root.mkdir(parents=True, exist_ok=True)

    drive_file_id = str(candidate.get("drive_video_file_id") or "").strip()
    if drive_file_id:
        public_url = f"https://drive.google.com/uc?id={drive_file_id}&export=download&confirm=t"
        filename = Path(
            str(candidate.get("source_filename") or candidate.get("video_title") or candidate.get("candidate_id") or drive_file_id)
        ).name
        if not Path(filename).suffix:
            filename = f"{filename}.mp4"
        target_path = source_root / filename
        _download_file(public_url, str(target_path))
        return target_path

    for key in ("available_link_video", "storage_path", "local_video_path"):
        value = str(candidate.get(key) or "").strip()
        if not value:
            continue
        if value.startswith(("http://", "https://")):
            suffix = Path(urlparse(value).path).suffix or ".mp4"
            target_path = source_root / f"{_tracking_slug(candidate.get('candidate_id') or 'candidate')}{suffix}"
            _download_file(value, str(target_path))
            return target_path
        path = Path(value).expanduser()
        if path.exists():
            return path

    raise FileNotFoundError(f"Unable to resolve source video for candidate {candidate.get('candidate_id')}")


def _prepare_remote_tracking_jobs(
    selected_candidates: list[dict],
    *,
    artifact_id: str,
    max_segments_per_candidate: int,
    source_resolve_workers_cap: int = 8,
) -> list[dict[str, object]]:
    if not selected_candidates:
        return []

    configured_workers = max(1, int(os.environ.get("MCPT_PARALLEL_VIDEO_JOBS", "4")))
    max_workers = max(1, min(len(selected_candidates), configured_workers, source_resolve_workers_cap))

    def resolve(candidate: dict) -> tuple[str, Path, list[dict[str, float | str]], dict]:
        source_path = _resolve_remote_candidate_source_path(candidate, artifact_id)
        segments = _normalize_tracking_segments(candidate, max_segments_per_candidate)
        return str(candidate.get("candidate_id") or ""), source_path, segments, candidate

    resolved_rows: list[tuple[str, Path, list[dict[str, float | str]], dict]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for row in executor.map(resolve, selected_candidates):
            resolved_rows.append(row)

    grouped_jobs: dict[str, dict[str, object]] = {}
    for candidate_id, source_path, segments, candidate in resolved_rows:
        if not segments:
            continue
        source_key = str(source_path)
        job = grouped_jobs.setdefault(
            source_key,
            {"source_path": source_path, "clips": []},
        )
        for segment in segments:
            job["clips"].append(
                {
                    "candidate_id": candidate_id,
                    "camera_id": candidate.get("camera_id"),
                    "track_id": candidate.get("track_id"),
                    "search_text": candidate.get("search_text"),
                    "start_second": float(segment["start_second"]),
                    "end_second": float(segment["end_second"]),
                    "action_summary": str(segment["action_summary"]),
                }
            )
    jobs = list(grouped_jobs.values())
    for job in jobs:
        clips = job.get("clips")
        if isinstance(clips, list):
            clips.sort(
                key=lambda clip: (
                    float(clip.get("start_second") or 0.0),
                    float(clip.get("end_second") or 0.0),
                    str(clip.get("candidate_id") or ""),
                )
            )
    return jobs


def build_tracking_video_remote(
    *,
    selected_candidate_id: str,
    candidates: list[dict],
    candidate_ids: list[str] | None = None,
    query_text: str | None = None,
    max_segments_per_candidate: int = 2,
) -> dict:
    import cv2

    runtime = prepare_runtime_for_inference()
    pipeline = _dict_or_empty(runtime.get("pipeline"))
    trace_cfg = _dict_or_empty(_dict_or_empty(pipeline.get("hyperparameters")).get("trace"))
    runtime_defaults = _dict_or_empty(pipeline.get("runtime_defaults"))
    selected_candidates = _global_tracking_matches(
        selected_candidate_id=selected_candidate_id,
        candidates=candidates,
        query_text=query_text,
        max_matches=max(12, min(len(candidates), 24)),
    )
    if not selected_candidates:
        raise FileNotFoundError("No candidates found for remote tracking build")

    if candidate_ids:
        allowed_ids = {str(candidate_id or "").strip() for candidate_id in candidate_ids if str(candidate_id or "").strip()}
        allowed_ids.add(str(selected_candidate_id or "").strip())
        selected_candidates = [
            candidate
            for candidate in selected_candidates
            if str(candidate.get("candidate_id") or "").strip() in allowed_ids
        ]
        if not selected_candidates:
            raise FileNotFoundError("No selected candidates remained after candidate_ids filtering")

    artifact_id = uuid4().hex
    output_path, manifest_path = resolve_tracking_artifact_paths(artifact_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    written_frames = 0
    output_fps_cap = float(runtime_defaults.get("trace_output_fps_cap") or 12.0)
    output_fps = output_fps_cap
    output_size: tuple[int, int] | None = None
    clips_manifest: list[dict[str, object]] = []
    grouped_jobs = _prepare_remote_tracking_jobs(
        selected_candidates,
        artifact_id=artifact_id,
        max_segments_per_candidate=max_segments_per_candidate,
        source_resolve_workers_cap=max(1, int(trace_cfg.get("source_resolve_workers_cap") or 8)),
    )

    for job in grouped_jobs:
        source_path = job["source_path"]
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            continue
        fps = float(cap.get(cv2.CAP_PROP_FPS) or output_fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            continue
        if output_size is None:
            output_size = (width, height)
            output_fps = max(8.0, min(fps or output_fps_cap, output_fps_cap))
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                output_fps,
                output_size,
            )
        current_frame = 0
        for clip in job["clips"]:
            start_second = float(clip["start_second"])
            end_second = float(clip["end_second"])
            start_frame = max(0, int(start_second * fps))
            end_frame = max(start_frame, int(end_second * fps))
            if start_frame < current_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                current_frame = start_frame
            elif start_frame > current_frame:
                while current_frame < start_frame:
                    if not cap.grab():
                        break
                    current_frame += 1
            while current_frame <= end_frame:
                ok, frame = cap.read()
                if not ok:
                    break
                current_frame += 1
                if output_size and (frame.shape[1], frame.shape[0]) != output_size:
                    frame = cv2.resize(frame, output_size)
                overlay_1 = f"Query: {query_text or 'candidate tracking'}"
                overlay_2 = f"{clip.get('camera_id') or 'camera'} | track {clip.get('track_id') or '?'}"
                overlay_3 = str(clip.get("action_summary") or clip.get("search_text") or "")[:120]
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
                cv2.putText(frame, overlay_1[:120], (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
                cv2.putText(frame, overlay_2[:120], (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, overlay_3[:120], (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
                if writer is not None:
                    writer.write(frame)
                    written_frames += 1
            clips_manifest.append(
                {
                    "candidate_id": clip.get("candidate_id"),
                    "camera_id": clip.get("camera_id"),
                    "track_id": clip.get("track_id"),
                    "source_video": str(source_path),
                    "start_second": start_second,
                    "end_second": end_second,
                    "action_summary": clip.get("action_summary"),
                }
            )
        cap.release()

    if writer is not None:
        writer.release()

    if written_frames <= 0 or not output_path.exists():
        output_path.unlink(missing_ok=True)
        raise RuntimeError("Could not create remote tracking compilation from the selected candidates")

    manifest = {
        "artifact_id": artifact_id,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "query_text": query_text,
        "selected_candidate_id": selected_candidate_id,
        "candidate_ids": [candidate.get("candidate_id") for candidate in selected_candidates],
        "global_tracking_mode": "anchor_similarity_gpu",
        "clip_count": len(clips_manifest),
        "written_frames": written_frames,
        "video_path": str(output_path),
        "clips": clips_manifest,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "artifact_id": artifact_id,
        "video_url": f"/api/v1/tracking-artifacts/{artifact_id}",
        "manifest_url": f"/api/v1/tracking-artifacts/{artifact_id}/manifest",
        "manifest": manifest,
        "selected_candidate_id": selected_candidate_id,
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
        source_url=str(payload.get("source_url") or "").strip(),
        source_filename=str(payload.get("source_filename") or "").strip() or None,
        camera_id=payload.get("camera_id"),
        recorded_start=payload.get("recorded_start"),
        output_video_dir=payload.get("output_video_dir"),
        output_metadata_dir=payload.get("output_metadata_dir"),
        output_basename=payload.get("output_basename"),
        metadata=_dict_or_empty(payload.get("metadata")),
    )


def process_video_ingestion_background(job_id: str, payload: dict) -> None:
    with _INGESTION_STATUS_LOCK:
        _INGESTION_BACKGROUND_STATUS[job_id] = {"status": "running"}
    try:
        result = process_video_ingestion(payload)
        with _INGESTION_STATUS_LOCK:
            _INGESTION_BACKGROUND_STATUS[job_id] = {"status": "completed", "result": result}
    except Exception as exc:
        logger.exception("Background ingestion failed job_id=%s", job_id)
        with _INGESTION_STATUS_LOCK:
            _INGESTION_BACKGROUND_STATUS[job_id] = {
                "status": "failed",
                "error": type(exc).__name__,
                "message": str(exc),
            }


def get_ingestion_background_status(job_id: str) -> dict[str, object]:
    with _INGESTION_STATUS_LOCK:
        state = _INGESTION_BACKGROUND_STATUS.get(job_id)
        if state is None:
            return {"status": "not_found", "job_id": job_id}
        return {"job_id": job_id, **state}
