from __future__ import annotations

import json
import logging
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
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
_SHARED_TEXT_MODEL = None
_SHARED_TEXT_TOKENIZER = None
_SHARED_TEXT_DEVICE = None
_SHARED_TEXT_MODEL_LOCK = threading.Lock()


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
        "matched_segments",
        "embedding_vector",
        "candidate_vector",
        "itself_features",
    ):
        if payload.get(key) not in (None, "", []):
            merged[key] = payload.get(key)
    return merged


def _get_text_model_components() -> tuple[object, object, str]:
    global _SHARED_TEXT_MODEL
    global _SHARED_TEXT_TOKENIZER
    global _SHARED_TEXT_DEVICE

    if _SHARED_TEXT_MODEL is not None and _SHARED_TEXT_TOKENIZER is not None and _SHARED_TEXT_DEVICE is not None:
        return _SHARED_TEXT_MODEL, _SHARED_TEXT_TOKENIZER, _SHARED_TEXT_DEVICE

    with _SHARED_TEXT_MODEL_LOCK:
        if _SHARED_TEXT_MODEL is None or _SHARED_TEXT_TOKENIZER is None or _SHARED_TEXT_DEVICE is None:
            import open_clip

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, _preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
            model = model.to(device)
            model.eval()
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            _SHARED_TEXT_MODEL = model
            _SHARED_TEXT_TOKENIZER = tokenizer
            _SHARED_TEXT_DEVICE = device
    return _SHARED_TEXT_MODEL, _SHARED_TEXT_TOKENIZER, _SHARED_TEXT_DEVICE


def _compute_query_embedding(query_text: str) -> np.ndarray | None:
    cleaned_query = str(query_text or "").strip()
    if not cleaned_query:
        return None
    try:
        model, tokenizer, device = _get_text_model_components()
        text_tokens = tokenizer([cleaned_query]).to(device)
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" else nullcontext()
        with torch.inference_mode():
            with autocast_ctx:
                query_emb = model.encode_text(text_tokens)
            query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
        return query_emb.detach().cpu().numpy()[0].astype(np.float32)
    except Exception as exc:
        logger.warning("Failed to compute query embedding: %s", exc)
        return None


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
    precomputed_embedding_similarity: float | None = None,
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


def _precompute_candidate_embedding_scores(query_embedding: np.ndarray | None, candidates: list[dict]) -> dict[int, float]:
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
        raw_embedding = candidate.get("embedding_vector")
        if not isinstance(raw_embedding, list) or not raw_embedding:
            continue
        candidate_vector = np.asarray(raw_embedding, dtype=np.float32)
        if candidate_vector.ndim != 1 or candidate_vector.shape[0] != normalized_query.shape[0]:
            continue
        candidate_norm = float(np.linalg.norm(candidate_vector))
        if candidate_norm <= 1e-8:
            continue
        row_indices.append(index)
        matrix_rows.append(candidate_vector / candidate_norm)

    if not matrix_rows:
        return {}

    matrix = np.stack(matrix_rows, axis=0)
    scores = matrix @ normalized_query
    return {row_index: float(score) for row_index, score in zip(row_indices, scores)}


def _candidate_embedding_values(candidate: dict) -> list[float]:
    for key in ("embedding_vector", "candidate_vector", "itself_features"):
        values = candidate.get(key)
        if not isinstance(values, list) or not values:
            continue
        try:
            vector = [float(value) for value in values]
        except (TypeError, ValueError):
            continue
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

    row_indices: list[int] = []
    matrix_rows: list[np.ndarray] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        candidate_embedding = _candidate_embedding_values(candidate)
        if not candidate_embedding or len(candidate_embedding) != anchor_vector.shape[0]:
            continue
        candidate_vector = np.asarray(candidate_embedding, dtype=np.float32)
        candidate_norm = float(np.linalg.norm(candidate_vector))
        if candidate_norm <= 1e-8:
            continue
        row_indices.append(index)
        matrix_rows.append(candidate_vector / candidate_norm)

    if not matrix_rows:
        return {}

    if torch.cuda.is_available():
        device = torch.device("cuda")
        matrix_tensor = torch.as_tensor(np.stack(matrix_rows, axis=0), device=device)
        anchor_tensor = torch.as_tensor(anchor_vector, device=device)
        scores_tensor = matrix_tensor @ anchor_tensor
        scores = scores_tensor.detach().cpu().numpy()
    else:
        matrix = np.stack(matrix_rows, axis=0)
        scores = matrix @ anchor_vector
    return {row_index: float(score) for row_index, score in zip(row_indices, scores)}


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

    input_path: Path | None = None
    h265_path: Path | None = None
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

        if compressed_video_path and settings.download_remote_outputs:
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
    finally:
        cleanup_targets: list[Path] = []
        if h265_path is not None and h265_path.exists():
            try:
                if input_path is None or h265_path.resolve() != input_path.resolve():
                    cleanup_targets.append(h265_path)
            except Exception:
                cleanup_targets.append(h265_path)
        if input_path is not None and "query-inputs" in input_path.parts:
            cleanup_targets.append(input_path)
        _cleanup_remote_query_files(*cleanup_targets)


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

    query_embedding = _compute_query_embedding(query_text)

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


def search_candidates_remote(query_text: str, candidates: list[dict], limit: int = 5) -> dict:
    cleaned_query = str(query_text or "").strip()
    if not cleaned_query:
        return {"query_text": cleaned_query, "count": 0, "items": []}

    query_embedding = _compute_query_embedding(cleaned_query)
    embedding_scores = _precompute_candidate_embedding_scores(query_embedding, candidates)
    ranked: list[dict] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        candidate_view = _candidate_payload_view(candidate)
        score = _rank_candidate_itself(
            cleaned_query,
            query_embedding,
            candidate_view,
            precomputed_embedding_similarity=embedding_scores.get(index),
        )
        if score <= 0:
            continue
        enriched = dict(candidate)
        enriched["score"] = score
        if not enriched.get("matched_segments"):
            enriched["matched_segments"] = _normalize_tracking_segments(candidate_view, limit=3)
        ranked.append(enriched)

    ranked.sort(
        key=lambda item: (
            -float(item.get("score") or 0.0),
            str(item.get("camera_id") or ""),
            str(item.get("track_id") or ""),
            str(item.get("candidate_id") or ""),
        )
    )
    limited = ranked[: max(1, min(limit, 50))]
    return {"query_text": cleaned_query, "count": len(limited), "items": limited}


def _resolve_remote_candidate_source_path(candidate: dict, artifact_id: str) -> Path:
    runtime = get_ingestion_runtime()
    source_root = _artifact_root() / artifact_id / "sources"
    source_root.mkdir(parents=True, exist_ok=True)

    drive_file_id = str(candidate.get("drive_video_file_id") or "").strip()
    if drive_file_id:
        filename = Path(
            str(candidate.get("source_filename") or candidate.get("video_title") or candidate.get("candidate_id") or drive_file_id)
        ).name
        if not Path(filename).suffix:
            filename = f"{filename}.h265"
        target_path = source_root / filename
        runtime._download_drive_file(drive_file_id, target_path)
        return target_path

    for key in ("available_link_video", "storage_path", "local_video_path"):
        value = str(candidate.get(key) or "").strip()
        if not value:
            continue
        if value.startswith(("http://", "https://")):
            suffix = Path(urlparse(value).path).suffix or ".h265"
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
) -> list[dict[str, object]]:
    if not selected_candidates:
        return []

    max_workers = max(1, min(len(selected_candidates), 4))

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
    return list(grouped_jobs.values())


def build_tracking_video_remote(
    *,
    selected_candidate_id: str,
    candidates: list[dict],
    candidate_ids: list[str] | None = None,
    query_text: str | None = None,
    max_segments_per_candidate: int = 2,
) -> dict:
    import cv2

    selected_candidates = _global_tracking_matches(
        selected_candidate_id=selected_candidate_id,
        candidates=candidates,
        query_text=query_text,
        max_matches=max(12, min(len(candidates), 24)),
    )
    if not selected_candidates:
        raise FileNotFoundError("No candidates found for remote tracking build")

    artifact_id = uuid4().hex
    output_path, manifest_path = resolve_tracking_artifact_paths(artifact_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    written_frames = 0
    output_fps = 12.0
    output_size: tuple[int, int] | None = None
    clips_manifest: list[dict[str, object]] = []
    grouped_jobs = _prepare_remote_tracking_jobs(
        selected_candidates,
        artifact_id=artifact_id,
        max_segments_per_candidate=max_segments_per_candidate,
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
            output_fps = max(8.0, min(fps or 12.0, 24.0))
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                output_fps,
                output_size,
            )
        for clip in job["clips"]:
            start_second = float(clip["start_second"])
            end_second = float(clip["end_second"])
            start_frame = max(0, int(start_second * fps))
            end_frame = max(start_frame, int(end_second * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame
            while frame_idx <= end_frame:
                ok, frame = cap.read()
                if not ok:
                    break
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
                frame_idx += 1
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
