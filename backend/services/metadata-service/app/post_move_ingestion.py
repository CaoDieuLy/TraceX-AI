from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .storage_ingest import StorageVideoItem


STORAGE_INGEST_SOURCE_MODE = "storage_ingest"


@dataclass(frozen=True)
class DecodeSamplingPolicy:
    """Fixed decode/sampling contract for post-move storage ingestion."""

    container_suffix: str = ".mp4"
    codec: str = "h265"
    sample_fps: int = 5
    keep_container_as_mp4: bool = True

    def to_metadata(self) -> dict[str, Any]:
        return {
            "container_suffix": self.container_suffix,
            "codec": self.codec,
            "sample_fps": self.sample_fps,
            "keep_container_as_mp4": self.keep_container_as_mp4,
        }


@dataclass(frozen=True)
class TrackingPolicy:
    """Strict per-video tracking contract after decode/sampling."""

    detector_name: str = "RF-DETR 2x-large"
    detector_class_name: str = "RFDETR2XLarge"
    detector_inference_alias: str = "rfdetr-2xlarge"
    detector_package: str = "rfdetr[plus]"
    tracker_name: str = "OCMCTrack-style corrective cascade"
    execution_scope: str = "per_video"
    independence_rule: str = "each_10_min_video_is_independent"
    output_fields: tuple[str, ...] = ("video_id", "track_id", "bboxes")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "detector_class_name": self.detector_class_name,
            "detector_inference_alias": self.detector_inference_alias,
            "detector_package": self.detector_package,
            "tracker_name": self.tracker_name,
            "execution_scope": self.execution_scope,
            "independence_rule": self.independence_rule,
            "output_fields": list(self.output_fields),
        }


@dataclass(frozen=True)
class TrackletQualityPolicy:
    """Tracklet filtering contract for low-confidence / low-information tracks."""

    enabled: bool = True
    minimum_confidence_score: float = 0.3
    minimum_tracklet_duration_seconds: float = 2.0
    reject_blurry_tracklets: bool = True
    reject_incomplete_tracklets: bool = True

    def to_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "minimum_confidence_score": self.minimum_confidence_score,
            "minimum_tracklet_duration_seconds": self.minimum_tracklet_duration_seconds,
            "reject_blurry_tracklets": self.reject_blurry_tracklets,
            "reject_incomplete_tracklets": self.reject_incomplete_tracklets,
        }


@dataclass(frozen=True)
class TrackletFeaturePipelinePolicy:
    """Unified tracklet feature pipeline for attribute, appearance, and action."""

    selection_mode: str = "hybrid_track_mean_plus_topk"
    keyframe_selector: str = "laplacian_variance_with_bbox_area_rank"
    selection_score_inputs: tuple[str, ...] = ("laplacian_variance", "bbox_area", "detection_confidence")
    selected_frame_count: int = 5
    max_selected_frame_count: int = 10
    pooling_methods: tuple[str, ...] = (
        "mean_pooling",
        "max_pooling",
        "quality_weighted_mean",
    )
    static_attribute_extractor: str = "static_attribute_extraction"
    static_attribute_fields: tuple[str, ...] = ("gender", "age_group")
    static_attribute_embedding_fields: tuple[str, ...] = ("attribute_embedding_vector",)
    appearance_attribute_extractor: str = "appearance_attribute_extraction"
    appearance_attribute_fields: tuple[str, ...] = (
        "head_accessory",
        "hat",
        "hair_color",
        "skin_tone",
        "shirt",
        "pants",
        "shoes",
        "bag",
    )
    appearance_embedding_extractor: str = "SOLIDER + KPR"
    appearance_embedding_output_fields: tuple[str, ...] = (
        "appearance_embedding_vector",
        "embedding_vector",
        "tracklet_vectors",
    )
    action_clip_builder: str = "tracklet_action_clip_builder"
    action_clip_duration_seconds: float = 2.0
    action_clip_stride_seconds: float = 1.5
    action_behavior_analyzer: str = "action_behavior_analysis"
    action_semantic_embedding_model: str = "ITSELF"
    action_output_fields: tuple[str, ...] = ("timeline", "matched_segments", "action_semantic_embedding")
    aggregation_target_vectors: int = 3

    def to_metadata(self) -> dict[str, Any]:
        return {
            "always_run": True,
            "selection_mode": self.selection_mode,
            "keyframe_selector": self.keyframe_selector,
            "selection_score_inputs": list(self.selection_score_inputs),
            "selected_frame_count": self.selected_frame_count,
            "max_selected_frame_count": self.max_selected_frame_count,
            "pooling_methods": list(self.pooling_methods),
            "static_attribute_extractor": self.static_attribute_extractor,
            "static_attribute_fields": list(self.static_attribute_fields),
            "static_attribute_embedding_fields": list(self.static_attribute_embedding_fields),
            "appearance_attribute_extractor": self.appearance_attribute_extractor,
            "appearance_attribute_fields": list(self.appearance_attribute_fields),
            "appearance_embedding_extractor": self.appearance_embedding_extractor,
            "appearance_embedding_output_fields": list(self.appearance_embedding_output_fields),
            "action_clip_builder": self.action_clip_builder,
            "action_clip_duration_seconds": self.action_clip_duration_seconds,
            "action_clip_stride_seconds": self.action_clip_stride_seconds,
            "action_behavior_analyzer": self.action_behavior_analyzer,
            "action_semantic_embedding_model": self.action_semantic_embedding_model,
            "action_output_fields": list(self.action_output_fields),
            "aggregation_target_vectors": self.aggregation_target_vectors,
        }


@dataclass(frozen=True)
class PostMoveIngestionPolicy:
    """End-to-end contract for everything that happens after move.py finishes."""

    source_mode: str = STORAGE_INGEST_SOURCE_MODE
    decode_sampling: DecodeSamplingPolicy = field(default_factory=DecodeSamplingPolicy)
    tracking: TrackingPolicy = field(default_factory=TrackingPolicy)
    tracklet_quality: TrackletQualityPolicy = field(default_factory=TrackletQualityPolicy)
    tracklet_feature_pipeline: TrackletFeaturePipelinePolicy = field(default_factory=TrackletFeaturePipelinePolicy)

    def build_metadata(self, item: "StorageVideoItem") -> dict[str, Any]:
        return {
            "source_mode": self.source_mode,
            "source_storage_relative_path": item.relative_path,
            "source_size_bytes": item.size_bytes,
            "source_modified_ns": item.modified_ns,
            "ingestion_contract": {
                "decode_sampling": self.decode_sampling.to_metadata(),
                "tracking": self.tracking.to_metadata(),
                "tracklet_quality": self.tracklet_quality.to_metadata(),
                "tracklet_feature_pipeline": self.tracklet_feature_pipeline.to_metadata(),
            },
        }


@dataclass(frozen=True)
class StorageIngestionTask:
    """Typed request payload sent from metadata-service to strict tracking runtime."""

    source_path: Path | None
    source_drive_file_id: str | None
    source_filename: str
    camera_id: str
    recorded_at: datetime
    output_basename: str
    source_mode: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ProcessedStorageVideo:
    """Typed post-processing output used by queue persistence."""

    result: dict[str, Any]
    source_filename: str
    source_mode: str
    source_item: "StorageVideoItem"


class StorageTrackingRequestFactory:
    """Builds strict ingestion requests for videos already placed in storage/."""

    def __init__(self, policy: PostMoveIngestionPolicy | None = None) -> None:
        self.policy = policy or PostMoveIngestionPolicy()

    def build(self, item: "StorageVideoItem") -> StorageIngestionTask:
        return StorageIngestionTask(
            source_path=item.source_path,
            source_drive_file_id=item.source_drive_file_id,
            source_filename=item.source_filename,
            camera_id=item.camera_id,
            recorded_at=item.recorded_at,
            output_basename=item.output_basename,
            source_mode=self.policy.source_mode,
            metadata=self.policy.build_metadata(item),
        )


class StorageTrackingResultAssembler:
    """Normalizes strict runtime output into the local queue metadata shape."""

    def __init__(self, metadata_dir: Path) -> None:
        self.metadata_dir = metadata_dir

    def assemble(self, item: "StorageVideoItem", raw_result: dict[str, Any], source_mode: str) -> ProcessedStorageVideo:
        video_payload = dict(raw_result.get("video") or {})
        people = list(raw_result.get("people") or [])
        metadata_path = self.metadata_dir / f"{Path(item.source_filename).stem}.json"
        compressed_path = str(raw_result.get("compressed_path") or item.source_path or "")
        source_drive_file_id = str(item.source_drive_file_id or "").strip() or None
        if item.source_path is None and source_drive_file_id:
            compressed_path = ""

        video_payload["source_mode"] = source_mode
        video_payload["source_storage_relative_path"] = item.relative_path
        video_payload["source_drive_file_id"] = source_drive_file_id
        video_payload["compressed_path"] = compressed_path
        video_payload["metadata_path"] = str(metadata_path)

        normalized_result = dict(raw_result)
        normalized_result["compressed_path"] = compressed_path
        normalized_result["metadata_path"] = str(metadata_path)
        if source_drive_file_id and not str(normalized_result.get("drive_video_file_id") or "").strip():
            normalized_result["drive_video_file_id"] = source_drive_file_id
            normalized_result["drive_video_link"] = f"https://drive.google.com/file/d/{source_drive_file_id}/view"
            video_payload["drive_video_file_id"] = source_drive_file_id
            video_payload["drive_video_link"] = normalized_result["drive_video_link"]
        normalized_result["video"] = video_payload

        self._write_local_metadata_artifact(metadata_path, video_payload, people)
        return ProcessedStorageVideo(
            result=normalized_result,
            source_filename=item.source_filename,
            source_mode=source_mode,
            source_item=item,
        )

    @staticmethod
    def _write_local_metadata_artifact(metadata_path: Path, video_payload: dict[str, Any], people: list[dict[str, Any]]) -> None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"video": video_payload, "people": people}
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
