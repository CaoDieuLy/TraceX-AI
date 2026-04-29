from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import cv2
from dataclasses import dataclass, field
import math
import numpy as np
from statistics import mean
from typing import Protocol


EMBEDDING_VOCABULARY: tuple[str, ...] = (
    "standing_or_slow_motion",
    "walking_motion",
    "running_or_fast_motion",
    "bending_or_sit_like_motion",
    "fall_like_motion",
    "carrying_object_like_motion",
    "unknown_action",
)


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box for one sampled frame."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_xyxy(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class TrackletFrameObservation:
    """
    One sampled frame inside a tracklet.

    The strict runtime is expected to fill `laplacian_score` during preprocessing,
    because frame selection uses blur sharpness together with detection confidence
    and bbox area.
    """

    frame_index: int
    timestamp_second: float
    bbox: BoundingBox
    detection_confidence: float
    laplacian_score: float
    crop_bgr: np.ndarray | None = None


@dataclass(frozen=True)
class TrackletFeatureInput:
    """
    Input contract after tracklet quality scoring.

    Each tracklet belongs to exactly one video and one local object id. The track
    carries per-frame bounding boxes at the fixed sampled FPS.
    """

    video_id: str
    object_id: str
    sampled_fps: int
    frames: tuple[TrackletFrameObservation, ...]

    @property
    def duration_seconds(self) -> float:
        if not self.frames:
            return 0.0
        return max(self.frames[-1].timestamp_second - self.frames[0].timestamp_second, 0.0)


@dataclass(frozen=True)
class FrameSelectionWeights:
    """Weights for the hybrid frame quality score."""

    laplacian_weight: float = 0.4
    bbox_area_weight: float = 0.3
    confidence_weight: float = 0.3


@dataclass(frozen=True)
class FrameQualityScore:
    """Scored representation of one sampled frame."""

    frame_index: int
    timestamp_second: float
    bbox: list[int]
    laplacian_score: float
    bbox_area: int
    detection_confidence: float
    normalized_laplacian: float
    normalized_bbox_area: float
    quality_score: float


@dataclass(frozen=True)
class FrameSelectionOutput:
    """
    Hybrid frame selection result.

    The pipeline keeps both signals requested by the user:
    - average score over all sampled frames
    - best frame as the representative frame
    """

    average_tracklet_score: float
    representative_frame: FrameQualityScore
    selected_frames: tuple[FrameQualityScore, ...]
    ranked_frames: tuple[FrameQualityScore, ...]
    pooling_scores: dict[str, float]


@dataclass(frozen=True)
class TrackletStageExecutionConfig:
    """Execution settings for independent stages inside one tracklet pipeline."""

    max_workers: int = 3


@dataclass(frozen=True)
class StaticAttributeResult:
    """Tracklet-level static attributes extracted from the representative frame."""

    gender: str | None
    age_group: str | None
    confidence: float

    def semantic_tokens(self) -> list[str]:
        tokens: list[str] = []
        if self.gender:
            tokens.append(f"gender:{self.gender}")
        if self.age_group:
            tokens.append(f"age_group:{self.age_group}")
        return tokens


@dataclass(frozen=True)
class AttributeEmbeddingResult:
    """Deterministic attribute embedding used until the strict model is wired."""

    embedding_model: str
    embedding_vector: tuple[float, ...]


@dataclass(frozen=True)
class AppearanceAttributeResult:
    """Appearance attributes extracted from the representative frame."""

    head_accessory: str | None = None
    hat: str | None = None
    hair_color: str | None = None
    skin_tone: str | None = None
    shirt: str | None = None
    pants: str | None = None
    shoes: str | None = None
    bag: str | None = None

    def semantic_tokens(self) -> list[str]:
        mapping = {
            "head_accessory": self.head_accessory,
            "hat": self.hat,
            "hair_color": self.hair_color,
            "skin_tone": self.skin_tone,
            "shirt": self.shirt,
            "pants": self.pants,
            "shoes": self.shoes,
            "bag": self.bag,
        }
        return [f"{key}:{value}" for key, value in mapping.items() if value]


@dataclass(frozen=True)
class ActionClip:
    """Temporal clip extracted from one tracklet for action analysis."""

    clip_id: str
    video_id: str
    object_id: str
    start_frame_index: int
    end_frame_index: int
    center_frame_index: int
    start_second: float
    end_second: float
    frames: tuple[TrackletFrameObservation, ...]

    def bbox_samples(self) -> tuple[list[int], ...]:
        return tuple(frame.bbox.to_xyxy() for frame in self.frames[: min(len(self.frames), 5)])


@dataclass(frozen=True)
class BehaviorAnalysisResult:
    """Output of the action / behavior stage for one clip."""

    clip_id: str
    action_summary: str
    labels: tuple[str, ...]
    confidence: float
    metadata: dict[str, object]

    def semantic_tokens(self) -> list[str]:
        return [f"action:{label}" for label in self.labels]


@dataclass(frozen=True)
class SemanticEmbeddingResult:
    """Action semantic embedding for one analyzed clip."""

    clip_id: str
    embedding_model: str
    vocabulary: tuple[str, ...]
    embedding_vector: tuple[float, ...]


@dataclass(frozen=True)
class TrackletFeatureAggregationOutput:
    """
    Aggregated tracklet-level payload.

    Attribute, appearance, and action are treated as equal stages inside the
    same pipeline and land in one metadata payload.
    """

    video_id: str
    object_id: str
    representative_frame_index: int
    representative_bbox: list[int]
    average_tracklet_score: float
    attribute_summary: str
    appearance_summary: str
    semantic_attributes: tuple[str, ...]
    attribute_embedding_vector: tuple[float, ...]
    appearance_embedding_vector: tuple[float, ...]
    embedding_vector: tuple[float, ...]
    tracklet_vectors: tuple[tuple[float, ...], ...]
    bbox_samples: tuple[list[int], ...]
    visibility_scores: dict[str, float]
    timeline: tuple[dict[str, object], ...]
    matched_segments: tuple[dict[str, object], ...]
    action_semantic_embedding: dict[str, object]
    pipeline_metadata: dict[str, object]

    def to_metadata(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "track_id": self.object_id,
            "frame_idx": self.representative_frame_index,
            "bbox": self.representative_bbox,
            "representative_bbox": self.representative_bbox,
            "bbox_samples": list(self.bbox_samples),
            "attribute_summary": self.attribute_summary,
            "appearance_summary": self.appearance_summary,
            "semantic_attributes": list(self.semantic_attributes),
            "attribute_embedding_vector": list(self.attribute_embedding_vector),
            "appearance_embedding_vector": list(self.appearance_embedding_vector),
            "embedding_vector": list(self.embedding_vector),
            "tracklet_vectors": [list(item) for item in self.tracklet_vectors],
            "visibility_scores": self.visibility_scores,
            "timeline": list(self.timeline),
            "matched_segments": list(self.matched_segments),
            "action_semantic_embedding": self.action_semantic_embedding,
            "score": self.average_tracklet_score,
            "tracklet_feature_pipeline": self.pipeline_metadata,
        }


class StaticAttributeExtractor(Protocol):
    """Model adapter for gender + age-group extraction."""

    def extract(self, tracklet: TrackletFeatureInput, selection: FrameSelectionOutput) -> StaticAttributeResult:
        ...


class AppearanceAttributeExtractor(Protocol):
    """Model adapter for appearance extraction."""

    def extract(self, tracklet: TrackletFeatureInput, selection: FrameSelectionOutput) -> AppearanceAttributeResult:
        ...


class AttributeEmbeddingExtractor(Protocol):
    """Model adapter for attribute embedding extraction."""

    def extract(
        self,
        tracklet: TrackletFeatureInput,
        selection: FrameSelectionOutput,
        static_attributes: StaticAttributeResult,
    ) -> AttributeEmbeddingResult:
        ...


@dataclass(frozen=True)
class AppearanceEmbeddingResult:
    """Appearance embedding summary used by retrieval and feature aggregation."""

    embedding_model: str
    embedding_vector: tuple[float, ...]
    tracklet_vectors: tuple[tuple[float, ...], ...]


class ActionBehaviorAnalyzer(Protocol):
    """Model adapter for action / behavior analysis."""

    def analyze(self, clip: ActionClip) -> BehaviorAnalysisResult:
        ...


class ActionSemanticEmbedder(Protocol):
    """Model adapter for action semantic embedding."""

    def embed(self, analysis: BehaviorAnalysisResult) -> SemanticEmbeddingResult:
        ...


class AppearanceEmbeddingExtractor(Protocol):
    """Model adapter for appearance embedding extraction."""

    def extract(
        self,
        tracklet: TrackletFeatureInput,
        selection: FrameSelectionOutput,
        static_attributes: StaticAttributeResult,
        appearance_attributes: AppearanceAttributeResult,
    ) -> AppearanceEmbeddingResult:
        ...


@dataclass
class NullStaticAttributeExtractor:
    """
    Safe placeholder extractor until the strict runtime wires a real attribute model.

    Returning explicit null fields is safer than fabricating labels.
    """

    def extract(self, tracklet: TrackletFeatureInput, selection: FrameSelectionOutput) -> StaticAttributeResult:
        return StaticAttributeResult(gender=None, age_group=None, confidence=0.0)


@dataclass


@dataclass


@dataclass


@dataclass


@dataclass
class HybridFrameSelector:
    """
    Hybrid frame selector used before all feature stages.

    This selector feeds attribute, appearance, and action stages from the same
    representative frame decision.
    """

    weights: FrameSelectionWeights = field(default_factory=FrameSelectionWeights)
    selected_frame_count: int = 5
    max_selected_frame_count: int = 10

    def select(self, tracklet: TrackletFeatureInput) -> FrameSelectionOutput:
        if not tracklet.frames:
            raise ValueError("TrackletFeatureInput.frames must not be empty")

        laplacian_values = [max(frame.laplacian_score, 0.0) for frame in tracklet.frames]
        area_values = [max(frame.bbox.area, 0) for frame in tracklet.frames]
        max_laplacian = max(laplacian_values) or 1.0
        max_area = max(area_values) or 1

        scored_frames: list[FrameQualityScore] = []
        for frame in tracklet.frames:
            normalized_laplacian = max(frame.laplacian_score, 0.0) / max_laplacian
            normalized_bbox_area = max(frame.bbox.area, 0) / max_area
            confidence = min(max(frame.detection_confidence, 0.0), 1.0)
            quality_score = (
                normalized_laplacian * self.weights.laplacian_weight
                + normalized_bbox_area * self.weights.bbox_area_weight
                + confidence * self.weights.confidence_weight
            )
            scored_frames.append(
                FrameQualityScore(
                    frame_index=frame.frame_index,
                    timestamp_second=frame.timestamp_second,
                    bbox=frame.bbox.to_xyxy(),
                    laplacian_score=frame.laplacian_score,
                    bbox_area=frame.bbox.area,
                    detection_confidence=confidence,
                    normalized_laplacian=round(normalized_laplacian, 6),
                    normalized_bbox_area=round(normalized_bbox_area, 6),
                    quality_score=round(quality_score, 6),
                )
            )

        ranked_frames = tuple(sorted(scored_frames, key=lambda item: item.quality_score, reverse=True))
        selected_count = max(1, min(self.selected_frame_count, self.max_selected_frame_count, len(ranked_frames)))
        selected_frames = ranked_frames[:selected_count]
        average_tracklet_score = round(mean(item.quality_score for item in ranked_frames), 6)
        pooling_scores = self._pooling_scores(ranked_frames, selected_frames)
        return FrameSelectionOutput(
            average_tracklet_score=average_tracklet_score,
            representative_frame=ranked_frames[0],
            selected_frames=selected_frames,
            ranked_frames=ranked_frames,
            pooling_scores=pooling_scores,
        )

    @staticmethod
    def _pooling_scores(
        ranked_frames: tuple[FrameQualityScore, ...],
        selected_frames: tuple[FrameQualityScore, ...],
    ) -> dict[str, float]:
        all_scores = [item.quality_score for item in ranked_frames]
        selected_scores = [item.quality_score for item in selected_frames]
        quality_weight_denominator = sum(max(item.quality_score, 1e-6) for item in selected_frames) or 1.0
        quality_weighted_mean = sum(
            item.quality_score * max(item.quality_score, 1e-6) for item in selected_frames
        ) / quality_weight_denominator
        return {
            "mean_pooling_all_frames": round(mean(all_scores), 6),
            "max_pooling_all_frames": round(max(all_scores), 6),
            "mean_pooling_selected_frames": round(mean(selected_scores), 6),
            "max_pooling_selected_frames": round(max(selected_scores), 6),
            "quality_weighted_mean_selected_frames": round(quality_weighted_mean, 6),
        }


@dataclass
class ActionClipBuilder:
    """
    Builds one or more temporal clips from a tracklet.

    Clip extraction runs as a normal feature stage beside attribute and
    appearance.
    """

    clip_duration_seconds: float = 2.0
    clip_stride_seconds: float = 1.5
    minimum_tracklet_duration_seconds: float = 2.0
    max_segments: int = 10

    def build(
        self,
        tracklet: TrackletFeatureInput,
        representative_frame_index: int,
    ) -> tuple[ActionClip, ...]:
        frames = tracklet.frames
        if not frames:
            return ()
        if tracklet.duration_seconds < self.minimum_tracklet_duration_seconds:
            return ()

        clip_frame_span = max(1, int(round(tracklet.sampled_fps * self.clip_duration_seconds)))
        if len(frames) <= clip_frame_span:
            return (self._build_clip(tracklet, frames=frames, clip_index=0),)

        stride_frames = max(1, int(round(tracklet.sampled_fps * self.clip_stride_seconds)))
        ordered_frames = tuple(sorted(frames, key=lambda item: item.frame_index))
        start_centers: list[int] = []
        start_position = 0
        while start_position < len(ordered_frames):
            window_end = min(start_position + clip_frame_span, len(ordered_frames))
            window = ordered_frames[start_position:window_end]
            if not window:
                break
            start_centers.append(window[len(window) // 2].frame_index)
            if window_end >= len(ordered_frames):
                break
            start_position += stride_frames

        centers: list[int] = [representative_frame_index]
        centers.extend(start_centers)
        motion_peak = self._motion_peak_frame_index(frames)
        if motion_peak is not None and motion_peak not in centers:
            centers.append(motion_peak)

        clips: list[ActionClip] = []
        seen_ranges: set[tuple[int, int]] = set()
        unique_centers: list[int] = []
        for center_frame_index in centers:
            if center_frame_index not in unique_centers:
                unique_centers.append(center_frame_index)
        for clip_index, center_frame_index in enumerate(unique_centers[: self.max_segments]):
            clip_frames = self._slice_frames_around_center(
                frames,
                center_frame_index=center_frame_index,
                clip_frame_span=clip_frame_span,
            )
            clip_range = (clip_frames[0].frame_index, clip_frames[-1].frame_index)
            if clip_range in seen_ranges:
                continue
            seen_ranges.add(clip_range)
            clips.append(
                self._build_clip(
                    tracklet,
                    frames=clip_frames,
                    clip_index=len(clips),
                )
            )
        return tuple(clips)

    def _slice_frames_around_center(
        self,
        frames: tuple[TrackletFrameObservation, ...],
        *,
        center_frame_index: int,
        clip_frame_span: int,
    ) -> tuple[TrackletFrameObservation, ...]:
        sorted_frames = tuple(sorted(frames, key=lambda item: item.frame_index))
        index_map = {frame.frame_index: idx for idx, frame in enumerate(sorted_frames)}
        center_position = index_map.get(center_frame_index, len(sorted_frames) // 2)
        half_span = max(clip_frame_span // 2, 1)
        start = max(center_position - half_span, 0)
        end = min(start + clip_frame_span, len(sorted_frames))
        return sorted_frames[start:end]

    def _build_clip(
        self,
        tracklet: TrackletFeatureInput,
        *,
        frames: tuple[TrackletFrameObservation, ...],
        clip_index: int,
    ) -> ActionClip:
        ordered = tuple(sorted(frames, key=lambda item: item.frame_index))
        center_idx = ordered[len(ordered) // 2].frame_index
        return ActionClip(
            clip_id=f"{tracklet.video_id}:{tracklet.object_id}:clip-{clip_index}",
            video_id=tracklet.video_id,
            object_id=tracklet.object_id,
            start_frame_index=ordered[0].frame_index,
            end_frame_index=ordered[-1].frame_index,
            center_frame_index=center_idx,
            start_second=ordered[0].timestamp_second,
            end_second=ordered[-1].timestamp_second,
            frames=ordered,
        )

    @staticmethod
    def _motion_peak_frame_index(frames: tuple[TrackletFrameObservation, ...]) -> int | None:
        if len(frames) < 2:
            return frames[0].frame_index if frames else None

        import numpy as np
        ordered = sorted(frames, key=lambda item: item.frame_index)
        centers = np.array(
            [((f.bbox.x1 + f.bbox.x2) / 2.0, (f.bbox.y1 + f.bbox.y2) / 2.0) for f in ordered],
            dtype=np.float32,
        )
        displacements = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        confidences = np.array([max(f.detection_confidence, 0.0) for f in ordered[1:]], dtype=np.float32)
        scores = displacements + confidences * 10.0
        best_idx = int(np.argmax(scores)) + 1
        return ordered[best_idx].frame_index


@dataclass


@dataclass


@dataclass
class TrackletFeatureAggregator:
    """Aggregates attribute, appearance, and action into one tracklet metadata payload."""

    def aggregate(
        self,
        *,
        tracklet: TrackletFeatureInput,
        selection: FrameSelectionOutput,
        static_attributes: StaticAttributeResult,
        attribute_embedding: AttributeEmbeddingResult,
        appearance_attributes: AppearanceAttributeResult,
        appearance_embedding: AppearanceEmbeddingResult,
        clips: tuple[ActionClip, ...],
        behavior_results: tuple[BehaviorAnalysisResult, ...],
        semantic_embeddings: tuple[SemanticEmbeddingResult, ...],
        runtime_metadata: dict[str, object],
    ) -> TrackletFeatureAggregationOutput:
        semantic_tokens = tuple(
            static_attributes.semantic_tokens()
            + appearance_attributes.semantic_tokens()
            + self._action_tokens(behavior_results)
        )
        attribute_summary = self._attribute_summary(static_attributes)
        appearance_summary = self._appearance_summary(static_attributes, appearance_attributes)
        bbox_samples = tuple(item.bbox for item in selection.selected_frames)
        visibility_scores = {
            "average_tracklet_score": selection.average_tracklet_score,
            "representative_frame_score": selection.representative_frame.quality_score,
            "representative_frame_confidence": selection.representative_frame.detection_confidence,
        }
        timeline = tuple(
            {
                "clip_id": clip.clip_id,
                "start_second": clip.start_second,
                "end_second": clip.end_second,
                "center_frame_index": clip.center_frame_index,
                "action_summary": analysis.action_summary,
                "labels": list(analysis.labels),
                "confidence": analysis.confidence,
                "bbox_samples": list(clip.bbox_samples()),
            }
            for clip, analysis in zip(clips, behavior_results)
        )
        action_embedding_payload = {
            "model": semantic_embeddings[0].embedding_model if semantic_embeddings else "siglip2-vit-l16-512-action",
            "vocabulary": list(semantic_embeddings[0].vocabulary if semantic_embeddings else EMBEDDING_VOCABULARY),
            "segments": [
                {
                    "clip_id": item.clip_id,
                    "embedding_vector": list(item.embedding_vector),
                }
                for item in semantic_embeddings
            ],
        }
        pipeline_metadata = {
            "selection_mode": "hybrid_average_plus_best_frame",
            "sampled_fps": tracklet.sampled_fps,
            "frame_count": len(tracklet.frames),
            "tracklet_duration_seconds": round(tracklet.duration_seconds, 6),
            "representative_frame_index": selection.representative_frame.frame_index,
            "average_tracklet_score": selection.average_tracklet_score,
            "pooling_scores": selection.pooling_scores,
            "selected_frame_indices": [item.frame_index for item in selection.selected_frames],
            "appearance_embedding_model": appearance_embedding.embedding_model,
            "attribute_embedding_model": attribute_embedding.embedding_model,
            "ranked_frame_scores": [
                {
                    "frame_index": item.frame_index,
                    "timestamp_second": item.timestamp_second,
                    "quality_score": item.quality_score,
                }
                for item in selection.ranked_frames[: min(len(selection.ranked_frames), 10)]
            ],
            "action_clip_ids": [clip.clip_id for clip in clips],
            "action_labels": [analysis.action_summary for analysis in behavior_results],
        }
        pipeline_metadata.update(runtime_metadata)
        return TrackletFeatureAggregationOutput(
            video_id=tracklet.video_id,
            object_id=tracklet.object_id,
            representative_frame_index=selection.representative_frame.frame_index,
            representative_bbox=selection.representative_frame.bbox,
            average_tracklet_score=selection.average_tracklet_score,
            attribute_summary=attribute_summary,
            appearance_summary=appearance_summary,
            semantic_attributes=semantic_tokens,
            attribute_embedding_vector=attribute_embedding.embedding_vector,
            appearance_embedding_vector=appearance_embedding.embedding_vector,
            embedding_vector=appearance_embedding.embedding_vector,
            tracklet_vectors=appearance_embedding.tracklet_vectors,
            bbox_samples=bbox_samples,
            visibility_scores=visibility_scores,
            timeline=timeline,
            matched_segments=timeline,
            action_semantic_embedding=action_embedding_payload,
            pipeline_metadata=pipeline_metadata,
        )

    @staticmethod
    def _attribute_summary(static_attributes: StaticAttributeResult) -> str:
        parts = [item for item in (static_attributes.gender, static_attributes.age_group) if item]
        return ", ".join(parts)

    @staticmethod
    def _appearance_summary(
        static_attributes: StaticAttributeResult,
        appearance_attributes: AppearanceAttributeResult,
    ) -> str:
        parts: list[str] = []
        if static_attributes.gender:
            parts.append(static_attributes.gender)
        if static_attributes.age_group:
            parts.append(static_attributes.age_group)

        ordered_appearance = [
            appearance_attributes.head_accessory,
            appearance_attributes.hat,
            appearance_attributes.hair_color,
            appearance_attributes.skin_tone,
            appearance_attributes.shirt,
            appearance_attributes.pants,
            appearance_attributes.shoes,
            appearance_attributes.bag,
        ]
        parts.extend([item for item in ordered_appearance if item])
        return ", ".join(parts)

    @staticmethod
    def _action_tokens(behavior_results: tuple[BehaviorAnalysisResult, ...]) -> list[str]:
        tokens: list[str] = []
        for result in behavior_results:
            tokens.extend(result.semantic_tokens())
        return tokens


@dataclass(frozen=True)
class TrackletFeaturePipelineOutput:
    """Full output of the unified tracklet feature pipeline."""

    selection: FrameSelectionOutput
    static_attributes: StaticAttributeResult
    attribute_embedding: AttributeEmbeddingResult
    appearance_attributes: AppearanceAttributeResult
    appearance_embedding: AppearanceEmbeddingResult
    clips: tuple[ActionClip, ...]
    behavior_results: tuple[BehaviorAnalysisResult, ...]
    semantic_embeddings: tuple[SemanticEmbeddingResult, ...]
    aggregated: TrackletFeatureAggregationOutput


def _default_static_attribute_extractor():
    from model_adapters import ZeroShotAttributeAdapter
    return ZeroShotAttributeAdapter()


def _default_attribute_embedding_extractor():
    from model_adapters import CLIPAttributeEmbeddingAdapter
    return CLIPAttributeEmbeddingAdapter()


def _default_appearance_attribute_extractor():
    from model_adapters import ZeroShotAppearanceMetadataAdapter
    return ZeroShotAppearanceMetadataAdapter()


def _default_appearance_embedding_extractor():
    from model_adapters import SoliderKPRAppearanceEmbeddingAdapter
    return SoliderKPRAppearanceEmbeddingAdapter()


def _default_semantic_embedder():
    from model_adapters import ItselfSemanticEmbedder
    return ItselfSemanticEmbedder()


@dataclass
class TrackletFeaturePipelineProcessor:
    """
    End-to-end tracklet feature pipeline.

    Stage order:
    1. frame selection
    2. static attribute extraction   (SigLIP2 zero-shot)
    3. appearance extraction          (SigLIP2 zero-shot)
    4. action clip building
    5. action / behavior analysis
    6. action semantic embedding      (SigLIP2 text)
    7. feature aggregation
    """

    selector: HybridFrameSelector = field(default_factory=HybridFrameSelector)
    static_attribute_extractor: StaticAttributeExtractor = field(
        default_factory=_default_static_attribute_extractor
    )
    attribute_embedding_extractor: AttributeEmbeddingExtractor = field(
        default_factory=_default_attribute_embedding_extractor
    )
    appearance_attribute_extractor: AppearanceAttributeExtractor = field(
        default_factory=_default_appearance_attribute_extractor
    )
    appearance_embedding_extractor: AppearanceEmbeddingExtractor = field(
        default_factory=_default_appearance_embedding_extractor
    )
    clip_builder: ActionClipBuilder = field(default_factory=ActionClipBuilder)
    behavior_analyzer: ActionBehaviorAnalyzer = field(default_factory=lambda: __import__("model_adapters", fromlist=["SigLIP2BehaviorAnalyzer"]).SigLIP2BehaviorAnalyzer())
    semantic_embedder: ActionSemanticEmbedder = field(default_factory=_default_semantic_embedder)
    aggregator: TrackletFeatureAggregator = field(default_factory=TrackletFeatureAggregator)
    stage_execution: TrackletStageExecutionConfig = field(default_factory=TrackletStageExecutionConfig)

    def process(self, tracklet: TrackletFeatureInput) -> TrackletFeaturePipelineOutput:
        selection = self.selector.select(tracklet)
        max_workers = max(1, self.stage_execution.max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            static_future = executor.submit(self.static_attribute_extractor.extract, tracklet, selection)
            appearance_future = executor.submit(self.appearance_attribute_extractor.extract, tracklet, selection)
            clips_future = executor.submit(self.clip_builder.build, tracklet, selection.representative_frame.frame_index)

            static_attributes = static_future.result()
            appearance_attributes = appearance_future.result()
            clips = clips_future.result()

            attribute_embedding_future = executor.submit(
                self.attribute_embedding_extractor.extract,
                tracklet,
                selection,
                static_attributes,
            )
            appearance_embedding_future = executor.submit(
                self.appearance_embedding_extractor.extract,
                tracklet,
                selection,
                static_attributes,
                appearance_attributes,
            )

            if clips:
                behavior_results = tuple(executor.map(self.behavior_analyzer.analyze, clips))
                semantic_embeddings = tuple(executor.map(self.semantic_embedder.embed, behavior_results))
            else:
                behavior_results = ()
                semantic_embeddings = ()

            attribute_embedding = attribute_embedding_future.result()
            appearance_embedding = appearance_embedding_future.result()

        runtime_metadata = self._runtime_metadata()
        aggregated = self.aggregator.aggregate(
            tracklet=tracklet,
            selection=selection,
            static_attributes=static_attributes,
            attribute_embedding=attribute_embedding,
            appearance_attributes=appearance_attributes,
            appearance_embedding=appearance_embedding,
            clips=clips,
            behavior_results=behavior_results,
            semantic_embeddings=semantic_embeddings,
            runtime_metadata=runtime_metadata,
        )
        return TrackletFeaturePipelineOutput(
            selection=selection,
            static_attributes=static_attributes,
            attribute_embedding=attribute_embedding,
            appearance_attributes=appearance_attributes,
            appearance_embedding=appearance_embedding,
            clips=clips,
            behavior_results=behavior_results,
            semantic_embeddings=semantic_embeddings,
            aggregated=aggregated,
        )

    def _runtime_metadata(self) -> dict[str, object]:
        return {
            "stage_parallel_workers": max(1, self.stage_execution.max_workers),
            "model_runtime": {
                "static_attribute_extractor": self._model_descriptor(self.static_attribute_extractor),
                "attribute_embedding_extractor": self._model_descriptor(self.attribute_embedding_extractor),
                "appearance_attribute_extractor": self._model_descriptor(self.appearance_attribute_extractor),
                "appearance_embedding_extractor": self._model_descriptor(self.appearance_embedding_extractor),
                "behavior_analyzer": self._model_descriptor(self.behavior_analyzer),
                "semantic_embedder": self._model_descriptor(self.semantic_embedder),
            },
        }

    @staticmethod
    def _model_descriptor(model: object) -> dict[str, object]:
        class_name = type(model).__name__
        placeholder_classes = {
            "NullStaticAttributeExtractor",
        }
        approximate_classes = {
        }
        production_classes = {
            "ZeroShotAttributeAdapter",
            "ZeroShotAppearanceMetadataAdapter",
            "SoliderKPRAppearanceEmbeddingAdapter",
            "CLIPAttributeEmbeddingAdapter",
            "ItselfSemanticEmbedder",
        }
        return {
            "class_name": class_name,
            "placeholder": class_name in placeholder_classes,
            "approximate": class_name in approximate_classes,
            "production": class_name in production_classes,
        }
