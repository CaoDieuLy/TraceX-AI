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
class RuleBasedStaticAttributeExtractor:
    """Conservative visual extractor for static attributes from selected crops."""

    adult_height_threshold: int = 110
    confidence_floor: float = 0.15

    def extract(self, tracklet: TrackletFeatureInput, selection: FrameSelectionOutput) -> StaticAttributeResult:
        selected_frames = self._selected_observations(tracklet, selection)
        heights = [frame.bbox.height for frame in selected_frames]
        median_height = float(np.median(heights)) if heights else 0.0
        age_group = "adult_like" if median_height >= self.adult_height_threshold else "unknown"
        confidence = self.confidence_floor if age_group == "adult_like" else 0.05
        return StaticAttributeResult(gender=None, age_group=age_group, confidence=round(confidence, 6))

    @staticmethod
    def _selected_observations(
        tracklet: TrackletFeatureInput,
        selection: FrameSelectionOutput,
    ) -> tuple[TrackletFrameObservation, ...]:
        frame_map = {frame.frame_index: frame for frame in tracklet.frames}
        return tuple(
            frame_map[item.frame_index]
            for item in selection.selected_frames
            if item.frame_index in frame_map
        )


@dataclass
class ColorAppearanceAttributeExtractor:
    """Extract coarse appearance metadata from selected frame crops."""

    def extract(self, tracklet: TrackletFeatureInput, selection: FrameSelectionOutput) -> AppearanceAttributeResult:
        selected_frames = self._selected_observations(tracklet, selection)
        crops = [frame.crop_bgr for frame in selected_frames if frame.crop_bgr is not None and frame.crop_bgr.size > 0]
        if not crops:
            return AppearanceAttributeResult()

        upper_colors: list[str] = []
        lower_colors: list[str] = []
        hair_colors: list[str] = []
        shoe_colors: list[str] = []
        hat_votes: list[str] = []
        bag_votes: list[str] = []
        head_accessory_votes: list[str] = []
        skin_tones: list[str] = []

        for crop in crops:
            h, w = crop.shape[:2]
            if h < 12 or w < 8:
                continue
            upper = crop[max(0, int(h * 0.18)): max(1, int(h * 0.55)), :]
            lower = crop[max(0, int(h * 0.55)): max(1, int(h * 0.85)), :]
            head = crop[: max(1, int(h * 0.18)), :]
            shoes = crop[max(0, int(h * 0.85)):h, :]
            side_band = crop[max(0, int(h * 0.25)): max(1, int(h * 0.75)), max(0, int(w * 0.75)):w]
            face_band = crop[max(0, int(h * 0.12)): max(1, int(h * 0.28)), int(w * 0.3): int(w * 0.7) or 1]

            upper_colors.append(self._dominant_color_name(upper))
            lower_colors.append(self._dominant_color_name(lower))
            hair_colors.append(self._dominant_color_name(head))
            shoe_colors.append(self._dominant_color_name(shoes))
            hat_votes.append(self._hat_label(head))
            bag_votes.append(self._bag_label(side_band, w))
            head_accessory_votes.append(self._head_accessory_label(head))
            skin_tones.append(self._skin_tone_label(face_band))

        return AppearanceAttributeResult(
            head_accessory=self._majority_non_unknown(head_accessory_votes),
            hat=self._majority_non_unknown(hat_votes),
            hair_color=self._majority_non_unknown(hair_colors),
            skin_tone=self._majority_non_unknown(skin_tones),
            shirt=self._majority_non_unknown(upper_colors),
            pants=self._majority_non_unknown(lower_colors),
            shoes=self._majority_non_unknown(shoe_colors),
            bag=self._majority_non_unknown(bag_votes),
        )

    @staticmethod
    def _selected_observations(
        tracklet: TrackletFeatureInput,
        selection: FrameSelectionOutput,
    ) -> tuple[TrackletFrameObservation, ...]:
        frame_map = {frame.frame_index: frame for frame in tracklet.frames}
        return tuple(
            frame_map[item.frame_index]
            for item in selection.selected_frames
            if item.frame_index in frame_map
        )

    @staticmethod
    def _majority_non_unknown(values: list[str]) -> str | None:
        filtered = [value for value in values if value and value != "unknown"]
        if not filtered:
            return None
        counts: dict[str, int] = {}
        for value in filtered:
            counts[value] = counts.get(value, 0) + 1
        return max(counts.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _dominant_color_name(region: np.ndarray) -> str:
        if region.size == 0:
            return "unknown"
        hsv = np.asarray(region, dtype=np.uint8)
        hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
        hue = float(np.mean(hsv[:, :, 0]))
        sat = float(np.mean(hsv[:, :, 1]))
        val = float(np.mean(hsv[:, :, 2]))
        if val < 45:
            return "black"
        if sat < 35 and val > 185:
            return "white"
        if sat < 40:
            return "gray"
        if hue < 10 or hue >= 170:
            return "red"
        if hue < 20:
            return "orange"
        if hue < 34:
            return "yellow"
        if hue < 85:
            return "green"
        if hue < 130:
            return "blue"
        if hue < 160:
            return "purple"
        return "brown"

    def _hat_label(self, head_region: np.ndarray) -> str:
        if head_region.size == 0:
            return "unknown"
        h, _ = head_region.shape[:2]
        top_strip = head_region[: max(1, int(h * 0.45)), :]
        full_brightness = float(np.mean(head_region))
        top_brightness = float(np.mean(top_strip))
        return "present" if top_brightness + 18 < full_brightness else "unknown"

    def _head_accessory_label(self, head_region: np.ndarray) -> str:
        if head_region.size == 0:
            return "unknown"
        hsv = cv2.cvtColor(np.asarray(head_region, dtype=np.uint8), cv2.COLOR_BGR2HSV)
        sat = float(np.mean(hsv[:, :, 1]))
        return "present" if sat > 75 else "unknown"

    def _bag_label(self, side_region: np.ndarray, body_width: int) -> str:
        if side_region.size == 0 or body_width <= 0:
            return "unknown"
        mask = cv2.Canny(side_region, 40, 120)
        active_ratio = float(np.count_nonzero(mask)) / float(mask.size or 1)
        return "present" if active_ratio > 0.08 else "unknown"

    def _skin_tone_label(self, face_region: np.ndarray) -> str:
        if face_region.size == 0:
            return "unknown"
        hsv = cv2.cvtColor(np.asarray(face_region, dtype=np.uint8), cv2.COLOR_BGR2HSV)
        value = float(np.mean(hsv[:, :, 2]))
        if value < 85:
            return "dark"
        if value < 155:
            return "medium"
        return "light"


@dataclass
class VisualAttributeEmbeddingExtractor:
    """Attribute embedding derived from static attribute metadata and selection quality."""

    embedding_model: str = "visual_attribute_embedding_v1"
    target_dimensions: int = 8

    def extract(
        self,
        tracklet: TrackletFeatureInput,
        selection: FrameSelectionOutput,
        static_attributes: StaticAttributeResult,
    ) -> AttributeEmbeddingResult:
        seed_tokens = [
            tracklet.video_id,
            tracklet.object_id,
            static_attributes.gender or "",
            static_attributes.age_group or "",
            str(selection.representative_frame.detection_confidence),
            ",".join(str(item.frame_index) for item in selection.selected_frames),
        ]
        values = [0.0 for _ in range(self.target_dimensions)]
        for token_index, token in enumerate(seed_tokens):
            if not token:
                continue
            for char_index, char in enumerate(token):
                bucket = (token_index + char_index) % self.target_dimensions
                values[bucket] += (ord(char) % 89) / 89.0

        values[0] += selection.average_tracklet_score
        values[1] += selection.pooling_scores.get("quality_weighted_mean_selected_frames", 0.0)
        values[2] += float(len(selection.selected_frames)) / 10.0

        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        vector = tuple(round(value / norm, 6) for value in values)
        return AttributeEmbeddingResult(
            embedding_model=self.embedding_model,
            embedding_vector=vector,
        )


@dataclass
class VisualAppearanceEmbeddingExtractor:
    """Quality-weighted visual embedding from selected crops."""

    embedding_model: str = "visual_tracklet_embedding_v2"
    histogram_bins: int = 8

    def extract(
        self,
        tracklet: TrackletFeatureInput,
        selection: FrameSelectionOutput,
        static_attributes: StaticAttributeResult,
        appearance_attributes: AppearanceAttributeResult,
    ) -> AppearanceEmbeddingResult:
        selected_map = {item.frame_index: item for item in selection.selected_frames}
        per_frame_vectors: list[tuple[float, ...]] = []
        per_frame_weights: list[float] = []
        for frame in tracklet.frames:
            selected = selected_map.get(frame.frame_index)
            if selected is None or frame.crop_bgr is None or frame.crop_bgr.size == 0:
                continue
            vector = self._extract_crop_vector(frame.crop_bgr, selected, static_attributes, appearance_attributes)
            per_frame_vectors.append(vector)
            per_frame_weights.append(max(selected.quality_score, 1e-6))

        if not per_frame_vectors:
            fallback_vector = self._fallback_semantic_vector(selection, static_attributes, appearance_attributes)
            return AppearanceEmbeddingResult(
                embedding_model=self.embedding_model,
                embedding_vector=fallback_vector,
                tracklet_vectors=(fallback_vector,),
            )

        fused = np.average(np.asarray(per_frame_vectors, dtype=np.float32), axis=0, weights=np.asarray(per_frame_weights))
        norm = float(np.linalg.norm(fused)) or 1.0
        vector = tuple(round(float(value / norm), 6) for value in fused.tolist())
        return AppearanceEmbeddingResult(
            embedding_model=self.embedding_model,
            embedding_vector=vector,
            tracklet_vectors=tuple(per_frame_vectors),
        )

    def _fallback_semantic_vector(
        self,
        selection: FrameSelectionOutput,
        static_attributes: StaticAttributeResult,
        appearance_attributes: AppearanceAttributeResult,
    ) -> tuple[float, ...]:
        seed_tokens = [
            static_attributes.gender or "",
            static_attributes.age_group or "",
            appearance_attributes.head_accessory or "",
            appearance_attributes.hat or "",
            appearance_attributes.hair_color or "",
            appearance_attributes.skin_tone or "",
            appearance_attributes.shirt or "",
            appearance_attributes.pants or "",
            appearance_attributes.shoes or "",
            appearance_attributes.bag or "",
            str(selection.representative_frame.quality_score),
        ]
        values = [0.0 for _ in range(32)]
        for token_index, token in enumerate(seed_tokens):
            if not token:
                continue
            for char_index, char in enumerate(token):
                bucket = (token_index + char_index) % len(values)
                values[bucket] += (ord(char) % 97) / 97.0
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return tuple(round(value / norm, 6) for value in values)

    def _extract_crop_vector(
        self,
        crop_bgr: np.ndarray,
        selected_frame: FrameQualityScore,
        static_attributes: StaticAttributeResult,
        appearance_attributes: AppearanceAttributeResult,
    ) -> tuple[float, ...]:
        hsv = cv2.cvtColor(np.asarray(crop_bgr, dtype=np.uint8), cv2.COLOR_BGR2HSV)
        hist_features: list[float] = []
        for channel_index, bins in ((0, self.histogram_bins), (1, self.histogram_bins), (2, self.histogram_bins)):
            hist = cv2.calcHist([hsv], [channel_index], None, [bins], [0, 256]).flatten()
            hist_sum = float(hist.sum()) or 1.0
            hist_features.extend((hist / hist_sum).tolist())

        gray = cv2.cvtColor(np.asarray(crop_bgr, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (16, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        texture_features = cv2.HOGDescriptor(
            _winSize=(16, 32),
            _blockSize=(8, 8),
            _blockStride=(4, 4),
            _cellSize=(4, 4),
            _nbins=9,
        ).compute(resized.astype(np.uint8))
        texture_vector = texture_features.flatten()[:64] if texture_features is not None else np.zeros(64, dtype=np.float32)

        semantic_bits = np.array(
            [
                1.0 if static_attributes.age_group == "adult_like" else 0.0,
                1.0 if appearance_attributes.hat else 0.0,
                1.0 if appearance_attributes.bag else 0.0,
                float(selected_frame.detection_confidence),
                float(selected_frame.quality_score),
            ],
            dtype=np.float32,
        )
        vector = np.concatenate(
            [
                np.asarray(hist_features, dtype=np.float32),
                texture_vector.astype(np.float32),
                semantic_bits,
            ]
        )
        norm = float(np.linalg.norm(vector)) or 1.0
        return tuple(round(float(value / norm), 6) for value in vector.tolist())


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

        ordered = tuple(sorted(frames, key=lambda item: item.frame_index))
        best_score = -1.0
        best_index = ordered[len(ordered) // 2].frame_index
        previous = ordered[0]
        for current in ordered[1:]:
            previous_center = ((previous.bbox.x1 + previous.bbox.x2) / 2.0, (previous.bbox.y1 + previous.bbox.y2) / 2.0)
            current_center = ((current.bbox.x1 + current.bbox.x2) / 2.0, (current.bbox.y1 + current.bbox.y2) / 2.0)
            displacement = math.dist(previous_center, current_center)
            score = displacement + max(current.detection_confidence, 0.0) * 10.0
            if score > best_score:
                best_score = score
                best_index = current.frame_index
            previous = current
        return best_index


@dataclass
class HeuristicBehaviorAnalyzer:
    """
    Lightweight action analyzer based on bbox motion.

    This is a stable adapter contract before the strict runtime wires the final
    action model.
    """

    def analyze(self, clip: ActionClip) -> BehaviorAnalysisResult:
        if not clip.frames:
            return BehaviorAnalysisResult(
                clip_id=clip.clip_id,
                action_summary="unknown_action",
                labels=("unknown_action",),
                confidence=0.0,
                metadata={},
            )

        movement = self._movement_stats(clip.frames)
        labels, summary = self._classify(movement)
        confidence = round(min(max(movement["confidence_proxy"], 0.0), 1.0), 6)
        return BehaviorAnalysisResult(
            clip_id=clip.clip_id,
            action_summary=summary,
            labels=labels,
            confidence=confidence,
            metadata=movement,
        )

    @staticmethod
    def _movement_stats(frames: tuple[TrackletFrameObservation, ...]) -> dict[str, float]:
        ordered = tuple(sorted(frames, key=lambda item: item.frame_index))
        if len(ordered) < 2:
            return {
                "mean_center_displacement": 0.0,
                "height_change_ratio": 0.0,
                "confidence_proxy": ordered[0].detection_confidence if ordered else 0.0,
            }

        displacements: list[float] = []
        for previous, current in zip(ordered, ordered[1:]):
            previous_center = ((previous.bbox.x1 + previous.bbox.x2) / 2.0, (previous.bbox.y1 + previous.bbox.y2) / 2.0)
            current_center = ((current.bbox.x1 + current.bbox.x2) / 2.0, (current.bbox.y1 + current.bbox.y2) / 2.0)
            body_scale = max((previous.bbox.height + current.bbox.height) / 2.0, 1.0)
            displacements.append(math.dist(previous_center, current_center) / body_scale)

        first_height = max(float(ordered[0].bbox.height), 1.0)
        last_height = float(ordered[-1].bbox.height)
        height_change_ratio = (first_height - last_height) / first_height
        confidence_proxy = sum(frame.detection_confidence for frame in ordered) / len(ordered)
        return {
            "mean_center_displacement": round(sum(displacements) / len(displacements), 6),
            "height_change_ratio": round(height_change_ratio, 6),
            "confidence_proxy": round(confidence_proxy, 6),
        }

    @staticmethod
    def _classify(movement: dict[str, float]) -> tuple[tuple[str, ...], str]:
        displacement = float(movement.get("mean_center_displacement") or 0.0)
        height_change_ratio = float(movement.get("height_change_ratio") or 0.0)

        if height_change_ratio > 0.38:
            return ("bending_or_sit_like_motion",), "bending_or_sit_like_motion"
        if displacement > 0.38:
            return ("running_or_fast_motion",), "running_or_fast_motion"
        if displacement > 0.12:
            return ("walking_motion",), "walking_motion"
        return ("standing_or_slow_motion",), "standing_or_slow_motion"


@dataclass
class ActionVocabularyEmbedder:
    """
    Deterministic semantic embedder for action labels.

    The vector is a confidence-weighted projection over a fixed action
    vocabulary. This keeps the API stable until a real action semantic encoder
    is plugged in.
    """

    embedding_model: str = "heuristic_action_embedding_v1"
    vocabulary: tuple[str, ...] = EMBEDDING_VOCABULARY

    def embed(self, analysis: BehaviorAnalysisResult) -> SemanticEmbeddingResult:
        vector = [0.0 for _ in self.vocabulary]
        label_set = set(analysis.labels)
        for index, token in enumerate(self.vocabulary):
            if token in label_set:
                vector[index] = analysis.confidence or 1.0
        if not any(vector):
            unknown_index = self.vocabulary.index("unknown_action")
            vector[unknown_index] = max(analysis.confidence, 0.1)
        return SemanticEmbeddingResult(
            clip_id=analysis.clip_id,
            embedding_model=self.embedding_model,
            vocabulary=self.vocabulary,
            embedding_vector=tuple(round(value, 6) for value in vector),
        )


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
            "model": semantic_embeddings[0].embedding_model if semantic_embeddings else "heuristic_action_embedding_v1",
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
    try:
        from model_adapters import ZeroShotAttributeAdapter
        return ZeroShotAttributeAdapter()
    except Exception:
        return RuleBasedStaticAttributeExtractor()


def _default_attribute_embedding_extractor():
    try:
        from model_adapters import CLIPAttributeEmbeddingAdapter
        return CLIPAttributeEmbeddingAdapter()
    except Exception:
        return VisualAttributeEmbeddingExtractor()


def _default_appearance_attribute_extractor():
    try:
        from model_adapters import ZeroShotAppearanceMetadataAdapter
        return ZeroShotAppearanceMetadataAdapter()
    except Exception:
        return ColorAppearanceAttributeExtractor()


def _default_appearance_embedding_extractor():
    try:
        from model_adapters import SoliderKPRAppearanceEmbeddingAdapter
        return SoliderKPRAppearanceEmbeddingAdapter()
    except Exception:
        return VisualAppearanceEmbeddingExtractor()


def _default_semantic_embedder():
    try:
        from model_adapters import ItselfSemanticEmbedder
        return ItselfSemanticEmbedder()
    except Exception:
        return ActionVocabularyEmbedder()


@dataclass
class TrackletFeaturePipelineProcessor:
    """
    End-to-end tracklet feature pipeline.

    Stage order:
    1. frame selection
    2. static attribute extraction   (CLIP zero-shot / heuristic fallback)
    3. appearance extraction          (CLIP zero-shot / heuristic fallback)
    4. action clip building
    5. action / behavior analysis
    6. action semantic embedding      (CLIP text / vocab fallback)
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
    behavior_analyzer: ActionBehaviorAnalyzer = field(default_factory=HeuristicBehaviorAnalyzer)
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
            "RuleBasedStaticAttributeExtractor",
            "ColorAppearanceAttributeExtractor",
            "VisualAttributeEmbeddingExtractor",
            "VisualAppearanceEmbeddingExtractor",
            "HeuristicBehaviorAnalyzer",
            "ActionVocabularyEmbedder",
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
