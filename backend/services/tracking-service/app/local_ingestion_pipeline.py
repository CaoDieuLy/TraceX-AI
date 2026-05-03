from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gc
import json
import logging
import math
import os
from pathlib import Path
import threading

import cv2
import numpy as np

from .tracklet_feature_pipeline import (
    BoundingBox,
    TrackletFeatureInput,
    TrackletFeaturePipelineProcessor,
    TrackletFrameObservation,
)


LOGGER = logging.getLogger(__name__)
gpu_lock = threading.Semaphore(1)


def _default_tracklet_worker_count() -> int:
    configured = os.getenv("MCPT_METADATA_WORKERS", "").strip()
    if configured.isdigit():
        return max(1, int(configured))
    return max(1, min((os.cpu_count() or 1), 8))


@dataclass(frozen=True)
class SampledFrame:
    """Decoded frame sampled from the source video at the fixed ingest FPS."""

    frame_index: int
    timestamp_second: float
    image: np.ndarray
    laplacian_score: float


@dataclass(frozen=True)
class FrameDetection:
    """One person detection in a sampled frame."""

    frame_index: int
    timestamp_second: float
    bbox: BoundingBox
    confidence: float
    laplacian_score: float
    crop_bgr: np.ndarray | None = None


@dataclass(frozen=True)
class TrackletObservation:
    """One tracked observation attached to a local track id."""

    frame_index: int
    timestamp_second: float
    bbox: BoundingBox
    confidence: float
    laplacian_score: float
    crop_bgr: np.ndarray | None = None


@dataclass(frozen=True)
class LocalTracklet:
    """Independent tracklet inside one video."""

    video_id: str
    camera_id: str | None
    track_id: str
    observations: tuple[TrackletObservation, ...]


@dataclass(frozen=True)
class TrackletQualityResult:
    """Quality decision after tracklet scoring."""

    accepted: bool
    average_confidence: float
    average_laplacian: float
    frame_count: int
    frame_density: float
    duration_seconds: float
    rejection_reason: str | None


@dataclass(frozen=True)
class LocalIngestionOutput:
    """Final local ingestion output returned to the API layer."""

    video: dict[str, object]
    people: list[dict[str, object]]
    compressed_path: str
    metadata_path: str
    processed_at: datetime

    def to_response(self) -> dict[str, object]:
        return {
            "status": "completed",
            "processing_backend": "strict_tracking_service",
            "source_path": str(self.video.get("source_path") or ""),
            "compressed_path": self.compressed_path,
            "metadata_path": self.metadata_path,
            "video": self.video,
            "people": self.people,
            "person_count": len(self.people),
            "processed_at": self.processed_at,
        }


def _crop_from_bbox(image: np.ndarray, bbox: BoundingBox) -> np.ndarray | None:
    """Extract a BGR crop from image at bbox coordinates, clamped to image bounds."""
    h, w = image.shape[:2]
    x1 = max(0, min(bbox.x1, w))
    x2 = max(0, min(bbox.x2, w))
    y1 = max(0, min(bbox.y1, h))
    y2 = max(0, min(bbox.y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    return crop.copy() if crop.size > 0 else None


def _bbox_iou(lhs: BoundingBox, rhs: BoundingBox) -> float:
    """Intersection-over-Union for two bounding boxes."""
    inter_x1 = max(lhs.x1, rhs.x1)
    inter_y1 = max(lhs.y1, rhs.y1)
    inter_x2 = min(lhs.x2, rhs.x2)
    inter_y2 = min(lhs.y2, rhs.y2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area <= 0:
        return 0.0
    union_area = lhs.area + rhs.area - inter_area
    return float(inter_area / union_area) if union_area > 0 else 0.0


def _bbox_center(bbox: BoundingBox) -> tuple[float, float]:
    return (
        float(bbox.x1 + bbox.x2) / 2.0,
        float(bbox.y1 + bbox.y2) / 2.0,
    )


def _point_distance(lhs: tuple[float, float], rhs: tuple[float, float]) -> float:
    return math.hypot(lhs[0] - rhs[0], lhs[1] - rhs[1])


def _serialize_tracklet_frames(
    observations: tuple[TrackletObservation, ...] | list[TrackletObservation],
) -> list[dict[str, object]]:
    return [
        {
            "frame_idx": item.frame_index,
            "timestamp_second": item.timestamp_second,
            "bbox": item.bbox.to_xyxy(),
            "confidence": item.confidence,
        }
        for item in observations
    ]


def _compact_serialized_tracklet_frames(
    frames: list[dict[str, object]],
    *,
    max_frames: int = 30,
) -> list[dict[str, object]]:
    if len(frames) <= max_frames:
        return list(frames)
    step = max(1, len(frames) // max_frames)
    selected = list(frames[::step][:max_frames])
    if selected and frames:
        selected[-1] = frames[-1]
    return selected


def _compact_tracklet_frames(
    observations: tuple[TrackletObservation, ...] | list[TrackletObservation],
    *,
    max_frames: int = 30,
) -> list[dict[str, object]]:
    return _compact_serialized_tracklet_frames(
        _serialize_tracklet_frames(observations),
        max_frames=max_frames,
    )


def _quality_or_zero(person: dict[str, object], field_name: str) -> float:
    quality = person.get("tracklet_quality")
    if not isinstance(quality, dict):
        return 0.0
    value = quality.get(field_name)
    return float(value) if isinstance(value, (int, float)) else 0.0


@dataclass
class VideoFrameSampler:
    """Decode video and sample frames at the fixed ingest FPS."""

    sample_fps: int = 4

    def stream_batched(self, video_path: Path, batch_size: int = 150):
        """
        Generator: yields successive batches of SampledFrame.

        Each batch holds at most `batch_size` full-resolution frames.
        Callers must process and discard each batch before requesting the next
        so that peak RAM stays at  batch_size × frame_bytes  instead of
        total_frames × frame_bytes  (typically 900 MB vs 18 GB for a 10-min video).
        """
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open source video: {video_path}")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0.0:
            source_fps = float(self.sample_fps)

        next_emit_second = 0.0
        source_frame_index = 0
        sampled_index = 0
        batch: list[SampledFrame] = []

        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                timestamp_second = source_frame_index / source_fps
                if timestamp_second + 1e-9 < next_emit_second:
                    source_frame_index += 1
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                laplacian_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                batch.append(
                    SampledFrame(
                        frame_index=sampled_index,
                        timestamp_second=round(timestamp_second, 6),
                        image=frame,
                        laplacian_score=round(laplacian_score, 6),
                    )
                )
                sampled_index += 1
                next_emit_second += 1.0 / max(self.sample_fps, 1)
                source_frame_index += 1

                if len(batch) >= batch_size:
                    yield tuple(batch)
                    batch.clear()
        finally:
            capture.release()

        if batch:
            yield tuple(batch)

    def sample(self, video_path: Path) -> tuple[SampledFrame, ...]:
        """Load all frames at once — only safe for short clips or tests."""
        all_frames: list[SampledFrame] = []
        for batch in self.stream_batched(video_path):
            all_frames.extend(batch)
        return tuple(all_frames)


class RFDETRPersonDetector:
    """RF-DETR 2x-large person detector — strict production detector, class-level singleton."""

    confidence_threshold: float = 0.32
    max_detections_per_frame: int = 300

    _instance: "RFDETRPersonDetector | None" = None
    _class_lock = threading.Lock()

    def __new__(cls) -> "RFDETRPersonDetector":
        with cls._class_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._model = None
                inst._ready = False
                inst._load_lock = threading.Lock()
                cls._instance = inst
        return cls._instance

    def _ensure_loaded(self) -> None:
        if self._ready:
            return
        with self._load_lock:
            if self._ready:
                return
            self._model = self._try_load_rfdetr()
            self._ready = True

    @staticmethod
    def _try_load_rfdetr():
        # Checkpoint: storage/model-weights/rf-detr/rf-detr-xxlarge.pth
        # __file__ = app/local_ingestion_pipeline.py → parent^5 = A20-App-119/ (repo root)
        env_path = os.environ.get("MCPT_RF_DETR_WEIGHTS", "").strip()
        if env_path:
            weights_path = Path(env_path)
        else:
            _here = Path(__file__).resolve()
            repo_root = _here.parent.parent.parent.parent.parent  # 5 parents = A20-App-119/
            weights_path = repo_root / "storage" / "model-weights" / "rf-detr" / "rf-detr-xxlarge.pth"
        weights_str = str(weights_path) if weights_path.exists() else "rf-detr-xxlarge.pth"

        for loader in [  # noqa: RET503
            lambda: __import__("rfdetr", fromlist=["RFDETR2XLarge"]).RFDETR2XLarge(pretrain_weights=weights_str),
            lambda: __import__("rfdetr", fromlist=["RFDETRLarge"]).RFDETRLarge(),
            lambda: __import__("rfdetr", fromlist=["RFDETR"]).RFDETR(model_id="rf-detr-2xlarge"),
        ]:
            try:
                return loader()
            except Exception:
                continue
        return None

    def detect(self, frames: tuple[SampledFrame, ...]) -> dict[int, tuple[FrameDetection, ...]]:
        self._ensure_loaded()
        if self._model is None:
            raise RuntimeError("RF-DETR model failed to load. Ensure rfdetr[plus] is installed on the LightningAI GPU machine.")

        import numpy as np
        from PIL import Image as _PILImage

        batch_size = max(1, int(os.environ.get("MCPT_DETECTOR_BATCH_SIZE", "8")))
        result: dict[int, tuple[FrameDetection, ...]] = {}

        for batch_start in range(0, len(frames), batch_size):
            batch_frames = frames[batch_start: batch_start + batch_size]
            pils = [_PILImage.fromarray(cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB)) for f in batch_frames]
            try:
                with gpu_lock:
                    batch_dets = self._model.predict(pils, threshold=self.confidence_threshold)
            except Exception:
                for f in batch_frames:
                    result[f.frame_index] = ()
                continue

            if not isinstance(batch_dets, list):
                batch_dets = [batch_dets]

            for frame, dets in zip(batch_frames, batch_dets):
                xyxy = getattr(dets, "xyxy", None)
                confs = getattr(dets, "confidence", None)
                class_ids = getattr(dets, "class_id", None)
                if xyxy is None or confs is None:
                    result[frame.frame_index] = ()
                    continue

                frame_dets: list[FrameDetection] = []
                for i, (box, conf) in enumerate(zip(xyxy, confs)):
                    if class_ids is not None and int(class_ids[i]) != 1:  # RF-DETR COCO 1-indexed: person=1
                        continue
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                    frame_dets.append(
                        FrameDetection(
                            frame_index=frame.frame_index,
                            timestamp_second=frame.timestamp_second,
                            bbox=bbox,
                            confidence=float(np.clip(conf, 0.0, 1.0)),
                            laplacian_score=frame.laplacian_score,
                            crop_bgr=_crop_from_bbox(frame.image, bbox),
                        )
                    )
                frame_dets.sort(key=lambda d: d.confidence, reverse=True)
                result[frame.frame_index] = tuple(frame_dets[: self.max_detections_per_frame])
        return result


@dataclass
class HeadBoxTracker:
    """
    ByteTrack-style tracker using head-box geometry, center-distance,
    velocity prediction, and IoU. No ReID/GPU during tracking.

    Designed for top-down / bird's-eye surveillance cameras where person-ReID
    models trained on eye-level data (MSMT17) are unreliable. TransReID is still
    used downstream in _build_people_batch for appearance_embedding_vector.

    Head box: top head_ratio of bbox height, x-shrunk by shrink_x on each side.
    Matching cost = distance_weight * center_dist + velocity_weight * predicted_dist + iou_weight * (1 - iou).

    Matching cascade per frame:
      Stage 1 — high-conf dets  → active tracks
      Stage 2 — low-conf dets   → active tracks
      Stage 3 — high-conf unmatched → buffer tracks
      New track for remaining unmatched high-conf dets above new_track_threshold.
    """

    track_thresh: float = 0.40
    low_thresh: float = 0.10
    new_track_threshold: float = 0.45
    max_match_cost: float = 0.80
    max_buffer_match_cost: float = 0.90      # relaxed: long-gap re-entry cost runs higher
    max_head_center_distance: float = 120.0  # wider gate: 4fps = 0.25s between frames
    max_predicted_distance: float = 180.0    # wider gate: velocity estimate less reliable at 4fps
    iou_weight: float = 0.20
    distance_weight: float = 0.55
    velocity_weight: float = 0.25
    track_buffer: int = 20                   # 5s @ 4fps (was 3s)
    max_buffer_frames: int = 300             # 75s @ 4fps (was 7.5s) — people behind shelves
    head_ratio: float = 0.35
    shrink_x: float = 0.08
    min_track_frames: int = 9               # 9 frames × 0.25s = 2.25s ≥ minimum_duration_seconds
    min_track_density: float = 0.10         # re-entry obs span includes buffer gap → density deflated

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.active: dict[str, list[TrackletObservation]] = {}
        self.active_last_bbox: dict[str, BoundingBox] = {}
        self.active_last_frame: dict[str, int] = {}
        self.buffer: dict[str, list[TrackletObservation]] = {}
        self.buffer_last_bbox: dict[str, BoundingBox] = {}
        self.buffer_entry_frame: dict[str, int] = {}
        self.next_id = 1

    def _head_bbox(self, bbox: BoundingBox) -> BoundingBox:
        x1, y1, x2, y2 = float(bbox.x1), float(bbox.y1), float(bbox.x2), float(bbox.y2)
        w = x2 - x1
        h = y2 - y1
        return BoundingBox(
            x1=int(x1 + w * self.shrink_x),
            y1=int(y1),
            x2=int(x2 - w * self.shrink_x),
            y2=int(y1 + h * self.head_ratio),
        )

    def track_incremental(
        self,
        *,
        video_id: str,
        camera_id: str | None,
        detections_by_frame: dict[int, tuple[FrameDetection, ...]],
    ) -> tuple[LocalTracklet, ...]:
        completed: list[LocalTracklet] = []

        for fk in sorted(detections_by_frame):
            dets = list(detections_by_frame.get(fk) or ())
            frame_ts = dets[0].timestamp_second if dets else 0.0

            # Move stale active tracks to buffer (no match for > track_buffer frames)
            stale = [tid for tid in self.active if fk - self.active_last_frame.get(tid, fk) > self.track_buffer]
            for tid in stale:
                self.buffer[tid] = self.active.pop(tid)
                self.buffer_last_bbox[tid] = self.active_last_bbox.pop(tid)
                self.buffer_entry_frame[tid] = fk
                self.active_last_frame.pop(tid, None)

            # Finalize buffer tracks that have been waiting too long
            expired = [tid for tid in self.buffer if fk - self.buffer_entry_frame.get(tid, fk) > self.max_buffer_frames]
            for tid in expired:
                obs = self.buffer.pop(tid, [])
                if obs and self._should_keep_tracklet(obs):
                    completed.append(LocalTracklet(video_id=video_id, camera_id=camera_id, track_id=tid, observations=tuple(obs)))
                self.buffer_last_bbox.pop(tid, None)
                self.buffer_entry_frame.pop(tid, None)

            if not dets:
                continue

            high = [d for d in dets if d.confidence >= self.track_thresh]
            low  = [d for d in dets if self.low_thresh <= d.confidence < self.track_thresh]
            active_unmatched = set(self.active.keys())
            active_states = {
                tid: self._build_match_state(
                    observations=self.active[tid],
                    last_bbox=self.active_last_bbox[tid],
                    target_frame_idx=fk,
                )
                for tid in active_unmatched
            }

            # Stage 1: high-conf → active tracks
            unmatched_high: list[FrameDetection] = []
            for det in high:
                best_tid, best_cost = self._best_match(
                    self._head_bbox(det.bbox),
                    active_states,
                    active_unmatched,
                    max_cost=self.max_match_cost,
                )
                if best_tid is not None and best_cost <= self.max_match_cost:
                    self._update_active(best_tid, det, fk, frame_ts)
                    active_unmatched.discard(best_tid)
                    active_states.pop(best_tid, None)
                else:
                    unmatched_high.append(det)

            # Stage 2: low-conf → remaining active tracks
            for det in low:
                if not active_unmatched:
                    break
                best_tid, best_cost = self._best_match(
                    self._head_bbox(det.bbox),
                    active_states,
                    active_unmatched,
                    max_cost=self.max_match_cost,
                )
                if best_tid is not None and best_cost <= self.max_match_cost:
                    self._update_active(best_tid, det, fk, frame_ts)
                    active_unmatched.discard(best_tid)
                    active_states.pop(best_tid, None)

            # Stage 3: high-conf unmatched → buffer tracks (re-entry)
            buffer_candidates = set(self.buffer.keys())
            buffer_states = {
                tid: self._build_match_state(
                    observations=self.buffer[tid],
                    last_bbox=self.buffer_last_bbox[tid],
                    target_frame_idx=fk,
                )
                for tid in buffer_candidates
            }
            new_dets: list[FrameDetection] = []
            for det in unmatched_high:
                if not buffer_candidates:
                    new_dets.append(det)
                    continue
                best_tid, best_cost = self._best_match(
                    self._head_bbox(det.bbox),
                    buffer_states,
                    buffer_candidates,
                    max_cost=self.max_buffer_match_cost,
                )
                if best_tid is not None and best_cost <= self.max_buffer_match_cost:
                    obs = self.buffer.pop(best_tid)
                    obs.append(self._make_obs(det, frame_ts))
                    self.active[best_tid] = obs
                    self.active_last_bbox[best_tid] = det.bbox
                    self.active_last_frame[best_tid] = fk
                    self.buffer_last_bbox.pop(best_tid, None)
                    self.buffer_entry_frame.pop(best_tid, None)
                    buffer_candidates.discard(best_tid)
                    buffer_states.pop(best_tid, None)
                else:
                    new_dets.append(det)

            # New tracks for truly unmatched high-conf dets
            for det in new_dets:
                if det.confidence >= self.new_track_threshold:
                    tid = str(self.next_id)
                    self.next_id += 1
                    self.active[tid] = [self._make_obs(det, frame_ts)]
                    self.active_last_bbox[tid] = det.bbox
                    self.active_last_frame[tid] = fk

        return tuple(t for t in completed if t.observations)

    def finalize_all(self, *, video_id: str, camera_id: str | None) -> tuple[LocalTracklet, ...]:
        completed: list[LocalTracklet] = []
        for tid, obs_list in list(self.active.items()) + list(self.buffer.items()):
            if obs_list and self._should_keep_tracklet(obs_list):
                completed.append(LocalTracklet(video_id=video_id, camera_id=camera_id, track_id=tid, observations=tuple(obs_list)))
        self.reset()
        return tuple(completed)

    def track(
        self,
        *,
        video_id: str,
        camera_id: str | None,
        detections_by_frame: dict[int, tuple[FrameDetection, ...]],
    ) -> tuple[LocalTracklet, ...]:
        self.reset()
        completed = list(self.track_incremental(video_id=video_id, camera_id=camera_id, detections_by_frame=detections_by_frame))
        completed.extend(self.finalize_all(video_id=video_id, camera_id=camera_id))
        return tuple(t for t in completed if t.observations)

    def _update_active(self, tid: str, det: FrameDetection, frame_idx: int, frame_ts: float) -> None:
        self.active[tid].append(self._make_obs(det, frame_ts))
        self.active_last_bbox[tid] = det.bbox
        self.active_last_frame[tid] = frame_idx

    def _build_match_state(
        self,
        *,
        observations: list[TrackletObservation],
        last_bbox: BoundingBox,
        target_frame_idx: int,
    ) -> tuple[BoundingBox, tuple[float, float], tuple[float, float]]:
        head_bbox = self._head_bbox(last_bbox)
        last_center = _bbox_center(head_bbox)
        if len(observations) < 2:
            return head_bbox, last_center, last_center

        previous_head_bbox = self._head_bbox(observations[-2].bbox)
        previous_center = _bbox_center(previous_head_bbox)
        frame_delta = max(observations[-1].frame_index - observations[-2].frame_index, 1)
        vx = (last_center[0] - previous_center[0]) / float(frame_delta)
        vy = (last_center[1] - previous_center[1]) / float(frame_delta)
        raw_steps = max(target_frame_idx - observations[-1].frame_index, 0)
        # Cap extrapolation: beyond track_buffer * 2 frames the velocity estimate is stale.
        # Without this cap, buffer tracks with any velocity produce predicted positions
        # hundreds of pixels off-screen, making Stage-3 re-entry matching impossible.
        predict_steps = min(raw_steps, self.track_buffer * 2)
        predicted_center = (
            last_center[0] + vx * predict_steps,
            last_center[1] + vy * predict_steps,
        )
        return head_bbox, last_center, predicted_center

    def _best_match(
        self,
        det_bbox: BoundingBox,
        track_states: dict[str, tuple[BoundingBox, tuple[float, float], tuple[float, float]]],
        candidates: set[str],
        *,
        max_cost: float,
    ) -> tuple[str | None, float]:
        det_center = _bbox_center(det_bbox)
        best_tid = None
        best_cost = max_cost + 1.0
        best_iou = -1.0
        for tid in candidates:
            state = track_states.get(tid)
            if state is None:
                continue
            head_bbox, last_center, predicted_center = state
            center_distance = _point_distance(last_center, det_center)
            predicted_distance = _point_distance(predicted_center, det_center)
            if (
                center_distance > self.max_head_center_distance
                and predicted_distance > self.max_predicted_distance
            ):
                continue

            center_cost = min(center_distance / max(self.max_head_center_distance, 1e-6), 1.0)
            predicted_cost = min(predicted_distance / max(self.max_predicted_distance, 1e-6), 1.0)
            iou = _bbox_iou(head_bbox, det_bbox)
            cost = (
                self.distance_weight * center_cost
                + self.velocity_weight * predicted_cost
                + self.iou_weight * (1.0 - iou)
            )
            if cost < best_cost or (abs(cost - best_cost) <= 1e-6 and iou > best_iou):
                best_cost = cost
                best_iou = iou
                best_tid = tid
        return best_tid, best_cost

    def _should_keep_tracklet(self, observations: list[TrackletObservation]) -> bool:
        frame_count = len(observations)
        if frame_count < self.min_track_frames:
            return False
        frame_span = max(observations[-1].frame_index - observations[0].frame_index + 1, 1)
        density = frame_count / float(frame_span)
        return density >= self.min_track_density

    @staticmethod
    def _make_obs(det: FrameDetection, frame_ts: float) -> TrackletObservation:
        return TrackletObservation(
            frame_index=det.frame_index,
            timestamp_second=frame_ts,
            bbox=det.bbox,
            confidence=det.confidence,
            laplacian_score=det.laplacian_score,
            crop_bgr=det.crop_bgr,
        )


@dataclass
class TrackletQualityScorer:
    """Filter blurry or weak tracklets before feature extraction."""

    minimum_confidence_score: float = 0.3
    minimum_frame_count: int = 9            # aligned with HeadBoxTracker.min_track_frames
    minimum_frame_density: float = 0.10    # re-entry tracklets span buffer gap → density deflated
    minimum_average_laplacian: float = 12.0
    minimum_duration_seconds: float = 2.0

    def score(self, tracklet: LocalTracklet) -> TrackletQualityResult:
        confidences = [item.confidence for item in tracklet.observations]
        laplacians = [item.laplacian_score for item in tracklet.observations]
        average_confidence = round(sum(confidences) / max(len(confidences), 1), 6)
        average_laplacian = round(sum(laplacians) / max(len(laplacians), 1), 6)
        frame_count = len(tracklet.observations)
        frame_span = max(tracklet.observations[-1].frame_index - tracklet.observations[0].frame_index + 1, 1) if frame_count else 1
        frame_density = round(frame_count / float(frame_span), 6)
        duration_seconds = 0.0
        if frame_count >= 2:
            duration_seconds = max(tracklet.observations[-1].timestamp_second - tracklet.observations[0].timestamp_second, 0.0)

        if frame_count < self.minimum_frame_count:
            return TrackletQualityResult(False, average_confidence, average_laplacian, frame_count, frame_density, duration_seconds, "insufficient_frames")
        if frame_density < self.minimum_frame_density:
            return TrackletQualityResult(False, average_confidence, average_laplacian, frame_count, frame_density, duration_seconds, "sparse_tracklet")
        if duration_seconds < self.minimum_duration_seconds:
            return TrackletQualityResult(False, average_confidence, average_laplacian, frame_count, frame_density, duration_seconds, "short_tracklet")
        if average_confidence < self.minimum_confidence_score:
            return TrackletQualityResult(False, average_confidence, average_laplacian, frame_count, frame_density, duration_seconds, "low_confidence")
        if average_laplacian < self.minimum_average_laplacian:
            return TrackletQualityResult(False, average_confidence, average_laplacian, frame_count, frame_density, duration_seconds, "blurry_tracklet")
        return TrackletQualityResult(True, average_confidence, average_laplacian, frame_count, frame_density, duration_seconds, None)


@dataclass
class LocalMetadataAssembler:
    """Convert accepted tracklets into the unified metadata payload."""

    feature_pipeline: TrackletFeaturePipelineProcessor = field(default_factory=TrackletFeaturePipelineProcessor)
    tracklet_worker_count: int = field(default_factory=_default_tracklet_worker_count)

    def build_people(
        self,
        *,
        video_id: str,
        camera_id: str | None,
        tracklets: tuple[LocalTracklet, ...],
        quality_results: dict[str, TrackletQualityResult],
        sampled_fps: int,
    ) -> list[dict[str, object]]:
        accepted_tracklets = [
            (tracklet, quality_results[tracklet.track_id])
            for tracklet in tracklets
            if tracklet.track_id in quality_results and quality_results[tracklet.track_id].accepted
        ]
        if not accepted_tracklets:
            return []
        return self._build_people_batch(
            video_id=video_id,
            camera_id=camera_id,
            accepted_tracklets=accepted_tracklets,
            sampled_fps=sampled_fps,
        )

    def _build_people_batch(
        self,
        *,
        video_id: str,
        camera_id: str | None,
        accepted_tracklets: list[tuple["LocalTracklet", "TrackletQualityResult"]],
        sampled_fps: int,
    ) -> list[dict[str, object]]:
        """
        Batch-optimized feature extraction: runs each model once for ALL tracklets.
        SigLIP2 image: 1 call instead of 2N  (static + appearance share the same batch)
        TransReID:     1 call instead of N
        VideoMAE:      1 batched call instead of N×clips calls
        Text features: already cached in SigLIP2, effectively free.
        """
        if not accepted_tracklets:
            return []

        import cv2 as _cv2
        from PIL import Image as _PILImage

        # Lazy import model hubs and helpers from model_adapters
        from .model_adapters import (
            SigLIP2ModelHub, DINOv2ReIDHub, VideoMAEHub,
            _crop_pil, _selected_observations, _quality_weighted_pool,
            _GENDER_PROMPTS, _AGE_PROMPTS,
            _SHIRT_PROMPTS, _PANTS_PROMPTS, _HAIR_PROMPTS, _SKIN_PROMPTS,
            _HAT_PROMPTS, _BAG_PROMPTS, _HEAD_ACCESSORY_PROMPTS, _SHOES_PROMPTS,
        )
        from .tracklet_feature_pipeline import (
            ActionClipBuilder, TrackletFeatureAggregator,
            EMBEDDING_VOCABULARY,
            StaticAttributeResult, AppearanceAttributeResult,
            AppearanceEmbeddingResult, AttributeEmbeddingResult,
            BehaviorAnalysisResult, SemanticEmbeddingResult,
        )

        siglip = SigLIP2ModelHub()
        reid   = DINOv2ReIDHub()
        vmae   = VideoMAEHub()

        # ── Build TrackletFeatureInput for every tracklet ──────────────────────
        tracklet_inputs = [
            TrackletFeatureInput(
                video_id=video_id,
                object_id=tracklet.track_id,
                sampled_fps=sampled_fps,
                frames=tuple(
                    TrackletFrameObservation(
                        frame_index=item.frame_index,
                        timestamp_second=item.timestamp_second,
                        bbox=item.bbox,
                        detection_confidence=item.confidence,
                        laplacian_score=item.laplacian_score,
                        crop_bgr=item.crop_bgr,
                    )
                    for item in tracklet.observations
                ),
            )
            for tracklet, _ in accepted_tracklets
        ]
        selections = [self.feature_pipeline.selector.select(ti) for ti in tracklet_inputs]

        # ── Phase 1: Batch SigLIP2 image encoding ──────────────────────────────
        per_t_valid: list[list[tuple]] = []   # [(frame_obs, pil_img), ...]  per tracklet
        all_siglip_pils: list = []
        siglip_slices: list[tuple[int, int]] = []

        for ti, selection in zip(tracklet_inputs, selections):
            selected_obs = _selected_observations(ti, selection)
            valid = [(f, _crop_pil(f, None)) for f in selected_obs]
            valid = [(f, p) for f, p in valid if p is not None]
            per_t_valid.append(valid)
            s = len(all_siglip_pils)
            all_siglip_pils.extend(p for _, p in valid)
            siglip_slices.append((s, len(all_siglip_pils)))

        # ONE SigLIP2 image forward pass for ALL tracklets combined
        all_img_feats = (
            siglip.image_features(all_siglip_pils)
            if all_siglip_pils
            else np.zeros((0, 1024), dtype=np.float32)
        )

        # Pre-warm all text feature caches (one-time; subsequent calls hit cache)
        _prompt_map = {
            "shirt": _SHIRT_PROMPTS, "pants": _PANTS_PROMPTS,
            "hair_color": _HAIR_PROMPTS, "skin_tone": _SKIN_PROMPTS,
            "hat": _HAT_PROMPTS, "bag": _BAG_PROMPTS,
            "head_accessory": _HEAD_ACCESSORY_PROMPTS, "shoes": _SHOES_PROMPTS,
        }
        for _, prompts in _GENDER_PROMPTS + _AGE_PROMPTS:
            siglip.text_features(prompts)
        for label_prompts in _prompt_map.values():
            for _, prompts in label_prompts:
                siglip.text_features(prompts)

        # ── Phase 2: Classify attributes using pre-computed image features (CPU) ──
        all_static: list[StaticAttributeResult] = []
        all_appearance: list[AppearanceAttributeResult] = []

        for idx, selection in enumerate(selections):
            s, e = siglip_slices[idx]
            img_feats = all_img_feats[s:e]   # [N_frames, 1024]
            valid     = per_t_valid[idx]
            quality_map = {item.frame_index: item.quality_score for item in selection.selected_frames}

            if img_feats.shape[0] == 0:
                all_static.append(StaticAttributeResult(gender=None, age_group=None, confidence=0.0))
                all_appearance.append(AppearanceAttributeResult())
                continue

            # Static: gender + age_group
            gender_votes: dict = {}
            age_votes: dict    = {}
            for i, (frame, _) in enumerate(valid):
                w  = quality_map.get(frame.frame_index, 0.1)
                feat = img_feats[i]
                best_g, bs = None, -float("inf")
                for label, prompts in _GENDER_PROMPTS:
                    sc = float(np.mean(siglip.text_features(prompts) @ feat))
                    if sc > bs: bs, best_g = sc, label
                gender_votes[best_g] = gender_votes.get(best_g, 0.0) + w
                best_a, bs = None, -float("inf")
                for label, prompts in _AGE_PROMPTS:
                    sc = float(np.mean(siglip.text_features(prompts) @ feat))
                    if sc > bs: bs, best_a = sc, label
                age_votes[best_a] = age_votes.get(best_a, 0.0) + w
            gender    = max(gender_votes, key=gender_votes.get) if gender_votes else None
            age_group = max(age_votes,    key=age_votes.get)    if age_votes    else None
            all_static.append(StaticAttributeResult(
                gender=gender, age_group=age_group,
                confidence=0.45 if (gender or age_group) else 0.0,
            ))

            # Appearance: 8 attribute fields
            fields: dict[str, dict] = {k: {} for k in _prompt_map}
            for i, (frame, _) in enumerate(valid):
                w    = max(quality_map.get(frame.frame_index, 0.1), 1e-4)
                feat = img_feats[i]
                for field_name, label_prompts in _prompt_map.items():
                    best_label, bs = None, -float("inf")
                    for label, prompts in label_prompts:
                        sc = float(np.mean(siglip.text_features(prompts) @ feat))
                        if sc > bs: bs, best_label = sc, label
                    fields[field_name][best_label] = fields[field_name].get(best_label, 0.0) + w

            def _best(votes: dict):
                return max(votes, key=votes.get) if votes else None

            all_appearance.append(AppearanceAttributeResult(
                head_accessory=_best(fields["head_accessory"]),
                hat=_best(fields["hat"]),
                hair_color=_best(fields["hair_color"]),
                skin_tone=_best(fields["skin_tone"]),
                shirt=_best(fields["shirt"]),
                pants=_best(fields["pants"]),
                shoes=_best(fields["shoes"]),
                bag=_best(fields["bag"]),
            ))

        # ── Phase 3: Batch DINOv2 Re-ID ────────────────────────────────────────
        # DINOv2 ViT-L/14 replaces TransReID: view-invariant CLS-token embeddings
        # handle overhead / angled cameras that break MSMT17-trained side-view models.
        # No part-cropping needed — the global CLS token already captures full-body
        # appearance from any viewpoint.
        per_t_pil_crops: list[list]        = []
        per_t_weights:   list[list[float]] = []
        all_fulls:  list = []
        reid_slices: list[tuple[int, int]] = []

        for ti, selection in zip(tracklet_inputs, selections):
            selected_obs = _selected_observations(ti, selection)
            quality_map  = {item.frame_index: item.quality_score for item in selection.selected_frames}
            pil_crops, weights = [], []
            for frame in selected_obs:
                pil = _crop_pil(frame, None)
                if pil is None:
                    continue
                pil_crops.append(pil)
                weights.append(quality_map.get(frame.frame_index, 0.1))
            per_t_pil_crops.append(pil_crops)
            per_t_weights.append(weights)
            s = len(all_fulls)
            all_fulls.extend(pil_crops)
            reid_slices.append((s, len(all_fulls)))

        # ONE DINOv2 call with all crops → [M, 1024]
        gf_all = reid.embed_crops(all_fulls) if all_fulls else np.zeros((0, 1024), dtype=np.float32)

        all_appearance_embed: list[AppearanceEmbeddingResult] = []
        for idx, (pil_crops, weights) in enumerate(zip(per_t_pil_crops, per_t_weights)):
            s, e = reid_slices[idx]
            if not pil_crops:
                empty = tuple(0.0 for _ in range(1024))
                all_appearance_embed.append(AppearanceEmbeddingResult(
                    embedding_model="dinov2-vitl14",
                    embedding_vector=empty, tracklet_vectors=(empty,),
                ))
                continue
            gf = gf_all[s:e]  # [K, 1024] already L2-normalised per frame
            pooled = _quality_weighted_pool(list(gf), weights)
            all_appearance_embed.append(AppearanceEmbeddingResult(
                embedding_model="dinov2-vitl14",
                embedding_vector=tuple(round(float(v), 6) for v in pooled.tolist()),
                tracklet_vectors=tuple(
                    tuple(round(float(v), 6) for v in gf[i].tolist())
                    for i in range(len(pil_crops))
                ),
            ))

        # ── Phase 4: Batch VideoMAE ─────────────────────────────────────────────
        clip_builder = ActionClipBuilder()
        all_clips_per_t = [
            clip_builder.build(ti, sel.representative_frame.frame_index)
            for ti, sel in zip(tracklet_inputs, selections)
        ]
        all_clips_flat = [clip for clips in all_clips_per_t for clip in clips]

        def _clip_pils(clip) -> list:
            pils = []
            for frame in clip.frames:
                crop = frame.crop_bgr
                if crop is not None and crop.size > 0:
                    rgb = _cv2.cvtColor(np.asarray(crop, dtype=np.uint8), _cv2.COLOR_BGR2RGB)
                    pils.append(_PILImage.fromarray(rgb))
            return pils

        # ONE batched VideoMAE call
        clips_pil_lists = [_clip_pils(clip) for clip in all_clips_flat]
        if clips_pil_lists:
            vmae_feats = vmae.extract_features_batch(clips_pil_lists)  # [total_clips, 1024]
        else:
            vmae_feats = np.zeros((0, 1024), dtype=np.float32)

        # Action vocabulary text features (cached after first call)
        vocab_labels  = list(EMBEDDING_VOCABULARY)
        vocab_prompts = [
            f"a surveillance footage of a person who is {lbl.replace('_', ' ')} in an indoor space"
            for lbl in vocab_labels
        ]
        text_vocab_feats = siglip.text_features(vocab_prompts)  # [K, 1024]

        clip_offset = 0
        all_behavior_per_t:  list[tuple] = []
        all_semantic_per_t:  list[tuple] = []

        for clips in all_clips_per_t:
            behaviors: list[BehaviorAnalysisResult]  = []
            semantics: list[SemanticEmbeddingResult] = []
            for clip in clips:
                if clip_offset >= len(vmae_feats):
                    break
                vid_feat = vmae_feats[clip_offset]
                clip_offset += 1
                scores    = text_vocab_feats @ vid_feat          # [K]
                best_idx  = int(np.argmax(scores))
                best_lbl  = vocab_labels[best_idx]
                confidence = round(float(np.clip(scores[best_idx], 0.0, 1.0)), 6)
                behaviors.append(BehaviorAnalysisResult(
                    clip_id=clip.clip_id,
                    action_summary=best_lbl,
                    labels=(best_lbl,),
                    confidence=confidence,
                    metadata={"scores": {l: round(float(s), 4) for l, s in zip(vocab_labels, scores)}},
                ))
                sem_text  = f"a surveillance footage of a person who is {best_lbl.replace('_', ' ')} in an indoor space"
                sem_feat  = siglip.text_features([sem_text])[0]
                semantics.append(SemanticEmbeddingResult(
                    clip_id=clip.clip_id,
                    embedding_model="siglip2-vit-l16-512-action",
                    vocabulary=EMBEDDING_VOCABULARY,
                    embedding_vector=tuple(round(float(v), 6) for v in sem_feat.tolist()),
                ))
            all_behavior_per_t.append(tuple(behaviors))
            all_semantic_per_t.append(tuple(semantics))

        # ── Phase 5: Attribute text embeddings (text-only, cached) ─────────────
        all_attr_embed: list[AttributeEmbeddingResult] = []
        for static_attr in all_static:
            parts = [p for p in [static_attr.gender, static_attr.age_group] if p]
            text  = "a photo of a " + (" ".join(parts) if parts else "person")
            feat  = siglip.text_features([text])[0]
            all_attr_embed.append(AttributeEmbeddingResult(
                embedding_model="siglip2-vit-l16-512-attr",
                embedding_vector=tuple(round(float(v), 6) for v in feat.tolist()),
            ))

        # ── Phase 6: Aggregate per tracklet (CPU) ──────────────────────────────
        aggregator = TrackletFeatureAggregator()
        runtime_meta = {"batch_optimized": True, "batch_size": len(accepted_tracklets)}
        people: list[dict[str, object]] = []

        for idx, ((tracklet, quality), ti, selection) in enumerate(
            zip(accepted_tracklets, tracklet_inputs, selections)
        ):
            try:
                agg = aggregator.aggregate(
                    tracklet=ti,
                    selection=selection,
                    static_attributes=all_static[idx],
                    attribute_embedding=all_attr_embed[idx],
                    appearance_attributes=all_appearance[idx],
                    appearance_embedding=all_appearance_embed[idx],
                    clips=all_clips_per_t[idx],
                    behavior_results=all_behavior_per_t[idx],
                    semantic_embeddings=all_semantic_per_t[idx],
                    runtime_metadata=runtime_meta,
                )
            except Exception as exc:
                LOGGER.warning("batch aggregation failed for track %s: %s — falling back", tracklet.track_id, exc)
                try:
                    agg = self.feature_pipeline.process(ti).aggregated
                except Exception:
                    continue

            meta = agg.to_metadata()
            candidate_id = f"{video_id}:{tracklet.track_id}"
            compact_tracklet_frames = _compact_tracklet_frames(tracklet.observations, max_frames=30)
            tracklet_frame_count_full = len(tracklet.observations)
            tracklet_frame_start_idx = int(tracklet.observations[0].frame_index) if tracklet.observations else 0
            tracklet_frame_end_idx = int(tracklet.observations[-1].frame_index) if tracklet.observations else 0
            meta.update({
                "candidate_id": candidate_id,
                "camera_id": camera_id,
                "video_id": video_id,
                "track_id": tracklet.track_id,
                "human_key": f"{camera_id or video_id}:{tracklet.track_id}",
                "tracklet_frames": compact_tracklet_frames,
                "tracklet_frame_count_full": tracklet_frame_count_full,
                "tracklet_frame_start_idx": tracklet_frame_start_idx,
                "tracklet_frame_end_idx": tracklet_frame_end_idx,
                "tracklet_quality": {
                    "accepted": quality.accepted,
                    "average_confidence": quality.average_confidence,
                    "average_laplacian": quality.average_laplacian,
                    "frame_count": quality.frame_count,
                    "frame_density": quality.frame_density,
                    "duration_seconds": round(quality.duration_seconds, 6),
                },
            })
            people.append(meta)

        return people

def _default_detector():
    """RF-DETR 2x-large — strict production detector on LightningAI GPU."""
    return RFDETRPersonDetector()


def _default_tracker():
    """ByteTrack-style tracker using head boxes plus motion-aware matching."""
    return HeadBoxTracker()


@dataclass
class LocalVideoIngestionPipeline:
    """
    Video ingestion pipeline.

    RF-DETR 2x-large detector + HeadBoxTracker (head box + distance + velocity) +
    TrackletFeaturePipelineProcessor (SigLIP2, TransReID, VideoMAE).
    """

    sample_fps: int = 4
    sampler: VideoFrameSampler = field(default_factory=VideoFrameSampler)
    detector: RFDETRPersonDetector = field(default_factory=_default_detector)
    tracker: HeadBoxTracker = field(default_factory=_default_tracker)
    quality_scorer: TrackletQualityScorer = field(default_factory=TrackletQualityScorer)
    metadata_assembler: LocalMetadataAssembler = field(default_factory=LocalMetadataAssembler)

    def __post_init__(self) -> None:
        self.sampler = VideoFrameSampler(sample_fps=self.sample_fps)

    @staticmethod
    def _person_merge_weight(person: dict[str, object]) -> float:
        quality = person.get("tracklet_quality")
        quality_map = quality if isinstance(quality, dict) else {}
        frame_count = max(1.0, float(quality_map.get("frame_count") or 1.0))
        confidence = max(0.1, float(quality_map.get("average_confidence") or 0.1))
        score = max(0.1, float(person.get("score") or 0.1))
        return frame_count * confidence * score

    @staticmethod
    def _merge_embedding_vectors(
        people: list[dict[str, object]],
        *,
        field_name: str,
    ) -> list[float] | None:
        weighted_vectors: list[np.ndarray] = []
        weights: list[float] = []
        for person in people:
            raw_vector = person.get(field_name)
            if not isinstance(raw_vector, list) or not raw_vector:
                continue
            vector = np.asarray(raw_vector, dtype=np.float32)
            if vector.ndim != 1:
                continue
            vector_norm = float(np.linalg.norm(vector))
            if vector_norm <= 1e-8:
                continue
            weighted_vectors.append(vector / vector_norm)
            weights.append(LocalVideoIngestionPipeline._person_merge_weight(person))
        if not weighted_vectors:
            return None
        weight_array = np.asarray(weights, dtype=np.float32)
        weight_array = weight_array / max(float(weight_array.sum()), 1e-8)
        matrix = np.stack(weighted_vectors, axis=0)
        merged = (matrix * weight_array[:, None]).sum(axis=0)
        merged_norm = float(np.linalg.norm(merged))
        if merged_norm > 1e-8:
            merged = merged / merged_norm
        return [round(float(value), 6) for value in merged.tolist()]

    @staticmethod
    def _resolve_within_camera_identities(
        people: list[dict[str, object]],
        *,
        reid_threshold: float = 0.78,
        max_overlap_ratio: float = 0.15,
    ) -> list[dict[str, object]]:
        """
        Merge same-camera tracklet fragments belonging to the same person by
        comparing TransReID appearance embeddings.

        Two fragments are merged when BOTH conditions hold:
          1. Temporal non-overlap: the shorter tracklet overlaps the other by
             at most max_overlap_ratio — enforces "one person, one location at
             a time" and prevents merging concurrently visible different people.
          2. Embedding similarity: cosine similarity of appearance_embedding_vector
             >= reid_threshold — enforces visual identity.

        Union-Find gives transitive closure so fragments A≈B and B≈C → same
        identity even if A and C are never directly compared.
        All members of a merged group adopt the human_key of the highest-quality
        fragment so that the existing _merge_people_by_identity machinery handles
        the data-level merge.
        """
        if len(people) < 2:
            return people

        n = len(people)
        parent = list(range(n))

        def _find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        def _union(x: int, y: int) -> None:
            px, py = _find(x), _find(y)
            if px != py:
                parent[py] = px

        # Group indices by camera so comparisons stay within-camera only
        by_camera: dict[str, list[int]] = {}
        for i, person in enumerate(people):
            cam = str(person.get("camera_id") or "")
            by_camera.setdefault(cam, []).append(i)

        for cam_indices in by_camera.values():
            if len(cam_indices) < 2:
                continue

            # Pre-compute L2-normalised TransReID vectors and frame ranges
            vectors: dict[int, np.ndarray] = {}
            starts:  dict[int, int] = {}
            ends:    dict[int, int] = {}
            for i in cam_indices:
                p   = people[i]
                raw = p.get("appearance_embedding_vector")
                if not isinstance(raw, list) or not raw:
                    continue
                v   = np.asarray(raw, dtype=np.float32)
                nrm = float(np.linalg.norm(v))
                if nrm < 1e-8:
                    continue
                vectors[i] = v / nrm
                starts[i]  = int(p.get("tracklet_frame_start_idx") or 0)
                ends[i]    = int(p.get("tracklet_frame_end_idx")   or 0)

            eligible = [i for i in cam_indices if i in vectors]
            for a_pos in range(len(eligible)):
                ia = eligible[a_pos]
                for b_pos in range(a_pos + 1, len(eligible)):
                    ib = eligible[b_pos]
                    if _find(ia) == _find(ib):
                        continue

                    # Guard 1: temporal — same person cannot be in two places at once
                    s_a, e_a = starts[ia], ends[ia]
                    s_b, e_b = starts[ib], ends[ib]
                    overlap  = max(0, min(e_a, e_b) - max(s_a, s_b))
                    shorter  = min(max(e_a - s_a, 1), max(e_b - s_b, 1))
                    if overlap / shorter > max_overlap_ratio:
                        continue

                    # Guard 2: embedding similarity
                    if float(np.dot(vectors[ia], vectors[ib])) >= reid_threshold:
                        _union(ia, ib)

        # Elect a shared human_key per merged group (best quality fragment wins)
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(_find(i), []).append(i)

        result = [dict(p) for p in people]
        for members in groups.values():
            if len(members) < 2:
                continue
            best = max(members, key=lambda i: LocalVideoIngestionPipeline._person_merge_weight(people[i]))
            shared_key = str(people[best].get("human_key") or "")
            if not shared_key:
                continue
            for i in members:
                result[i]["human_key"] = shared_key

        return result

    @staticmethod
    def _merge_people_by_identity(
        *,
        video_id: str,
        people: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        grouped: dict[str, list[dict[str, object]]] = {}
        for person in people:
            human_key = str(person.get("human_key") or "").strip()
            if not human_key:
                continue
            grouped.setdefault(human_key, []).append(person)

        merged_people: list[dict[str, object]] = []
        for human_key, group in grouped.items():
            ordered_group = sorted(
                group,
                key=lambda item: (
                    LocalVideoIngestionPipeline._person_merge_weight(item),
                    float(_quality_or_zero(item, "average_confidence")),
                    float(item.get("frame_idx") or 0),
                ),
                reverse=True,
            )
            base = deepcopy(ordered_group[0])
            track_ids = [
                str(item.get("track_id") or "").strip()
                for item in ordered_group
                if str(item.get("track_id") or "").strip()
            ]
            unique_track_ids = list(dict.fromkeys(track_ids))
            full_frame_counts = [
                int(item.get("tracklet_frame_count_full") or _quality_or_zero(item, "frame_count") or 0)
                for item in ordered_group
            ]
            frame_start_indices = [
                int(item.get("tracklet_frame_start_idx") or item.get("frame_idx") or 0)
                for item in ordered_group
            ]
            frame_end_indices = [
                int(item.get("tracklet_frame_end_idx") or item.get("frame_idx") or 0)
                for item in ordered_group
            ]
            merged_frame_count_full = sum(full_frame_counts)
            merged_frame_start_idx = min(frame_start_indices) if frame_start_indices else 0
            merged_frame_end_idx = max(frame_end_indices) if frame_end_indices else 0

            all_frames: list[dict[str, object]] = []
            for item in ordered_group:
                frames = item.get("tracklet_frames")
                if isinstance(frames, list):
                    all_frames.extend(frame for frame in frames if isinstance(frame, dict))
            all_frames.sort(
                key=lambda frame: (
                    float(frame.get("timestamp_second") or 0.0),
                    int(frame.get("frame_idx") or 0),
                )
            )

            all_timeline: list[dict[str, object]] = []
            for item in ordered_group:
                timeline = item.get("timeline")
                if isinstance(timeline, list):
                    all_timeline.extend(segment for segment in timeline if isinstance(segment, dict))
            all_timeline.sort(
                key=lambda segment: (
                    float(segment.get("start_second") or 0.0),
                    float(segment.get("end_second") or 0.0),
                )
            )

            semantic_tokens: list[str] = []
            for item in ordered_group:
                values = item.get("semantic_attributes")
                if isinstance(values, list):
                    semantic_tokens.extend(str(value).strip() for value in values if str(value).strip())
            merged_semantic_attributes = list(dict.fromkeys(semantic_tokens))

            merged_visibility: dict[str, float] = {}
            visibility_weights: dict[str, float] = {}
            for item in ordered_group:
                visibility = item.get("visibility_scores")
                if not isinstance(visibility, dict):
                    continue
                weight = LocalVideoIngestionPipeline._person_merge_weight(item)
                for key, value in visibility.items():
                    if not isinstance(value, (int, float)):
                        continue
                    merged_visibility[key] = merged_visibility.get(key, 0.0) + float(value) * weight
                    visibility_weights[key] = visibility_weights.get(key, 0.0) + weight
            for key, total_weight in visibility_weights.items():
                if total_weight > 1e-8:
                    merged_visibility[key] = round(merged_visibility[key] / total_weight, 6)

            merged_action_payload = deepcopy(base.get("action_semantic_embedding")) if isinstance(base.get("action_semantic_embedding"), dict) else {}
            merged_segments: list[dict[str, object]] = []
            if isinstance(merged_action_payload.get("segments"), list):
                merged_segments.extend(segment for segment in merged_action_payload["segments"] if isinstance(segment, dict))
            for item in ordered_group[1:]:
                action_payload = item.get("action_semantic_embedding")
                if not isinstance(action_payload, dict):
                    continue
                segments = action_payload.get("segments")
                if isinstance(segments, list):
                    merged_segments.extend(segment for segment in segments if isinstance(segment, dict))
            merged_segments.sort(key=lambda segment: str(segment.get("clip_id") or ""))
            if merged_action_payload:
                merged_action_payload["segments"] = merged_segments

            merged_quality = {
                "accepted": True,
                "average_confidence": round(
                    sum(float(_quality_or_zero(item, "average_confidence")) for item in ordered_group) / max(len(ordered_group), 1),
                    6,
                ),
                "average_laplacian": round(
                    sum(float(_quality_or_zero(item, "average_laplacian")) for item in ordered_group) / max(len(ordered_group), 1),
                    6,
                ),
                "frame_count": merged_frame_count_full,
                "frame_density": round(
                    (
                        merged_frame_count_full
                        / float(
                            max(
                                merged_frame_end_idx - merged_frame_start_idx + 1,
                                1,
                            )
                        )
                    ) if merged_frame_count_full > 0 else 0.0,
                    6,
                ),
                "duration_seconds": round(
                    sum(float(_quality_or_zero(item, "duration_seconds")) for item in ordered_group),
                    6,
                ),
                "merged_tracklet_count": len(ordered_group),
            }

            representative_frame = all_frames[0] if all_frames else {}
            base["candidate_id"] = human_key
            base["video_id"] = video_id
            base["track_id"] = unique_track_ids[0] if unique_track_ids else str(base.get("track_id") or "")
            base["track_ids"] = unique_track_ids
            base["human_key"] = human_key
            base["merged_tracklet_count"] = len(ordered_group)
            base["tracklet_frames"] = _compact_serialized_tracklet_frames(all_frames, max_frames=30)
            base["tracklet_frame_count_full"] = merged_frame_count_full
            base["tracklet_frame_start_idx"] = merged_frame_start_idx
            base["tracklet_frame_end_idx"] = merged_frame_end_idx
            base["timeline"] = all_timeline
            base["matched_segments"] = all_timeline
            base["semantic_attributes"] = merged_semantic_attributes
            base["visibility_scores"] = merged_visibility or (base.get("visibility_scores") if isinstance(base.get("visibility_scores"), dict) else {})
            base["tracklet_quality"] = merged_quality
            base["frame_idx"] = int(representative_frame.get("frame_idx") or base.get("frame_idx") or 0)
            if "bbox" in representative_frame and isinstance(representative_frame.get("bbox"), list):
                base["bbox"] = representative_frame["bbox"]
                base["representative_bbox"] = representative_frame["bbox"]
            for field_name in ("attribute_embedding_vector", "appearance_embedding_vector", "embedding_vector"):
                merged_vector = LocalVideoIngestionPipeline._merge_embedding_vectors(ordered_group, field_name=field_name)
                if merged_vector is not None:
                    base[field_name] = merged_vector
            if merged_action_payload:
                base["action_semantic_embedding"] = merged_action_payload
            merged_people.append(base)

        merged_people.sort(
            key=lambda item: (
                str(item.get("camera_id") or ""),
                float(item.get("frame_idx") or 0),
                str(item.get("candidate_id") or ""),
            )
        )
        return merged_people

    @staticmethod
    def _metadata_people_with_full_tracklet_frames(
        *,
        people: list[dict[str, object]],
        tracklets: tuple[LocalTracklet, ...],
    ) -> list[dict[str, object]]:
        tracklets_by_id = {tracklet.track_id: tracklet for tracklet in tracklets}
        metadata_people: list[dict[str, object]] = []
        for person in people:
            metadata_person = deepcopy(person)
            raw_track_ids = metadata_person.get("track_ids")
            track_ids = [
                str(track_id).strip()
                for track_id in raw_track_ids
                if str(track_id).strip()
            ] if isinstance(raw_track_ids, list) else [str(metadata_person.get("track_id") or "").strip()]

            full_observations: list[TrackletObservation] = []
            for track_id in track_ids:
                tracklet = tracklets_by_id.get(track_id)
                if tracklet is not None:
                    full_observations.extend(tracklet.observations)
            full_observations.sort(key=lambda item: (float(item.timestamp_second), int(item.frame_index)))

            metadata_person["tracklet_frames"] = _serialize_tracklet_frames(full_observations)
            metadata_person["tracklet_frame_count_full"] = len(full_observations)
            metadata_person["tracklet_frame_start_idx"] = int(full_observations[0].frame_index) if full_observations else 0
            metadata_person["tracklet_frame_end_idx"] = int(full_observations[-1].frame_index) if full_observations else 0
            metadata_people.append(metadata_person)
        return metadata_people

    def _materialize_people(
        self,
        *,
        video_id: str,
        camera_id: str | None,
        tracklets: tuple[LocalTracklet, ...],
        sampled_fps: int,
        metadata_path: Path,
    ) -> tuple[list[dict[str, object]], int]:
        if not tracklets:
            return [], 0

        quality_results = {
            tracklet.track_id: self.quality_scorer.score(tracklet)
            for tracklet in tracklets
        }
        people = self.metadata_assembler.build_people(
            video_id=video_id,
            camera_id=camera_id,
            tracklets=tracklets,
            quality_results=quality_results,
            sampled_fps=sampled_fps,
        )
        accepted_tracklet_count = len(people)
        if not people:
            return [], 0

        # Within-camera embedding merge is DISABLED.
        # GT analysis shows DINOv2 intra/inter similarity distributions overlap
        # completely for overhead warehouse cameras (best F1=0.146 at any threshold).
        # At t=0.80: 9 correct merges vs 91 false merges (10:1 false positive ratio).
        # Tracker improvements alone reduce 209 → ~46 tracklets (vs GT 25).
        # Accepting 1.84× fragmentation is safer than risking wrong identity merges.
        merged_people = self._merge_people_by_identity(video_id=video_id, people=people)
        metadata_people = self._metadata_people_with_full_tracklet_frames(
            people=merged_people,
            tracklets=tracklets,
        )
        with metadata_path.open("a", encoding="utf-8") as fh:
            for person in metadata_people:
                fh.write(json.dumps(person, ensure_ascii=False))
                fh.write("\n")
        return merged_people, accepted_tracklet_count

    def run(
        self,
        *,
        source_path: Path | str,
        compressed_path: Path,
        metadata_path: Path,
        camera_id: str | None,
        recorded_start: datetime | None,
        metadata: dict[str, object],
    ) -> LocalIngestionOutput:
        LOGGER.info(
            "Local ingestion started source=%s sample_fps=%s",
            source_path, self.sample_fps,
        )

        video_id = compressed_path.name

        # Probe video duration so the tracker buffer never expires mid-video.
        # All buffer tracks survive until finalize_all() at the end of the stream,
        # eliminating hard-break fragmentation from buffer timeout.
        try:
            _cap = cv2.VideoCapture(str(source_path))
            _src_fps = float(_cap.get(cv2.CAP_PROP_FPS) or self.sample_fps)
            _total_src = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            _cap.release()
            _duration_s = _total_src / max(_src_fps, 1.0) if _total_src > 0 else 0.0
            _total_sampled = int(math.ceil(_duration_s * self.sample_fps)) + 1 if _duration_s > 0 else self.tracker.max_buffer_frames
        except Exception:
            _total_sampled = self.tracker.max_buffer_frames
        self.tracker.max_buffer_frames = max(self.tracker.max_buffer_frames, _total_sampled)

        self.tracker.reset()
        people: list[dict[str, object]] = []
        finalized_tracklets_buffer: list[LocalTracklet] = []
        sampled_frame_count = 0
        tracklet_count = 0
        detector_batch_size = max(1, int(os.environ.get("MCPT_DETECTOR_BATCH_SIZE", "8")))
        batch_size = max(
            detector_batch_size * 3,
            int(os.environ.get("MCPT_STREAM_BATCH_SIZE", str(max(detector_batch_size * 3, 150)))),
        )

        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text("", encoding="utf-8")
        LOGGER.info("Local ingestion streaming started source=%s sample_fps=%s batch=%s", source_path, self.sample_fps, batch_size)

        for batch_idx, sampled_batch in enumerate(
            self.sampler.stream_batched(source_path, batch_size=batch_size),
            start=1,
        ):
            sampled_frame_count += len(sampled_batch)
            detections_by_frame = self.detector.detect(sampled_batch)
            finalized_tracklets = self.tracker.track_incremental(
                video_id=video_id,
                camera_id=camera_id,
                detections_by_frame=detections_by_frame,
            )
            if finalized_tracklets:
                finalized_tracklets_buffer.extend(finalized_tracklets)
                tracklet_count += len(finalized_tracklets)

            # Release raw frame memory aggressively after each batch.
            del sampled_batch
            del detections_by_frame
            del finalized_tracklets
            gc.collect()
            LOGGER.info(
                "Local ingestion batch completed batch=%s sampled_frames=%s people=%s",
                batch_idx,
                sampled_frame_count,
                len(finalized_tracklets_buffer),
            )

        tail_tracklets = self.tracker.finalize_all(video_id=video_id, camera_id=camera_id)
        if tail_tracklets:
            finalized_tracklets_buffer.extend(tail_tracklets)
            tracklet_count += len(tail_tracklets)

        LOGGER.info(
            "Local ingestion tracking phase completed sampled_frames=%s finalized_tracklets=%s",
            sampled_frame_count,
            len(finalized_tracklets_buffer),
        )
        people, accepted_tracklet_count = self._materialize_people(
            video_id=video_id,
            camera_id=camera_id,
            tracklets=tuple(finalized_tracklets_buffer),
            sampled_fps=self.sample_fps,
            metadata_path=metadata_path,
        )
        LOGGER.info(
            "Local ingestion feature phase completed tracklets=%s accepted_people=%s",
            tracklet_count,
            len(people),
        )

        processed_at = datetime.now(timezone.utc).replace(microsecond=0)
        video_payload: dict[str, object] = {
            "video_id": video_id,
            "camera_id": camera_id,
            "source_path": str(source_path),
            "compressed_path": str(compressed_path),
            "metadata_path": str(metadata_path),
            "recorded_start": recorded_start.isoformat() if recorded_start else None,
            "sample_fps": self.sample_fps,
            "sampled_frame_count": sampled_frame_count,
            "tracklet_count": tracklet_count,
            "accepted_tracklet_count": accepted_tracklet_count,
            "rejected_tracklet_count": max(tracklet_count - accepted_tracklet_count, 0),
            "processing_backend": "strict_tracking_service",
            "ingestion_metadata": metadata,
            "processed_at": processed_at.isoformat().replace("+00:00", "Z"),
        }
        sidecar_path = metadata_path.with_suffix(".summary.json")
        sidecar_path.write_text(
            json.dumps({"video": video_payload, "people": people}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("Local ingestion metadata written stream=%s summary=%s", metadata_path, sidecar_path)
        return LocalIngestionOutput(
            video=video_payload,
            people=people,
            compressed_path=str(compressed_path),
            metadata_path=str(metadata_path),
            processed_at=processed_at,
        )
