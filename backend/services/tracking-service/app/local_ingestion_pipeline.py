from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
import gc
import json
import logging
import math
import os
from pathlib import Path
import threading
from typing import Iterator

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
_APPEARANCE_DESCRIPTOR_DIM = 40


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
class OCMCTrackStyleTracker:
    """
    OCMCTrack-style corrective cascade tracker — strict production tracker.

    Two-stage matching cascade:
    Stage 1 – high-confidence detections (conf ≥ high_confidence_threshold) matched
              to active tracks via IoU gate + appearance gate.
    Stage 2 – low-confidence detections (conf ≥ low_confidence_threshold) matched
              to remaining unmatched tracks with relaxed criteria.
    Corrective buffer – recently-lost tracks (< corrective_buffer_seconds) kept alive
                        for re-association to resolve temporal occlusions.
    New tracks – created only when detection confidence ≥ new_track_threshold.

    Hyperparameters match strict_pipeline.py exactly.
    """

    high_confidence_threshold: float = 0.45
    low_confidence_threshold: float = 0.12
    new_track_threshold: float = 0.55
    iou_gate: float = 0.18
    appearance_gate: float = 0.22
    motion_proximity_gate: float = 0.30
    center_distance_gate: float = 1.85
    min_scale_similarity: float = 0.45
    corrective_buffer_seconds: float = 45.0
    max_frame_gap: int = 8
    inactive_finalize_seconds: float = 15.0

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.active: dict[str, list[TrackletObservation]] = {}
        self.active_last_bbox: dict[str, BoundingBox] = {}
        self.active_last_ts: dict[str, float] = {}
        self.active_appearance: dict[str, np.ndarray] = {}
        self.buffer: dict[str, list[TrackletObservation]] = {}
        self.buffer_last_bbox: dict[str, BoundingBox] = {}
        self.buffer_last_ts: dict[str, float] = {}
        self.buffer_appearance: dict[str, np.ndarray] = {}
        self.next_id = 1

    def track_incremental(
        self,
        *,
        video_id: str,
        camera_id: str | None,
        detections_by_frame: dict[int, tuple[FrameDetection, ...]],
    ) -> tuple[LocalTracklet, ...]:
        completed: list[LocalTracklet] = []
        for frame_idx in sorted(detections_by_frame):
            dets = list(detections_by_frame.get(frame_idx) or ())
            if not dets:
                continue
            frame_ts = dets[0].timestamp_second
            completed.extend(self._expire_stale(video_id=video_id, camera_id=camera_id, frame_idx=frame_idx, frame_ts=frame_ts))

            high_dets = [d for d in dets if d.confidence >= self.high_confidence_threshold]
            low_dets = [d for d in dets if self.low_confidence_threshold <= d.confidence < self.high_confidence_threshold]
            unmatched_high: list[FrameDetection] = []
            active_unmatched = set(self.active.keys())

            for det in high_dets:
                best_tid, best_score = self._best_match(det, self.active_last_bbox, self.active_appearance, active_unmatched)
                if best_tid is not None and best_score >= self.iou_gate:
                    self._update_track(self.active, self.active_last_bbox, self.active_last_ts, self.active_appearance, best_tid, det, frame_ts)
                    active_unmatched.discard(best_tid)
                else:
                    unmatched_high.append(det)

            for det in low_dets:
                best_tid, best_score = self._best_match(det, self.active_last_bbox, self.active_appearance, active_unmatched)
                if best_tid is not None and best_score >= self.iou_gate:
                    self._update_track(self.active, self.active_last_bbox, self.active_last_ts, self.active_appearance, best_tid, det, frame_ts)
                    active_unmatched.discard(best_tid)

            buffer_unmatched = set(self.buffer.keys())
            for det in list(unmatched_high):
                best_tid, best_score = self._best_match(det, self.buffer_last_bbox, self.buffer_appearance, buffer_unmatched)
                if best_tid is not None and best_score >= self.iou_gate:
                    obs_list = self.buffer.pop(best_tid)
                    self.buffer_last_bbox.pop(best_tid, None)
                    self.buffer_last_ts.pop(best_tid, None)
                    self.buffer_appearance.pop(best_tid, None)
                    obs = self._make_observation(det, frame_ts)
                    self.active[best_tid] = obs_list + [obs]
                    self.active_last_bbox[best_tid] = det.bbox
                    self.active_last_ts[best_tid] = frame_ts
                    self.active_appearance[best_tid] = self._appearance_descriptor(det)
                    buffer_unmatched.discard(best_tid)
                    unmatched_high.remove(det)

            for det in unmatched_high:
                if det.confidence >= self.new_track_threshold:
                    tid = str(self.next_id)
                    self.next_id += 1
                    self.active[tid] = [self._make_observation(det, frame_ts)]
                    self.active_last_bbox[tid] = det.bbox
                    self.active_last_ts[tid] = frame_ts
                    self.active_appearance[tid] = self._appearance_descriptor(det)
        return tuple(t for t in completed if t.observations)

    def _expire_stale(
        self,
        *,
        video_id: str,
        camera_id: str | None,
        frame_idx: int,
        frame_ts: float,
    ) -> list[LocalTracklet]:
        completed: list[LocalTracklet] = []
        expired = [tid for tid, ts in self.buffer_last_ts.items() if frame_ts - ts > self.corrective_buffer_seconds]
        for tid in expired:
            completed.append(LocalTracklet(video_id=video_id, camera_id=camera_id, track_id=tid, observations=tuple(self.buffer.pop(tid, []))))
            self.buffer_last_bbox.pop(tid, None)
            self.buffer_last_ts.pop(tid, None)
            self.buffer_appearance.pop(tid, None)

        stale = [tid for tid in self.active.keys() if frame_idx - self._last_frame_idx(self.active[tid]) > self.max_frame_gap]
        for tid in stale:
            self.buffer[tid] = self.active.pop(tid)
            self.buffer_last_bbox[tid] = self.active_last_bbox.pop(tid)
            self.buffer_last_ts[tid] = self.active_last_ts.pop(tid)
            self.buffer_appearance[tid] = self.active_appearance.pop(
                tid,
                np.zeros(_APPEARANCE_DESCRIPTOR_DIM, dtype=np.float32),
            )

        finalized = [tid for tid, ts in self.buffer_last_ts.items() if frame_ts - ts >= self.inactive_finalize_seconds]
        for tid in finalized:
            completed.append(LocalTracklet(video_id=video_id, camera_id=camera_id, track_id=tid, observations=tuple(self.buffer.pop(tid, []))))
            self.buffer_last_bbox.pop(tid, None)
            self.buffer_last_ts.pop(tid, None)
            self.buffer_appearance.pop(tid, None)
        return completed

    def finalize_all(self, *, video_id: str, camera_id: str | None) -> tuple[LocalTracklet, ...]:
        completed: list[LocalTracklet] = []
        for tracks in (self.active, self.buffer):
            for tid, obs_list in tracks.items():
                if obs_list:
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

    def _best_match(
        self,
        det: FrameDetection,
        last_bbox: dict[str, BoundingBox],
        appearance: dict[str, np.ndarray],
        candidates: set[str],
    ) -> tuple[str | None, float]:
        best_tid = None
        best_score = -1.0
        det_app = self._appearance_descriptor(det)
        for tid in candidates:
            iou = _bbox_iou(last_bbox[tid], det.bbox)
            motion_score = self._motion_proximity(last_bbox[tid], det.bbox)
            geometry_score = max(iou, motion_score)
            if iou < self.iou_gate and motion_score < self.motion_proximity_gate:
                continue
            app_sim = self._cosine_sim(
                appearance.get(tid, np.zeros(_APPEARANCE_DESCRIPTOR_DIM, dtype=np.float32)),
                det_app,
            )
            # Enforce appearance consistency unless geometry is very strong.
            if app_sim < self.appearance_gate and geometry_score < max(self.motion_proximity_gate + 0.22, 0.58):
                continue
            score = (
                0.45 * geometry_score +
                0.35 * iou +
                0.20 * max(app_sim, 0.0)
            )
            if score > best_score:
                best_score = score
                best_tid = tid
        return best_tid, best_score

    @staticmethod
    def _update_track(
        active: dict,
        bbox_map: dict,
        ts_map: dict,
        app_map: dict,
        tid: str,
        det: FrameDetection,
        frame_ts: float,
    ) -> None:
        active[tid].append(OCMCTrackStyleTracker._make_observation(det, frame_ts))
        bbox_map[tid] = det.bbox
        ts_map[tid] = frame_ts
        app_map[tid] = OCMCTrackStyleTracker._appearance_descriptor(det)

    @staticmethod
    def _make_observation(det: FrameDetection, frame_ts: float) -> TrackletObservation:
        return TrackletObservation(
            frame_index=det.frame_index,
            timestamp_second=frame_ts,
            bbox=det.bbox,
            confidence=det.confidence,
            laplacian_score=det.laplacian_score,
            crop_bgr=det.crop_bgr,
        )

    @staticmethod
    def _appearance_descriptor(det: FrameDetection) -> np.ndarray:
        """Compact HSV descriptor for fast appearance gating in the tracker."""
        crop = det.crop_bgr
        if crop is None or crop.size == 0:
            return np.zeros(_APPEARANCE_DESCRIPTOR_DIM, dtype=np.float32)
        hsv = cv2.cvtColor(np.asarray(crop, dtype=np.uint8), cv2.COLOR_BGR2HSV)
        hue_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten().astype(np.float32)
        sat_hist = cv2.calcHist([hsv], [1], None, [12], [0, 256]).flatten().astype(np.float32)
        val_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten().astype(np.float32)
        descriptor = np.concatenate([hue_hist, sat_hist, val_hist], axis=0)
        norm = float(descriptor.sum()) or 1.0
        return descriptor / norm

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a)) or 1.0
        nb = float(np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / (na * nb))

    def _motion_proximity(self, lhs: BoundingBox, rhs: BoundingBox) -> float:
        if lhs.area <= 0 or rhs.area <= 0:
            return 0.0
        lhs_center_x = (lhs.x1 + lhs.x2) * 0.5
        lhs_center_y = (lhs.y1 + lhs.y2) * 0.5
        rhs_center_x = (rhs.x1 + rhs.x2) * 0.5
        rhs_center_y = (rhs.y1 + rhs.y2) * 0.5
        center_distance = math.hypot(lhs_center_x - rhs_center_x, lhs_center_y - rhs_center_y)
        mean_diag = math.hypot((lhs.width + rhs.width) * 0.5, (lhs.height + rhs.height) * 0.5)
        if mean_diag <= 1e-6:
            return 0.0
        distance_ratio = center_distance / mean_diag
        distance_score = max(0.0, 1.0 - distance_ratio / max(self.center_distance_gate, 1e-6))
        scale_similarity = min(lhs.area, rhs.area) / max(lhs.area, rhs.area)
        if scale_similarity < self.min_scale_similarity:
            return 0.0
        return 0.7 * distance_score + 0.3 * scale_similarity

    @staticmethod
    def _last_frame_idx(obs_list: list[TrackletObservation]) -> int:
        return obs_list[-1].frame_index if obs_list else 0


@dataclass
class TrackletQualityScorer:
    """Filter blurry or weak tracklets before feature extraction."""

    minimum_confidence_score: float = 0.3
    minimum_frame_count: int = 3
    minimum_average_laplacian: float = 12.0
    minimum_duration_seconds: float = 2.0

    def score(self, tracklet: LocalTracklet) -> TrackletQualityResult:
        confidences = [item.confidence for item in tracklet.observations]
        laplacians = [item.laplacian_score for item in tracklet.observations]
        average_confidence = round(sum(confidences) / max(len(confidences), 1), 6)
        average_laplacian = round(sum(laplacians) / max(len(laplacians), 1), 6)
        frame_count = len(tracklet.observations)
        duration_seconds = 0.0
        if frame_count >= 2:
            duration_seconds = max(tracklet.observations[-1].timestamp_second - tracklet.observations[0].timestamp_second, 0.0)

        if frame_count < self.minimum_frame_count:
            return TrackletQualityResult(False, average_confidence, average_laplacian, frame_count, duration_seconds, "insufficient_frames")
        if duration_seconds < self.minimum_duration_seconds:
            return TrackletQualityResult(False, average_confidence, average_laplacian, frame_count, duration_seconds, "short_tracklet")
        if average_confidence < self.minimum_confidence_score:
            return TrackletQualityResult(False, average_confidence, average_laplacian, frame_count, duration_seconds, "low_confidence")
        if average_laplacian < self.minimum_average_laplacian:
            return TrackletQualityResult(False, average_confidence, average_laplacian, frame_count, duration_seconds, "blurry_tracklet")
        return TrackletQualityResult(True, average_confidence, average_laplacian, frame_count, duration_seconds, None)


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
        GPU-efficient batch feature extraction.

        Runs each model once for ALL tracklets combined instead of N separate calls:
          SigLIP2 image  — 1 forward pass (all selected crops concatenated)
          TransReID      — 1 forward pass (all part-crops concatenated)
          VideoMAE       — 1 forward pass (all action clips stacked)
        Text features are cached inside SigLIP2ModelHub after the first call.
        """
        import cv2 as _cv2
        from PIL import Image as _PILImage
        from .model_adapters import (
            SigLIP2ModelHub, TransReIDHub, VideoMAEHub,
            _crop_pil, _selected_observations, _part_crops, _quality_weighted_pool,
            _GENDER_PROMPTS, _AGE_PROMPTS,
            _SHIRT_PROMPTS, _PANTS_PROMPTS, _HAIR_PROMPTS, _SKIN_PROMPTS,
            _HAT_PROMPTS, _BAG_PROMPTS, _HEAD_ACCESSORY_PROMPTS, _SHOES_PROMPTS,
        )
        from .tracklet_feature_pipeline import (
            ActionClipBuilder, TrackletFeatureAggregator, EMBEDDING_VOCABULARY,
            StaticAttributeResult, AppearanceAttributeResult,
            AppearanceEmbeddingResult, AttributeEmbeddingResult,
            BehaviorAnalysisResult, SemanticEmbeddingResult,
        )

        siglip = SigLIP2ModelHub()
        reid   = TransReIDHub()
        vmae   = VideoMAEHub()

        _PROMPT_MAP = {
            "shirt": _SHIRT_PROMPTS, "pants": _PANTS_PROMPTS,
            "hair_color": _HAIR_PROMPTS, "skin_tone": _SKIN_PROMPTS,
            "hat": _HAT_PROMPTS, "bag": _BAG_PROMPTS,
            "head_accessory": _HEAD_ACCESSORY_PROMPTS, "shoes": _SHOES_PROMPTS,
        }

        # ── Build TrackletFeatureInput payloads ──────────────────────────────
        payloads: list[TrackletFeatureInput] = []
        for tracklet, _ in accepted_tracklets:
            payloads.append(TrackletFeatureInput(
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
            ))

        # ── Frame selection (CPU) ────────────────────────────────────────────
        selections = [self.feature_pipeline.selector.select(p) for p in payloads]

        # ── Batch SigLIP2 image features: 1 GPU call for all tracklets ───────
        per_t_valid: list[list] = []
        all_siglip_pils: list = []
        siglip_slices: list[tuple[int, int]] = []

        for payload, selection in zip(payloads, selections):
            obs = _selected_observations(payload, selection)
            valid = [(f, _crop_pil(f, None)) for f in obs]
            valid = [(f, p) for f, p in valid if p is not None]
            per_t_valid.append(valid)
            s = len(all_siglip_pils)
            all_siglip_pils.extend(p for _, p in valid)
            siglip_slices.append((s, len(all_siglip_pils)))

        all_img_feats = (
            siglip.image_features(all_siglip_pils)
            if all_siglip_pils
            else np.zeros((0, 1024), dtype=np.float32)
        )

        # Pre-warm text cache (one shot — subsequent calls are dict lookups)
        for _, prompts in _GENDER_PROMPTS + _AGE_PROMPTS:
            siglip.text_features(prompts)
        for prompt_list in _PROMPT_MAP.values():
            for _, prompts in prompt_list:
                siglip.text_features(prompts)

        # ── Classify attributes per tracklet (pure CPU matmul) ───────────────
        all_static:     list[StaticAttributeResult]     = []
        all_appearance: list[AppearanceAttributeResult] = []

        for idx, selection in enumerate(selections):
            s, e    = siglip_slices[idx]
            img_feats = all_img_feats[s:e]
            valid   = per_t_valid[idx]
            qmap    = {item.frame_index: item.quality_score for item in selection.selected_frames}

            if len(img_feats) == 0:
                all_static.append(StaticAttributeResult(gender=None, age_group=None, confidence=0.0))
                all_appearance.append(AppearanceAttributeResult())
                continue

            gender_votes: dict = {}
            age_votes:    dict = {}
            for i, (frame, _) in enumerate(valid):
                w    = qmap.get(frame.frame_index, 0.1)
                feat = img_feats[i]
                best_g, bg_s = None, -float("inf")
                for label, prompts in _GENDER_PROMPTS:
                    sc = float(np.mean(siglip.text_features(prompts) @ feat))
                    if sc > bg_s:
                        bg_s, best_g = sc, label
                gender_votes[best_g] = gender_votes.get(best_g, 0.0) + w
                best_a, ba_s = None, -float("inf")
                for label, prompts in _AGE_PROMPTS:
                    sc = float(np.mean(siglip.text_features(prompts) @ feat))
                    if sc > ba_s:
                        ba_s, best_a = sc, label
                age_votes[best_a] = age_votes.get(best_a, 0.0) + w

            gender    = max(gender_votes, key=gender_votes.get) if gender_votes else None
            age_group = max(age_votes,    key=age_votes.get)    if age_votes    else None
            all_static.append(StaticAttributeResult(
                gender=gender, age_group=age_group,
                confidence=0.45 if (gender or age_group) else 0.0,
            ))

            fields: dict[str, dict] = {k: {} for k in _PROMPT_MAP}
            for i, (frame, _) in enumerate(valid):
                w    = max(qmap.get(frame.frame_index, 0.1), 1e-4)
                feat = img_feats[i]
                for field_name, label_prompts in _PROMPT_MAP.items():
                    best_label, bs = None, -float("inf")
                    for label, prompts in label_prompts:
                        sc = float(np.mean(siglip.text_features(prompts) @ feat))
                        if sc > bs:
                            bs, best_label = sc, label
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

        # ── Batch TransReID: 1 GPU call for all crops ────────────────────────
        per_t_pil_crops: list[list] = []
        per_t_weights:   list[list[float]] = []
        all_fulls: list = []
        all_uppers: list = []
        all_lowers: list = []
        reid_slices: list[tuple[int, int]] = []

        for payload, selection in zip(payloads, selections):
            obs  = _selected_observations(payload, selection)
            qmap = {item.frame_index: item.quality_score for item in selection.selected_frames}
            pil_crops: list = []
            weights:   list[float] = []
            for frame in obs:
                pil = _crop_pil(frame, None)
                if pil is None:
                    continue
                pil_crops.append(pil)
                weights.append(qmap.get(frame.frame_index, 0.1))
            per_t_pil_crops.append(pil_crops)
            per_t_weights.append(weights)
            s = len(all_fulls)
            for pil in pil_crops:
                parts = _part_crops(pil)
                all_fulls.append(parts[0])
                all_uppers.append(parts[1])
                all_lowers.append(parts[2])
            reid_slices.append((s, len(all_fulls)))

        _GW, _PW = 0.55, 0.45
        if all_fulls:
            all_reid = reid.embed_crops(all_fulls + all_uppers + all_lowers)  # [3M, 768]
            M        = len(all_fulls)
            gf_all   = all_reid[:M]
            uf_all   = all_reid[M:2 * M]
            lf_all   = all_reid[2 * M:]
        else:
            gf_all = uf_all = lf_all = np.zeros((0, 768), dtype=np.float32)

        all_app_embed: list[AppearanceEmbeddingResult] = []
        for idx, (pil_crops, weights) in enumerate(zip(per_t_pil_crops, per_t_weights)):
            s, e = reid_slices[idx]
            if not pil_crops:
                empty = tuple(0.0 for _ in range(512))
                all_app_embed.append(AppearanceEmbeddingResult(
                    embedding_model="transreid-vit-base-msmt17-kpr",
                    embedding_vector=empty, tracklet_vectors=(empty,),
                ))
                continue
            gf, uf, lf = gf_all[s:e], uf_all[s:e], lf_all[s:e]
            per_frame_vecs = []
            for i in range(len(pil_crops)):
                pm   = (uf[i] + lf[i]) / 2.0
                pm  /= float(np.linalg.norm(pm)) or 1.0
                fv   = _GW * gf[i] + _PW * pm
                fv  /= float(np.linalg.norm(fv)) or 1.0
                per_frame_vecs.append(fv.astype(np.float32))
            fused   = _quality_weighted_pool(per_frame_vecs, weights)
            fused_t = tuple(round(float(v), 6) for v in fused.tolist())
            pf_t    = [tuple(round(float(v), 6) for v in gf[i].tolist()) for i in range(len(pil_crops))]
            all_app_embed.append(AppearanceEmbeddingResult(
                embedding_model="transreid-vit-base-msmt17-kpr",
                embedding_vector=fused_t, tracklet_vectors=tuple(pf_t),
            ))

        # ── Batch VideoMAE: 1 GPU call for all clips ─────────────────────────
        clip_builder = ActionClipBuilder()
        all_clips_per_t: list[tuple] = [
            clip_builder.build(p, sel.representative_frame.frame_index)
            for p, sel in zip(payloads, selections)
        ]
        all_clips_flat = [clip for clips in all_clips_per_t for clip in clips]

        def _clip_pils(clip) -> list:
            out = []
            for frame in clip.frames:
                crop = frame.crop_bgr
                if crop is not None and crop.size > 0:
                    rgb = _cv2.cvtColor(np.asarray(crop, dtype=np.uint8), _cv2.COLOR_BGR2RGB)
                    out.append(_PILImage.fromarray(rgb))
            return out

        clip_pils_list = [_clip_pils(clip) for clip in all_clips_flat]
        vmae_feats = (
            vmae.extract_features_batch(clip_pils_list)
            if clip_pils_list
            else np.zeros((0, 1024), dtype=np.float32)
        )

        vocab_labels  = list(EMBEDDING_VOCABULARY)
        vocab_prompts = [
            f"a surveillance footage of a person who is {lbl.replace('_', ' ')} in an indoor space"
            for lbl in vocab_labels
        ]
        text_feats_vocab = siglip.text_features(vocab_prompts)  # [K, 1024], cached

        clip_offset = 0
        all_behavior_per_t: list[tuple] = []
        all_semantic_per_t: list[tuple] = []

        for clips in all_clips_per_t:
            beh_results: list[BehaviorAnalysisResult] = []
            sem_results: list[SemanticEmbeddingResult] = []
            for clip in clips:
                video_feat = vmae_feats[clip_offset] if clip_offset < len(vmae_feats) else np.zeros(1024, dtype=np.float32)
                clip_offset += 1
                scores    = text_feats_vocab @ video_feat
                best_idx  = int(np.argmax(scores))
                best_lbl  = vocab_labels[best_idx]
                conf      = round(float(np.clip(scores[best_idx], 0.0, 1.0)), 6)
                beh_results.append(BehaviorAnalysisResult(
                    clip_id=clip.clip_id, action_summary=best_lbl,
                    labels=(best_lbl,), confidence=conf,
                    metadata={"scores": {lbl: round(float(s), 4) for lbl, s in zip(vocab_labels, scores)}},
                ))
                sem_text = f"a surveillance footage of a person who is {best_lbl.replace('_', ' ')} in an indoor space"
                sem_feat = siglip.text_features([sem_text])[0]
                sem_results.append(SemanticEmbeddingResult(
                    clip_id=clip.clip_id,
                    embedding_model="siglip2-vit-l16-512-action",
                    vocabulary=tuple(EMBEDDING_VOCABULARY),
                    embedding_vector=tuple(round(float(v), 6) for v in sem_feat.tolist()),
                ))
            all_behavior_per_t.append(tuple(beh_results))
            all_semantic_per_t.append(tuple(sem_results))

        # ── Attribute text embedding (SigLIP2 text, all cached) ──────────────
        all_attr_embed: list[AttributeEmbeddingResult] = []
        for static_attr in all_static:
            parts = [p for p in [static_attr.gender, static_attr.age_group] if p]
            text  = "a photo of a " + (" ".join(parts) if parts else "person")
            feat  = siglip.text_features([text])[0]
            all_attr_embed.append(AttributeEmbeddingResult(
                embedding_model="siglip2-vit-l16-512-attr",
                embedding_vector=tuple(round(float(v), 6) for v in feat.tolist()),
            ))

        # ── Aggregate and build final metadata dicts ─────────────────────────
        aggregator  = TrackletFeatureAggregator()
        runtime_meta = {"batch_optimized": True, "batch_size": len(accepted_tracklets)}
        people: list[dict[str, object]] = []

        for idx, ((tracklet, quality), payload, selection) in enumerate(
            zip(accepted_tracklets, payloads, selections)
        ):
            try:
                agg = aggregator.aggregate(
                    tracklet=payload,
                    selection=selection,
                    static_attributes=all_static[idx],
                    attribute_embedding=all_attr_embed[idx],
                    appearance_attributes=all_appearance[idx],
                    appearance_embedding=all_app_embed[idx],
                    clips=all_clips_per_t[idx],
                    behavior_results=all_behavior_per_t[idx],
                    semantic_embeddings=all_semantic_per_t[idx],
                    runtime_metadata=runtime_meta,
                )
                aggregated_metadata = agg.to_metadata()
            except Exception:
                LOGGER.warning("Batch aggregate failed for track_id=%s, falling back.", tracklet.track_id)
                try:
                    aggregated_metadata = self.feature_pipeline.process(payload).aggregated.to_metadata()
                except Exception:
                    continue

            candidate_id = f"{video_id}:{tracklet.track_id}"
            aggregated_metadata.update({
                "candidate_id": candidate_id,
                "camera_id": camera_id,
                "video_id": video_id,
                "track_id": tracklet.track_id,
                "human_key": f"{camera_id or video_id}:{tracklet.track_id}",
                "tracklet_frames": [
                    {
                        "frame_idx": item.frame_index,
                        "timestamp_second": item.timestamp_second,
                        "bbox": item.bbox.to_xyxy(),
                        "confidence": item.confidence,
                    }
                    for item in tracklet.observations
                ],
                "tracklet_quality": {
                    "accepted": quality.accepted,
                    "average_confidence": quality.average_confidence,
                    "average_laplacian": quality.average_laplacian,
                    "frame_count": quality.frame_count,
                    "duration_seconds": round(quality.duration_seconds, 6),
                },
            })
            people.append(aggregated_metadata)

        return people

    def _build_person_metadata_from_item(
        self,
        item: tuple[LocalTracklet, TrackletQualityResult],
        *,
        video_id: str,
        camera_id: str | None,
        sampled_fps: int,
    ) -> dict[str, object]:
        tracklet, quality = item
        return self._build_person_metadata(
            video_id=video_id,
            camera_id=camera_id,
            tracklet=tracklet,
            quality=quality,
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
            SigLIP2ModelHub, TransReIDHub, VideoMAEHub,
            _crop_pil, _selected_observations, _part_crops, _quality_weighted_pool,
            _GENDER_PROMPTS, _AGE_PROMPTS,
            _SHIRT_PROMPTS, _PANTS_PROMPTS, _HAIR_PROMPTS, _SKIN_PROMPTS,
            _HAT_PROMPTS, _BAG_PROMPTS, _HEAD_ACCESSORY_PROMPTS, _SHOES_PROMPTS,
            StaticAttributeResult, AppearanceAttributeResult,
            AppearanceEmbeddingResult, AttributeEmbeddingResult,
            BehaviorAnalysisResult, SemanticEmbeddingResult,
        )
        from .tracklet_feature_pipeline import (
            ActionClipBuilder, TrackletFeatureAggregator,
            EMBEDDING_VOCABULARY,
        )

        siglip = SigLIP2ModelHub()
        reid   = TransReIDHub()
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

        # ── Phase 3: Batch TransReID ────────────────────────────────────────────
        per_t_pil_crops: list[list]       = []
        per_t_weights:   list[list[float]]= []
        all_fulls:  list = []
        all_uppers: list = []
        all_lowers: list = []
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
            if pil_crops:
                parts = [_part_crops(pil) for pil in pil_crops]
                all_fulls.extend(p[0] for p in parts)
                all_uppers.extend(p[1] for p in parts)
                all_lowers.extend(p[2] for p in parts)
            reid_slices.append((s, len(all_fulls)))

        # ONE TransReID call with all crops
        _G_W, _P_W = 0.55, 0.45
        if all_fulls:
            total_M   = len(all_fulls)
            reid_raw  = reid.embed_crops(all_fulls + all_uppers + all_lowers)  # [3M, 768]
            gf_all    = reid_raw[:total_M]
            uf_all    = reid_raw[total_M:2 * total_M]
            lf_all    = reid_raw[2 * total_M:]
        else:
            gf_all = uf_all = lf_all = np.zeros((0, 768), dtype=np.float32)

        all_appearance_embed: list[AppearanceEmbeddingResult] = []
        for idx, (pil_crops, weights) in enumerate(zip(per_t_pil_crops, per_t_weights)):
            s, e = reid_slices[idx]
            if not pil_crops:
                empty = tuple(0.0 for _ in range(512))
                all_appearance_embed.append(AppearanceEmbeddingResult(
                    embedding_model="transreid-vit-base-msmt17-kpr",
                    embedding_vector=empty, tracklet_vectors=(empty,),
                ))
                continue
            gf, uf, lf = gf_all[s:e], uf_all[s:e], lf_all[s:e]
            per_frame_vecs = []
            for i in range(len(pil_crops)):
                pm   = (uf[i] + lf[i]) / 2.0
                pn   = float(np.linalg.norm(pm)) or 1.0
                fused = _G_W * gf[i] + _P_W * (pm / pn)
                fn   = float(np.linalg.norm(fused)) or 1.0
                per_frame_vecs.append((fused / fn).astype(np.float32))
            pooled = _quality_weighted_pool(per_frame_vecs, weights)
            all_appearance_embed.append(AppearanceEmbeddingResult(
                embedding_model="transreid-vit-base-msmt17-kpr",
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
            meta.update({
                "candidate_id": candidate_id,
                "camera_id": camera_id,
                "video_id": video_id,
                "track_id": tracklet.track_id,
                "human_key": f"{camera_id or video_id}:{tracklet.track_id}",
                "tracklet_frames": [
                    {
                        "frame_idx": item.frame_index,
                        "timestamp_second": item.timestamp_second,
                        "bbox": item.bbox.to_xyxy(),
                        "confidence": item.confidence,
                    }
                    for item in tracklet.observations
                ],
                "tracklet_quality": {
                    "accepted": quality.accepted,
                    "average_confidence": quality.average_confidence,
                    "average_laplacian": quality.average_laplacian,
                    "frame_count": quality.frame_count,
                    "duration_seconds": round(quality.duration_seconds, 6),
                },
            })
            people.append(meta)

        return people

    def _build_person_metadata(
        self,
        *,
        video_id: str,
        camera_id: str | None,
        tracklet: LocalTracklet,
        quality: TrackletQualityResult,
        sampled_fps: int,
    ) -> dict[str, object]:
        payload = TrackletFeatureInput(
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
        feature_output = self.feature_pipeline.process(payload)
        aggregated_metadata = feature_output.aggregated.to_metadata()
        candidate_id = f"{video_id}:{tracklet.track_id}"
        aggregated_metadata.update(
            {
                "candidate_id": candidate_id,
                "camera_id": camera_id,
                "video_id": video_id,
                "track_id": tracklet.track_id,
                "human_key": f"{camera_id or video_id}:{tracklet.track_id}",
                "tracklet_frames": [
                    {
                        "frame_idx": item.frame_index,
                        "timestamp_second": item.timestamp_second,
                        "bbox": item.bbox.to_xyxy(),
                        "confidence": item.confidence,
                    }
                    for item in tracklet.observations
                ],
                "tracklet_quality": {
                    "accepted": quality.accepted,
                    "average_confidence": quality.average_confidence,
                    "average_laplacian": quality.average_laplacian,
                    "frame_count": quality.frame_count,
                    "duration_seconds": round(quality.duration_seconds, 6),
                },
            }
        )
        return aggregated_metadata


@dataclass
class TrackletMemoryBank:
    """
    Keep short-term identity history and resolve stable human keys across tracklets.

    One centroid per identity (no chaining): each identity is represented by a
    running-average unit vector so that the 1st and 166th tracklet of the same
    person are compared against the same stable reference, not a drifted chain.
    """

    similarity_threshold: float = 0.75  # cosine; 0.60 caused chain-merging of different people
    history_seconds: float = 600.0
    next_global_id: int = 1
    # id → {"ts": float, "uv": np.ndarray | None, "count": int}
    _centroids: dict[str, dict[str, object]] = field(default_factory=dict)

    # Keep dataclass field name stable for any external pickle/copy consumers.
    @property
    def memory(self) -> dict[str, dict[str, object]]:
        return self._centroids

    def resolve_identity(
        self,
        *,
        embedding_vector: list[float] | None,
        timestamp_second: float,
    ) -> str:
        # Evict stale identities.
        stale = [k for k, v in self._centroids.items()
                 if timestamp_second - float(v["ts"]) > self.history_seconds]
        for k in stale:
            del self._centroids[k]

        if not embedding_vector:
            gid = f"global-{self.next_global_id}"
            self.next_global_id += 1
            self._centroids[gid] = {"ts": timestamp_second, "uv": None, "count": 1}
            return gid

        q = np.asarray(embedding_vector, dtype=np.float32)
        n = float(np.linalg.norm(q))
        unit = q / n if n > 0 else q

        # Compare query against one centroid per identity — no chaining.
        identity: str | None = None
        valid = [(k, v) for k, v in self._centroids.items() if v["uv"] is not None]
        if valid:
            ids = [k for k, _ in valid]
            mat = np.stack([v["uv"] for _, v in valid])  # [M, dim]
            scores = mat @ unit                           # [M] cosine similarities
            best_idx = int(np.argmax(scores))
            if float(scores[best_idx]) >= self.similarity_threshold:
                identity = ids[best_idx]

        if identity is None:
            identity = f"global-{self.next_global_id}"
            self.next_global_id += 1

        # Update centroid: running weighted average keeps the reference stable.
        if identity in self._centroids and self._centroids[identity]["uv"] is not None:
            count = int(self._centroids[identity]["count"])
            old_uv = self._centroids[identity]["uv"]
            new_uv = (old_uv * count + unit) / (count + 1)
            norm = float(np.linalg.norm(new_uv))
            new_uv = new_uv / norm if norm > 1e-8 else new_uv
            self._centroids[identity] = {"ts": timestamp_second, "uv": new_uv, "count": count + 1}
        else:
            self._centroids[identity] = {"ts": timestamp_second, "uv": unit, "count": 1}

        return identity


def _default_detector():
    """RF-DETR 2x-large — strict production detector on LightningAI GPU."""
    return RFDETRPersonDetector()


def _default_tracker():
    """OCMCTrack-style corrective cascade — always available (pure Python + numpy)."""
    return OCMCTrackStyleTracker()


@dataclass
class LocalVideoIngestionPipeline:
    """
    Video ingestion pipeline.

    Production path: RF-DETR 2x-large (detector) + OCMCTrack-style corrective
    cascade (tracker) + TrackletFeaturePipelineProcessor with CLIP / SOLIDER+KPR
    adapters.  Falls back to HOG + GreedyIoU when rfdetr package is absent.
    """

    sample_fps: int = 4
    sampler: VideoFrameSampler = field(default_factory=VideoFrameSampler)
    detector: RFDETRPersonDetector = field(default_factory=_default_detector)
    tracker: OCMCTrackStyleTracker = field(default_factory=_default_tracker)
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
                "frame_count": sum(int(_quality_or_zero(item, "frame_count")) for item in ordered_group),
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
            base["tracklet_frames"] = all_frames
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

    def _materialize_people(
        self,
        *,
        video_id: str,
        camera_id: str | None,
        tracklets: tuple[LocalTracklet, ...],
        sampled_fps: int,
        memory_bank: TrackletMemoryBank,
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

        for person in people:
            track_frames = person.get("tracklet_frames") or []
            last_ts = float(track_frames[-1].get("timestamp_second") or 0.0) if track_frames else 0.0
            embedding = person.get("embedding_vector")
            if not isinstance(embedding, list):
                embedding = None
            global_identity = memory_bank.resolve_identity(
                embedding_vector=embedding,
                timestamp_second=last_ts,
            )
            person["human_key"] = f"{camera_id or video_id}:{global_identity}"

        merged_people = self._merge_people_by_identity(video_id=video_id, people=people)
        with metadata_path.open("a", encoding="utf-8") as fh:
            for person in merged_people:
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
        self.tracker.reset()
        memory_bank = TrackletMemoryBank()
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
            memory_bank=memory_bank,
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
