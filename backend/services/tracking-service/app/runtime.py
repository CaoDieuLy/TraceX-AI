from __future__ import annotations

import cv2
import logging
import math
import os
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Any, Optional

import numpy as np

from .cuda_runtime import get_homography_inv, project_to_world
from .types import BoundingBox

logger = logging.getLogger(__name__)
gpu_lock = threading.Semaphore(1)


def _crop_from_bbox(image: np.ndarray, bbox: BoundingBox) -> np.ndarray | None:
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
    return (float(bbox.x1 + bbox.x2) / 2.0, float(bbox.y1 + bbox.y2) / 2.0)


def _point_distance(lhs: tuple[float, float], rhs: tuple[float, float]) -> float:
    return math.hypot(lhs[0] - rhs[0], lhs[1] - rhs[1])


@dataclass(frozen=True)
class SampledFrame:
    frame_index: int
    timestamp_second: float
    image: np.ndarray
    laplacian_score: float


@dataclass(frozen=True)
class FrameDetection:
    frame_index: int
    timestamp_second: float
    bbox: BoundingBox
    confidence: float
    laplacian_score: float
    crop_bgr: np.ndarray | None = None
    world_x: float | None = None
    world_y: float | None = None


@dataclass(frozen=True)
class TrackletObservation:
    frame_index: int
    timestamp_second: float
    bbox: BoundingBox
    confidence: float
    laplacian_score: float
    crop_bgr: np.ndarray | None = None
    world_x: float | None = None
    world_y: float | None = None


@dataclass(frozen=True)
class LocalTracklet:
    video_id: str
    camera_id: str | None
    track_id: str
    observations: tuple[TrackletObservation, ...]


@dataclass(frozen=True)
class TrackletQualityResult:
    accepted: bool
    average_confidence: float
    average_laplacian: float
    frame_count: int
    frame_density: float
    duration_seconds: float
    rejection_reason: str | None


@dataclass
class VideoFrameSampler:
    sample_fps: int = 4

    def stream_batched(self, video_path: Path, batch_size: int = 150):
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
        all_frames: list[SampledFrame] = []
        for batch in self.stream_batched(video_path):
            all_frames.extend(batch)
        return tuple(all_frames)


@dataclass
class RFDETRPersonDetector:
    confidence_threshold: float = 0.32
    max_detections_per_frame: int = 300
    _model: Any = field(default=None, repr=False)
    _ready: bool = False
    _load_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

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
        env_path = os.environ.get("MCPT_RF_DETR_WEIGHTS", "").strip()
        if env_path:
            weights_path = Path(env_path)
        else:
            _here = Path(__file__).resolve()
            repo_root = _here.parent.parent.parent.parent
            weights_path = repo_root / "storage" / "model-weights" / "rf-detr" / "rf-detr-xxlarge.pth"
        weights_str = str(weights_path) if weights_path.exists() else "rf-detr-xxlarge.pth"

        for loader in [
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
            raise RuntimeError("RF-DETR model failed to load")

        from PIL import Image as PILImage

        batch_size = max(1, int(os.environ.get("MCPT_DETECTOR_BATCH_SIZE", "8")))
        result: dict[int, tuple[FrameDetection, ...]] = {}

        for batch_start in range(0, len(frames), batch_size):
            batch_frames = frames[batch_start: batch_start + batch_size]
            pils = [PILImage.fromarray(cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB)) for f in batch_frames]
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
                    if class_ids is not None and int(class_ids[i]) != 1:
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
    track_thresh: float = 0.40
    low_thresh: float = 0.10
    new_track_threshold: float = 0.45
    max_match_cost: float = 0.80
    max_buffer_match_cost: float = 0.90
    max_head_center_distance: float = 120.0
    max_predicted_distance: float = 180.0
    max_world_distance: float = 0.42
    max_reentry_world_distance: float = 13.0
    iou_weight: float = 0.20
    distance_weight: float = 0.55
    velocity_weight: float = 0.25
    track_buffer: int = 20
    max_buffer_frames: int = 300
    head_ratio: float = 0.35
    shrink_x: float = 0.08
    min_track_frames: int = 9
    min_track_density: float = 0.10

    active: dict = field(default_factory=dict, repr=False)
    active_last_bbox: dict = field(default_factory=dict, repr=False)
    active_last_frame: dict = field(default_factory=dict, repr=False)
    active_last_world: dict = field(default_factory=dict, repr=False)
    buffer: dict = field(default_factory=dict, repr=False)
    buffer_last_bbox: dict = field(default_factory=dict, repr=False)
    buffer_entry_frame: dict = field(default_factory=dict, repr=False)
    buffer_last_world: dict = field(default_factory=dict, repr=False)
    next_id: int = 1

    def reset(self) -> None:
        self.active.clear()
        self.active_last_bbox.clear()
        self.active_last_frame.clear()
        self.active_last_world.clear()
        self.buffer.clear()
        self.buffer_last_bbox.clear()
        self.buffer_entry_frame.clear()
        self.buffer_last_world.clear()
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

    def _build_match_state(self, observations: list, last_bbox: BoundingBox, last_world, target_frame_idx: int):
        head_bbox = self._head_bbox(last_bbox)
        last_center = _bbox_center(head_bbox)
        if len(observations) < 2:
            return head_bbox, last_center, last_center, last_world

        previous_head_bbox = self._head_bbox(observations[-2].bbox)
        previous_center = _bbox_center(previous_head_bbox)
        frame_delta = max(observations[-1].frame_index - observations[-2].frame_index, 1)
        vx = (last_center[0] - previous_center[0]) / float(frame_delta)
        vy = (last_center[1] - previous_center[1]) / float(frame_delta)
        raw_steps = max(target_frame_idx - observations[-1].frame_index, 0)
        predict_steps = min(raw_steps, self.track_buffer * 2)
        predicted_center = (last_center[0] + vx * predict_steps, last_center[1] + vy * predict_steps)
        return head_bbox, last_center, predicted_center, last_world

    def _best_match(self, det_bbox: BoundingBox, det_world, track_states: dict, candidates: set, max_cost: float, stage3: bool = False):
        det_center = _bbox_center(det_bbox)
        det_h = max(det_bbox.y2 - det_bbox.y1, 1)
        best_tid = None
        best_cost_val = max_cost + 1.0
        best_iou = -1.0

        for tid in candidates:
            state = track_states.get(tid)
            if state is None:
                continue
            head_bbox, last_center, predicted_center, last_world = state

            if stage3:
                if det_world is None or last_world is None:
                    continue
                world_dist = _point_distance(det_world, last_world)
                if world_dist > self.max_reentry_world_distance:
                    continue
                buf_full_h = max((head_bbox.y2 - head_bbox.y1) / max(self.head_ratio, 1e-6), 1.0)
                world_cost = world_dist / max(self.max_reentry_world_distance, 1e-6)
                size_cost = abs(det_h - buf_full_h) / max(det_h, buf_full_h)
                cost = 0.65 * world_cost + 0.35 * size_cost
                if cost < best_cost_val:
                    best_cost_val = cost
                    best_tid = tid
                continue

            if det_world is not None and last_world is not None:
                if _point_distance(det_world, last_world) > self.max_world_distance:
                    continue

            center_distance = _point_distance(last_center, det_center)
            predicted_distance = _point_distance(predicted_center, det_center)
            if center_distance > self.max_head_center_distance and predicted_distance > self.max_predicted_distance:
                continue

            center_cost = min(center_distance / max(self.max_head_center_distance, 1e-6), 1.0)
            predicted_cost = min(predicted_distance / max(self.max_predicted_distance, 1e-6), 1.0)
            iou = _bbox_iou(head_bbox, det_bbox)
            cost = self.distance_weight * center_cost + self.velocity_weight * predicted_cost + self.iou_weight * (1.0 - iou)
            if cost < best_cost_val or (abs(cost - best_cost_val) <= 1e-6 and iou > best_iou):
                best_cost_val = cost
                best_iou = iou
                best_tid = tid

        return best_tid, best_cost_val

    def track_incremental(self, video_id: str, camera_id: str | None, detections_by_frame: dict) -> list[LocalTracklet]:
        completed: list[LocalTracklet] = []

        for fk in sorted(detections_by_frame):
            dets = list(detections_by_frame.get(fk) or ())
            frame_ts = dets[0].timestamp_second if dets else 0.0

            stale = [tid for tid in self.active if fk - self.active_last_frame.get(tid, fk) > self.track_buffer]
            for tid in stale:
                self.buffer[tid] = self.active.pop(tid)
                self.buffer_last_bbox[tid] = self.active_last_bbox.pop(tid)
                self.buffer_last_world[tid] = self.active_last_world.pop(tid, None)
                self.buffer_entry_frame[tid] = fk
                self.active_last_frame.pop(tid, None)

            expired = [tid for tid in self.buffer if fk - self.buffer_entry_frame.get(tid, fk) > self.max_buffer_frames]
            for tid in expired:
                obs = self.buffer.pop(tid, [])
                if obs and self._should_keep_tracklet(obs):
                    completed.append(LocalTracklet(video_id=video_id, camera_id=camera_id, track_id=tid, observations=tuple(obs)))
                self.buffer_last_bbox.pop(tid, None)
                self.buffer_last_world.pop(tid, None)
                self.buffer_entry_frame.pop(tid, None)

            if not dets:
                continue

            high = [d for d in dets if d.confidence >= self.track_thresh]
            low = [d for d in dets if self.low_thresh <= d.confidence < self.track_thresh]
            active_unmatched = set(self.active.keys())
            active_states = {
                tid: self._build_match_state(self.active[tid], self.active_last_bbox[tid], self.active_last_world.get(tid), fk)
                for tid in active_unmatched
            }

            def _det_world(det: FrameDetection):
                return (det.world_x, det.world_y) if det.world_x is not None else None

            unmatched_high: list[FrameDetection] = []
            for det in high:
                best_tid, best_cost = self._best_match(self._head_bbox(det.bbox), _det_world(det), active_states, active_unmatched, max_cost=self.max_match_cost)
                if best_tid is not None and best_cost <= self.max_match_cost:
                    self._update_active(best_tid, det, fk, frame_ts)
                    active_unmatched.discard(best_tid)
                    active_states.pop(best_tid, None)
                else:
                    unmatched_high.append(det)

            for det in low:
                if not active_unmatched:
                    break
                best_tid, best_cost = self._best_match(self._head_bbox(det.bbox), _det_world(det), active_states, active_unmatched, max_cost=self.max_match_cost)
                if best_tid is not None and best_cost <= self.max_match_cost:
                    self._update_active(best_tid, det, fk, frame_ts)
                    active_unmatched.discard(best_tid)
                    active_states.pop(best_tid, None)

            buffer_candidates = set(self.buffer.keys())
            buffer_states = {
                tid: self._build_match_state(self.buffer[tid], self.buffer_last_bbox[tid], self.buffer_last_world.get(tid), fk)
                for tid in buffer_candidates
            }
            new_dets: list[FrameDetection] = []
            for det in unmatched_high:
                if not buffer_candidates:
                    new_dets.append(det)
                    continue
                best_tid, best_cost = self._best_match(self._head_bbox(det.bbox), _det_world(det), buffer_states, buffer_candidates, max_cost=self.max_buffer_match_cost, stage3=True)
                if best_tid is not None and best_cost <= self.max_buffer_match_cost:
                    obs = self.buffer.pop(best_tid)
                    obs.append(self._make_obs(det, frame_ts))
                    self.active[best_tid] = obs
                    self.active_last_bbox[best_tid] = det.bbox
                    self.active_last_frame[best_tid] = fk
                    self.active_last_world[best_tid] = _det_world(det)
                    self.buffer_last_bbox.pop(best_tid, None)
                    self.buffer_last_world.pop(best_tid, None)
                    self.buffer_entry_frame.pop(best_tid, None)
                    buffer_candidates.discard(best_tid)
                    buffer_states.pop(best_tid, None)
                else:
                    new_dets.append(det)

            for det in new_dets:
                if det.confidence >= self.new_track_threshold:
                    tid = str(self.next_id)
                    self.next_id += 1
                    self.active[tid] = [self._make_obs(det, frame_ts)]
                    self.active_last_bbox[tid] = det.bbox
                    self.active_last_frame[tid] = fk
                    self.active_last_world[tid] = _det_world(det)

        return completed

    def finalize_all(self, video_id: str, camera_id: str | None) -> tuple[LocalTracklet, ...]:
        completed: list[LocalTracklet] = []
        for tid, obs_list in list(self.active.items()) + list(self.buffer.items()):
            if obs_list and self._should_keep_tracklet(obs_list):
                completed.append(LocalTracklet(video_id=video_id, camera_id=camera_id, track_id=tid, observations=tuple(obs_list)))
        self.reset()
        return tuple(completed)

    def _update_active(self, tid: str, det: FrameDetection, frame_idx: int, frame_ts: float) -> None:
        self.active[tid].append(self._make_obs(det, frame_ts))
        self.active_last_bbox[tid] = det.bbox
        self.active_last_frame[tid] = frame_idx
        if det.world_x is not None and det.world_y is not None:
            self.active_last_world[tid] = (det.world_x, det.world_y)

    def _should_keep_tracklet(self, observations: list) -> bool:
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
            world_x=det.world_x,
            world_y=det.world_y,
        )


@dataclass
class TrackletQualityScorer:
    minimum_confidence_score: float = 0.3
    minimum_frame_count: int = 9
    minimum_frame_density: float = 0.10
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
class LocalVideoIngestionPipeline:
    sample_fps: int = 4
    sampler: VideoFrameSampler = field(default_factory=VideoFrameSampler)
    detector: RFDETRPersonDetector = field(default_factory=RFDETRPersonDetector)
    tracker: HeadBoxTracker = field(default_factory=HeadBoxTracker)
    quality_scorer: TrackletQualityScorer = field(default_factory=TrackletQualityScorer)

    def __post_init__(self) -> None:
        self.sampler = VideoFrameSampler(sample_fps=self.sample_fps)

    def run(self, source_path: Path | str, video_id: str, camera_id: str | None = None, recorded_start: datetime | None = None) -> dict:
        logger.info("Local ingestion started source=%s sample_fps=%s", source_path, self.sample_fps)
        source_path = Path(source_path)

        _H_inv = get_homography_inv(camera_id)
        self.tracker.reset()
        sampled_frame_count = 0
        tracklet_count = 0
        finalized_tracklets_buffer: list[LocalTracklet] = []
        batch_size = max(150, int(os.environ.get("MCPT_STREAM_BATCH_SIZE", "150")))

        for batch_idx, sampled_batch in enumerate(self.sampler.stream_batched(source_path, batch_size=batch_size), start=1):
            sampled_frame_count += len(sampled_batch)
            detections_by_frame = self.detector.detect(sampled_batch)

            if _H_inv is not None:
                projected: dict[int, tuple[FrameDetection, ...]] = {}
                for fi, frame_dets in detections_by_frame.items():
                    new_dets = []
                    for det in frame_dets:
                        w = project_to_world(_H_inv, det.bbox)
                        new_dets.append(FrameDetection(
                            frame_index=det.frame_index,
                            timestamp_second=det.timestamp_second,
                            bbox=det.bbox,
                            confidence=det.confidence,
                            laplacian_score=det.laplacian_score,
                            crop_bgr=det.crop_bgr,
                            world_x=w[0] if w else None,
                            world_y=w[1] if w else None,
                        ))
                    projected[fi] = tuple(new_dets)
                detections_by_frame = projected

            finalized_tracklets = self.tracker.track_incremental(video_id=video_id, camera_id=camera_id, detections_by_frame=detections_by_frame)
            if finalized_tracklets:
                finalized_tracklets_buffer.extend(finalized_tracklets)
                tracklet_count += len(finalized_tracklets)

            logger.info("Batch completed batch=%s sampled_frames=%s people=%s", batch_idx, sampled_frame_count, len(finalized_tracklets_buffer))

        tail_tracklets = self.tracker.finalize_all(video_id=video_id, camera_id=camera_id)
        if tail_tracklets:
            finalized_tracklets_buffer.extend(tail_tracklets)
            tracklet_count += len(tail_tracklets)

        logger.info("Tracking phase completed sampled_frames=%s finalized_tracklets=%s", sampled_frame_count, len(finalized_tracklets_buffer))

        quality_results = {
            tracklet.track_id: self.quality_scorer.score(tracklet)
            for tracklet in finalized_tracklets_buffer
        }

        accepted_tracklets = [
            (tracklet, quality_results[tracklet.track_id])
            for tracklet in finalized_tracklets_buffer
            if tracklet.track_id in quality_results and quality_results[tracklet.track_id].accepted
        ]

        people = self._build_people(video_id, camera_id, accepted_tracklets)

        processed_at = datetime.now(timezone.utc).replace(microsecond=0)
        return {
            "video": {
                "video_id": video_id,
                "camera_id": camera_id,
                "sample_fps": self.sample_fps,
                "sampled_frame_count": sampled_frame_count,
                "tracklet_count": tracklet_count,
                "accepted_tracklet_count": len(people),
            },
            "people": people,
            "person_count": len(people),
            "processed_at": processed_at.isoformat().replace("+00:00", "Z"),
        }

    def _build_people(self, video_id: str, camera_id: str | None, accepted_tracklets: list) -> list[dict]:
        people = []
        for tracklet, quality in accepted_tracklets:
            if not tracklet.observations:
                continue
            first_obs = tracklet.observations[0]
            candidate_id = f"{video_id}:{tracklet.track_id}"
            people.append({
                "candidate_id": candidate_id,
                "camera_id": camera_id,
                "video_id": video_id,
                "track_id": tracklet.track_id,
                "human_key": f"{camera_id or video_id}:{tracklet.track_id}",
                "frame_idx": first_obs.frame_index,
                "bbox": [first_obs.bbox.x1, first_obs.bbox.y1, first_obs.bbox.x2, first_obs.bbox.y2],
                "confidence": first_obs.confidence,
                "timestamp_second": first_obs.timestamp_second,
                "tracklet_quality": {
                    "accepted": quality.accepted,
                    "average_confidence": quality.average_confidence,
                    "average_laplacian": quality.average_laplacian,
                    "frame_count": quality.frame_count,
                    "frame_density": quality.frame_density,
                    "duration_seconds": quality.duration_seconds,
                },
                "tracklet_frame_count_full": len(tracklet.observations),
                "tracklet_frame_start_idx": tracklet.observations[0].frame_index,
                "tracklet_frame_end_idx": tracklet.observations[-1].frame_index,
            })
        return people
