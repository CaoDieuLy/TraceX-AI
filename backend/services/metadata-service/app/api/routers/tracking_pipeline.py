"""
Tracking pipeline adapted from production HeadBoxTracker.
Replaces simple MCBLT grouping with ByteTrack-style head-box tracker.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── Vectorized geometry helpers (module-level, used by HeadBoxTracker) ────────

def _head_bbox_arr(bboxes: np.ndarray, head_ratio: float, shrink_x: float) -> np.ndarray:
    """Vectorized head bbox for [N, 4] array → [N, 4]."""
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    w = x2 - x1
    return np.stack([x1 + w * shrink_x, y1, x2 - w * shrink_x, y1 + (y2 - y1) * head_ratio], axis=1)


def _batch_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU: a [M,4], b [N,4] → [M,N]."""
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


# ── Data types ────────────────────────────────────────────────────────────────

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
    bbox: tuple[int, int, int, int]   # x1 y1 x2 y2
    confidence: float
    laplacian_score: float
    crop_bgr: Optional[np.ndarray] = None


@dataclass(frozen=True)
class TrackletObservation:
    frame_index: int
    timestamp_second: float
    bbox: tuple[int, int, int, int]
    confidence: float
    laplacian_score: float
    crop_bgr: Optional[np.ndarray] = None


@dataclass(frozen=True)
class LocalTracklet:
    video_id: str
    camera_id: Optional[str]
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
    rejection_reason: Optional[str]


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _bbox_iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _center(bbox: tuple) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _crop_from_bbox(image: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    return crop.copy() if crop.size > 0 else None


def _head_bbox(bbox: tuple, head_ratio: float = 0.35, shrink_x: float = 0.08) -> tuple:
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    w, h = x2 - x1, y2 - y1
    return (
        int(x1 + w * shrink_x),
        int(y1),
        int(x2 - w * shrink_x),
        int(y1 + h * head_ratio),
    )


# ── VideoFrameSampler ─────────────────────────────────────────────────────────

class VideoFrameSampler:
    """Sample frames at fixed FPS with Laplacian blur scoring."""

    def __init__(self, sample_fps: int = 4):
        self.sample_fps = max(1, sample_fps)

    def sample(self, video_path: str) -> tuple[SampledFrame, ...]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or self.sample_fps)
        if source_fps <= 0:
            source_fps = float(self.sample_fps)

        frames: list[SampledFrame] = []
        next_emit = 0.0
        src_idx = 0
        sampled_idx = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                ts = src_idx / source_fps
                if ts + 1e-9 >= next_emit:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    frames.append(SampledFrame(
                        frame_index=sampled_idx,
                        timestamp_second=round(ts, 6),
                        image=frame,
                        laplacian_score=round(lap, 6),
                    ))
                    sampled_idx += 1
                    next_emit += 1.0 / self.sample_fps
                src_idx += 1
        finally:
            cap.release()

        return tuple(frames)


# ── HeadBoxTracker ────────────────────────────────────────────────────────────

class HeadBoxTracker:
    """
    ByteTrack-style tracker using head-box + center distance + velocity + IoU.
    Adapted from production tracking service.
    """

    def __init__(
        self,
        track_thresh: float = 0.40,
        low_thresh: float = 0.10,
        new_track_threshold: float = 0.45,
        max_match_cost: float = 0.80,
        max_buffer_match_cost: float = 0.90,
        max_head_center_distance: float = 120.0,
        max_predicted_distance: float = 180.0,
        iou_weight: float = 0.20,
        distance_weight: float = 0.55,
        velocity_weight: float = 0.25,
        track_buffer: int = 20,
        max_buffer_frames: int = 300,
        head_ratio: float = 0.35,
        shrink_x: float = 0.08,
        min_track_frames: int = 3,
        min_track_density: float = 0.05,
    ):
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.new_track_threshold = new_track_threshold
        self.max_match_cost = max_match_cost
        self.max_buffer_match_cost = max_buffer_match_cost
        self.max_head_center_distance = max_head_center_distance
        self.max_predicted_distance = max_predicted_distance
        self.iou_weight = iou_weight
        self.distance_weight = distance_weight
        self.velocity_weight = velocity_weight
        self.track_buffer = track_buffer
        self.max_buffer_frames = max_buffer_frames
        self.head_ratio = head_ratio
        self.shrink_x = shrink_x
        self.min_track_frames = min_track_frames
        self.min_track_density = min_track_density
        self._reset()

    def _reset(self) -> None:
        self.active: dict[str, list[TrackletObservation]] = {}
        self.active_last_bbox: dict[str, tuple] = {}
        self.active_last_frame: dict[str, int] = {}
        self.buffer: dict[str, list[TrackletObservation]] = {}
        self.buffer_last_bbox: dict[str, tuple] = {}
        self.buffer_entry_frame: dict[str, int] = {}
        self.next_id = 1

    def _hbox(self, bbox: tuple) -> tuple:
        return _head_bbox(bbox, self.head_ratio, self.shrink_x)

    def _build_state(self, obs: list[TrackletObservation], last_bbox: tuple, target_fk: int):
        hbox = self._hbox(last_bbox)
        lc = _center(hbox)
        if len(obs) < 2:
            return hbox, lc, lc
        prev_hbox = self._hbox(obs[-2].bbox)
        pc = _center(prev_hbox)
        delta = max(obs[-1].frame_index - obs[-2].frame_index, 1)
        vx = (lc[0] - pc[0]) / delta
        vy = (lc[1] - pc[1]) / delta
        steps = min(max(target_fk - obs[-1].frame_index, 0), self.track_buffer * 2)
        pred = (lc[0] + vx * steps, lc[1] + vy * steps)
        return hbox, lc, pred

    def _match(self, det_bbox: tuple, states: dict, candidates: set, max_cost: float) -> tuple[Optional[str], float]:
        det_hbox = self._hbox(det_bbox)
        dc = _center(det_hbox)
        best_tid, best_cost, best_iou = None, max_cost + 1.0, -1.0
        for tid in candidates:
            hbox, lc, pred = states[tid]
            cd = _dist(lc, dc)
            pd = _dist(pred, dc)
            if cd > self.max_head_center_distance and pd > self.max_predicted_distance:
                continue
            cc = min(cd / max(self.max_head_center_distance, 1e-6), 1.0)
            pc_cost = min(pd / max(self.max_predicted_distance, 1e-6), 1.0)
            iou = _bbox_iou(hbox, det_hbox)
            cost = self.distance_weight * cc + self.velocity_weight * pc_cost + self.iou_weight * (1.0 - iou)
            if cost < best_cost or (abs(cost - best_cost) < 1e-6 and iou > best_iou):
                best_cost, best_iou, best_tid = cost, iou, tid
        return best_tid, best_cost

    def _match_frame_greedy(
        self,
        det_bboxes: list[tuple],
        states: dict,
        candidates: set,
        max_cost: float,
    ) -> list[Optional[tuple[str, float]]]:
        """Vectorized greedy matching: numpy cost matrix, same greedy semantics as _match()."""
        if not candidates or not det_bboxes:
            return [None] * len(det_bboxes)

        track_ids = list(candidates)
        N, M = len(track_ids), len(det_bboxes)

        t_hboxes = np.array([states[tid][0] for tid in track_ids], dtype=np.float32)  # [N, 4]
        t_lcs    = np.array([states[tid][1] for tid in track_ids], dtype=np.float32)  # [N, 2]
        t_preds  = np.array([states[tid][2] for tid in track_ids], dtype=np.float32)  # [N, 2]
        d_hboxes = _head_bbox_arr(
            np.array(det_bboxes, dtype=np.float32), self.head_ratio, self.shrink_x
        )                                                                               # [M, 4]
        d_centers = (d_hboxes[:, :2] + d_hboxes[:, 2:]) / 2                           # [M, 2]

        cd = np.linalg.norm(d_centers[:, None, :] - t_lcs[None, :, :], axis=-1)       # [M, N]
        pd = np.linalg.norm(d_centers[:, None, :] - t_preds[None, :, :], axis=-1)     # [M, N]
        iou = _batch_iou_np(d_hboxes, t_hboxes)                                        # [M, N]

        cost = (self.distance_weight * np.minimum(cd / max(self.max_head_center_distance, 1e-6), 1.0) +
                self.velocity_weight * np.minimum(pd / max(self.max_predicted_distance, 1e-6), 1.0) +
                self.iou_weight * (1.0 - iou))
        cost = np.where((cd > self.max_head_center_distance) & (pd > self.max_predicted_distance), 1e9, cost)

        results: list[Optional[tuple[str, float]]] = [None] * M
        used: list[int] = []
        for m in range(M):
            row = cost[m].copy()
            if used:
                row[used] = 1e9
            best_j = int(row.argmin())
            if row[best_j] <= max_cost:
                results[m] = (track_ids[best_j], float(row[best_j]))
                used.append(best_j)
        return results

    def _should_keep(self, obs: list[TrackletObservation]) -> bool:
        if len(obs) < self.min_track_frames:
            return False
        span = max(obs[-1].frame_index - obs[0].frame_index + 1, 1)
        return (len(obs) / span) >= self.min_track_density

    @staticmethod
    def _make_obs(det: FrameDetection, ts: float) -> TrackletObservation:
        return TrackletObservation(
            frame_index=det.frame_index,
            timestamp_second=ts,
            bbox=det.bbox,
            confidence=det.confidence,
            laplacian_score=det.laplacian_score,
            crop_bgr=det.crop_bgr,
        )

    def track(
        self,
        video_id: str,
        camera_id: Optional[str],
        detections_by_frame: dict[int, list[FrameDetection]],
    ) -> tuple[LocalTracklet, ...]:
        self._reset()
        completed: list[LocalTracklet] = []

        for fk in sorted(detections_by_frame):
            dets = list(detections_by_frame.get(fk) or [])
            ts = dets[0].timestamp_second if dets else 0.0

            # Move stale active → buffer
            stale = [tid for tid in self.active if fk - self.active_last_frame.get(tid, fk) > self.track_buffer]
            for tid in stale:
                self.buffer[tid] = self.active.pop(tid)
                self.buffer_last_bbox[tid] = self.active_last_bbox.pop(tid)
                self.buffer_entry_frame[tid] = fk
                self.active_last_frame.pop(tid, None)

            # Expire old buffer tracks
            expired = [tid for tid in self.buffer if fk - self.buffer_entry_frame.get(tid, fk) > self.max_buffer_frames]
            for tid in expired:
                obs = self.buffer.pop(tid, [])
                if obs and self._should_keep(obs):
                    completed.append(LocalTracklet(video_id, camera_id, tid, tuple(obs)))
                self.buffer_last_bbox.pop(tid, None)
                self.buffer_entry_frame.pop(tid, None)

            if not dets:
                continue

            high = [d for d in dets if d.confidence >= self.track_thresh]
            low  = [d for d in dets if self.low_thresh <= d.confidence < self.track_thresh]
            active_unmatched = set(self.active.keys())
            states = {tid: self._build_state(self.active[tid], self.active_last_bbox[tid], fk) for tid in active_unmatched}

            unmatched_high: list[FrameDetection] = []
            if high and active_unmatched:
                matches = self._match_frame_greedy(
                    [d.bbox for d in high], states, active_unmatched, self.max_match_cost
                )
                for det, m in zip(high, matches):
                    if m:
                        tid, _ = m
                        self.active[tid].append(self._make_obs(det, ts))
                        self.active_last_bbox[tid] = det.bbox
                        self.active_last_frame[tid] = fk
                        active_unmatched.discard(tid)
                        states.pop(tid, None)
                    else:
                        unmatched_high.append(det)
            else:
                unmatched_high = list(high)

            if low and active_unmatched:
                matches = self._match_frame_greedy(
                    [d.bbox for d in low], states, active_unmatched, self.max_match_cost
                )
                for det, m in zip(low, matches):
                    if m:
                        tid, _ = m
                        self.active[tid].append(self._make_obs(det, ts))
                        self.active_last_bbox[tid] = det.bbox
                        self.active_last_frame[tid] = fk
                        active_unmatched.discard(tid)
                        states.pop(tid, None)

            for det in unmatched_high:
                if det.confidence >= self.new_track_threshold:
                    tid = str(self.next_id); self.next_id += 1
                    self.active[tid] = [self._make_obs(det, ts)]
                    self.active_last_bbox[tid] = det.bbox
                    self.active_last_frame[tid] = fk

        # Finalize all remaining tracks
        for tid, obs in list(self.active.items()) + list(self.buffer.items()):
            if obs and self._should_keep(obs):
                completed.append(LocalTracklet(video_id, camera_id, tid, tuple(obs)))
        self._reset()
        return tuple(completed)


# ── TrackletQualityScorer ─────────────────────────────────────────────────────

class TrackletQualityScorer:
    def __init__(
        self,
        min_confidence: float = 0.30,
        min_frames: int = 3,
        min_density: float = 0.05,
        min_duration_s: float = 0.5,
        min_laplacian: float = 8.0,
    ):
        self.min_confidence = min_confidence
        self.min_frames = min_frames
        self.min_density = min_density
        self.min_duration_s = min_duration_s
        self.min_laplacian = min_laplacian

    def score(self, tracklet: LocalTracklet) -> TrackletQualityResult:
        obs = tracklet.observations
        if not obs:
            return TrackletQualityResult(False, 0.0, 0.0, 0, 0.0, 0.0, "empty")

        confs = [o.confidence for o in obs]
        laps  = [o.laplacian_score for o in obs]
        avg_conf = sum(confs) / len(confs)
        avg_lap  = sum(laps) / len(laps)
        n = len(obs)
        span = max(obs[-1].frame_index - obs[0].frame_index + 1, 1)
        density = n / span
        duration = max(obs[-1].timestamp_second - obs[0].timestamp_second, 0.0) if n >= 2 else 0.0

        if n < self.min_frames:
            return TrackletQualityResult(False, avg_conf, avg_lap, n, density, duration, "insufficient_frames")
        if density < self.min_density:
            return TrackletQualityResult(False, avg_conf, avg_lap, n, density, duration, "sparse_tracklet")
        if duration < self.min_duration_s:
            return TrackletQualityResult(False, avg_conf, avg_lap, n, density, duration, "short_tracklet")
        if avg_conf < self.min_confidence:
            return TrackletQualityResult(False, avg_conf, avg_lap, n, density, duration, "low_confidence")
        if avg_lap < self.min_laplacian:
            return TrackletQualityResult(False, avg_conf, avg_lap, n, density, duration, "blurry_tracklet")
        return TrackletQualityResult(True, avg_conf, avg_lap, n, density, duration, None)
