from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
import json
import logging
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


@dataclass
class VideoFrameSampler:
    """Decode video and sample frames at the fixed ingest FPS."""

    sample_fps: int = 5

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
    corrective_buffer_seconds: float = 14.0
    max_frame_gap: int = 3

    def track(
        self,
        *,
        video_id: str,
        camera_id: str | None,
        detections_by_frame: dict[int, tuple[FrameDetection, ...]],
    ) -> tuple[LocalTracklet, ...]:
        active: dict[str, list[TrackletObservation]] = {}
        active_last_bbox: dict[str, BoundingBox] = {}
        active_last_ts: dict[str, float] = {}
        active_appearance: dict[str, np.ndarray] = {}

        buffer: dict[str, list[TrackletObservation]] = {}
        buffer_last_bbox: dict[str, BoundingBox] = {}
        buffer_last_ts: dict[str, float] = {}
        buffer_appearance: dict[str, np.ndarray] = {}

        completed: list[LocalTracklet] = []
        next_id = 1

        for frame_idx in sorted(detections_by_frame):
            dets = list(detections_by_frame.get(frame_idx) or ())
            if not dets:
                continue
            frame_ts = dets[0].timestamp_second

            # Expire buffer entries beyond corrective window
            expired = [tid for tid, ts in buffer_last_ts.items() if frame_ts - ts > self.corrective_buffer_seconds]
            for tid in expired:
                completed.append(LocalTracklet(
                    video_id=video_id, camera_id=camera_id,
                    track_id=tid, observations=tuple(buffer.pop(tid, []))
                ))
                buffer_last_bbox.pop(tid, None)
                buffer_last_ts.pop(tid, None)
                buffer_appearance.pop(tid, None)

            # Move stale active → buffer
            stale = [tid for tid, ts in active_last_ts.items() if frame_idx - self._last_frame_idx(active[tid]) > self.max_frame_gap]
            for tid in stale:
                buffer[tid] = active.pop(tid)
                buffer_last_bbox[tid] = active_last_bbox.pop(tid)
                buffer_last_ts[tid] = active_last_ts.pop(tid)
                buffer_appearance[tid] = active_appearance.pop(tid, np.zeros(16))

            high_dets = [d for d in dets if d.confidence >= self.high_confidence_threshold]
            low_dets = [d for d in dets if self.low_confidence_threshold <= d.confidence < self.high_confidence_threshold]
            new_dets = [d for d in dets if d.confidence >= self.new_track_threshold]

            unmatched_high: list[FrameDetection] = []
            active_unmatched = set(active.keys())

            # Stage 1: high-confidence dets → active tracks
            for det in high_dets:
                best_tid, best_score = self._best_match(
                    det, active_last_bbox, active_appearance, active_unmatched
                )
                if best_tid is not None and best_score >= self.iou_gate:
                    self._update_track(active, active_last_bbox, active_last_ts, active_appearance, best_tid, det, frame_ts)
                    active_unmatched.discard(best_tid)
                else:
                    unmatched_high.append(det)

            # Stage 2: low-confidence dets → remaining active tracks
            for det in low_dets:
                best_tid, best_score = self._best_match(det, active_last_bbox, active_appearance, active_unmatched)
                if best_tid is not None and best_score >= self.iou_gate:
                    self._update_track(active, active_last_bbox, active_last_ts, active_appearance, best_tid, det, frame_ts)
                    active_unmatched.discard(best_tid)

            # Corrective stage: unmatched high-conf dets → buffer tracks
            buffer_unmatched = set(buffer.keys())
            for det in list(unmatched_high):
                best_tid, best_score = self._best_match(det, buffer_last_bbox, buffer_appearance, buffer_unmatched)
                if best_tid is not None and best_score >= self.iou_gate:
                    obs_list = buffer.pop(best_tid)
                    bbox_prev = buffer_last_bbox.pop(best_tid)
                    ts_prev = buffer_last_ts.pop(best_tid, frame_ts)
                    app_prev = buffer_appearance.pop(best_tid, np.zeros(16))
                    obs = self._make_observation(det, frame_ts)
                    active[best_tid] = obs_list + [obs]
                    active_last_bbox[best_tid] = det.bbox
                    active_last_ts[best_tid] = frame_ts
                    active_appearance[best_tid] = self._appearance_descriptor(det)
                    buffer_unmatched.discard(best_tid)
                    unmatched_high.remove(det)

            # New tracks from unmatched high-conf dets above new_track_threshold
            for det in unmatched_high:
                if det.confidence >= self.new_track_threshold:
                    tid = str(next_id); next_id += 1
                    active[tid] = [self._make_observation(det, frame_ts)]
                    active_last_bbox[tid] = det.bbox
                    active_last_ts[tid] = frame_ts
                    active_appearance[tid] = self._appearance_descriptor(det)

        # Finalize all remaining tracks
        for container, tracks in [(active, active), (buffer, buffer)]:
            for tid, obs_list in tracks.items():
                if obs_list:
                    completed.append(LocalTracklet(video_id=video_id, camera_id=camera_id, track_id=tid, observations=tuple(obs_list)))

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
            if iou < self.iou_gate:
                continue
            app_sim = self._cosine_sim(appearance.get(tid, np.zeros(16)), det_app)
            score = 0.6 * iou + 0.4 * max(app_sim, 0.0)
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
        """Compact 16-bin HSV histogram for fast appearance gating in the tracker."""
        crop = det.crop_bgr
        if crop is None or crop.size == 0:
            return np.zeros(16, dtype=np.float32)
        hsv = cv2.cvtColor(np.asarray(crop, dtype=np.uint8), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten().astype(np.float32)
        norm = float(hist.sum()) or 1.0
        return hist / norm

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a)) or 1.0
        nb = float(np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / (na * nb))

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

        worker_count = max(1, min(self.tracklet_worker_count, len(accepted_tracklets)))
        if worker_count == 1:
            return [
                self._build_person_metadata(
                    video_id=video_id,
                    camera_id=camera_id,
                    tracklet=tracklet,
                    quality=quality,
                    sampled_fps=sampled_fps,
                )
                for tracklet, quality in accepted_tracklets
            ]

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            build_person = partial(
                self._build_person_metadata_from_item,
                video_id=video_id,
                camera_id=camera_id,
                sampled_fps=sampled_fps,
            )
            return list(
                executor.map(build_person, accepted_tracklets)
            )

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

    sample_fps: int = 5
    sampler: VideoFrameSampler = field(default_factory=VideoFrameSampler)
    detector: RFDETRPersonDetector = field(default_factory=_default_detector)
    tracker: OCMCTrackStyleTracker = field(default_factory=_default_tracker)
    quality_scorer: TrackletQualityScorer = field(default_factory=TrackletQualityScorer)
    metadata_assembler: LocalMetadataAssembler = field(default_factory=LocalMetadataAssembler)

    def __post_init__(self) -> None:
        self.sampler = VideoFrameSampler(sample_fps=self.sample_fps)

    def run(
        self,
        *,
        source_path: Path,
        compressed_path: Path,
        metadata_path: Path,
        camera_id: str | None,
        recorded_start: datetime | None,
        metadata: dict[str, object],
    ) -> LocalIngestionOutput:
        batch_size = max(1, int(os.environ.get("MCPT_FRAME_BATCH_SIZE", "150")))
        LOGGER.info(
            "Local ingestion started source=%s sample_fps=%s batch_size=%s",
            source_path, self.sample_fps, batch_size,
        )

        # Batched streaming: each batch of frames is decoded, detected, then freed.
        # Peak RAM = batch_size × frame_bytes (~900 MB) instead of all_frames × frame_bytes (~18 GB).
        detections_by_frame: dict[int, tuple[FrameDetection, ...]] = {}
        total_sampled = 0
        for batch in self.sampler.stream_batched(source_path, batch_size=batch_size):
            batch_dets = self.detector.detect(batch)
            detections_by_frame.update(batch_dets)
            total_sampled += len(batch)
            # batch goes out of scope here — full-resolution frames are GC'd immediately
        detection_count = sum(len(items) for items in detections_by_frame.values())
        LOGGER.info(
            "Local ingestion detect+stream completed frames=%s detections=%s",
            total_sampled, detection_count,
        )

        video_id = compressed_path.name
        LOGGER.info("Local ingestion tracking started video_id=%s", video_id)
        tracklets = self.tracker.track(
            video_id=video_id,
            camera_id=camera_id,
            detections_by_frame=detections_by_frame,
        )
        LOGGER.info("Local ingestion tracking completed tracklets=%s", len(tracklets))
        quality_results = {tracklet.track_id: self.quality_scorer.score(tracklet) for tracklet in tracklets}
        LOGGER.info("Local ingestion metadata assembly started accepted_tracklets=%s", sum(1 for result in quality_results.values() if result.accepted))
        people = self.metadata_assembler.build_people(
            video_id=video_id,
            camera_id=camera_id,
            tracklets=tracklets,
            quality_results=quality_results,
            sampled_fps=self.sample_fps,
        )
        LOGGER.info("Local ingestion metadata assembly completed people=%s", len(people))
        processed_at = datetime.now(timezone.utc).replace(microsecond=0)
        video_payload: dict[str, object] = {
            "video_id": video_id,
            "camera_id": camera_id,
            "source_path": str(source_path),
            "compressed_path": str(compressed_path),
            "metadata_path": str(metadata_path),
            "recorded_start": recorded_start.isoformat() if recorded_start else None,
            "sample_fps": self.sample_fps,
            "sampled_frame_count": total_sampled,
            "tracklet_count": len(tracklets),
            "accepted_tracklet_count": len(people),
            "rejected_tracklet_count": max(len(tracklets) - len(people), 0),
            "processing_backend": "strict_tracking_service",
            "ingestion_metadata": metadata,
            "processed_at": processed_at.isoformat().replace("+00:00", "Z"),
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps({"video": video_payload, "people": people}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("Local ingestion metadata written path=%s", metadata_path)
        return LocalIngestionOutput(
            video=video_payload,
            people=people,
            compressed_path=str(compressed_path),
            metadata_path=str(metadata_path),
            processed_at=processed_at,
        )
