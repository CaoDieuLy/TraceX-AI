from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .bbox import BBox


@dataclass
class Detection:
    bbox: BBox
    track_id: Optional[int] = None
    frame_idx: int = 0
    timestamp_second: float = 0.0


@dataclass
class Tracklet:
    track_id: int
    detections: List[Detection] = field(default_factory=list)
    feature_vector: Optional[np.ndarray] = None
    appearance_embedding: Optional[np.ndarray] = None
    attribute_embedding: Optional[np.ndarray] = None

    @property
    def start_frame(self) -> int:
        return self.detections[0].frame_idx if self.detections else 0

    @property
    def end_frame(self) -> int:
        return self.detections[-1].frame_idx if self.detections else 0

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def representative_bbox(self) -> Optional[BBox]:
        if not self.detections:
            return None
        bboxes = [d.bbox for d in self.detections if d.bbox.confidence > 0.5]
        if not bboxes:
            return self.detections[0].bbox
        return max(bboxes, key=lambda b: b.area)


@dataclass
class Frame:
    frame_idx: int
    timestamp_second: float
    image: np.ndarray
    detections: List[Detection] = field(default_factory=list)
    original_height: int = 0
    original_width: int = 0

    def __post_init__(self):
        if self.original_height == 0 and self.image is not None:
            self.original_height = self.image.shape[0]
        if self.original_width == 0 and self.image is not None:
            self.original_width = self.image.shape[1]


@dataclass
class VideoMetadata:
    video_id: str
    total_frames: int
    fps: float
    width: int
    height: int
    duration_seconds: float
    camera_id: Optional[str] = None
    recorded_at: Optional[str] = None
    source_path: Optional[str] = None


@dataclass
class ProcessingResult:
    video_metadata: VideoMetadata
    tracklets: List[Tracklet]
    frame_detections: Dict[int, List[Detection]] = field(default_factory=dict)
    processing_time_seconds: float = 0.0
    model_info: Dict[str, str] = field(default_factory=dict)
