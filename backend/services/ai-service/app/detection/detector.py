import logging
from typing import List, Optional

import cv2
import numpy as np
import torch

from ..config import settings
from ..core.types.bbox import BBox
from ..core.types.frame import Detection

logger = logging.getLogger(__name__)


class Detector:
    def __init__(
        self,
        model_name: str = "yolov8n",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model: {self.model_name} on {self.device}")
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            logger.info(f"YOLO model loaded successfully")
        except ImportError:
            logger.warning("ultralytics not installed, using fallback detection")
            self.model = None

    def detect_persons(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        timestamp: float = 0.0,
    ) -> List[Detection]:
        if self.model is None:
            return self._fallback_detect(frame, frame_idx, timestamp)

        results = self.model(frame, verbose=False, conf=self.confidence_threshold)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls.item())
                if cls_id != 0:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf.item())

                bbox = BBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name="person",
                )
                detections.append(Detection(
                    bbox=bbox,
                    frame_idx=frame_idx,
                    timestamp_second=timestamp,
                ))

        return detections

    def _fallback_detect(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
    ) -> List[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = cv2.createBackgroundSubtractorMOG2().apply(gray)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        h, w = frame.shape[:2]
        min_area = (w * h) * 0.01
        max_area = (w * h) * 0.8

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, cw, ch = cv2.boundingRect(contour)
                bbox = BBox(
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + cw),
                    y2=float(y + ch),
                    confidence=0.5,
                    class_id=0,
                    class_name="person",
                )
                detections.append(Detection(
                    bbox=bbox,
                    frame_idx=frame_idx,
                    timestamp_second=timestamp,
                ))

        return detections

    def detect_batch(
        self,
        frames: List[np.ndarray],
        start_frame_idx: int = 0,
        fps: float = 30.0,
    ) -> List[List[Detection]]:
        all_detections = []
        for i, frame in enumerate(frames):
            timestamp = (start_frame_idx + i) / fps
            detections = self.detect_persons(frame, start_frame_idx + i, timestamp)
            all_detections.append(detections)
        return all_detections

    @property
    def model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
        }
