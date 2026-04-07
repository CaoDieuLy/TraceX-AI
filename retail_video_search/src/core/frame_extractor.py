import cv2
from pathlib import Path
from typing import Iterator, Tuple
import numpy as np
from src.utils.logger import logger
from src.utils.config import config

class FrameExtractor:
    def __init__(self, interval_sec: float = None):
        self.interval_sec = interval_sec or config.frame_sample_interval_sec

    def extract_frames(self, video_path: Path) -> Iterator[Tuple[float, np.ndarray]]:
        """Yield (timestamp_sec, frame_bgr) for each sampled frame."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        frame_interval = int(fps * self.interval_sec)
        if frame_interval < 1:
            frame_interval = 1
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                yield timestamp, frame
            frame_count += 1
        cap.release()

    def save_frame(self, frame: np.ndarray, output_path: Path):
        cv2.imwrite(str(output_path), frame)
