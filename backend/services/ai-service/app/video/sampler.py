import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import decord
import numpy as np

decord.bridge.set_bridge("native")

from ..config import settings

logger = logging.getLogger(__name__)


class VideoSampler:
    def __init__(
        self,
        target_fps: int = 5,
        max_frames: int = 300,
    ):
        self.target_fps = target_fps
        self.max_frames = max_frames

    def sample_video(
        self,
        video_path: Union[str, Path],
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[int]]:
        path = str(video_path)
        try:
            vr = decord.VideoReader(path, ctx=decord.cpu(0))
            total_frames = len(vr)
            fps = vr.get_avg_fps()

            if fps <= 0:
                fps = 30.0

            frame_indices = self._compute_sample_indices(
                total_frames=total_frames,
                fps=fps,
                start_frame=start_frame,
                end_frame=end_frame,
            )

            frames = []
            for idx in frame_indices:
                if 0 <= idx < len(vr):
                    frame = vr[idx].asnumpy()
                    frames.append(frame)

            return frames, frame_indices

        except Exception as e:
            logger.error(f"Failed to sample video {path}: {e}")
            return [], []

    def _compute_sample_indices(
        self,
        total_frames: int,
        fps: float,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> List[int]:
        if total_frames == 0:
            return []

        start = start_frame if start_frame is not None else 0
        end = end_frame if end_frame is not None else total_frames

        start = max(0, min(start, total_frames - 1))
        end = max(start + 1, min(end, total_frames))

        if fps <= 0:
            fps = 30.0

        sample_interval = max(1, int(round(fps / self.target_fps)))

        indices = list(range(start, end, sample_interval))

        if len(indices) > self.max_frames:
            step = len(indices) // self.max_frames
            indices = indices[::step][:self.max_frames]

        return indices

    def get_frame_at_index(
        self,
        video_path: Union[str, Path],
        frame_idx: int,
    ) -> Optional[np.ndarray]:
        path = str(video_path)
        try:
            vr = decord.VideoReader(path, ctx=decord.cpu(0))
            if 0 <= frame_idx < len(vr):
                return vr[frame_idx].asnumpy()
        except Exception as e:
            logger.warning(f"Failed to read frame {frame_idx} from {path}: {e}")
        return None

    def get_video_info(self, video_path: Union[str, Path]) -> dict:
        path = str(video_path)
        try:
            vr = decord.VideoReader(path, ctx=decord.cpu(0))
            total_frames = len(vr)
            fps = vr.get_avg_fps()
            h, w = vr[0].shape[:2]
            duration = total_frames / fps if fps > 0 else 0

            return {
                "total_frames": total_frames,
                "fps": fps,
                "width": w,
                "height": h,
                "duration": duration,
            }
        except Exception as e:
            logger.error(f"Failed to get video info for {path}: {e}")
            return {
                "total_frames": 0,
                "fps": 0,
                "width": 0,
                "height": 0,
                "duration": 0,
            }
