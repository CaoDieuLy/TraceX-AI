import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union

import decord
from decord import VideoReader, cpu, gpu

decord.bridge.set_bridge("native")


def read_video_frames(
    video_path: Union[str, Path],
    frame_indices: Optional[List[int]] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    path = str(video_path)
    vr = VideoReader(path, ctx=cpu(0))

    frames = []
    if frame_indices is None:
        for i in range(len(vr)):
            frame = vr[i].asnumpy()
            if target_size is not None:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
    else:
        for idx in frame_indices:
            if 0 <= idx < len(vr):
                frame = vr[idx].asnumpy()
                if target_size is not None:
                    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
                frames.append(frame)

    return frames


def get_video_info(video_path: Union[str, Path]) -> dict:
    path = str(video_path)
    vr = VideoReader(path, ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    height, width = vr[0].shape[:2]
    duration = total_frames / fps if fps > 0 else 0

    return {
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
    }


def extract_frame_at_index(
    video_path: Union[str, Path],
    frame_idx: int,
    target_size: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    path = str(video_path)
    try:
        vr = VideoReader(path, ctx=cpu(0))
        if 0 <= frame_idx < len(vr):
            frame = vr[frame_idx].asnumpy()
            if target_size is not None:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            return frame
    except Exception:
        pass
    return None


def save_frame(frame: np.ndarray, output_path: Union[str, Path], quality: int = 95) -> bool:
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded = cv2.imencode(".jpg", frame, encode_param)
        if success:
            output_path.write_bytes(encoded.tobytes())
            return True
    except Exception:
        pass
    return False


def compute_frame_laplacian_variance(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())
