from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .models import Moment


def sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value)


def select_visual_path(moment: Moment) -> str | None:
    if moment.crop_paths:
        return moment.crop_paths[0]
    if moment.frame_paths:
        return moment.frame_paths[0]
    return None


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python-headless is required for frame extraction."
        ) from exc
    return cv2


def _clip_bbox(bbox: list[int], width: int, height: int, padding_ratio: float) -> list[int]:
    left, top, right, bottom = bbox
    box_width = max(right - left, 1)
    box_height = max(bottom - top, 1)
    pad_x = int(box_width * padding_ratio)
    pad_y = int(box_height * padding_ratio)
    return [
        max(left - pad_x, 0),
        max(top - pad_y, 0),
        min(right + pad_x, width),
        min(bottom + pad_y, height),
    ]


def extract_visual_assets(
    moments: list[Moment],
    output_dir: Path,
    max_assets_per_moment: int = 3,
    crop_padding: float = 0.08,
    overwrite: bool = False,
) -> list[Moment]:
    cv2 = _require_cv2()
    output_dir.mkdir(parents=True, exist_ok=True)

    for moment in moments:
        moment.frame_paths = []
        moment.crop_paths = []

    moments_by_video: dict[str, list[Moment]] = {}
    for moment in moments:
        moments_by_video.setdefault(moment.video_path, []).append(moment)

    for video_path, video_moments in moments_by_video.items():
        video_file = Path(video_path)
        if not video_file.exists():
            continue

        capture = cv2.VideoCapture(str(video_file))
        if not capture.isOpened():
            continue

        try:
            for moment in video_moments:
                chosen_frames = moment.sample_frames[:max_assets_per_moment] or [moment.start_frame]
                moment_dir = output_dir / sanitize_name(moment.id)
                moment_dir.mkdir(parents=True, exist_ok=True)

                for frame_idx in chosen_frames:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ok, frame = capture.read()
                    if not ok:
                        continue

                    frame_path = moment_dir / f"frame_{frame_idx:06d}.jpg"
                    if overwrite or not frame_path.exists():
                        cv2.imwrite(str(frame_path), frame)
                    moment.frame_paths.append(str(frame_path))

                    bbox = moment.representative_bbox
                    if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        height, width = frame.shape[:2]
                        clip_box = _clip_bbox(bbox, width, height, crop_padding)
                        crop = frame[clip_box[1] : clip_box[3], clip_box[0] : clip_box[2]]
                        if crop.size:
                            crop_path = moment_dir / f"crop_{frame_idx:06d}.jpg"
                            if overwrite or not crop_path.exists():
                                cv2.imwrite(str(crop_path), crop)
                            moment.crop_paths.append(str(crop_path))
        finally:
            capture.release()

    return moments


def extract_clip(
    video_path: Path,
    output_path: Path,
    start_second: float,
    duration_seconds: float,
    ffmpeg_bin: str = "ffmpeg",
) -> Path:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-ss",
        f"{start_second:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        "-c",
        "copy",
        "-y",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path
