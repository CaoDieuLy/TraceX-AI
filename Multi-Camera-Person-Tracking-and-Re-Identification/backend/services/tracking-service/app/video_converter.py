"""FFmpeg video conversion utility for H.265 encoding."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

VideoCodec = Literal["h265", "hevc", "libx265"]


class VideoConversionError(Exception):
    """Raised when video conversion fails."""

    pass


def get_video_info(video_path: str | Path) -> dict[str, str]:
    """
    Get video information using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video metadata (codec, duration, resolution, etc.)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        # Get video stream info
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,duration,r_frame_rate",
            "-show_entries",
            "format=size",
            "-of",
            "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json

        info = json.loads(result.stdout)

        stream_info = info.get("streams", [{}])[0]
        format_info = info.get("format", {})

        return {
            "codec": stream_info.get("codec_name", "unknown"),
            "width": stream_info.get("width", 0),
            "height": stream_info.get("height", 0),
            "duration": float(stream_info.get("duration", 0)),
            "frame_rate": stream_info.get("r_frame_rate", "0/1"),
            "size_bytes": int(format_info.get("size", 0)),
        }
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning(f"Could not extract video info from {video_path}: {e}")
        return {}


def is_h265(video_path: str | Path) -> bool:
    """
    Check if video is already H.265/HEVC encoded.

    Args:
        video_path: Path to video file

    Returns:
        True if video codec is h265/hevc
    """
    info = get_video_info(video_path)
    codec = info.get("codec", "").lower()
    return codec in ("hevc", "h265")


def convert_to_h265(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    crf: int = 28,
    preset: str = "medium",
    audio_codec: str = "aac",
    overwrite: bool = False,
) -> str:
    """
    Convert video to H.265 (HEVC) format using FFmpeg.

    Args:
        input_path: Source video file path
        output_path: Destination path (auto-generated if None)
        crf: Constant Rate Factor (0-51, lower = better quality, 28 is default)
        preset: Encoding speed preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
        audio_codec: Audio codec (aac, copy, etc.)
        overwrite: Overwrite output file if exists

    Returns:
        Path to converted H.265 video file

    Raises:
        VideoConversionError: If conversion fails
        FileNotFoundError: If input file doesn't exist
    """
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    # Skip conversion if already H.265
    if is_h265(input_path):
        logger.info(f"Video already H.265: {input_path}")
        return str(input_path)

    # Generate output path
    if output_path is None:
        output_path = input_path.with_suffix(".h265.mp4")
    else:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if output exists
    if output_path.exists() and not overwrite:
        logger.info(f"Converted file already exists: {output_path}")
        return str(output_path)

    logger.info(f"Converting {input_path} → {output_path} (H.265, CRF={crf}, preset={preset})")

    # FFmpeg command
    # -c:v libx265: H.265 video codec
    # -crf: quality (lower = better, 28 is good default)
    # -preset: encoding speed/compression tradeoff
    # -c:a aac: keep audio as AAC
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",  # Overwrite or no-clobber
        "-i",
        str(input_path),
        "-c:v",
        "libx265",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-c:a",
        audio_codec,
        "-progress",
        "pipe:1",  # Progress to stdout
        str(output_path),
    ]

    try:
        # Run FFmpeg with realtime output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Read output for progress (optional)
        while True:
            output = process.stdout.readline() if process.stdout else ""
            if output == "" and process.poll() is not None:
                break
            if output:
                logger.debug(f"FFmpeg: {output.strip()}")

        # Wait for completion
        returncode = process.wait()
        if returncode != 0:
            stderr = process.stderr.read() if process.stderr else ""
            raise VideoConversionError(f"FFmpeg failed (code {returncode}): {stderr}")

        logger.info(f"Conversion successful: {output_path}")
        return str(output_path)

    except FileNotFoundError as e:
        raise VideoConversionError(
            "FFmpeg not found. Install with: apt-get install ffmpeg (Ubuntu) or brew install ffmpeg (macOS)"
        ) from e
    except Exception as e:
        raise VideoConversionError(f"Conversion failed: {e}") from e


def validate_video_file(video_path: str | Path) -> tuple[bool, str]:
    """
    Validate that a file is a valid video.

    Args:
        video_path: Path to check

    Returns:
        (is_valid, message)
    """
    path = Path(video_path)
    if not path.exists():
        return False, f"File not found: {path}"

    if path.stat().st_size == 0:
        return False, "File is empty"

    try:
        # Quick probe with ffprobe
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False, f"Not a valid video file: {result.stderr}"
        return True, "Valid video file"
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        return False, f"Video validation error: {e}"
