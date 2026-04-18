#!/usr/bin/env python3
"""
Batch convert all MP4 videos in source directory to H.265 + generate metadata.
This script processes the existing dataset and populates the Queue folder.

Source: /teamspace/studios/this_studio/A20-App-119/Multi-Camera-Person-Tracking-and-Re-Identification/data/NVIDIA_SmartSpaces/MTMC_Tracking_2025/val/Hospital_000/videos
Destination Queue: Queue/.h265 + Queue/Metadata
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ==================== Configuration ====================

SOURCE_DIR = Path(
    "/teamspace/studios/this_studio/A20-App-119/Multi-Camera-Person-Tracking-and-Re-Identification/data/NVIDIA_SmartSpaces/MTMC_Tracking_2025/val/Hospital_000/videos"
)

# Local working directories (will be synced to GDrive later)
WORK_DIR = Path("./batch_work")
H265_DIR = WORK_DIR / ".h265"
METADATA_DIR = WORK_DIR / "Metadata"

# FFmpeg settings
FFMPEG_CRF = 28
FFMPEG_PRESET = "medium"
FFMPEG_AUDIO_CODEC = "aac"


# ==================== Utilities ====================

def run_ffmpeg_conversion(input_mp4: Path, output_h265: Path) -> Path:
    """Convert MP4 to H.265 using FFmpeg."""
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", str(input_mp4),
        "-c:v", "libx265",
        "-crf", str(FFMPEG_CRF),
        "-preset", FFMPEG_PRESET,
        "-c:a", FFMPEG_AUDIO_CODEC,
        "-progress", "pipe:1",
        str(output_h265),
    ]

    logger.info(f"Converting: {input_mp4.name} → {output_h265.name}")
    logger.debug(f"FFmpeg command: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Optional: read progress
        while True:
            output = process.stdout.readline() if process.stdout else ""
            if output == "" and process.poll() is not None:
                break
            if output:
                logger.debug(f"  FFmpeg: {output.strip()}")

        returncode = process.wait()
        if returncode != 0:
            stderr = process.stderr.read() if process.stderr else ""
            raise RuntimeError(f"FFmpeg failed (code {returncode}): {stderr}")

        logger.info(f"✓ Converted: {output_h265.name}")
        return output_h265

    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Install: apt-get install ffmpeg")
    except Exception as e:
        raise RuntimeError(f"Conversion error: {e}")


def generate_metadata(
    h265_path: Path,
    metadata_path: Path,
    video_id: str,
) -> Dict[str, Any]:
    """
    Generate metadata for video using detection + embedding models.
    PLACEHOLDER – replace with actual model inference code.
    """
    logger.info(f"Generating metadata: {metadata_path.name}")

    # TODO: Integrate real detection + embedding model here
    # For now, return mock metadata based on video file info

    metadata = {
        "video_id": video_id,
        "source_file": h265_path.name,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "detection": {
            "model": "YOLOv8x (placeholder)",
            "total_detections": 0,
            "confidence_threshold": 0.5,
            "classes": ["person"],
        },
        "embedding": {
            "model": "CLIP-RN50 (placeholder)",
            "embedding_dim": 512,
            "num_persons": 0,
        },
        "pipeline": {
            "profile": "accuracy_first",
            "tracker": "ByteTrack",
            "reid": "StrongReID",
        },
        "note": "Placeholder metadata – replace with actual model inference",
    }

    # Save to JSON
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Metadata saved: {metadata_path.name}")
    return metadata


def get_video_info(video_path: Path) -> Dict[str, Any]:
    """Get basic video info using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,duration,r_frame_rate",
        "-show_entries", "format=size",
        "-of", "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        stream = info.get("streams", [{}])[0]
        fmt = info.get("format", {})
        return {
            "codec": stream.get("codec_name", "unknown"),
            "width": stream.get("width", 0),
            "height": stream.get("height", 0),
            "duration": float(stream.get("duration", 0)),
            "frame_rate": stream.get("r_frame_rate", "0/1"),
            "size_bytes": int(fmt.get("size", 0)),
        }
    except Exception as e:
        logger.warning(f"Could not get video info for {video_path}: {e}")
        return {}


# ==================== Main ====================

def main():
    logger.info("=" * 60)
    logger.info("BATCH VIDEO CONVERTER – MP4 → H.265 + Metadata")
    logger.info("=" * 60)

    # Validate source directory
    if not SOURCE_DIR.exists():
        logger.error(f"Source directory not found: {SOURCE_DIR}")
        sys.exit(1)

    # Create working directories
    H265_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Find all MP4 files
    mp4_files = list(SOURCE_DIR.glob("*.mp4"))
    if not mp4_files:
        logger.warning(f"No MP4 files found in {SOURCE_DIR}")
        sys.exit(0)

    logger.info(f"Found {len(mp4_files)} MP4 files to process")
    logger.info(f"Source: {SOURCE_DIR}")
    logger.info(f"Destination: {WORK_DIR}")

    # Process each video
    stats = {"success": 0, "failed": 0}
    for idx, mp4_path in enumerate(mp4_files, 1):
        logger.info(f"[{idx}/{len(mp4_files)}] Processing: {mp4_path.name}")

        video_id = mp4_path.stem
        h265_filename = f"{video_id}.h265.mp4"
        metadata_filename = f"{video_id}.json"

        h265_dest = H265_DIR / h265_filename
        metadata_dest = METADATA_DIR / metadata_filename

        try:
            # Step 1: Convert MP4 → H.265
            h265_path = run_ffmpeg_conversion(mp4_path, h265_dest)

            # Step 2: Get video info
            video_info = get_video_info(h265_path)

            # Step 3: Generate metadata
            metadata = generate_metadata(h265_path, metadata_dest, video_id)
            metadata["video_info"] = video_info

            # Step 4: (Optional) Upload to Google Drive
            # TODO: Implement if google_drive_enabled=True

            logger.info(f"✓ Completed: {mp4_path.name}")
            stats["success"] += 1

        except Exception as e:
            logger.error(f"✗ Failed: {mp4_path.name} – {e}")
            stats["failed"] += 1

    # Summary
    logger.info("=" * 60)
    logger.info(f"BATCH CONVERSION COMPLETE")
    logger.info(f"  Success: {stats['success']}")
    logger.info(f"  Failed:  {stats['failed']}")
    logger.info(f"  Output:  {WORK_DIR}")
    logger.info(f"  H.265:   {H265_DIR} ({len(list(H265_DIR.glob('*.h265.mp4')))} files)")
    logger.info(f"  Metadata:{METADATA_DIR} ({len(list(METADATA_DIR.glob('*.json')))} files)")
    logger.info("=" * 60)

    return stats


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result["failed"] == 0 else 1)
