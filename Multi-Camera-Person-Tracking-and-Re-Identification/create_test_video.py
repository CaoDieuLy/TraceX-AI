#!/usr/bin/env python3
"""
Create a dummy .mp4 test video for exchange.py pipeline testing.
"""

import subprocess
import sys
from pathlib import Path

# Output path
output_dir = Path("/teamspace/studios/this_studio/A20-App-119/Multi-Camera-Person-Tracking-and-Re-Identification/test_videos")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "test_sample.mp4"

print(f"Creating test video: {output_path}")

# Generate 5-second test video with test pattern
cmd = [
    "ffmpeg", "-y",
    "-f", "lavfi",
    "-i", "testsrc=duration=5:size=640x480:rate=30",
    "-f", "lavfi",
    "-i", "sine=frequency=440:duration=5",
    "-c:v", "libx264",
    "-c:a", "aac",
    "-pix_fmt", "yuv420p",
    str(output_path)
]

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print(f"✅ Test video created: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
else:
    print(f"❌ ffmpeg error: {result.stderr[-300:]}")
    sys.exit(1)
