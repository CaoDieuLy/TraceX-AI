#!/usr/bin/env python3
"""
Generate a longer test video (~30s) with people walking.
"""

import subprocess
import sys
from pathlib import Path

print("Creating longer test video with people...")

# Use ffmpeg to create a 30-second test video with moving rectangles (simulate people)
output = Path("test_videos/test_sample_long.mp4")
duration = 30  # seconds

cmd = [
    "ffmpeg", "-y", "-f", "lavfi",
    "-i", f"testsrc=duration={duration}:size=1920x1080:rate=30",
    "-vf", (
        "drawbox=x='if(lt(mod(t,4),2),100,800)':y=400:w=100:h=200:color=red@0.7,"
        "drawbox=x='if(lt(mod(t+2,4),2),700,200)':y=500:w=80:h=150:color=blue@0.7,"
        "drawtext=text='Person 1':x=100:y=380:fontsize=24:fontcolor=white,"
        "drawtext=text='Person 2':x=700:y=480:fontsize=24:fontcolor=white"
    ),
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    str(output)
]

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    print(f"✅ Created: {output} ({output.stat().st_size / 1024 / 1024:.2f} MB)")
    # Also create h265 version
    h265_output = Path("test_videos/test_sample_long.h265")
    cmd2 = [
        "ffmpeg", "-y", "-i", str(output),
        "-c:v", "libx265", "-crf", "28", "-preset", "medium",
        "-c:a", "aac", str(h265_output)
    ]
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    if result2.returncode == 0:
        print(f"✅ Created: {h265_output} ({h265_output.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        print(f"❌ H265 conversion failed: {result2.stderr}")
else:
    print(f"❌ Failed: {result.stderr}")
    sys.exit(1)
