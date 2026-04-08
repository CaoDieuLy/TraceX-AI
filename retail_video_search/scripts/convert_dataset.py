"""
Chuyển đổi các dataset bán lẻ (Surveillance for Retail Stores, PhysicalAI-SmartSpaces)
về cấu trúc thư mục yêu cầu: data/videos/{store_id}/{camera_id}/{date}.mp4
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.logger import logger
import shutil
import cv2

def convert_surveillance_frames_to_video(frames_dir: Path, output_path: Path, fps=25):
    """Chuyển sequence ảnh từ Surveillance for Retail Stores thành video MP4."""
    images = sorted(frames_dir.glob("*.jpg"))
    if not images:
        logger.error(f"No frames in {frames_dir}")
        return False
    frame = cv2.imread(str(images[0]))
    h, w = frame.shape[:2]
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for img in images:
        frame = cv2.imread(str(img))
        out.write(frame)
    out.release()
    logger.info(f"Saved {output_path}")
    return True

if __name__ == "__main__":
    # Example: convert a dataset folder
    # python scripts/convert_dataset.py --input /path/to/frames --output data/videos/store_A/counter/2025-04-01.mp4
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert_surveillance_frames_to_video(args.input, args.output)
