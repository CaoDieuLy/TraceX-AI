"""
Unified data converter for multiple retail datasets.
Converts various dataset formats to standard structure:
    data/videos/{store_id}/{camera_id}/{date}.mp4
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.config import config
from src.utils.logger import logger
import cv2
import shutil
import json
from typing import Optional

class DatasetConverter:
    def __init__(self):
        self.video_root = config.video_root
        self.raw_root = config.raw_data_root

    def convert_surveillance_retail(self, source_path: Path, target_store: str = "store_001"):
        """
        Convert Surveillance for Retail Stores dataset.
        Source: folder with 01/, 02/, 03/ sequences (each has img1/ with jpg frames)
        """
        logger.info(f"Converting Surveillance for Retail Stores from {source_path}")
        sequences = ["01", "02", "03"]
        cameras = ["counter", "seating", "entrance"]
        
        for seq_idx, seq in enumerate(sequences):
            img_dir = source_path / seq / "img1"
            if not img_dir.exists():
                logger.warning(f"Sequence {seq} not found at {img_dir}")
                continue
            
            # Create video from frames
            camera_id = cameras[seq_idx % len(cameras)]
            output_video = self.video_root / target_store / camera_id / "2025-04-01.mp4"
            output_video.parent.mkdir(parents=True, exist_ok=True)
            
            # Get all jpg files sorted
            frames = sorted(img_dir.glob("*.jpg"))
            if not frames:
                logger.warning(f"No frames found in {img_dir}")
                continue
            
            # Read first frame to get dimensions
            first_frame = cv2.imread(str(frames[0]))
            h, w = first_frame.shape[:2]
            
            # Write video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video), fourcc, 25.0, (w, h))
            for frame_path in frames:
                frame = cv2.imread(str(frame_path))
                out.write(frame)
            out.release()
            logger.info(f"Created video: {output_video} ({len(frames)} frames)")

    def convert_physicalai_smartspaces(self, source_path: Path):
        """
        Convert NVIDIA PhysicalAI-SmartSpaces dataset.
        Source: HuggingFace dataset with warehouse folders.
        """
        logger.info(f"Converting PhysicalAI-SmartSpaces from {source_path}")
        # The dataset contains folders like Warehouse_000, Warehouse_001, etc.
        for warehouse_dir in source_path.glob("Warehouse_*"):
            if not warehouse_dir.is_dir():
                continue
            store_id = warehouse_dir.name.lower()
            # Look for video files in subdirectories
            for video_file in warehouse_dir.rglob("*.mp4"):
                # Parse camera info from filename or path
                camera_id = video_file.parent.name if video_file.parent != warehouse_dir else "camera_001"
                date_str = "2025-04-01"  # Use default date
                target_path = self.video_root / store_id / camera_id / f"{date_str}.mp4"
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(video_file, target_path)
                logger.info(f"Copied {video_file} -> {target_path}")

    def convert_retailaction(self):
        """
        Convert RetailAction dataset using FiftyOne.
        This dataset requires internet and will be downloaded on first use.
        """
        logger.info("Converting RetailAction dataset...")
        try:
            import fiftyone as fo
            from fiftyone.utils.huggingface import load_from_hub
            
            # Load dataset from HuggingFace
            dataset = load_from_hub("Voxel51/RetailAction", max_samples=100)  # Limit for demo
            
            for sample in dataset:
                # Each sample is a group with two video slices
                for slice_name in ["rank0", "rank1"]:
                    video_path = sample[slice_name].filepath
                    if video_path and Path(video_path).exists():
                        # Parse store and camera from metadata
                        store_id = f"store_{sample.sample_id[:3]}" if hasattr(sample, 'sample_id') else "retail_store"
                        camera_id = f"camera_{slice_name}"
                        date_str = "2025-04-01"
                        target_path = self.video_root / store_id / camera_id / f"{date_str}.mp4"
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(video_path, target_path)
                        logger.info(f"Copied RetailAction video -> {target_path}")
        except ImportError:
            logger.error("FiftyOne not installed. Run: pip install fiftyone")
        except Exception as e:
            logger.error(f"Error converting RetailAction: {e}")

    def convert_mimex_images(self, source_path: Path, target_store: str = "store_mimex"):
        """
        Convert MIMEX fine-grained product images to video format.
        Creates simple slideshow videos from image collections.
        """
        logger.info(f"Converting MIMEX dataset from {source_path}")
        # MIMEX has product categories as subfolders
        for category_dir in source_path.glob("*"):
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            images = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
            if not images:
                continue
            
            # Create a video from images (slideshow mode)
            camera_id = f"product_{category}"
            output_video = self.video_root / target_store / camera_id / f"{category}.mp4"
            output_video.parent.mkdir(parents=True, exist_ok=True)
            
            # Read first image for dimensions
            first_img = cv2.imread(str(images[0]))
            h, w = first_img.shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video), fourcc, 1.0, (w, h))  # 1 fps slideshow
            
            for img_path in images:
                img = cv2.imread(str(img_path))
                # Display each image for 2 seconds (2 frames at 1 fps)
                for _ in range(2):
                    out.write(img)
            out.release()
            logger.info(f"Created product video: {output_video} with {len(images)} images")

    def run_all_conversions(self):
        """Run all dataset conversions."""
        # Check for Surveillance for Retail Stores
        surveillance_path = self.raw_root / "surveillance_retail"
        if surveillance_path.exists():
            self.convert_surveillance_retail(surveillance_path)
        else:
            logger.info(f"Surveillance dataset not found at {surveillance_path}, skipping")
        
        # Check for PhysicalAI-SmartSpaces
        physicalai_path = self.raw_root / "physicalai_smartspaces"
        if physicalai_path.exists():
            self.convert_physicalai_smartspaces(physicalai_path)
        else:
            logger.info(f"PhysicalAI dataset not found at {physicalai_path}, skipping")
        
        # Convert RetailAction (requires internet)
        self.convert_retailaction()
        
        # Check for MIMEX
        mimex_path = self.raw_root / "mimex"
        if mimex_path.exists():
            self.convert_mimex_images(mimex_path)
        else:
            logger.info(f"MIMEX dataset not found at {mimex_path}, skipping")

if __name__ == "__main__":
    converter = DatasetConverter()
    converter.run_all_conversions()
