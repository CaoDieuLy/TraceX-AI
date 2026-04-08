import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config
from src.utils.logger import logger
from src.core.clip_encoder import CLIPEncoder
from src.core.frame_extractor import FrameExtractor
from src.core.vector_store import QdrantStore
from src.pipeline.data_converter import DatasetConverter
import argparse
from tqdm import tqdm

def index_videos(video_root: Path, force_reindex: bool = False):
    """Index all videos in the unified video directory."""
    encoder = CLIPEncoder()
    extractor = FrameExtractor()
    store = QdrantStore()
    
    video_files = list(video_root.glob("**/*.mp4"))
    logger.info(f"Found {len(video_files)} video files")
    
    for vpath in tqdm(video_files):
        rel = vpath.relative_to(video_root)
        parts = rel.parts
        if len(parts) >= 3:
            store_id = parts[0]
            camera_id = parts[1]
            date_str = vpath.stem  # YYYY-MM-DD
        else:
            logger.warning(f"Skipping {vpath}: invalid structure")
            continue
        
        # Determine dataset source from path
        dataset_source = "unknown"
        if "surveillance" in str(vpath).lower():
            dataset_source = "surveillance_retail"
        elif "warehouse" in str(vpath).lower():
            dataset_source = "physicalai"
        elif "product" in camera_id.lower():
            dataset_source = "mimex"
        elif "retailaction" in str(vpath).lower():
            dataset_source = "retailaction"
        
        vectors = []
        payloads = []
        for timestamp, frame_bgr in extractor.extract_frames(vpath):
            frame_rgb = frame_bgr[:, :, ::-1]
            from PIL import Image
            pil_img = Image.fromarray(frame_rgb)
            emb = encoder.encode_image(pil_img)
            vectors.append(emb.tolist())
            payloads.append({
                "store_id": store_id,
                "camera_id": camera_id,
                "timestamp": timestamp,
                "date": date_str,
                "video_path": str(vpath),
                "dataset": dataset_source,
            })
        if vectors:
            store.insert_batch(vectors, payloads)
            logger.info(f"Inserted {len(vectors)} frames from {vpath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, default=config.video_root)
    parser.add_argument("--convert-only", action="store_true", help="Only convert datasets, don't index")
    parser.add_argument("--index-only", action="store_true", help="Only index, don't convert")
    args = parser.parse_args()
    
    if not args.index_only:
        converter = DatasetConverter()
        converter.run_all_conversions()
    
    if not args.convert_only:
        index_videos(args.video_root)
