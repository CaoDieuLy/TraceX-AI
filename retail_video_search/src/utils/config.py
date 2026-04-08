import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class Config(BaseModel):
    video_root: Path = Path(os.getenv("VIDEO_ROOT", "data/videos"))
    frame_cache: Path = Path(os.getenv("FRAME_CACHE", "data/frames_cache"))
    raw_data_root: Path = Path(os.getenv("RAW_DATA_ROOT", "data/raw"))
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "retail_video_frames")
    clip_model_name: str = os.getenv("CLIP_MODEL_NAME", "ViT-B/32")
    frame_sample_interval_sec: float = float(os.getenv("FRAME_SAMPLE_INTERVAL_SEC", "2.0"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def create_dirs(self):
        self.video_root.mkdir(parents=True, exist_ok=True)
        self.frame_cache.mkdir(parents=True, exist_ok=True)
        self.raw_data_root.mkdir(parents=True, exist_ok=True)

config = Config()
config.create_dirs()
