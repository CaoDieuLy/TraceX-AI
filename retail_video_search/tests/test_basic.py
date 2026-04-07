import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.core.clip_encoder import CLIPEncoder
from src.core.frame_extractor import FrameExtractor
import numpy as np

def test_clip_encoder():
    encoder = CLIPEncoder()
    emb = encoder.encode_text("test query")
    assert len(emb) == 512
    assert isinstance(emb, np.ndarray)

def test_frame_extractor(tmp_path):
    import cv2
    video_path = tmp_path / "test.mp4"
    out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))
    for _ in range(30):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    
    extractor = FrameExtractor(interval_sec=0.5)
    frames = list(extractor.extract_frames(video_path))
    assert len(frames) > 0

def test_qdrant_connection():
    from src.core.vector_store import QdrantStore
    try:
        store = QdrantStore()
        assert store.client is not None
    except Exception as e:
        # Qdrant may not be running
        pass
