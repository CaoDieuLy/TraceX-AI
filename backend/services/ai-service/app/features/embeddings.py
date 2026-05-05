import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..config import settings
from ..core.utils.geometry import batch_extract_crops
from ..core.types.bbox import BBox
from ..core.types.frame import Tracklet

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        crop_size: tuple = (224, 224),
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.crop_size = crop_size
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import AutoModel, AutoProcessor
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Embedding model loaded successfully")
        except ImportError as e:
            logger.warning(f"Transformers not available: {e}, using fallback")
            self.model = None
            self.processor = None

    def extract_tracklet_embedding(self, tracklet: Tracklet, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        if not tracklet.detections or not frames:
            return None

        bbox = tracklet.representative_bbox
        if bbox is None:
            for detection in tracklet.detections:
                if detection.bbox.confidence > 0.5:
                    bbox = detection.bbox
                    break

        if bbox is None:
            return self._extract_random_crop_embedding(frames[0])

        crop = self._extract_single_crop(frames[0], bbox)
        if crop is None:
            return self._extract_random_crop_embedding(frames[0])

        return self._compute_embedding([crop])[0] if self._compute_embedding([crop]) else None

    def extract_batch_embeddings(
        self,
        tracklets: List[Tracklet],
        frames: List[np.ndarray],
    ) -> List[Optional[np.ndarray]]:
        all_crops = []
        valid_indices = []

        for i, tracklet in enumerate(tracklets):
            if not tracklet.detections:
                all_crops.append(None)
                continue

            det = tracklet.detections[len(tracklet.detections) // 2]
            crop = self._extract_single_crop(frames[det.frame_idx] if det.frame_idx < len(frames) else frames[0], det.bbox)
            if crop is not None:
                all_crops.append(crop)
                valid_indices.append(i)
            else:
                all_crops.append(None)

        if not valid_indices:
            return [None] * len(tracklets)

        valid_crops = [c for c in all_crops if c is not None]
        embeddings = self._compute_embedding(valid_crops)

        result = [None] * len(tracklets)
        for idx, emb in zip(valid_indices, embeddings):
            result[idx] = emb

        return result

    def _extract_single_crop(self, frame: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        try:
            crops = batch_extract_crops(frame, [bbox], target_size=self.crop_size)
            return crops[0] if crops else None
        except Exception as e:
            logger.warning(f"Failed to extract crop: {e}")
            return None

    def _extract_random_crop_embedding(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        crop_h, crop_w = self.crop_size
        scale = min(h / crop_h, w / crop_w)
        new_h, new_w = int(crop_h * scale), int(crop_w * scale)
        import cv2
        resized = cv2.resize(frame, (new_w, new_h))
        top = (crop_h - new_h) // 2
        left = (crop_w - new_w) // 2
        canvas = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        canvas[top:top + new_h, left:left + new_w] = resized
        return canvas

    def _compute_embedding(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        if self.model is None or self.processor is None:
            return [self._random_embedding() for _ in crops]

        try:
            pil_images = [Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)) for crop in crops]
            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0]
                embeddings = F.normalize(embeddings, p=2, dim=1)

            return [emb.cpu().numpy() for emb in embeddings]
        except Exception as e:
            logger.warning(f"Embedding computation failed: {e}")
            return [self._random_embedding() for _ in crops]

    def _random_embedding(self, dim: int = 384) -> np.ndarray:
        emb = np.random.randn(dim).astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    @property
    def model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "crop_size": self.crop_size,
        }
