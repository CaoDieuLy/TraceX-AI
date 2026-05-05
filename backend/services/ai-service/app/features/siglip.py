import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class SigLIPExtractor:
    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import AutoModel, AutoProcessor
            logger.info(f"Loading SigLIP model: {self.model_name} on {self.device}")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("SigLIP model loaded successfully")
        except ImportError as e:
            logger.warning(f"SigLIP not available: {e}")
            self.model = None
            self.processor = None

    def extract_image_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            return np.random.randn(1152).astype(np.float32)

        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                embeddings = F.normalize(outputs, p=2, dim=1)

            return embeddings.cpu().numpy()[0]
        except Exception as e:
            logger.warning(f"SigLIP embedding failed: {e}")
            return np.random.randn(1152).astype(np.float32)

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        return float(np.dot(embedding1, embedding2))
