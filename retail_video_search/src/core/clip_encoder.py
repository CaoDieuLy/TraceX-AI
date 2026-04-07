import torch
import clip
from PIL import Image
import numpy as np
from src.utils.logger import logger
from src.utils.config import config

class CLIPEncoder:
    def __init__(self, model_name: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = model_name or config.clip_model_name
        logger.info(f"Loading CLIP model {model_name} on {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode PIL image to normalized embedding (512-dim)."""
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()

    def encode_image_path(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        return self.encode_image(image)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text query to normalized embedding."""
        text_tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_text(text_tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()
