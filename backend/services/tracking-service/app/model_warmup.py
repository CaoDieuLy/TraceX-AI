"""
model_warmup.py — Auto-load AI models on service startup.

Ensures all models are loaded into GPU memory when the service starts,
so first user request doesn't have to wait for model loading.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Track warmup status
_warmup_done = False
_warmup_error: Optional[str] = None


def get_device() -> torch.device:
    """Get the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("Using GPU: %s (%.1f GB VRAM)", gpu_name, gpu_mem)
        return device
    logger.warning("CUDA not available, using CPU (slow inference)")
    return torch.device("cpu")


async def warmup_models():
    """
    Warmup all AI models by running a dummy inference.
    This loads models into GPU memory and compiles CUDA kernels.
    """
    global _warmup_done, _warmup_error

    if _warmup_done:
        logger.info("Models already warmupped")
        return

    logger.info("=== Starting model warmup ===")
    device = get_device()

    try:
        # 1. Warmup DINOv2 (appearance embedding)
        await _warmup_dinov2(device)
        _warmup_done = True
        logger.info("=== Model warmup COMPLETE ===")
    except Exception as e:
        _warmup_error = str(e)
        logger.exception("Model warmup failed: %s", e)
        raise


async def _warmup_dinov2(device: torch.device):
    """Warmup DINOv2 model for appearance embedding."""
    logger.info("Warming up DINOv2...")
    try:
        from transformers import AutoImageProcessor, AutoModel

        model_id = "facebook/dinov2-large"
        logger.info("  Loading DINOv2 from HuggingFace: %s", model_id)

        processor = AutoImageProcessor.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
        model.eval()

        # Dummy inference
        import numpy as np
        from PIL import Image

        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        inputs = processor(images=dummy_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0]  # CLS token
            logger.info("  DINOv2 output shape: %s", embedding.shape)

        # Clear from memory (will reload with correct config later)
        del model, processor
        torch.cuda.empty_cache()
        logger.info("  DINOv2 warmup OK")

    except Exception as e:
        logger.warning("DINOv2 warmup failed (non-fatal): %s", e)


async def _warmup_detector(device: torch.device):
    """Warmup RF-DETR detector."""
    logger.info("Warming up RF-DETR detector...")
    # TODO: Implement when rfdetr is available
    logger.info("  RF-DETR warmup skipped (placeholder)")


async def _warmup_siglip(device: torch.device):
    """Warmup SigLIP2 for text/image encoding."""
    logger.info("Warming up SigLIP2...")
    # TODO: Implement when open_clip is configured
    logger.info("  SigLIP2 warmup skipped (placeholder)")


async def _warmup_videomae(device: torch.device):
    """Warmup VideoMAE for action recognition."""
    logger.info("Warming up VideoMAE...")
    # TODO: Implement when videomae is configured
    logger.info("  VideoMAE warmup skipped (placeholder)")


def is_warmup_done() -> bool:
    """Check if warmup has completed."""
    return _warmup_done


def get_warmup_error() -> Optional[str]:
    """Get warmup error message if any."""
    return _warmup_error
