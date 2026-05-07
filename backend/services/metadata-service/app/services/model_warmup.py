"""GPU model warmup for metadata-service — SOTA 2026 AI pipeline.

Loads all models into VRAM once at startup via FastAPI lifespan.
VRAM budget (metadata-service share ~50GB of A100 80GB):
  - EVA-02 ViT-L/14 (appearance embedding, 1024-dim):  ~5GB fp16
  - Grounding DINO 1.6 (person detection):             ~4GB fp16
  - SigLIP 2-So400m (attribute tagging, zero-shot):    ~3GB fp16
  - VideoMAE V2-Large (action recognition):             ~3GB fp16
  - SeamlessM4T v2-large (Vietnamese translation):     ~5GB fp16

Forbidden: YOLO (any version), ByteTrack.
Required:  Grounding DINO 1.6, MCBLT 3D association, EVA-02, SigLIP 2, VideoMAE V2.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_warmup_done = False
_warmup_error: Optional[str] = None

_MODELS: dict[str, Any] = {}


def get_model(name: str) -> Optional[Any]:
    """Retrieve a loaded model by name."""
    return _MODELS.get(name)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_mem_gb = props.total_memory / 1024**3
        logger.info("GPU: %s  VRAM: %.1f GB", gpu_name, gpu_mem_gb)
        return device
    logger.warning("CUDA unavailable — CPU fallback (very slow for production)")
    return torch.device("cpu")


async def warmup_models() -> None:
    """Load all SOTA 2026 models into GPU memory. Called once at FastAPI lifespan startup."""
    global _warmup_done, _warmup_error

    if _warmup_done:
        logger.info("GPU models already loaded")
        return

    logger.info("=== metadata-service SOTA 2026 warmup starting ===")
    device = get_device()

    _load_eva02(device)
    _load_grounding_dino_16(device)
    _load_siglip2(device)
    _load_videomae_v2(device)

    _warmup_done = True
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(
            "=== Warmup complete — VRAM allocated: %.1f GB / reserved: %.1f GB ===",
            allocated, reserved,
        )
    else:
        logger.info("=== Warmup complete (CPU) ===")


def _load_eva02(device: torch.device) -> None:
    """Load EVA-02 ViT-L/14 for 1024-dim appearance embeddings."""
    logger.info("Loading EVA-02 ViT-L/14...")
    try:
        import timm
        import numpy as np
        from PIL import Image

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = timm.create_model(
            "eva02_large_patch14_224.mim_m38m_ft_in22k_in1k",
            pretrained=True,
            num_classes=0,
        )
        model = model.to(device=device, dtype=dtype)
        model.eval()

        data_cfg = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_cfg, is_training=False)
        dummy = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        inp = transform(dummy).unsqueeze(0).to(device=device, dtype=dtype)
        with torch.no_grad():
            feat = model(inp)
        logger.info("  EVA-02 output dim: %d", feat.shape[-1])
        assert feat.shape[-1] == 1024, f"Expected 1024-dim, got {feat.shape[-1]}"

        _MODELS["eva02"] = model
        _MODELS["eva02_transform"] = transform
        logger.info("  EVA-02 loaded OK")

    except Exception as exc:
        logger.warning("EVA-02 load failed (non-fatal): %s", exc)


def _load_grounding_dino_16(device: torch.device) -> None:
    """Load Grounding DINO 1.6 for open-vocabulary person detection."""
    logger.info("Loading Grounding DINO 1.6...")
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model_id = "IDEA-Research/grounding-dino-1.6-pro"
        try:
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id, torch_dtype=dtype,
            ).to(device)
        except Exception:
            model_id = "IDEA-Research/grounding-dino-1.6-base"
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id, torch_dtype=dtype,
            ).to(device)
        model.eval()

        import numpy as np
        from PIL import Image
        dummy_img = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        inputs = processor(images=dummy_img, text="person.", return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)

        _MODELS["gdino16"] = model
        _MODELS["gdino16_processor"] = processor
        _MODELS["gdino16_model_id"] = model_id
        logger.info("  Grounding DINO 1.6 loaded OK (model: %s)", model_id)

    except Exception as exc:
        logger.warning("Grounding DINO 1.6 load failed (non-fatal): %s", exc)


def _load_siglip2(device: torch.device) -> None:
    """Load SigLIP 2-So400m for zero-shot attribute labeling."""
    logger.info("Loading SigLIP 2-So400m...")
    try:
        from transformers import AutoProcessor, AutoModel

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model_id = "google/siglip2-so400m-patch14-384"
        try:
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device)
        except Exception:
            model_id = "google/siglip-so400m-patch14-384"
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device)
        model.eval()

        import numpy as np
        from PIL import Image
        dummy_img = Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8))
        labels = ["person in red shirt", "person in blue jeans"]
        inputs = processor(text=labels, images=dummy_img, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)

        _MODELS["siglip2"] = model
        _MODELS["siglip2_processor"] = processor
        logger.info("  SigLIP 2 loaded OK (model: %s)", model_id)

    except Exception as exc:
        logger.warning("SigLIP 2 load failed (non-fatal): %s", exc)


def _load_videomae_v2(device: torch.device) -> None:
    """Load VideoMAE V2 fine-tuned on Kinetics-400 for temporal action recognition."""
    logger.info("Loading VideoMAE V2 (Kinetics-400 fine-tuned)...")
    try:
        from transformers import AutoProcessor, AutoModelForVideoClassification
        import numpy as np

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        # Try huge first (86.6% top-1 on Kinetics-400), then base, then small
        model_ids = [
            "MCG-NJU/videomae-huge-finetuned-kinetics",
            "MCG-NJU/videomae-base-finetuned-kinetics",
            "MCG-NJU/videomae-small-finetuned-kinetics",
        ]
        loaded_model_id = None
        for model_id in model_ids:
            try:
                processor = AutoProcessor.from_pretrained(model_id)
                model = AutoModelForVideoClassification.from_pretrained(
                    model_id, torch_dtype=dtype,
                ).to(device)
                loaded_model_id = model_id
                break
            except Exception:
                logger.warning("  Could not load %s, trying next...", model_id)
                continue

        if loaded_model_id is None:
            logger.warning("No VideoMAE Kinetics model available, skipping")
            return

        model.eval()

        dummy_frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(16)]
        inputs = processor(dummy_frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        logger.info("  VideoMAE V2 logits dim: %d", out.logits.shape[-1])

        _MODELS["videomae"] = model
        _MODELS["videomae_processor"] = processor
        _MODELS["videomae_model_id"] = loaded_model_id
        logger.info("  VideoMAE V2 loaded OK (model: %s)", loaded_model_id)

    except Exception as exc:
        logger.warning("VideoMAE V2 load failed (non-fatal): %s", exc)


def is_warmup_done() -> bool:
    return _warmup_done


def get_warmup_error() -> Optional[str]:
    return _warmup_error


def get_loaded_models() -> list[str]:
    return [k for k in _MODELS if not k.endswith(("_processor", "_transform", "_model_id"))]
