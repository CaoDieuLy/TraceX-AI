"""Multilingual translation using SeamlessM4T v2-large.

Replaces Helsinki-NLP opus-mt models with Facebook SeamlessM4T v2,
which handles Vietnamese ↔ English with better contextual accuracy
(especially surveillance/appearance description domain).

VRAM budget: ~5GB fp16 on query-service GPU partition.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Vietnamese diacritics pattern
_VI_PATTERN = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)

# Common Vietnamese function words for detection heuristic
_VI_STOPWORDS = frozenset({
    "của", "và", "là", "có", "được", "trong", "cho", "với", "không",
    "người", "này", "khi", "đã", "một", "những", "ra", "hay", "về",
    "theo", "từ", "bởi", "vào", "ở", "để", "như", "trên", "các",
    "ai", "họ", "bạn", "tôi", "chúng", "nào", "làm", "gì", "sao",
    "bao", "nhiêu", "mấy", "muốn", "cần", "phải", "mặc", "áo", "quần",
    "đang", "đi", "bước", "mang", "đeo", "cầm", "xách",
})

# Singleton SeamlessM4T components
_seamless_model = None
_seamless_processor = None
_seamless_device = None


def _get_model_cache_dir() -> Path:
    cache = Path(os.getenv("TRANSLATION_MODEL_PATH", "/workspace/models/translation"))
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _load_seamless() -> tuple:
    """Lazy-load SeamlessM4T v2-large. Returns (model, processor, device) or (None, None, None)."""
    global _seamless_model, _seamless_processor, _seamless_device
    if _seamless_model is not None:
        return _seamless_model, _seamless_processor, _seamless_device

    try:
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2Model

        model_id = os.getenv("SEAMLESS_M4T_MODEL", "facebook/seamless-m4t-v2-large")
        cache_dir = str(_get_model_cache_dir())

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        logger.info("Loading SeamlessM4T v2: %s  device=%s", model_id, device)
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        model = SeamlessM4Tv2Model.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=dtype,
        ).to(device)
        model.eval()

        _seamless_model = model
        _seamless_processor = processor
        _seamless_device = device
        logger.info("SeamlessM4T v2 loaded OK")

    except Exception as exc:
        logger.warning("SeamlessM4T v2 load failed: %s — translation will pass through", exc)
        _seamless_model = None
        _seamless_processor = None
        _seamless_device = None

    return _seamless_model, _seamless_processor, _seamless_device


def warmup() -> None:
    """Pre-load SeamlessM4T v2 into GPU memory at service startup."""
    logger.info("Warming up SeamlessM4T v2...")
    model, processor, device = _load_seamless()
    if model is None:
        logger.warning("SeamlessM4T v2 warmup skipped (model unavailable)")
        return

    # Dummy inference
    try:
        import torch
        text = "người mặc áo đỏ"
        inputs = processor(text=text, src_lang="vie", return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                tgt_lang="eng",
                generate_speech=False,
            )
        result = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        logger.info("SeamlessM4T v2 warmup OK: '%s' → '%s'", text, result)
    except Exception as exc:
        logger.warning("SeamlessM4T v2 warmup inference failed: %s", exc)


# Alias expected by callers in candidates.py
warmup_models = warmup


def detect_vietnamese(text: Optional[str]) -> bool:
    """Return True if text appears to be Vietnamese."""
    if not text:
        return False
    vi_chars = _VI_PATTERN.findall(text.lower())
    words = set(text.lower().split())
    stopword_hits = len(words & _VI_STOPWORDS)

    if len(vi_chars) >= 2:
        return True
    if stopword_hits >= 2:
        return True
    if len(vi_chars) >= 1 and stopword_hits >= 1:
        return True
    return False


def translate_to_english(text: Optional[str]) -> str:
    """Translate Vietnamese text → English using SeamlessM4T v2."""
    if not text:
        return text or ""
    model, processor, device = _load_seamless()
    if model is None:
        logger.debug("SeamlessM4T unavailable — returning original text")
        return text
    try:
        import torch
        inputs = processor(text=text, src_lang="vie", return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                tgt_lang="eng",
                generate_speech=False,
            )
        return processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    except Exception as exc:
        logger.warning("translate_to_english failed: %s", exc)
        return text


def translate_to_vietnamese(text: Optional[str]) -> str:
    """Translate English text → Vietnamese using SeamlessM4T v2."""
    if not text:
        return text or ""
    model, processor, device = _load_seamless()
    if model is None:
        return text
    try:
        import torch
        inputs = processor(text=text, src_lang="eng", return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                tgt_lang="vie",
                generate_speech=False,
            )
        return processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    except Exception as exc:
        logger.warning("translate_to_vietnamese failed: %s", exc)
        return text
