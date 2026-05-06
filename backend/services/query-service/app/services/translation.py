"""Vietnamese translation service using Helsinki-NLP models."""

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Vietnamese character pattern
VIETNAMESE_PATTERN = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.IGNORECASE)

# Singleton translation pipelines
_vi_to_en_pipeline = None
_en_to_vi_pipeline = None


def _get_model_cache_dir() -> Path:
    """Get model cache directory."""
    cache_dir = Path(os.getenv("TRANSLATION_MODEL_PATH", "/workspace/models/translation"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@lru_cache(maxsize=1)
def get_vi_to_en_pipeline():
    """Get or create VI→EN translation pipeline."""
    global _vi_to_en_pipeline
    if _vi_to_en_pipeline is None:
        try:
            from transformers import pipeline
            model_name = os.getenv("TRANSLATION_VI_EN_MODEL", "Helsinki-NLP/opus-mt-vi-en")
            cache_dir = str(_get_model_cache_dir())
            
            logger.info(f"Loading translation model: {model_name}")
            _vi_to_en_pipeline = pipeline(
                "translation",
                model=model_name,
                tokenizer=model_name,
                cache_dir=cache_dir,
                device=-1,  # CPU
            )
            logger.info("VI→EN translation pipeline loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load VI→EN model: {e}. Translation will return original text.")
            _vi_to_en_pipeline = None
    return _vi_to_en_pipeline


@lru_cache(maxsize=1)
def get_en_to_vi_pipeline():
    """Get or create EN→VI translation pipeline."""
    global _en_to_vi_pipeline
    if _en_to_vi_pipeline is None:
        try:
            from transformers import pipeline
            model_name = os.getenv("TRANSLATION_EN_VI_MODEL", "Helsinki-NLP/opus-mt-en-vi")
            cache_dir = str(_get_model_cache_dir())
            
            logger.info(f"Loading translation model: {model_name}")
            _en_to_vi_pipeline = pipeline(
                "translation",
                model=model_name,
                tokenizer=model_name,
                cache_dir=cache_dir,
                device=-1,  # CPU
            )
            logger.info("EN→VI translation pipeline loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load EN→VI model: {e}. Translation will return original text.")
            _en_to_vi_pipeline = None
    return _en_to_vi_pipeline


def detect_vietnamese(text: Optional[str]) -> bool:
    """
    Detect if text contains Vietnamese characters.
    
    Args:
        text: Input text
        
    Returns:
        True if Vietnamese characters detected, False otherwise
    """
    if not text:
        return False
    
    text_lower = text.lower()
    # Count Vietnamese characters
    vi_chars = VIETNAMESE_PATTERN.findall(text_lower)
    
    # Also check for common Vietnamese words
    common_vi_words = {
        "của", "và", "là", "có", "được", "trong", "cho", "với", "không", 
        "người", "này", "khi", "đã", "một", "những", "ra", "hay", "về",
        "theo", "từ", "bởi", "vào", "ở", "để", "như", "trên", "các",
        "ai", "ấy", "họ", "bạn", "tôi", "chúng", "nào", "ở đâu", "làm",
        "gì", "sao", "bao", "nhiêu", "mấy", "muốn", "cần", "phải"
    }
    
    words = set(text_lower.split())
    common_word_matches = len(words.intersection(common_vi_words))
    
    # Heuristic: if enough Vietnamese indicators
    if len(vi_chars) >= 2:
        return True
    if common_word_matches >= 2:
        return True
    if len(vi_chars) >= 1 and common_word_matches >= 1:
        return True
    
    return False


def translate_to_english(text: Optional[str]) -> str:
    """
    Translate Vietnamese text to English.
    
    Args:
        text: Vietnamese text
        
    Returns:
        English translation or original text if translation fails
    """
    if not text:
        return text
    
    pipeline = get_vi_to_en_pipeline()
    if pipeline is None:
        logger.warning("VI→EN translation unavailable, returning original text")
        return text
    
    try:
        result = pipeline(text, max_length=512)
        return result[0]["translation_text"]
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text


def translate_to_vietnamese(text: Optional[str]) -> str:
    """
    Translate English text to Vietnamese.
    
    Args:
        text: English text
        
    Returns:
        Vietnamese translation or original text if translation fails
    """
    if not text:
        return text
    
    pipeline = get_en_to_vi_pipeline()
    if pipeline is None:
        logger.warning("EN→VI translation unavailable, returning original text")
        return text
    
    try:
        result = pipeline(text, max_length=512)
        return result[0]["translation_text"]
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text


def warmup_models():
    """Pre-load translation models on service startup."""
    logger.info("Warming up translation models...")
    get_vi_to_en_pipeline()
    get_en_to_vi_pipeline()
    logger.info("Translation models warmed up")
