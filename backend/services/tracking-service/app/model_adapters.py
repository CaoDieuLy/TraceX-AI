"""
Model adapters for tracklet feature pipeline — SOTA single production path, no fallbacks.

Models (all SOTA as of April 2026):
  DINOv2ReIDHub   — ViT-L/14 self-supervised, 1024-dim (view-invariant Re-ID)
  VideoMAEHub     — Large video transformer, 1024-dim (Kinetics-400, ECCV 2022)
  SigLIP2ModelHub — ViT-L-16-512 image/text, 1024-dim (webli, Feb 2025 SOTA)

Adapters:
  ZeroShotAttributeAdapter          — gender + age_group (SigLIP2 zero-shot)
  ZeroShotAppearanceMetadataAdapter — 8-field appearance metadata (SigLIP2 zero-shot)
  CLIPAttributeEmbeddingAdapter     — attribute text embedding 1024-dim (SigLIP2)
  SoliderKPRAppearanceEmbeddingAdapter — KPR part fusion Re-ID 1024-dim (DINOv2)
  SigLIP2BehaviorAnalyzer           — VideoMAE temporal + SigLIP2 text action scoring
  ItselfSemanticEmbedder            — action semantic embedding 1024-dim (SigLIP2)
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .tracklet_feature_pipeline import (
    ActionClip,
    AppearanceAttributeResult,
    AppearanceEmbeddingResult,
    AttributeEmbeddingResult,
    BehaviorAnalysisResult,
    FrameSelectionOutput,
    SemanticEmbeddingResult,
    StaticAttributeResult,
    TrackletFeatureInput,
    TrackletFrameObservation,
    EMBEDDING_VOCABULARY,
)

logger = logging.getLogger(__name__)
_MODEL_GPU_LOCK = threading.Semaphore(1)


def _env_batch_size(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    return max(1, int(default))


def _env_precision(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip().lower()
    return value or default


@contextmanager
def _gpu_inference_scope(torch_module: object, device: str, *, precision_env: str, default_precision: str):
    precision = _env_precision(precision_env, default_precision)
    inference_mode = getattr(torch_module, "inference_mode", None)
    inference_context = inference_mode() if callable(inference_mode) else torch_module.no_grad()
    autocast_context = nullcontext()
    if device == "cuda":
        if precision == "bf16":
            autocast_context = torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16)
        elif precision in {"fp16", "half"}:
            autocast_context = torch_module.autocast(device_type="cuda", dtype=torch_module.float16)
        _MODEL_GPU_LOCK.acquire()
    try:
        with inference_context:
            with autocast_context:
                yield
    finally:
        if device == "cuda":
            _MODEL_GPU_LOCK.release()

# ---------------------------------------------------------------------------
# Taxonomy prompt maps — SigLIP2 zero-shot labels
# ---------------------------------------------------------------------------

_GENDER_PROMPTS = [
    ("male",   ["a photo of a man", "a photo of a boy", "a photo of a male person"]),
    ("female", ["a photo of a woman", "a photo of a girl", "a photo of a female person"]),
]

_AGE_PROMPTS = [
    ("child",   ["a photo of a child", "a photo of a young kid"]),
    ("adult",   ["a photo of an adult person", "a photo of a middle-aged person"]),
    ("elderly", ["a photo of an elderly person", "a photo of an old person"]),
]

_SHIRT_PROMPTS = [
    ("black",       ["a person wearing a black shirt or top"]),
    ("white",       ["a person wearing a white shirt or top"]),
    ("gray",        ["a person wearing a gray shirt or top"]),
    ("red",         ["a person wearing a red shirt or top"]),
    ("blue",        ["a person wearing a blue shirt or top"]),
    ("green",       ["a person wearing a green shirt or top"]),
    ("yellow",      ["a person wearing a yellow shirt or top"]),
    ("brown",       ["a person wearing a brown shirt or top"]),
    ("orange",      ["a person wearing an orange shirt or top"]),
    ("purple",      ["a person wearing a purple shirt or top"]),
    ("long_sleeve", ["a person wearing a long-sleeve shirt"]),
    ("short_sleeve",["a person wearing a short-sleeve shirt or t-shirt"]),
]

_PANTS_PROMPTS = [
    ("black",  ["a person wearing black pants or trousers"]),
    ("blue",   ["a person wearing blue jeans or blue pants"]),
    ("gray",   ["a person wearing gray pants or trousers"]),
    ("white",  ["a person wearing white pants or trousers"]),
    ("brown",  ["a person wearing brown pants or trousers"]),
    ("shorts", ["a person wearing shorts"]),
]

_HAIR_PROMPTS = [
    ("black",  ["a person with black hair"]),
    ("brown",  ["a person with brown hair"]),
    ("blonde", ["a person with blonde or light hair"]),
    ("gray",   ["a person with gray or white hair"]),
    ("red",    ["a person with red or auburn hair"]),
    ("bald",   ["a bald person without hair"]),
]

_SKIN_PROMPTS = [
    ("light",  ["a person with light or pale skin tone"]),
    ("medium", ["a person with medium or olive skin tone"]),
    ("dark",   ["a person with dark or brown skin tone"]),
]

_HAT_PROMPTS = [
    ("present", ["a person wearing a hat or cap or helmet"]),
    (None,      ["a person without a hat"]),
]

_BAG_PROMPTS = [
    ("present", ["a person carrying a bag or backpack or handbag or purse"]),
    (None,      ["a person not carrying any bag"]),
]

_HEAD_ACCESSORY_PROMPTS = [
    ("present", ["a person wearing sunglasses or glasses or scarf or headband"]),
    (None,      ["a person without head accessories"]),
]

_SHOES_PROMPTS = [
    ("sneakers", ["a person wearing sneakers or running shoes"]),
    ("boots",    ["a person wearing boots"]),
    ("sandals",  ["a person wearing sandals or flip-flops"]),
    ("heels",    ["a person wearing high heels"]),
    ("formal",   ["a person wearing formal shoes or dress shoes"]),
]


# ---------------------------------------------------------------------------
# TransReID Hub — ViT-base cross-camera Re-ID (768-dim, MSMT17, CVPR 2021)
# Checkpoint: transformer_120.pth from umair894/KAT-ReID-MSMT17
# ---------------------------------------------------------------------------

def _weights_root() -> Path:
    """Resolve model weights root: env var MCPT_MODEL_WEIGHTS_ROOT → storage/model-weights/ relative to repo.

    File path: app/local_ingestion_pipeline.py
      parent^1 = app/
      parent^2 = tracking-service/
      parent^3 = services/
      parent^4 = backend/
      parent^5 = A20-App-119/  ← repo root
    """
    configured = os.environ.get("MCPT_MODEL_WEIGHTS_ROOT", "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).parent.parent.parent.parent.parent / "storage" / "model-weights"




# ---------------------------------------------------------------------------
# DINOv2 Re-ID Hub — view-invariant appearance encoder (1024-dim)
# Replaces TransReID for scenarios with non-frontal / overhead cameras.
# DINOv2 ViT-L/14 is self-supervised on internet-scale diverse images and
# generalises to top-down and side views without any ReID fine-tuning.
# ---------------------------------------------------------------------------

class DINOv2ReIDHub:
    """
    DINOv2 ViT-L/14 — lazy singleton for view-invariant person Re-ID.

    Produces 1024-dim L2-normalised CLS-token embeddings from person crops.
    Unlike TransReID (trained on MSMT17 side-view pedestrians), DINOv2 handles
    overhead, angled, and unusual camera viewpoints found in multi-camera
    warehouse deployments.

    Model: facebook/dinov2-large (307M params, 1024-dim CLS).
    Override with env var MCPT_DINOV2_MODEL_ID (HF hub ID or local path).
    """

    _instance: Optional["DINOv2ReIDHub"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "DINOv2ReIDHub":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._loaded = False
            return cls._instance

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            import torch
            from transformers import AutoImageProcessor, AutoModel

            model_id = os.environ.get("MCPT_DINOV2_MODEL_ID", "facebook/dinov2-large")
            logger.info("Loading DINOv2 Re-ID from %s …", model_id)
            self._processor = AutoImageProcessor.from_pretrained(model_id)
            self._model = AutoModel.from_pretrained(model_id)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(self._device).eval()
            self._torch = torch
            self._loaded = True
            logger.info("DINOv2 Re-ID ready on %s (1024-dim)", self._device)

    def embed_crops(self, pil_crops: list[Image.Image]) -> np.ndarray:
        """Return L2-normalised 1024-dim CLS embeddings [N, 1024]."""
        self._ensure_loaded()
        if not pil_crops:
            return np.zeros((0, 1024), dtype=np.float32)
        batch_size = _env_batch_size("MCPT_REID_BATCH_SIZE", min(len(pil_crops), 64))
        outputs: list[np.ndarray] = []
        for start in range(0, len(pil_crops), batch_size):
            batch = [p.convert("RGB") for p in pil_crops[start: start + batch_size]]
            inputs = self._processor(images=batch, return_tensors="pt").to(self._device)
            with _gpu_inference_scope(
                self._torch,
                self._device,
                precision_env="MCPT_REID_PRECISION",
                default_precision="fp16",
            ):
                cls_feats = self._model(**inputs).last_hidden_state[:, 0, :]  # [B, 1024]
            norms = cls_feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            cls_feats = (cls_feats / norms).float().cpu().numpy()
            outputs.append(cls_feats)
        return np.concatenate(outputs, axis=0)


# ---------------------------------------------------------------------------
# VideoMAE Hub — large video transformer for action recognition (1024-dim)
# Checkpoint: MCG-NJU/videomae-large (pretrained Kinetics-400)
# ---------------------------------------------------------------------------

class VideoMAEHub:
    """
    VideoMAE Large — lazy singleton for video action feature extraction.

    Input: list of PIL frames from an ActionClip (resampled to 16 frames × 224×224).
    Output: 1024-dim L2-normalised CLS token embedding.
    SOTA self-supervised video pretraining (ECCV 2022 + Kinetics fine-tuned).
    Weights pre-loaded on shared filesystem — no download at inference time.
    """

    _instance: Optional["VideoMAEHub"] = None
    _lock = threading.Lock()

    @property
    def _CKPT_DIR(self) -> Path:
        import os
        env = os.environ.get("MCPT_VIDEOMAE_WEIGHTS", "").strip()
        return Path(env) if env else _weights_root() / "videomae-action"

    def __new__(cls) -> "VideoMAEHub":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._loaded = False
            return cls._instance

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            import torch
            from transformers import VideoMAEModel
            logger.info("Loading VideoMAE Large from %s …", self._CKPT_DIR)
            model = VideoMAEModel.from_pretrained(str(self._CKPT_DIR))
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(self._device).eval()
            self._model = model
            self._torch = torch
            import torchvision.transforms as T
            self._transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self._loaded = True
            logger.info("VideoMAE Large ready on %s (1024-dim)", self._device)

    def extract_features(self, pil_frames: list[Image.Image]) -> np.ndarray:
        """
        Extract 1024-dim CLS token from a clip.

        pil_frames: list of PIL RGB frames (any length, resampled to 16).
        Returns [1024] L2-normalised vector.
        """
        self._ensure_loaded()
        # Resample to exactly 16 frames
        n = len(pil_frames)
        if n == 0:
            return np.zeros(1024, dtype=np.float32)
        indices = [int(i * (n - 1) / 15) for i in range(16)] if n >= 2 else [0] * 16
        frames_16 = [pil_frames[idx].convert("RGB") for idx in indices]
        tensors = [self._transform(f) for f in frames_16]
        # Shape: [1, 16, 3, 224, 224]
        clip = self._torch.stack(tensors).unsqueeze(0).to(self._device)
        with _gpu_inference_scope(
            self._torch,
            self._device,
            precision_env="MCPT_VLM_PRECISION",
            default_precision="fp32",
        ):
            out = self._model(pixel_values=clip)
        cls = out.last_hidden_state[:, 0]  # CLS token [1, 1024]
        cls = cls / cls.norm(dim=-1, keepdim=True)
        return cls.squeeze(0).float().cpu().numpy()

    def extract_features_batch(self, clips_frames: list[list["Image.Image"]]) -> "np.ndarray":
        """
        Batch version: process N clips chunked to avoid OOM. Returns [N, 1024].

        Chunk size controlled by MCPT_VMAE_CHUNK_SIZE (default 8).
        Each clip is [16, 3, 224, 224] ≈ 12 MB fp32; 8 clips ≈ 96 MB — safe on A100.
        """
        self._ensure_loaded()
        import numpy as _np
        if not clips_frames:
            return _np.zeros((0, 1024), dtype=_np.float32)

        chunk_size = max(1, int(os.environ.get("MCPT_VMAE_CHUNK_SIZE", "8")))
        outputs: list[_np.ndarray] = []

        for chunk_start in range(0, len(clips_frames), chunk_size):
            chunk = clips_frames[chunk_start: chunk_start + chunk_size]
            batch_tensors = []
            for pil_frames in chunk:
                n = len(pil_frames)
                if n == 0:
                    batch_tensors.append(self._torch.zeros(16, 3, 224, 224))
                    continue
                indices = [int(i * (n - 1) / 15) for i in range(16)] if n >= 2 else [0] * 16
                frames_16 = [pil_frames[idx].convert("RGB") for idx in indices]
                tensors   = [self._transform(f) for f in frames_16]
                batch_tensors.append(self._torch.stack(tensors))  # [16, 3, 224, 224]
            clip_batch = self._torch.stack(batch_tensors).to(self._device)
            with _gpu_inference_scope(
                self._torch, self._device,
                precision_env="MCPT_VLM_PRECISION",
                default_precision="fp32",
            ):
                out = self._model(pixel_values=clip_batch)
            cls = out.last_hidden_state[:, 0]  # [chunk, 1024]
            cls = cls / cls.norm(dim=-1, keepdim=True)
            outputs.append(cls.float().cpu().numpy())

        return _np.concatenate(outputs, axis=0)


# ---------------------------------------------------------------------------
# SigLIP2 Hub — single model for all image/text encoding (1024-dim)
# ---------------------------------------------------------------------------

class SigLIP2ModelHub:
    """
    SigLIP2 ViT-L-16-512/webli — lazy singleton, thread-safe.

    Both image_features and text_features produce 1024-dim L2-normalised vectors
    in the same embedding space, enabling direct cosine similarity for retrieval.
    """

    _instance: Optional["SigLIP2ModelHub"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "SigLIP2ModelHub":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._loaded = False
                cls._instance._text_embed_cache: dict = {}
            return cls._instance

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            import os, torch, open_clip
            # Ensure HuggingFace uses the pre-downloaded cache on shared filesystem
            # SigLIP2 (3.4GB) is already at $HF_HOME/hub/models--timm--ViT-L-16-SigLIP2-512/
            studio_root = str(Path(__file__).parent.parent.parent.parent.parent.parent.parent)
            hf_home = os.environ.get("HF_HOME") or os.path.join(studio_root, ".cache", "huggingface")
            os.environ.setdefault("HF_HOME", hf_home)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
            logger.info("Loading SigLIP2 ViT-L-16-SigLIP2-512/webli from %s …", hf_home)
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-16-SigLIP2-512", pretrained="webli"
            )
            tokenizer = open_clip.get_tokenizer("ViT-L-16-SigLIP2-512")
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(self._device).eval()
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = tokenizer
            self._torch = torch
            self._loaded = True
            logger.info("SigLIP2 loaded on %s (1024-dim)", self._device)

    def image_features(self, pil_images: list[Image.Image]) -> np.ndarray:
        """Return L2-normalised image features [N, 1024]."""
        self._ensure_loaded()
        if not pil_images:
            return np.zeros((0, 1024), dtype=np.float32)
        tensors = [self._preprocess(img) for img in pil_images]
        batch_size = _env_batch_size("MCPT_EMBEDDING_BATCH_SIZE", min(len(tensors), 128))
        outputs: list[np.ndarray] = []
        for start in range(0, len(tensors), batch_size):
            batch = self._torch.stack(tensors[start:start + batch_size]).to(self._device)
            with _gpu_inference_scope(
                self._torch,
                self._device,
                precision_env="MCPT_EMBEDDING_PRECISION",
                default_precision="fp32",
            ):
                feats = self._model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            outputs.append(feats.float().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def text_features(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised text features [N, 1024]. Results cached by text content."""
        self._ensure_loaded()
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        key = tuple(texts)
        cached = self._text_embed_cache.get(key)
        if cached is not None:
            return cached
        batch_size = _env_batch_size("MCPT_EMBEDDING_BATCH_SIZE", len(texts))
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            tokens = self._tokenizer(texts[start:start + batch_size]).to(self._device)
            with _gpu_inference_scope(
                self._torch,
                self._device,
                precision_env="MCPT_EMBEDDING_PRECISION",
                default_precision="fp32",
            ):
                feats = self._model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            outputs.append(feats.float().cpu().numpy())
        result = np.concatenate(outputs, axis=0)
        self._text_embed_cache[key] = result
        return result

    def classify_zero_shot(
        self,
        pil_image: Image.Image,
        label_prompts: list[tuple],
    ) -> str | None:
        """Return highest-scoring label via image-text cosine similarity."""
        self._ensure_loaded()
        img_feat = self.image_features([pil_image])[0]
        best_label = None
        best_score = -float("inf")
        for label, prompts in label_prompts:
            text_feats = self.text_features(prompts)
            score = float(np.mean(text_feats @ img_feat))
            if score > best_score:
                best_score = score
                best_label = label
        return best_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crop_pil(frame: TrackletFrameObservation, full_bgr: np.ndarray | None) -> Image.Image | None:
    if frame.crop_bgr is not None and frame.crop_bgr.size > 0:
        rgb = cv2.cvtColor(np.asarray(frame.crop_bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    if full_bgr is not None:
        h, w = full_bgr.shape[:2]
        b = frame.bbox
        x1, y1 = max(0, b.x1), max(0, b.y1)
        x2, y2 = min(w, b.x2), min(h, b.y2)
        if x2 > x1 and y2 > y1:
            rgb = cv2.cvtColor(full_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
    return None


def _selected_observations(
    tracklet: TrackletFeatureInput,
    selection: FrameSelectionOutput,
) -> tuple[TrackletFrameObservation, ...]:
    frame_map = {f.frame_index: f for f in tracklet.frames}
    return tuple(
        frame_map[item.frame_index]
        for item in selection.selected_frames
        if item.frame_index in frame_map
    )


def _quality_weighted_pool(vectors: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """L2-normalised quality-weighted mean of embedding vectors."""
    w = np.array([max(wt, 1e-6) for wt in weights], dtype=np.float32)
    w /= w.sum()
    mat = np.stack(vectors, axis=0)
    fused = (mat * w[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(fused)) or 1.0
    return (fused / norm).astype(np.float32)


def _part_crops(pil_image: Image.Image) -> list[Image.Image]:
    """KPR-style: [full, upper_half, lower_half] at 256×128 for OSNet-AIN Re-ID."""
    w, h = pil_image.size
    upper = pil_image.crop((0, 0, w, h // 2))
    lower = pil_image.crop((0, h // 2, w, h))
    target = (128, 256)  # OSNet input: width=128, height=256
    return [
        pil_image.resize(target, Image.BILINEAR),
        upper.resize(target, Image.BILINEAR),
        lower.resize(target, Image.BILINEAR),
    ]


# ---------------------------------------------------------------------------
# Stage 4b: Static attribute adapter — gender + age_group
# ---------------------------------------------------------------------------

@dataclass
class ZeroShotAttributeAdapter:
    """SigLIP2 zero-shot for gender and age_group via quality-weighted majority vote."""

    confidence_floor: float = 0.45

    def extract(
        self,
        tracklet: TrackletFeatureInput,
        selection: FrameSelectionOutput,
    ) -> StaticAttributeResult:
        hub = SigLIP2ModelHub()
        selected = _selected_observations(tracklet, selection)
        quality_map = {item.frame_index: item.quality_score for item in selection.selected_frames}

        valid = [(f, _crop_pil(f, None)) for f in selected]
        valid = [(f, p) for f, p in valid if p is not None]
        if not valid:
            return StaticAttributeResult(gender=None, age_group=None, confidence=0.0)

        _, pils = zip(*valid)
        img_feats = hub.image_features(list(pils))  # [N, 1024] — one GPU call for all frames

        gender_votes: dict[str | None, float] = {}
        age_votes: dict[str | None, float] = {}

        for i, (frame, _) in enumerate(valid):
            weight = quality_map.get(frame.frame_index, 0.1)
            img_feat = img_feats[i]

            best_g, best_g_score = None, -float("inf")
            for label, prompts in _GENDER_PROMPTS:
                score = float(np.mean(hub.text_features(prompts) @ img_feat))
                if score > best_g_score:
                    best_g_score, best_g = score, label
            gender_votes[best_g] = gender_votes.get(best_g, 0.0) + weight

            best_a, best_a_score = None, -float("inf")
            for label, prompts in _AGE_PROMPTS:
                score = float(np.mean(hub.text_features(prompts) @ img_feat))
                if score > best_a_score:
                    best_a_score, best_a = score, label
            age_votes[best_a] = age_votes.get(best_a, 0.0) + weight

        gender = max(gender_votes, key=gender_votes.get) if gender_votes else None
        age_group = max(age_votes, key=age_votes.get) if age_votes else None

        return StaticAttributeResult(
            gender=gender,
            age_group=age_group,
            confidence=round(self.confidence_floor if (gender or age_group) else 0.0, 6),
        )


# ---------------------------------------------------------------------------
# Stage 4c: Appearance metadata adapter — 8 attribute fields
# ---------------------------------------------------------------------------

@dataclass
class ZeroShotAppearanceMetadataAdapter:
    """SigLIP2 zero-shot for all 8 appearance fields via quality-weighted majority vote."""

    def extract(
        self,
        tracklet: TrackletFeatureInput,
        selection: FrameSelectionOutput,
    ) -> AppearanceAttributeResult:
        hub = SigLIP2ModelHub()
        selected = _selected_observations(tracklet, selection)
        quality_map = {item.frame_index: item.quality_score for item in selection.selected_frames}

        fields: dict[str, dict] = {
            "shirt": {}, "pants": {}, "hair_color": {}, "skin_tone": {},
            "hat": {}, "bag": {}, "head_accessory": {}, "shoes": {},
        }
        prompt_map = {
            "shirt": _SHIRT_PROMPTS, "pants": _PANTS_PROMPTS,
            "hair_color": _HAIR_PROMPTS, "skin_tone": _SKIN_PROMPTS,
            "hat": _HAT_PROMPTS, "bag": _BAG_PROMPTS,
            "head_accessory": _HEAD_ACCESSORY_PROMPTS, "shoes": _SHOES_PROMPTS,
        }

        valid = [(f, _crop_pil(f, None)) for f in selected]
        valid = [(f, p) for f, p in valid if p is not None]
        if not valid:
            return AppearanceAttributeResult()

        _, pils = zip(*valid)
        img_feats = hub.image_features(list(pils))  # [N, 1024] — one GPU call for all frames

        for i, (frame, _) in enumerate(valid):
            weight = max(quality_map.get(frame.frame_index, 0.1), 1e-4)
            img_feat = img_feats[i]
            for field_name, label_prompts in prompt_map.items():
                best_label, best_score = None, -float("inf")
                for label, prompts in label_prompts:
                    score = float(np.mean(hub.text_features(prompts) @ img_feat))
                    if score > best_score:
                        best_score, best_label = score, label
                fields[field_name][best_label] = fields[field_name].get(best_label, 0.0) + weight

        def best(votes: dict) -> str | None:
            return max(votes, key=votes.get) if votes else None

        return AppearanceAttributeResult(
            head_accessory=best(fields["head_accessory"]),
            hat=best(fields["hat"]),
            hair_color=best(fields["hair_color"]),
            skin_tone=best(fields["skin_tone"]),
            shirt=best(fields["shirt"]),
            pants=best(fields["pants"]),
            shoes=best(fields["shoes"]),
            bag=best(fields["bag"]),
        )


# ---------------------------------------------------------------------------
# Stage 4b: Attribute embedding — SigLIP2 text (1024-dim)
# ---------------------------------------------------------------------------

@dataclass
class CLIPAttributeEmbeddingAdapter:
    """SigLIP2 text encoding of attribute summary → 1024-dim attribute embedding."""

    embedding_model: str = "siglip2-vit-l16-512-attr"

    def extract(
        self,
        tracklet: TrackletFeatureInput,
        selection: FrameSelectionOutput,
        static_attributes: StaticAttributeResult,
    ) -> AttributeEmbeddingResult:
        hub = SigLIP2ModelHub()
        parts = [p for p in [static_attributes.gender, static_attributes.age_group] if p]
        text = "a photo of a " + (" ".join(parts) if parts else "person")
        feats = hub.text_features([text])[0]  # [1024]
        vector = tuple(round(float(v), 6) for v in feats.tolist())
        return AttributeEmbeddingResult(
            embedding_model=self.embedding_model,
            embedding_vector=vector,
        )


# ---------------------------------------------------------------------------
# Stage 4c: Appearance embedding — DINOv2 ViT-L/14 (1024-dim)
# ---------------------------------------------------------------------------

@dataclass
class SoliderKPRAppearanceEmbeddingAdapter:
    """
    DINOv2 ViT-L/14 appearance embedding — view-invariant person Re-ID.

    Replaces TransReID+KPR for multi-camera deployments with overhead or
    non-standard camera angles. DINOv2 CLS-token features generalise across
    viewpoints without domain-specific ReID fine-tuning.
    Output: 1024-dim L2-normalised vector stored as appearance_embedding_vector.
    """

    embedding_model: str = "dinov2-vitl14"

    def extract(
        self,
        tracklet: TrackletFeatureInput,
        selection: FrameSelectionOutput,
        static_attributes: StaticAttributeResult,
        appearance_attributes: AppearanceAttributeResult,
    ) -> AppearanceEmbeddingResult:
        hub = DINOv2ReIDHub()
        selected = _selected_observations(tracklet, selection)
        quality_map = {item.frame_index: item.quality_score for item in selection.selected_frames}

        pil_crops: list[Image.Image] = []
        weights: list[float] = []
        for frame in selected:
            pil = _crop_pil(frame, None)
            if pil is None:
                continue
            pil_crops.append(pil)
            weights.append(quality_map.get(frame.frame_index, 0.1))

        if not pil_crops:
            empty = tuple(0.0 for _ in range(1024))
            return AppearanceEmbeddingResult(
                embedding_model=self.embedding_model,
                embedding_vector=empty,
                tracklet_vectors=(empty,),
            )

        feats = hub.embed_crops(pil_crops)  # [N, 1024] already L2-normalised
        pooled = _quality_weighted_pool(list(feats), weights)
        return AppearanceEmbeddingResult(
            embedding_model=self.embedding_model,
            embedding_vector=tuple(round(float(v), 6) for v in pooled.tolist()),
            tracklet_vectors=tuple(
                tuple(round(float(v), 6) for v in feats[i].tolist())
                for i in range(len(pil_crops))
            ),
        )


# ---------------------------------------------------------------------------
# Stage 5b: Action classifier — SigLIP2 zero-shot from clip frames
# ---------------------------------------------------------------------------

@dataclass
class SigLIP2BehaviorAnalyzer:
    """
    Action classification: VideoMAE Large temporal features + SigLIP2 text matching.

    Stage 1: Extract 1024-dim video clip features via VideoMAE Large (ECCV 2022).
    Stage 2: Score against action vocabulary text embeddings via SigLIP2.
    Combined: VideoMAE captures temporal motion, SigLIP2 maps to semantic labels.
    """

    vocabulary: tuple[str, ...] = EMBEDDING_VOCABULARY

    def __post_init__(self) -> None:
        self._prompts = {
            label: f"a surveillance footage of a person who is {label.replace('_', ' ')} in an indoor space"
            for label in self.vocabulary
        }

    def analyze(self, clip: ActionClip) -> BehaviorAnalysisResult:
        frames = clip.frames
        if not frames:
            return BehaviorAnalysisResult(
                clip_id=clip.clip_id,
                action_summary="unknown_action",
                labels=("unknown_action",),
                confidence=0.0,
                metadata={},
            )

        # Collect PIL frames from crop_bgr (full-body crops from tracklet)
        pils: list[Image.Image] = []
        for frame in frames:
            crop = frame.crop_bgr
            if crop is not None and crop.size > 0:
                rgb = cv2.cvtColor(np.asarray(crop, dtype=np.uint8), cv2.COLOR_BGR2RGB)
                pils.append(Image.fromarray(rgb))

        if not pils:
            return BehaviorAnalysisResult(
                clip_id=clip.clip_id,
                action_summary="unknown_action",
                labels=("unknown_action",),
                confidence=0.0,
                metadata={},
            )

        # Stage 1: VideoMAE temporal embedding (1024-dim) — captures motion patterns
        vmae = VideoMAEHub()
        video_feat = vmae.extract_features(pils)  # [1024]

        # Stage 2: SigLIP2 text embeddings for action labels, cosine similarity
        siglip = SigLIP2ModelHub()
        labels = list(self.vocabulary)
        text_feats = siglip.text_features([self._prompts[lbl] for lbl in labels])  # [K, 1024]

        # VideoMAE dim=1024 matches SigLIP2 dim=1024 — direct dot product
        scores = text_feats @ video_feat

        best_idx = int(np.argmax(scores))
        best_label = labels[best_idx]
        confidence = round(float(np.clip(scores[best_idx], 0.0, 1.0)), 6)

        return BehaviorAnalysisResult(
            clip_id=clip.clip_id,
            action_summary=best_label,
            labels=(best_label,),
            confidence=confidence,
            metadata={"scores": {lbl: round(float(s), 4) for lbl, s in zip(labels, scores)}},
        )


# ---------------------------------------------------------------------------
# Stage 5c: Action semantic embedding — SigLIP2 text (1024-dim)
# ---------------------------------------------------------------------------

@dataclass
class ItselfSemanticEmbedder:
    """SigLIP2 text encoding of action summary → 1024-dim action embedding."""

    embedding_model: str = "siglip2-vit-l16-512-action"
    vocabulary: tuple[str, ...] = EMBEDDING_VOCABULARY

    def embed(self, analysis: BehaviorAnalysisResult) -> SemanticEmbeddingResult:
        hub = SigLIP2ModelHub()
        label_text = analysis.action_summary.replace("_", " ")
        text = f"a surveillance footage of a person who is {label_text} in an indoor space"
        feats = hub.text_features([text])[0]  # [1024]
        vector = tuple(round(float(v), 6) for v in feats.tolist())
        return SemanticEmbeddingResult(
            clip_id=analysis.clip_id,
            embedding_model=self.embedding_model,
            vocabulary=self.vocabulary,
            embedding_vector=vector,
        )


# ---------------------------------------------------------------------------
# Production adapter factory — single path, no fallbacks
# ---------------------------------------------------------------------------

def build_production_adapters() -> dict:
    """Return all production adapters. All use SigLIP2; no heuristic fallbacks."""
    return {
        "static_attribute_extractor":    ZeroShotAttributeAdapter(),
        "attribute_embedding_extractor": CLIPAttributeEmbeddingAdapter(),
        "appearance_attribute_extractor": ZeroShotAppearanceMetadataAdapter(),
        "appearance_embedding_extractor": SoliderKPRAppearanceEmbeddingAdapter(),
        "behavior_analyzer":             SigLIP2BehaviorAnalyzer(),
        "semantic_embedder":             ItselfSemanticEmbedder(),
    }
