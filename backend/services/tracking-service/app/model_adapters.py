"""
Model adapters for tracklet feature pipeline — single production path, no fallbacks.

All adapters use SigLIP2 ViT-L-16-512/webli (1024-dim) for both image and text
encoding, enabling direct cosine similarity between query text and crop embeddings.

Hub:
  SigLIP2ModelHub — lazy singleton, handles image_features, text_features,
                    and classify_zero_shot for all adapters.

Adapters (all SigLIP2-backed):
  ZeroShotAttributeAdapter          — gender + age_group zero-shot
  ZeroShotAppearanceMetadataAdapter — 8-field appearance zero-shot
  CLIPAttributeEmbeddingAdapter     — attribute text embedding (1024-dim)
  SoliderKPRAppearanceEmbeddingAdapter — KPR-style appearance embedding (1024-dim)
  ItselfSemanticEmbedder            — action semantic embedding (1024-dim)
  SigLIP2BehaviorAnalyzer           — zero-shot action classification from clip frames
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from tracklet_feature_pipeline import (
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
            return cls._instance

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            import torch
            import open_clip
            logger.info("Loading SigLIP2 ViT-L-16-SigLIP2-512/webli …")
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
        tensors = [self._preprocess(img) for img in pil_images]
        batch = self._torch.stack(tensors).to(self._device)
        with self._torch.no_grad():
            feats = self._model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    def text_features(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised text features [N, 1024]."""
        self._ensure_loaded()
        tokens = self._tokenizer(texts).to(self._device)
        with self._torch.no_grad():
            feats = self._model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

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
    """KPR-style: [full, upper_half, lower_half] at 512×512 for SigLIP2."""
    w, h = pil_image.size
    upper = pil_image.crop((0, 0, w, h // 2))
    lower = pil_image.crop((0, h // 2, w, h))
    target = (512, 512)
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

        gender_votes: dict[str | None, float] = {}
        age_votes: dict[str | None, float] = {}

        for frame in selected:
            pil = _crop_pil(frame, None)
            if pil is None:
                continue
            weight = quality_map.get(frame.frame_index, 0.1)
            g = hub.classify_zero_shot(pil, _GENDER_PROMPTS)
            a = hub.classify_zero_shot(pil, _AGE_PROMPTS)
            gender_votes[g] = gender_votes.get(g, 0.0) + weight
            age_votes[a] = age_votes.get(a, 0.0) + weight

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

        for frame in selected:
            pil = _crop_pil(frame, None)
            if pil is None:
                continue
            weight = max(quality_map.get(frame.frame_index, 0.1), 1e-4)
            for field_name, prompts in prompt_map.items():
                label = hub.classify_zero_shot(pil, prompts)
                fields[field_name][label] = fields[field_name].get(label, 0.0) + weight

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
# Stage 4c: Appearance embedding — SigLIP2 + KPR-style part fusion (1024-dim)
# ---------------------------------------------------------------------------

@dataclass
class SoliderKPRAppearanceEmbeddingAdapter:
    """
    SigLIP2 + KPR-style part appearance embedding (1024-dim).

    Part strategy: global (full crop) + upper + lower body crops.
    Fusion: 0.55 * global + 0.45 * mean(upper, lower), quality-weighted mean
    across selected frames, L2-normalised. Shares embedding space with text queries.
    """

    embedding_model: str = "siglip2-vit-l16-512-kpr"
    global_weight: float = 0.55
    part_weight: float = 0.45

    def extract(
        self,
        tracklet: TrackletFeatureInput,
        selection: FrameSelectionOutput,
        static_attributes: StaticAttributeResult,
        appearance_attributes: AppearanceAttributeResult,
    ) -> AppearanceEmbeddingResult:
        hub = SigLIP2ModelHub()
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

        per_frame_vecs, global_vecs = self._embed_frames(hub, pil_crops)
        fused = _quality_weighted_pool(per_frame_vecs, weights)
        fused_tuple = tuple(round(float(v), 6) for v in fused.tolist())
        per_frame_tuples = [tuple(round(float(v), 6) for v in row) for row in global_vecs]

        return AppearanceEmbeddingResult(
            embedding_model=self.embedding_model,
            embedding_vector=fused_tuple,
            tracklet_vectors=tuple(per_frame_tuples),
        )

    def _embed_frames(
        self,
        hub: SigLIP2ModelHub,
        pil_crops: list[Image.Image],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        all_parts = [_part_crops(pil) for pil in pil_crops]
        full_crops  = [p[0] for p in all_parts]
        upper_crops = [p[1] for p in all_parts]
        lower_crops = [p[2] for p in all_parts]

        global_feats = hub.image_features(full_crops)   # [N, 1024]
        upper_feats  = hub.image_features(upper_crops)  # [N, 1024]
        lower_feats  = hub.image_features(lower_crops)  # [N, 1024]

        per_frame_vecs: list[np.ndarray] = []
        for i in range(len(pil_crops)):
            part_mean = (upper_feats[i] + lower_feats[i]) / 2.0
            part_norm = float(np.linalg.norm(part_mean)) or 1.0
            part_mean /= part_norm
            fused = self.global_weight * global_feats[i] + self.part_weight * part_mean
            fused_norm = float(np.linalg.norm(fused)) or 1.0
            per_frame_vecs.append((fused / fused_norm).astype(np.float32))

        return per_frame_vecs, [global_feats[i] for i in range(len(pil_crops))]


# ---------------------------------------------------------------------------
# Stage 5b: Action classifier — SigLIP2 zero-shot from clip frames
# ---------------------------------------------------------------------------

@dataclass
class SigLIP2BehaviorAnalyzer:
    """
    Zero-shot action classification via SigLIP2 image-text matching.

    Extracts mean image features from up to 4 key frames per ActionClip,
    then ranks action vocabulary labels by cosine similarity with their
    SigLIP2 text embeddings. No heuristic motion rules.
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

        hub = SigLIP2ModelHub()
        step = max(1, len(frames) // 4)
        sampled = [frames[i] for i in range(0, len(frames), step)][:4]

        pils: list[Image.Image] = []
        for frame in sampled:
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

        img_feats = hub.image_features(pils)     # [N, 1024]
        mean_feat = img_feats.mean(axis=0)
        norm = float(np.linalg.norm(mean_feat))
        if norm > 1e-8:
            mean_feat /= norm

        labels = list(self.vocabulary)
        text_feats = hub.text_features([self._prompts[lbl] for lbl in labels])  # [K, 1024]
        scores = text_feats @ mean_feat

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
