"""
Real model adapters for tracklet feature pipeline.

Architecture (strict_pipeline.py compliance):
  Appearance Embedding:
    SoliderKPRAppearanceEmbeddingAdapter — SOLIDER (Swin-L backbone) + KPR-style
    part-based embedding.  Uses timm Swin-Large as SOLIDER proxy until the
    official checkpoint is deployed. global_weight=0.55, part_weight=0.45.

  Attribute + Appearance Metadata:
    ZeroShotAttributeAdapter — CLIP ViT-L/14 zero-shot for gender + age_group.
    ZeroShotAppearanceMetadataAdapter — CLIP zero-shot for clothing taxonomy.
    (Attribute metadata model not specified in strict_pipeline.py; CLIP is used.)

  Semantic Embedding:
    ItselfSemanticEmbedder — proxy for ITSELF vision-language retrieval via
    CLIP text encoder.  Produces 768-dim vectors compatible with ITSELF retrieval.

  Hub:
    CLIPModelHub — lazy singleton, loads CLIP ViT-L/14 once per process.
    SwinLModelHub  — lazy singleton, loads timm Swin-Large once per process.

All adapters satisfy the Protocol contracts in tracklet_feature_pipeline.py.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2
import math
import numpy as np
from PIL import Image

from tracklet_feature_pipeline import (
    ActionClip,
    AppearanceAttributeResult,
    AppearanceEmbeddingResult,
    AttributeEmbeddingResult,
    BehaviorAnalysisResult,
    BoundingBox,
    FrameSelectionOutput,
    SemanticEmbeddingResult,
    StaticAttributeResult,
    TrackletFeatureInput,
    TrackletFrameObservation,
    EMBEDDING_VOCABULARY,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Taxonomy maps — internal labels only, never raw benchmark labels
# ---------------------------------------------------------------------------

_GENDER_PROMPTS = [
    ("male", ["a photo of a man", "a photo of a boy", "a photo of a male person"]),
    ("female", ["a photo of a woman", "a photo of a girl", "a photo of a female person"]),
]

_AGE_PROMPTS = [
    ("child", ["a photo of a child", "a photo of a young kid", "a photo of a toddler"]),
    ("adult", ["a photo of an adult person", "a photo of a middle-aged person"]),
    ("elderly", ["a photo of an elderly person", "a photo of an old person"]),
]

_SHIRT_PROMPTS = [
    ("black", ["a person wearing a black shirt or top"]),
    ("white", ["a person wearing a white shirt or top"]),
    ("gray",  ["a person wearing a gray shirt or top"]),
    ("red",   ["a person wearing a red shirt or top"]),
    ("blue",  ["a person wearing a blue shirt or top"]),
    ("green", ["a person wearing a green shirt or top"]),
    ("yellow",["a person wearing a yellow shirt or top"]),
    ("brown", ["a person wearing a brown shirt or top"]),
    ("orange",["a person wearing an orange shirt or top"]),
    ("purple",["a person wearing a purple shirt or top"]),
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
# Singleton CLIP hub
# ---------------------------------------------------------------------------

class CLIPModelHub:
    """Load CLIP once per process; thread-safe lazy init."""

    _instance: Optional[CLIPModelHub] = None
    _lock = threading.Lock()

    def __new__(cls) -> CLIPModelHub:
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
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor
                model_id = "openai/clip-vit-large-patch14"
                logger.info("Loading CLIP model %s …", model_id)
                self._processor = CLIPProcessor.from_pretrained(model_id)
                self._model = CLIPModel.from_pretrained(model_id)
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = self._model.to(self._device)
                self._model.eval()
                self._torch = torch
                self._loaded = True
                logger.info("CLIP loaded on %s", self._device)
            except Exception as exc:
                logger.error("CLIP load failed: %s", exc)
                raise

    def image_features(self, pil_images: list[Image.Image]) -> "np.ndarray":
        """Return L2-normalised image features [N, 768]."""
        self._ensure_loaded()
        import torch
        inputs = self._processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self._model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    def text_features(self, texts: list[str]) -> "np.ndarray":
        """Return L2-normalised text features [N, 768]."""
        self._ensure_loaded()
        import torch
        inputs = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    def classify_zero_shot(
        self,
        pil_image: Image.Image,
        label_prompts: list[tuple],
    ) -> str | None:
        """
        Zero-shot label selection.

        label_prompts: list of (label, [prompt, ...]) pairs.
        Returns the label with highest mean cosine similarity, or None
        if the top label maps to None.
        """
        self._ensure_loaded()
        img_feat = self.image_features([pil_image])[0]  # [768]
        best_label = None
        best_score = -float("inf")
        for label, prompts in label_prompts:
            text_feats = self.text_features(prompts)  # [P, 768]
            score = float(np.mean(text_feats @ img_feat))
            if score > best_score:
                best_score = score
                best_label = label
        return best_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crop_pil(frame: TrackletFrameObservation, full_bgr: np.ndarray | None) -> Image.Image | None:
    """Return a PIL crop for one frame, or None if crop_bgr missing."""
    if frame.crop_bgr is not None and frame.crop_bgr.size > 0:
        rgb = cv2.cvtColor(np.asarray(frame.crop_bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    if full_bgr is not None:
        h, w = full_bgr.shape[:2]
        b = frame.bbox
        x1 = max(0, b.x1); y1 = max(0, b.y1)
        x2 = min(w, b.x2); y2 = min(h, b.y2)
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


def _quality_weighted_pool(
    vectors: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    """L2-normalised quality-weighted mean of embedding vectors."""
    w = np.array([max(wt, 1e-6) for wt in weights], dtype=np.float32)
    w /= w.sum()
    mat = np.stack(vectors, axis=0)  # [N, D]
    fused = (mat * w[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(fused)) or 1.0
    return (fused / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# Attribute adapter
# ---------------------------------------------------------------------------

@dataclass
class ZeroShotAttributeAdapter:
    """
    CLIP zero-shot for gender + age_group.

    Uses the best available crop from selected frames and aggregates
    via majority vote weighted by frame quality score.
    """

    confidence_floor: float = 0.45

    def extract(
        self,
        tracklet: TrackletFeatureInput,
        selection: FrameSelectionOutput,
    ) -> StaticAttributeResult:
        hub = CLIPModelHub()
        selected = _selected_observations(tracklet, selection)
        quality_map = {item.frame_index: item.quality_score for item in selection.selected_frames}

        gender_votes: dict[str | None, float] = {}
        age_votes: dict[str | None, float] = {}

        for frame in selected:
            pil = _crop_pil(frame, None)
            if pil is None:
                continue
            weight = quality_map.get(frame.frame_index, 0.1)
            try:
                g = hub.classify_zero_shot(pil, _GENDER_PROMPTS)
                a = hub.classify_zero_shot(pil, _AGE_PROMPTS)
            except Exception:
                continue
            gender_votes[g] = gender_votes.get(g, 0.0) + weight
            age_votes[a] = age_votes.get(a, 0.0) + weight

        gender = max(gender_votes, key=gender_votes.get) if gender_votes else None
        age_group = max(age_votes, key=age_votes.get) if age_votes else None
        confidence = self.confidence_floor if (gender or age_group) else 0.0

        return StaticAttributeResult(
            gender=gender,
            age_group=age_group,
            confidence=round(confidence, 6),
        )


# ---------------------------------------------------------------------------
# Appearance metadata adapter
# ---------------------------------------------------------------------------

@dataclass
class ZeroShotAppearanceMetadataAdapter:
    """
    CLIP zero-shot for all appearance metadata fields.

    Aggregates per-field labels across selected frames using quality-weighted
    majority voting. Maps results to internal taxonomy.
    """

    def extract(
        self,
        tracklet: TrackletFeatureInput,
        selection: FrameSelectionOutput,
    ) -> AppearanceAttributeResult:
        hub = CLIPModelHub()
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
                try:
                    label = hub.classify_zero_shot(pil, prompts)
                except Exception:
                    label = None
                fields[field_name][label] = fields[field_name].get(label, 0.0) + weight

        def best(votes: dict) -> str | None:
            if not votes:
                return None
            label = max(votes, key=votes.get)
            return label  # may be None if "no X" won

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
# Swin-L hub (SOLIDER backbone proxy)
# ---------------------------------------------------------------------------

class SwinLModelHub:
    """
    Lazy singleton loading timm Swin-Large as SOLIDER backbone proxy.

    SOLIDER uses Swin Transformer trained on LUPerson. We load Swin-L with
    ImageNet-22k weights as a proxy until the official SOLIDER checkpoint
    is deployed.  Outputs 1536-dim features from the norm head.
    """

    _instance: Optional[SwinLModelHub] = None
    _lock = threading.Lock()

    def __new__(cls) -> SwinLModelHub:
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
            try:
                import torch
                import timm
                model_id = "swin_large_patch4_window12_384.ms_in22k"
                logger.info("Loading SOLIDER proxy backbone %s …", model_id)
                self._model = timm.create_model(model_id, pretrained=True, num_classes=0)
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = self._model.to(self._device).eval()
                data_cfg = timm.data.resolve_model_data_config(self._model)
                self._transform = timm.data.create_transform(**data_cfg, is_training=False)
                self._torch = torch
                self._embedding_dim = self._model.num_features
                self._loaded = True
                logger.info("SOLIDER proxy loaded (%d-dim) on %s", self._embedding_dim, self._device)
            except Exception as exc:
                logger.error("SOLIDER proxy load failed: %s", exc)
                raise

    def embed_crops(self, pil_crops: list[Image.Image]) -> np.ndarray:
        """Return L2-normalised embeddings [N, D] for a batch of crops."""
        self._ensure_loaded()
        import torch
        tensors = [self._transform(c) for c in pil_crops]
        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            feats = self._model(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# SOLIDER + KPR appearance embedding adapter
# ---------------------------------------------------------------------------

def _part_crops(pil_image: Image.Image) -> list[Image.Image]:
    """
    KPR-style body-part crops: full body + upper + lower.

    Returns [full, upper_half, lower_half] — matched with KPR's global +
    part-based embedding strategy.  Crops are resized to 384×384 to match
    Swin-L input resolution.
    """
    w, h = pil_image.size
    upper = pil_image.crop((0, 0, w, h // 2))
    lower = pil_image.crop((0, h // 2, w, h))
    target = (384, 384)
    return [
        pil_image.resize(target, Image.BILINEAR),
        upper.resize(target, Image.BILINEAR),
        lower.resize(target, Image.BILINEAR),
    ]


@dataclass
class SoliderKPRAppearanceEmbeddingAdapter:
    """
    SOLIDER + KPR appearance embedding — strict pipeline compliance.

    Backbone: timm Swin-Large (SOLIDER proxy, ImageNet-22k pretrained).
    Part strategy: global (full crop) + upper + lower body parts (KPR-style).
    Fusion: global_weight=0.55 * global_embed + part_weight=0.45 * mean(parts).
    Pooling: quality-weighted mean across selected frames, L2-normalised.

    embedding_dim: 1536 (Swin-L feature dimension; aligns with SOLIDER-REID).
    """

    embedding_model: str = "solider-kpr-swin-large-proxy"
    global_weight: float = 0.55
    part_weight: float = 0.45

    def extract(
        self,
        tracklet: TrackletFeatureInput,
        selection: FrameSelectionOutput,
        static_attributes: StaticAttributeResult,
        appearance_attributes: AppearanceAttributeResult,
    ) -> AppearanceEmbeddingResult:
        hub = SwinLModelHub()
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
            return self._fallback(selection, static_attributes, appearance_attributes)

        try:
            per_frame_vecs, per_frame_global = self._embed_frames(hub, pil_crops, weights)
        except Exception as exc:
            logger.error("SOLIDER embed failed: %s", exc)
            return self._fallback(selection, static_attributes, appearance_attributes)

        fused = _quality_weighted_pool(per_frame_vecs, weights)
        fused_tuple = tuple(round(float(v), 6) for v in fused.tolist())
        per_frame_tuples = [tuple(round(float(v), 6) for v in row) for row in per_frame_global]

        return AppearanceEmbeddingResult(
            embedding_model=self.embedding_model,
            embedding_vector=fused_tuple,
            tracklet_vectors=tuple(per_frame_tuples),
        )

    def _embed_frames(
        self,
        hub: SwinLModelHub,
        pil_crops: list[Image.Image],
        weights: list[float],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        all_parts: list[list[Image.Image]] = [_part_crops(pil) for pil in pil_crops]
        full_crops = [parts[0] for parts in all_parts]
        upper_crops = [parts[1] for parts in all_parts]
        lower_crops = [parts[2] for parts in all_parts]

        global_feats = hub.embed_crops(full_crops)   # [N, D]
        upper_feats = hub.embed_crops(upper_crops)   # [N, D]
        lower_feats = hub.embed_crops(lower_crops)   # [N, D]

        # KPR-style part fusion: mean of part embeddings, weighted by global
        per_frame_vecs: list[np.ndarray] = []
        for i in range(len(pil_crops)):
            part_mean = (upper_feats[i] + lower_feats[i]) / 2.0
            part_mean /= (float(np.linalg.norm(part_mean)) or 1.0)
            fused = self.global_weight * global_feats[i] + self.part_weight * part_mean
            fused /= (float(np.linalg.norm(fused)) or 1.0)
            per_frame_vecs.append(fused.astype(np.float32))

        return per_frame_vecs, [global_feats[i] for i in range(len(pil_crops))]

    def _fallback(
        self,
        selection: FrameSelectionOutput,
        static_attributes: StaticAttributeResult,
        appearance_attributes: AppearanceAttributeResult,
    ) -> AppearanceEmbeddingResult:
        seed = " ".join(filter(None, [
            static_attributes.gender, static_attributes.age_group,
            appearance_attributes.shirt, appearance_attributes.pants,
        ]))
        dim = 1536
        values = [0.0] * dim
        for i, ch in enumerate(seed):
            values[i % dim] += (ord(ch) % 97) / 97.0
        values[0] += selection.average_tracklet_score
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        vector = tuple(round(v / norm, 6) for v in values)
        return AppearanceEmbeddingResult(
            embedding_model=f"{self.embedding_model}_fallback",
            embedding_vector=vector,
            tracklet_vectors=(vector,),
        )


# ---------------------------------------------------------------------------
# Attribute embedding adapter (CLIP text projection of attribute tokens)
# ---------------------------------------------------------------------------

@dataclass
class CLIPAttributeEmbeddingAdapter:
    """
    Attribute embedding via CLIP text encoding of the attribute summary.

    Produces a 768-dim vector from the textual description of the tracklet's
    static attributes so downstream retrieval can match attributes by text.
    """

    embedding_model: str = "openai/clip-vit-large-patch14-attr"

    def extract(
        self,
        tracklet: TrackletFeatureInput,
        selection: FrameSelectionOutput,
        static_attributes: StaticAttributeResult,
    ) -> AttributeEmbeddingResult:
        hub = CLIPModelHub()
        parts = [p for p in [static_attributes.gender, static_attributes.age_group] if p]
        text = "a photo of a " + (" ".join(parts) if parts else "person")
        try:
            feats = hub.text_features([text])[0]  # [768]
            vector = tuple(round(float(v), 6) for v in feats.tolist())
        except Exception as exc:
            logger.error("CLIP text_features failed: %s", exc)
            vector = tuple(0.0 for _ in range(768))
        return AttributeEmbeddingResult(
            embedding_model=self.embedding_model,
            embedding_vector=vector,
        )


# ---------------------------------------------------------------------------
# Action adapter — heuristic + CLIP text for semantic embedding
# ---------------------------------------------------------------------------

@dataclass
class ItselfSemanticEmbedder:
    """
    ITSELF-compatible semantic embedding for action/behavior.

    ITSELF (Attention Guided Fine-Grained Alignment, WACV 2026) is the
    strict pipeline target for vision-language retrieval.  This adapter
    generates CLIP ViT-L/14 text embeddings that are compatible with the
    ITSELF retrieval space — both models share the same CLIP text backbone.
    When the ITSELF model endpoint is deployed on LightningAI, this adapter
    can be hot-swapped by pointing to that endpoint.
    """

    embedding_model: str = "itself-wacv2026-clip-proxy"
    vocabulary: tuple[str, ...] = EMBEDDING_VOCABULARY

    def embed(self, analysis: BehaviorAnalysisResult) -> SemanticEmbeddingResult:
        hub = CLIPModelHub()
        label_text = analysis.action_summary.replace("_", " ")
        # ITSELF-style fine-grained prompt: behavioural description
        text = f"a surveillance footage of a person who is {label_text} in an indoor space"
        try:
            feats = hub.text_features([text])[0]  # [768]
            vector = tuple(round(float(v), 6) for v in feats.tolist())
        except Exception:
            vector = tuple(0.0 for _ in range(768))
        return SemanticEmbeddingResult(
            clip_id=analysis.clip_id,
            embedding_model=self.embedding_model,
            vocabulary=self.vocabulary,
            embedding_vector=vector,
        )


# ---------------------------------------------------------------------------
# Factory: return production adapters, or log + fall back to heuristic
# ---------------------------------------------------------------------------

def build_production_adapters() -> dict:
    """
    Build strict-pipeline-compliant adapters.

    Appearance embedding: SoliderKPRAppearanceEmbeddingAdapter (timm Swin-L proxy).
    Attribute / metadata:  ZeroShotAttributeAdapter + ZeroShotAppearanceMetadataAdapter (CLIP).
    Semantic embedding:    ItselfSemanticEmbedder (CLIP text proxy for ITSELF space).

    Falls back to heuristic adapters only when torch/timm/transformers are absent.
    """
    from tracklet_feature_pipeline import (
        RuleBasedStaticAttributeExtractor,
        ColorAppearanceAttributeExtractor,
        VisualAttributeEmbeddingExtractor,
        VisualAppearanceEmbeddingExtractor,
        HeuristicBehaviorAnalyzer,
        ActionVocabularyEmbedder,
    )

    try:
        import transformers  # noqa: F401
        import torch          # noqa: F401
        import timm           # noqa: F401
        logger.info("torch + transformers + timm available — using production adapters")
        return {
            "static_attribute_extractor": ZeroShotAttributeAdapter(),
            "attribute_embedding_extractor": CLIPAttributeEmbeddingAdapter(),
            "appearance_attribute_extractor": ZeroShotAppearanceMetadataAdapter(),
            "appearance_embedding_extractor": SoliderKPRAppearanceEmbeddingAdapter(),
            "behavior_analyzer": HeuristicBehaviorAnalyzer(),
            "semantic_embedder": ItselfSemanticEmbedder(),
        }
    except Exception as exc:
        logger.warning("ML libs not available (%s) — using heuristic fallback adapters", exc)
        return {
            "static_attribute_extractor": RuleBasedStaticAttributeExtractor(),
            "attribute_embedding_extractor": VisualAttributeEmbeddingExtractor(),
            "appearance_attribute_extractor": ColorAppearanceAttributeExtractor(),
            "appearance_embedding_extractor": VisualAppearanceEmbeddingExtractor(),
            "behavior_analyzer": HeuristicBehaviorAnalyzer(),
            "semantic_embedder": ActionVocabularyEmbedder(),
        }
