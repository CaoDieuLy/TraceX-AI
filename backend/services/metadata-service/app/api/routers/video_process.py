"""Video processing pipeline for metadata-service — SOTA 2026 AI.

Full 7-stage pipeline (per video):
  1. Sample frames (uniform)
  2. RT-DETR R50 person detection (primary)
  3. BEVProjector (2D→3D via homography)
  4. MCBLT Hungarian cross-camera association
  5. DINOv2 ViT-L/14 appearance embedding (1024-dim)
  6. Qwen2-VL-7B-Instruct open-vocabulary attribute captioning
  7. VideoMAE V2 action classification

Batch pipeline (across cameras):
  - Stage 1-3: Run per video in parallel
  - Stage 4: ONE MCBLT call across all cameras
  - Stage 5-7: Per unified tracklet (cross-camera)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from ...services.model_warmup import get_model
from .video_process_schemas import (
    BatchProcessRequest,
    BatchProcessResponse,
    BatchVideoEntry,
    ProcessVideoRequest,
    ProcessVideoResponse,
    ProcessVideoStreamRequest,
    TrackletResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["video"])


def _get_positive_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default


DEFAULT_SAMPLE_INTERVAL = 15
DEFAULT_MIN_BBOX_AREA = 400
DEFAULT_BEV_MAX_DIST = 1.5
MAX_WORKERS = int(os.getenv("BATCH_MAX_WORKERS", "8"))
VLM_BATCH_SIZE = _get_positive_env_int("VLM_BATCH_SIZE", 1)
VLM_BATCH_MAX_NEW_TOKENS_PER_CROP = _get_positive_env_int(
    "VLM_BATCH_MAX_NEW_TOKENS_PER_CROP", 512
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    return torch.device("cuda")


def _video_to_frames(video_path: str, max_frames: int = 500) -> tuple[list[np.ndarray], float]:
    """Load frames from video file using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def _sample_frames_uniform(
    frames: list[np.ndarray], fps: float, sample_interval: int = 15
) -> list[tuple[int, np.ndarray]]:
    """Sample frames at uniform intervals with their indices."""
    sampled = []
    for frame_idx in range(0, len(frames), sample_interval):
        sampled.append((frame_idx, frames[frame_idx]))
    return sampled


# ---------------------------------------------------------------------------
# Stage 2: Person Detection (RT-DETR primary / GDINO legacy fallback)
# ---------------------------------------------------------------------------

def _detect_persons(frame: np.ndarray, threshold: float = 0.3) -> list[dict]:
    """Legacy single-frame GDINO detection — only called from batch error fallback."""
    model = get_model("gdino16")
    processor = get_model("gdino16_processor")
    if model is None or processor is None:
        logger.warning("Grounding DINO not available, returning empty detections")
        return []

    device = _get_device()
    dtype = torch.float16
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil_img, text="person.", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    autocast_ctx = torch.autocast("cuda", dtype=dtype)
    with torch.no_grad(), autocast_ctx:
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=threshold,
        text_threshold=threshold,
    )[0]

    detections = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        if score < threshold:
            continue
        if label.lower() != "person":
            continue
        x1, y1, x2, y2 = box.tolist()
        detections.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "score": float(score),
            "label": label,
        })

    return detections


def _detect_persons_rtdetr(
    frames: list[np.ndarray],
    threshold: float = 0.4,
    batch_size: int = 128,
) -> list[list[dict]] | None:
    """RT-DETR R50 person detection — primary detector when loaded.
    Returns None if not available (caller falls back to GDINO).
    batch_size=128 on A100 80GB (256 causes OOM).
    CPU preprocessing is pipelined with GPU inference via ThreadPoolExecutor.
    """
    from concurrent.futures import ThreadPoolExecutor
    model = get_model("rtdetr")
    processor = get_model("rtdetr_processor")
    if model is None or processor is None:
        return None

    person_ids: set = get_model("rtdetr_person_ids") or {0, 1}
    device = _get_device()
    dtype = torch.float16

    batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]

    def _preprocess(batch: list) -> tuple:
        sizes = [(f.shape[0], f.shape[1]) for f in batch]
        pil_imgs = [Image.fromarray(f[:, :, ::-1]) for f in batch]
        inputs = processor(images=pil_imgs, return_tensors="pt")
        return inputs, sizes

    all_dets: list[list[dict]] = []
    autocast_ctx = torch.autocast("cuda", dtype=dtype)

    with ThreadPoolExecutor(max_workers=2) as ex:
        # Submit first batch preprocessing
        futures = [ex.submit(_preprocess, b) for b in batches[:2]]

        for idx, batch in enumerate(batches):
            # Prefetch next+1 batch while current is on GPU
            if idx + 2 < len(batches):
                futures.append(ex.submit(_preprocess, batches[idx + 2]))

            inputs_raw, sizes = futures[idx].result()
            inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
                      for k, v in inputs_raw.items()}

            try:
                with torch.no_grad(), autocast_ctx:
                    outputs = model(**inputs)
                results = processor.post_process_object_detection(
                    outputs, threshold=threshold,
                    target_sizes=torch.tensor(sizes, device=device),
                )
            except Exception as exc:
                logger.warning("[rtdetr] batch failed: %s", exc)
                all_dets.extend([[] for _ in batch])
                continue

            for res in results:
                dets = []
                for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
                    if label.item() not in person_ids:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    dets.append({"bbox": [float(x1), float(y1), float(x2), float(y2)],
                                 "score": float(score), "label": "person"})
                all_dets.append(dets)

    return all_dets


def _detect_persons_batch(frames: list[np.ndarray], threshold: float = 0.25) -> list[list[dict]]:
    """Person detection: RT-DETR primary (fast), GDINO fallback."""
    rtdetr_result = _detect_persons_rtdetr(frames, threshold=max(threshold, 0.4))
    if rtdetr_result is not None:
        n_dets = sum(len(d) for d in rtdetr_result)
        logger.debug("[detect] RT-DETR: %d frames → %d detections", len(frames), n_dets)
        return rtdetr_result

    # GDINO fallback
    model = get_model("gdino16")
    processor = get_model("gdino16_processor")
    if model is None or processor is None:
        logger.error(
            "[detect] RT-DETR unavailable AND GDINO not loaded — "
            "returning 0 detections for %d frames. "
            "Check GPU OOM or model warmup logs.",
            len(frames),
        )
        return [[] for _ in frames]

    device = _get_device()
    dtype = torch.float16
    autocast_ctx = torch.autocast("cuda", dtype=dtype)

    def _preprocess(batch_frames: list[np.ndarray]):
        pil_imgs = []
        sizes = []
        for f in batch_frames:
            h, w = f.shape[:2]
            sizes.append((h, w))
            pil_imgs.append(Image.fromarray(f[:, :, ::-1]))
        texts = ["person."] * len(pil_imgs)
        inputs = processor(images=pil_imgs, text=texts, return_tensors="pt", padding=True)
        return inputs, sizes

    BATCH_SIZE = 64
    batches = [frames[i: i + BATCH_SIZE] for i in range(0, len(frames), BATCH_SIZE)]
    if not batches:
        return []

    all_dets: list[list[dict]] = []
    prefetch_exec = ThreadPoolExecutor(max_workers=1)

    prefetch_fut = prefetch_exec.submit(_preprocess, batches[0])
    try:
        for b_idx, batch in enumerate(batches):
            inputs_cpu, sizes = prefetch_fut.result()
            if b_idx + 1 < len(batches):
                prefetch_fut = prefetch_exec.submit(_preprocess, batches[b_idx + 1])

            try:
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs_cpu.items()}
                target_sizes = torch.tensor(sizes, dtype=torch.int64).to(device)
                with torch.no_grad(), autocast_ctx:
                    outputs = model(**inputs)
                results = processor.post_process_grounded_object_detection(
                    outputs, inputs["input_ids"],
                    box_threshold=threshold, text_threshold=threshold,
                    target_sizes=target_sizes,
                )
            except Exception:
                results = None

            if results is None:
                for f in batch:
                    all_dets.append(_detect_persons(f, threshold))
                continue

            for res in results:
                dets = []
                for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
                    if float(score) < threshold or label.lower() != "person":
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    dets.append({"bbox": [float(x1), float(y1), float(x2), float(y2)],
                                 "score": float(score), "label": label})
                all_dets.append(dets)
    finally:
        prefetch_exec.shutdown(wait=False)

    return all_dets


# ---------------------------------------------------------------------------
# Stage 3: BEV Projection
# ---------------------------------------------------------------------------

def _project_to_bev_single(
    detections: list[dict],
    camera_id: str,
    cal_path: str | None,
) -> list[dict]:
    """Project 2D bboxes to BEV for a single camera."""
    from shared.core.geometry import BEVProjector

    try:
        projector = BEVProjector(camera_id, cal_path)
    except Exception as exc:
        logger.warning("BEVProjector failed for %s: %s", camera_id, exc)
        for d in detections:
            d["bev_x"] = 0.0
            d["bev_y"] = 0.0
        return detections

    for det in detections:
        bbox = det["bbox"]
        bev_x, bev_y = projector.bbox_bottom_center_to_bev(*bbox)
        det["bev_x"] = bev_x
        det["bev_y"] = bev_y
        det["camera_id"] = camera_id

    return detections


# ---------------------------------------------------------------------------
# Per-video pipeline (stages 1-3: load → detect → BEV)
# ---------------------------------------------------------------------------

def _process_single_video(entry: BatchVideoEntry) -> dict:
    """Run stages 1-3 for one video: load frames, detect, project to BEV."""
    video_path = entry.video_path
    camera_id = entry.camera_id or f"cam_{entry.video_id.split('_')[0]}"

    try:
        frames, fps = _video_to_frames(video_path, max_frames=500)
    except Exception as exc:
        logger.warning("Failed to load video %s: %s", video_path, exc)
        return {
            "video_id": entry.video_id,
            "camera_id": camera_id,
            "detections": [],
            "error": str(exc),
        }

    if not frames:
        return {
            "video_id": entry.video_id,
            "camera_id": camera_id,
            "detections": [],
            "error": "No frames extracted",
        }

    sampled = _sample_frames_uniform(frames, fps, entry.sample_interval)
    all_detections: list[dict] = []

    sampled_imgs = [frame for _, frame in sampled]
    batch_dets = _detect_persons_batch(sampled_imgs, threshold=0.25)
    for (frame_idx, _frame), dets in zip(sampled, batch_dets):
        for det in dets:
            det["frame_idx"] = frame_idx
            det["timestamp"] = frame_idx / fps if fps > 0 else 0
            det["video_id"] = entry.video_id
        all_detections.extend(dets)

    cal_path = os.getenv("CAMERA_CALIBRATION_PATH")
    all_detections = _project_to_bev_single(all_detections, camera_id, cal_path)

    return {
        "video_id": entry.video_id,
        "camera_id": camera_id,
        "detections": all_detections,
        "fps": fps,
        "n_frames": len(frames),
    }


# ---------------------------------------------------------------------------
# Stage 4: MCBLT Cross-Camera Association (ONE call across all cameras)
# ---------------------------------------------------------------------------

def _mcblt_associate(
    detections_by_camera: dict[str, list[dict]],
    max_dist: float = 1.5,
) -> list[list[dict]]:
    """MCBLT cross-camera association using Hungarian matching."""
    from shared.core.mcblt import associate_cross_camera

    return associate_cross_camera(detections_by_camera, max_dist_meters=max_dist)


# ---------------------------------------------------------------------------
# Stage 5: DINOv2 Appearance Embedding
# ---------------------------------------------------------------------------

def _generate_dinov2_embeddings(
    frames: list[np.ndarray],
    bboxes: list[list[float]],
    tracklet_id: str,
) -> Optional[list[float]]:
    """Generate DINOv2 ViT-L/14 appearance embeddings (1024-dim)."""
    model = get_model("dinov2")
    processor = get_model("dinov2_processor")
    if model is None or processor is None:
        return None

    device = _get_device()
    dtype = torch.float16

    if not frames or not bboxes:
        return None

    n = min(5, len(frames))
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)

    pil_crops = []
    for idx in indices:
        frame = frames[idx]
        bbox = bboxes[min(idx, len(bboxes) - 1)]
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_h, crop_w = crop.shape[:2]
        max_dim = max(crop_h, crop_w)
        top = (max_dim - crop_h) // 2
        bottom = max_dim - crop_h - top
        left = (max_dim - crop_w) // 2
        right = max_dim - crop_w - left
        square = cv2.copyMakeBorder(
            crop, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        resized = cv2.resize(square, (224, 224), interpolation=cv2.INTER_LINEAR)
        pil_crops.append(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))

    if not pil_crops:
        return None

    with torch.no_grad():
        inputs = processor(images=pil_crops, return_tensors="pt")
        inputs = {
            k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
            for k, v in inputs.items()
        }
        feats = model(**inputs).pooler_output.float()  # [N, 1024] — stays on GPU

    avg = feats.mean(0)
    norm = avg.norm()
    if norm > 0:
        avg = avg / norm
    return avg.cpu().tolist()


# ---------------------------------------------------------------------------
# Stage 6: Legacy SigLIP 2 label maps (kept for reference / _legacy_ function only)
# ---------------------------------------------------------------------------

_AGE_RANGE_MAP: dict[str, str] = {
    "child person":       "child",
    "teenage person":     "teenager",
    "young adult person": "young_adult",
    "middle-aged person": "middle_aged",
    "elderly person":     "elderly",
}
_BAG_TYPE_MAP: dict[str, str] = {
    "person with backpack":     "backpack",
    "person with handbag":      "handbag",
    "person with shoulder bag": "shoulder_bag",
    "person with suitcase":     "suitcase",
    "person without bag":       "none",
}
_HAT_COLOR_MAP: dict[str, str] = {
    "person with red hat":    "red",
    "person with blue hat":   "blue",
    "person with black hat":  "black",
    "person with white hat":  "white",
    "person with gray hat":   "gray",
    "person with yellow hat": "yellow",
    "person without hat":     "none",
}
_HAIR_STYLE_MAP: dict[str, str] = {
    "person with short hair": "short",
    "person with long hair":  "long",
    "person with ponytail":   "ponytail",
    "person with tied hair":  "tied",
    "bald person":            "bald",
}
_HAIR_COLOR_MAP: dict[str, str] = {
    "person with black hair":  "black",
    "person with brown hair":  "brown",
    "person with blonde hair": "blonde",
    "person with gray hair":   "gray",
    "person with white hair":  "white",
}

def _legacy_run_siglip2_label_attributes(
    frames: list[np.ndarray],
    bbox: list[float],
) -> dict[str, Any]:
    """Legacy label-based SigLIP 2 attribute tagging — kept for debug/fallback only.
    Main pipeline uses _caption_crop_vlm() instead."""
    model = get_model("siglip2")
    processor = get_model("siglip2_processor")
    if model is None or processor is None:
        return _default_attributes()

    device = _get_device()
    dtype = torch.float16

    x1, y1, x2, y2 = map(int, bbox)
    h, w = frames[0].shape[:2] if frames else (1080, 1920)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frames[0][y1:y2, x1:x2] if frames else np.zeros((1, 1, 3), dtype=np.uint8)
    if crop.size == 0:
        return _default_attributes()

    crop_h, crop_w = crop.shape[:2]
    max_dim = max(crop_h, crop_w)
    top = (max_dim - crop_h) // 2
    bottom = max_dim - crop_h - top
    left = (max_dim - crop_w) // 2
    right = max_dim - crop_w - left
    square = cv2.copyMakeBorder(
        crop, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    resized = cv2.resize(square, (384, 384), interpolation=cv2.INTER_LINEAR)
    pil_crop = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    label_groups = {
        "top_color": [
            "red shirt", "blue shirt", "green shirt", "white shirt", "black shirt",
            "yellow shirt", "orange shirt", "purple shirt", "gray shirt", "brown shirt",
        ],
        "bottom_color": [
            "black pants", "blue jeans", "gray pants", "white pants", "brown pants",
            "black shorts", "gray shorts",
        ],
        "gender": ["man", "woman"],
        "bag": ["person carrying a bag", "person not carrying a bag"],
        "hat": ["person wearing a hat", "person not wearing a hat"],
        "age_range":      list(_AGE_RANGE_MAP.keys()),
        "hat_color":      list(_HAT_COLOR_MAP.keys()),
        "bag_type":       list(_BAG_TYPE_MAP.keys()),
        "is_wearing_mask": ["person wearing face mask", "person not wearing face mask"],
        "hair_style":     list(_HAIR_STYLE_MAP.keys()),
        "hair_color":     list(_HAIR_COLOR_MAP.keys()),
    }

    attributes: dict[str, str] = {}
    for attr_type, labels in label_groups.items():
        try:
            inputs = processor(
                text=labels, images=pil_crop,
                return_tensors="pt", padding=True
            )
            siglip_dtype = torch.float16
            inputs = {k: v.to(device, dtype=siglip_dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}
            siglip_ctx = torch.autocast("cuda", dtype=siglip_dtype)
            with torch.no_grad(), siglip_ctx:
                outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = torch.sigmoid(logits_per_image).squeeze()

            best_idx = int(probs.argmax())
            best_label = labels[best_idx]
            best_prob = float(probs[best_idx])

            if attr_type == "top_color":
                attributes["top_color"] = best_label.split()[0]
            elif attr_type == "bottom_color":
                attributes["bottom_color"] = best_label.split()[0]
            elif attr_type == "gender":
                attributes["gender"] = best_label.split()[0] if best_prob > 0.6 else "unknown"
            elif attr_type == "bag":
                attributes["bag"] = "carrying_bag" if "not" not in best_label else "no_bag"
            elif attr_type == "hat":
                attributes["hat"] = "wearing_hat" if "not" not in best_label else "no_hat"
            elif attr_type == "age_range":
                attributes["age_range"] = _AGE_RANGE_MAP.get(best_label, "unknown")
            elif attr_type == "hat_color":
                attributes["hat_color"] = _HAT_COLOR_MAP.get(best_label, "unknown")
            elif attr_type == "bag_type":
                attributes["bag_type"] = _BAG_TYPE_MAP.get(best_label, "unknown")
            elif attr_type == "is_wearing_mask":
                attributes["is_wearing_mask"] = "yes" if best_label == "person wearing face mask" else "no"
            elif attr_type == "hair_style":
                attributes["hair_style"] = _HAIR_STYLE_MAP.get(best_label, "unknown")
            elif attr_type == "hair_color":
                attributes["hair_color"] = _HAIR_COLOR_MAP.get(best_label, "unknown")

        except Exception as exc:
            logger.warning("SigLIP2 attribute failed for %s: %s", attr_type, exc)

    for key in ["top_color", "bottom_color", "gender", "bag", "hat",
                "age_range", "hat_color", "bag_type", "is_wearing_mask", "hair_style", "hair_color"]:
        if key not in attributes:
            attributes[key] = "unknown"

    return attributes


# ---------------------------------------------------------------------------
# Stage 7: VideoMAE V2 Action Classification
# ---------------------------------------------------------------------------

# Mapping from Kinetics-400 labels → TraceX simplified action taxonomy.
# Used by VideoMAE Kinetics fine-tuned model (400 classes) → 4TraceX classes.
_KINETICS_TO_SIMPLIFIED: dict[str, str] = {
    # Standing / static
    "standing": [
        "looking at person", "shaking hands", "applauding", "brushing teeth",
        "combing hair", "dancing", "fidgeting", "headbutting", "head massage",
        "holding baby", "hugging person", "kissing", "laughing", "looking at phone",
        "marching", "parade", "playing harmonica", "playing organ", "playing piano",
        "playing recorder", "playing violin", "playing accordion", "playing guitar",
        "playing drums", "playing cello", "singing", "tapping pen", "texting",
        "waving", "whistling", "wrestling", "yawning",
    ],
    # Walking
    "walking": [
        "walking the dog", "walking on stilts", "crossing street",
        "drumming fingers", "golf putting", "hula hooping", "juggling balls",
        "kicking soccer ball", "massaging back", "massaging feet", "massaging legs",
        "moving car", "moving trolley", "pushing car", "pushing cart",
        "pushing wheelchair", "shuffling cards", "sled dog racing",
        "sneaking", "snowkiting", "snowmobiling", "somersaulting",
        "speed walking", "strumming guitar", "surfing crowd", "tai chi",
        "tapping guitar", "tasting beer", "tasting food", "tasting wine",
        "throwing ball", "throwing discus", "throwing axe",
        "tickling", "tobogganing", "tossing salad", "towel snapping",
        "trapeze", "unboxing", "vault", "waiting in line", "walking on beam",
    ],
    # Running
    "running": [
        "running on treadmill", "sprinting", "jogging", "dribbling",
        "basketball", "burpee", "cartwheeling", "catching baseball",
        "catching cricket ball", "catching/disc throwing", "celebrating",
        "chopping wood", "climbing", "climbing rope", "climbing tree",
        "contact juggling", "crawling", "cricket batting", "croquet",
        "cutting pineapple", "diving", "dodgeball", "doing capoeira",
        "doing jigsaw puzzle", "dribbling basketball", "drop kicking",
        "exercising arm", "exercising with exercise ball", "faceplanting",
        "falling off bike", "falling off chair", "fencing", "flying disc",
        "freediving", "front raises", "golf driving", "hammering",
        "hand car wash", "hand washing", "headstands", "high kick",
        "hitball", "hitball with rake", "hockey", "horse race",
    ],
    # Sitting
    "sitting": [
        "sitting", "sitting on bed", "sitting on chair", "sitting on floor",
        "sitting on stairs", "sitting with something", "lying", "lying down",
        "sleeping", "taking a shower", "using computer", "typing",
        "writing", "reading", "drinking coffee", "drinking beer",
        "eating", "eating burger", "eating cake", "eating carrots",
        "eating chips", "eating corn", "eating doughnuts", "eating grapes",
        "eating hotdog", "eating ice cream", "eating spaghetti",
    ],
    "bending": [
        "bending back", "bending metal", "bowling", "clean and press",
        "clean and jerk", "cleaning floor", "cleaning gutters",
        "crouching", "curling (exercise)", "cutting nails",
        "cutting paper", "deadlifting", "digging", "dunking basketball",
    ],
    "carrying": [
        "carrying baby", "carrying cradles", "carrying rifle",
        "carrying something", "carrying water", "catching something",
    ],
    "pushing_pulling": [
        "pulling car", "pulling cart", "pulling rope", "pushing car",
        "pushing cart", "pushing wheelchair", "shovelling",
        "sweeping", "washing dishes", "washing windows",
    ],
    "sports": [
        "archery", "backflip", "badminton", "baseball batting",
        "basketball shooting", "bench pressing", "biking on trail",
        "billiards", "blowdrying hair", "blowing glass", "blowing leaves",
        "bobsledding", "body flying", "bodyweight stretching",
        "bouncing basketball", "bouncing on bouncy castle",
        "bouncing on trampoline", "breakdancing", "bungee jumping",
        "camel riding", "canoeing", "capoeira", "carrying baby",
        "catching/throwing baseball", "catching/flying disc",
        "chest press machine", "cheerleading", "chestfeeding",
        "chinning", "choirs", "chopping vegetables", "clapping",
        "climbing ladder", "climbing tree", "climbing wall",
        "counting money", "couple dancing", "cracking back",
        "cracking knuckles", "cricket bowling", "curling hair",
    ],
}

# Flatten mapping: kinetics_label → tracex_action
_KINETICS_MAP: dict[str, str] = {}
for tracex_action, kinetics_list in _KINETICS_TO_SIMPLIFIED.items():
    for k_label in kinetics_list:
        _KINETICS_MAP[k_label.lower()] = tracex_action

# Fallback: if no match, guess from logits distribution
_SIMPLIFIED_ACTIONS = ["standing", "walking", "running", "sitting", "bending", "carrying", "pushing_pulling", "sports"]


def _map_kinetics_to_tracex_action(kinetics_label: str) -> str:
    """Map a Kinetics-400 label to TraceX simplified action."""
    return _KINETICS_MAP.get(kinetics_label.lower(), "standing")


def _run_videomae_actions(
    frames: list[np.ndarray],
    bbox: list[float],
) -> str:
    """VideoMAE V2 action classification from tracklet frames.

    Uses VideoMAE fine-tuned on Kinetics-400 (400 classes).
    Maps Kinetics labels → TraceX simplified taxonomy:
      standing, walking, running, sitting, bending,
      carrying, pushing_pulling, sports
    """
    model = get_model("videomae")
    processor = get_model("videomae_processor")
    if model is None or processor is None:
        return "standing"

    device = _get_device()
    dtype = torch.float16

    if len(frames) < 2:
        return "standing"

    n = 16
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)

    crops = []
    for idx in indices:
        frame = frames[idx]
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            crops.append(np.zeros((224, 224, 3), dtype=np.uint8))
            continue
        crop = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        crops.append(resized)

    try:
        inputs = processor(crops, return_tensors="pt")
        inputs = {k: v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}
        vmae_ctx = torch.autocast("cuda", dtype=dtype)
        with torch.no_grad(), vmae_ctx:
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=5, dim=-1)

        # Try to map top predictions to TraceX actions
        if hasattr(model, "config") and hasattr(model.config, "id2label"):
            id2label = model.config.id2label
            for prob, idx in zip(top_probs[0].cpu(), top_indices[0].cpu()):
                kinetics_label = id2label.get(int(idx), "")
                tracex_action = _map_kinetics_to_tracex_action(kinetics_label)
                if tracex_action not in ("sports",):
                    return tracex_action
            return "sports"  # default fallback
        else:
            # No label mapping available — return from simplified
            return "standing"
    except Exception as exc:
        logger.warning("VideoMAE action failed: %s", exc)
        return "standing"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def _default_attributes() -> dict[str, Any]:
    return {
        "gender": "unknown", "age_range": "unknown",
        "upper_clothing_desc": None, "upper_clothing_color": "unknown",
        "upper_clothing_type": "unknown", "upper_clothing_conf": None,
        "lower_clothing_desc": None, "lower_clothing_color": "unknown",
        "lower_clothing_type": "unknown", "lower_clothing_conf": None,
        "shoes_desc": None, "shoes_color": "unknown", "shoes_conf": None,
        "bag_presence": "unknown", "bag_type": "unknown",
        "bag_desc": None, "bag_conf": None,
        "hat_presence": "unknown", "hat_color": "unknown",
        "hat_type": "unknown", "hat_desc": None, "hat_conf": None,
        "is_wearing_mask": "unknown", "mask_conf": None,
        "hair_style": "unknown", "hair_color": "unknown", "hair_conf": None,
        "appearance_summary": "person",
        # backward compat
        "top_color": "unknown", "bottom_color": "unknown",
        "bag": "unknown", "hat": "unknown",
    }


_VLM_PROMPT = """You are analyzing a person crop from a surveillance camera.
Describe this person's visible appearance accurately.
Do NOT choose from a fixed label list — use free text for clothing descriptions.
Return ONLY a valid JSON object with these exact fields:

{
  "gender": "man or woman or unknown",
  "gender_conf": 0.0,
  "age_range": "child or teenager or young_adult or middle_aged or elderly or unknown",
  "age_range_conf": 0.0,
  "upper_clothing_desc": "free text e.g. black suit jacket",
  "upper_clothing_color": "dominant color or unknown",
  "upper_clothing_type": "e.g. suit jacket or hoodie or t-shirt or vest or unknown",
  "upper_clothing_conf": 0.0,
  "lower_clothing_desc": "free text e.g. black formal trousers",
  "lower_clothing_color": "dominant color or unknown",
  "lower_clothing_type": "e.g. jeans or formal trousers or shorts or skirt or unknown",
  "lower_clothing_conf": 0.0,
  "shoes_desc": "free text or unknown",
  "shoes_color": "color or unknown",
  "shoes_conf": 0.0,
  "bag_presence": "yes or no or unknown",
  "bag_type": "e.g. backpack or handbag or suitcase or none or unknown",
  "bag_desc": "free text or none",
  "bag_conf": 0.0,
  "hat_presence": "yes or no or unknown",
  "hat_color": "color or none or unknown",
  "hat_type": "e.g. cap or hat or helmet or hood or none or unknown",
  "hat_desc": "free text or none",
  "hat_conf": 0.0,
  "is_wearing_mask": "yes or no or unknown",
  "mask_conf": 0.0,
  "hair_style": "short or long or ponytail or tied or bald or unknown",
  "hair_color": "color or unknown",
  "hair_conf": 0.0,
  "appearance_summary": "one concise sentence"
}

Use "unknown" for anything not clearly visible.
Do not infer gender or age from clothing alone.
Replace all 0.0 placeholders with your actual confidence (0.0–1.0)."""

_VLM_BATCH_PROMPT_TEMPLATE = """You are analyzing {n} person crops from surveillance cameras.
The images above show persons labeled (1) to ({n}) in order.
Describe each person's visible appearance accurately. Use free text for clothing descriptions.
Return ONLY a valid JSON array with exactly {n} objects in order (index 0 = person 1).
Each object must have the same fields as below:

{{
  "gender": "man or woman or unknown",
  "gender_conf": 0.0,
  "age_range": "child or teenager or young_adult or middle_aged or elderly or unknown",
  "age_range_conf": 0.0,
  "upper_clothing_desc": "free text",
  "upper_clothing_color": "dominant color or unknown",
  "upper_clothing_type": "e.g. suit jacket or hoodie or t-shirt or unknown",
  "upper_clothing_conf": 0.0,
  "lower_clothing_desc": "free text",
  "lower_clothing_color": "dominant color or unknown",
  "lower_clothing_type": "e.g. jeans or formal trousers or shorts or unknown",
  "lower_clothing_conf": 0.0,
  "shoes_desc": "free text or unknown",
  "shoes_color": "color or unknown",
  "shoes_conf": 0.0,
  "bag_presence": "yes or no or unknown",
  "bag_type": "backpack or handbag or none or unknown",
  "bag_desc": "free text or none",
  "bag_conf": 0.0,
  "hat_presence": "yes or no or unknown",
  "hat_color": "color or none or unknown",
  "hat_type": "cap or hat or helmet or none or unknown",
  "hat_desc": "free text or none",
  "hat_conf": 0.0,
  "is_wearing_mask": "yes or no or unknown",
  "mask_conf": 0.0,
  "hair_style": "short or long or ponytail or tied or bald or unknown",
  "hair_color": "color or unknown",
  "hair_conf": 0.0,
  "appearance_summary": "one concise sentence"
}}

Use "unknown" for anything not clearly visible. Replace 0.0 with actual confidence (0.0–1.0).
Return only the JSON array, no surrounding text."""


_GENDER_NORM   = {"male": "man", "man": "man", "female": "woman", "woman": "woman"}
_AGE_NORM      = {
    "young adult": "young_adult", "young_adult": "young_adult",
    "middle aged": "middle_aged", "middle-aged": "middle_aged", "middle_aged": "middle_aged",
    "teen": "teenager", "teenager": "teenager",
    "elder": "elderly", "elderly": "elderly",
    "child": "child",
}
_PRESENCE_NORM = {"yes": "yes", "no": "no", "true": "yes", "false": "no", "none": "no"}


def _parse_vlm_attrs(parsed: dict) -> dict:
    """Normalise a raw VLM JSON dict into the canonical attrs dict."""
    def _s(key: str, fallback: str = "unknown") -> str:
        v = parsed.get(key)
        return str(v).strip().lower() if v not in (None, "", "null") else fallback

    def _f(key: str) -> float | None:
        try:
            return float(parsed[key])
        except (KeyError, TypeError, ValueError):
            return None

    def _norm(val: str, mapping: dict) -> str:
        return mapping.get(val.lower().strip(), val) if val else "unknown"

    bag_pres = _norm(_s("bag_presence"), _PRESENCE_NORM)
    hat_pres = _norm(_s("hat_presence"), _PRESENCE_NORM)
    return {
        "gender":               _norm(_s("gender"), _GENDER_NORM),
        "gender_conf":          _f("gender_conf"),
        "age_range":            _norm(_s("age_range"), _AGE_NORM),
        "age_range_conf":       _f("age_range_conf"),
        "upper_clothing_desc":  parsed.get("upper_clothing_desc"),
        "upper_clothing_color": _s("upper_clothing_color"),
        "upper_clothing_type":  _s("upper_clothing_type"),
        "upper_clothing_conf":  _f("upper_clothing_conf"),
        "lower_clothing_desc":  parsed.get("lower_clothing_desc"),
        "lower_clothing_color": _s("lower_clothing_color"),
        "lower_clothing_type":  _s("lower_clothing_type"),
        "lower_clothing_conf":  _f("lower_clothing_conf"),
        "shoes_desc":           parsed.get("shoes_desc"),
        "shoes_color":          _s("shoes_color"),
        "shoes_conf":           _f("shoes_conf"),
        "bag_presence":         bag_pres,
        "bag_type":             _s("bag_type"),
        "bag_desc":             parsed.get("bag_desc"),
        "bag_conf":             _f("bag_conf"),
        "hat_presence":         hat_pres,
        "hat_color":            _s("hat_color"),
        "hat_type":             _s("hat_type"),
        "hat_desc":             parsed.get("hat_desc"),
        "hat_conf":             _f("hat_conf"),
        "is_wearing_mask":      _norm(_s("is_wearing_mask"), _PRESENCE_NORM),
        "mask_conf":            _f("mask_conf"),
        "hair_style":           _s("hair_style"),
        "hair_color":           _s("hair_color"),
        "hair_conf":            _f("hair_conf"),
        "appearance_summary":   parsed.get("appearance_summary") or "person",
        # backward compat
        "top_color":    _s("upper_clothing_color"),
        "bottom_color": _s("lower_clothing_color"),
        "bag": "no_bag"      if bag_pres == "no"  else ("carrying_bag" if bag_pres == "yes" else "unknown"),
        "hat": "no_hat"      if hat_pres == "no"  else ("wearing_hat"  if hat_pres == "yes" else "unknown"),
    }


def _extract_json_array(raw: str) -> list[dict]:
    """Extract the first complete JSON array from model output."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned, count=1).strip()

    start = cleaned.find("[")
    if start == -1:
        raise ValueError(f"JSON array not found: {cleaned[:200]}")

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                payload = cleaned[start:idx + 1]
                parsed = json.loads(payload)
                if not isinstance(parsed, list):
                    raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
                return parsed

    raise ValueError(f"JSON array incomplete: {cleaned[:200]}")


def _caption_crop_vlm(crop: "Image.Image") -> dict:
    """Generate open-vocabulary appearance attributes via Qwen2-VL-7B-Instruct (single crop)."""
    model = get_model("qwen2vl")
    processor = get_model("qwen2vl_processor")
    if model is None or processor is None:
        return _default_attributes()

    device = _get_device()
    try:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": crop},
            {"type": "text", "text": _VLM_PROMPT},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[crop], return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        input_len = inputs["input_ids"].shape[1]
        raw = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            logger.warning("[vlm] JSON not found in output: %s", raw[:200])
            return _default_attributes()

        return _parse_vlm_attrs(json.loads(json_match.group()))

    except Exception as exc:
        logger.warning("[vlm] _caption_crop_vlm failed: %s", exc)
        return _default_attributes()


def _caption_crops_vlm_batch(crops: list, batch_size: int = VLM_BATCH_SIZE) -> list:
    """Batch Qwen2-VL captioning — sends up to batch_size crops per call (~3-4x faster).

    Retries failed batches with smaller sub-batches before falling back to singles.
    """
    model = get_model("qwen2vl")
    processor = get_model("qwen2vl_processor")
    if model is None or processor is None:
        return [_default_attributes() for _ in crops]

    batch_size = max(1, batch_size)
    device = _get_device()
    results: list = []

    for i in range(0, len(crops), batch_size):
        batch = crops[i:i + batch_size]
        n = len(batch)

        if n == 1:
            results.append(_caption_crop_vlm(batch[0]))
            continue

        try:
            prompt = _VLM_BATCH_PROMPT_TEMPLATE.format(n=n)
            content = [{"type": "image", "image": c} for c in batch]
            content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=batch, return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=VLM_BATCH_MAX_NEW_TOKENS_PER_CROP * n,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )

            input_len = inputs["input_ids"].shape[1]
            raw = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

            parsed_list = _extract_json_array(raw)
            if not isinstance(parsed_list, list) or len(parsed_list) != n:
                raise ValueError(f"Expected {n} objects, got {len(parsed_list) if isinstance(parsed_list, list) else type(parsed_list)}")

            logger.debug("[vlm] batch(%d) OK at offset %d", n, i)
            for obj in parsed_list:
                results.append(_parse_vlm_attrs(obj))

        except Exception as exc:
            next_batch_size = min(max(1, batch_size // 2), max(1, (n + 1) // 2))
            if n > 1 and next_batch_size < n:
                logger.warning(
                    "[vlm] batch(%d) failed: %s — retrying with smaller batches of %d",
                    n,
                    exc,
                    next_batch_size,
                )
                results.extend(_caption_crops_vlm_batch(batch, batch_size=next_batch_size))
                continue

            logger.warning("[vlm] batch(%d) failed: %s — falling back to single crops", n, exc)
            for crop in batch:
                results.append(_caption_crop_vlm(crop))

    return results


def _build_appearance_summary(attrs: dict) -> str:
    # Prefer VLM-generated summary (open-vocabulary, accurate)
    vlm_summary = attrs.get("appearance_summary")
    if vlm_summary and str(vlm_summary).strip() and str(vlm_summary).strip().lower() != "person":
        return str(vlm_summary).strip()
    # Fallback: compose from VLM desc fields
    parts = []
    gender = (attrs.get("gender") or "").strip()
    upper = (attrs.get("upper_clothing_desc") or "").strip()
    lower = (attrs.get("lower_clothing_desc") or "").strip()
    if gender and gender != "unknown":
        parts.append(gender)
    if upper and upper != "unknown":
        parts.append(upper)
    if lower and lower != "unknown":
        parts.append(lower)
    return " ".join(parts) or "person"


def _extract_crop_for_vlm(frame: np.ndarray, bbox: list[float]) -> Optional[np.ndarray]:
    """Extract a square-padded 384×384 crop from frame for VLM input."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    max_dim = max(crop.shape[0], crop.shape[1])
    top = (max_dim - crop.shape[0]) // 2
    bottom = max_dim - crop.shape[0] - top
    left = (max_dim - crop.shape[1]) // 2
    right = max_dim - crop.shape[1] - left
    padded = cv2.copyMakeBorder(
        crop, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return cv2.resize(padded, (384, 384), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Per-video endpoint (single video, existing behaviour)
# ---------------------------------------------------------------------------

@router.post("/process", response_model=ProcessVideoResponse)
def process_video(req: ProcessVideoRequest) -> ProcessVideoResponse:
    """Process a single video (single-camera tracking pipeline)."""
    start = time.time()
    logger.info("Processing video: id=%s camera=%s", req.video_id, req.camera_id)

    video_path = req.video_path
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    camera_id = req.camera_id or "Camera_0000"

    try:
        frames, fps = _video_to_frames(video_path, max_frames=500)
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read video frames")

    if not frames:
        raise HTTPException(status_code=400, detail="No frames extracted from video")

    logger.info("Loaded %d frames (fps=%.1f)", len(frames), fps)

    sample_interval = req.sample_interval or DEFAULT_SAMPLE_INTERVAL
    sampled = _sample_frames_uniform(frames, fps, sample_interval)
    all_detections: list[dict] = []

    sampled_imgs = [frame for _, frame in sampled]
    batch_dets = _detect_persons_batch(sampled_imgs, threshold=0.25)
    for (frame_idx, _frame), dets in zip(sampled, batch_dets):
        for det in dets:
            det["frame_idx"] = frame_idx
            det["timestamp"] = frame_idx / fps if fps > 0 else 0
            det["video_id"] = req.video_id
        all_detections.extend(dets)

    logger.info("Detected %d person detections", len(all_detections))

    if not all_detections:
        return ProcessVideoResponse(
            video_id=req.video_id,
            camera_id=camera_id,
            tracklets=[],
            total_detections=0,
            processing_time_s=time.time() - start,
        )

    cal_path = os.getenv("CAMERA_CALIBRATION_PATH")
    all_detections = _project_to_bev_single(all_detections, camera_id, cal_path)

    detections_by_camera = {camera_id: all_detections}
    groups = _mcblt_associate(detections_by_camera, max_dist=req.bev_max_dist or DEFAULT_BEV_MAX_DIST)
    logger.info("MCBLT formed %d tracklet groups", len(groups))

    tracklets: list[TrackletResult] = []
    for group_idx, group in enumerate(groups):
        if len(group) < 1:
            continue

        group = sorted(group, key=lambda d: d.get("frame_idx", 0))
        start_frame = group[0].get("frame_idx", 0)
        end_frame = group[-1].get("frame_idx", len(frames) - 1)
        step = max(1, (end_frame - start_frame) // 16)
        tracklet_frames = frames[start_frame:end_frame + 1:step]
        if not tracklet_frames:
            tracklet_frames = [frames[min(start_frame, len(frames) - 1)]]

        mid_det = group[len(group) // 2]
        rep_bbox = mid_det["bbox"]
        rep_bev_x = mid_det.get("bev_x", 0.0)
        rep_bev_y = mid_det.get("bev_y", 0.0)

        embedding = _generate_dinov2_embeddings(
            tracklet_frames, [rep_bbox] * len(tracklet_frames),
            f"{req.video_id}_{group_idx}"
        )
        mid_frame = tracklet_frames[len(tracklet_frames) // 2]
        rep_crop_cv = _extract_crop_for_vlm(mid_frame, rep_bbox)
        rep_crop_pil = Image.fromarray(cv2.cvtColor(rep_crop_cv, cv2.COLOR_BGR2RGB)) if rep_crop_cv is not None \
            else Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8))
        attributes = _caption_crop_vlm(rep_crop_pil)
        action = _run_videomae_actions(tracklet_frames, rep_bbox)
        summary = _build_appearance_summary(attributes)

        tracklets.append(TrackletResult(
            tracklet_id=f"{req.video_id}_{camera_id}_{group_idx}",
            video_id=req.video_id,
            camera_id=camera_id,
            track_id=group_idx,
            start_time=group[0].get("timestamp", 0),
            end_time=group[-1].get("timestamp", 0),
            quality_score=float(mid_det.get("score", 0.5)),
            gender=attributes.get("gender", "unknown"),
            age_range=attributes.get("age_range", "unknown"),
            top_color=attributes.get("top_color", "unknown"),
            bottom_color=attributes.get("bottom_color", "unknown"),
            shoes_color=attributes.get("shoes_color", "unknown"),
            hat_color=attributes.get("hat_color", "unknown"),
            bag_type=attributes.get("bag_type", "unknown"),
            is_wearing_mask=attributes.get("is_wearing_mask", "unknown"),
            hair_style=attributes.get("hair_style", "unknown"),
            hair_color=attributes.get("hair_color", "unknown"),
            appearance_summary=summary,
            upper_clothing_desc=attributes.get("upper_clothing_desc"),
            upper_clothing_color=attributes.get("upper_clothing_color"),
            upper_clothing_type=attributes.get("upper_clothing_type"),
            upper_clothing_conf=attributes.get("upper_clothing_conf"),
            lower_clothing_desc=attributes.get("lower_clothing_desc"),
            lower_clothing_color=attributes.get("lower_clothing_color"),
            lower_clothing_type=attributes.get("lower_clothing_type"),
            lower_clothing_conf=attributes.get("lower_clothing_conf"),
            shoes_desc=attributes.get("shoes_desc"),
            shoes_type=attributes.get("shoes_type"),
            bag_presence=attributes.get("bag_presence"),
            bag_desc=attributes.get("bag_desc"),
            bag_conf=attributes.get("bag_conf"),
            hat_presence=attributes.get("hat_presence"),
            hat_type=attributes.get("hat_type"),
            hat_desc=attributes.get("hat_desc"),
            hat_conf=attributes.get("hat_conf"),
            mask_conf=attributes.get("mask_conf"),
            gender_conf=attributes.get("gender_conf"),
            age_range_conf=attributes.get("age_range_conf"),
            representative_bbox=[int(x) for x in rep_bbox],
            bev_x=rep_bev_x,
            bev_y=rep_bev_y,
            embedding_vector=embedding or [],
            action=action,
            occlusion_score=0.0,
            contributing_cameras=[camera_id],
            contributing_video_ids=[req.video_id],
        ))

    elapsed = time.time() - start
    logger.info("Processed %s: %d tracklets in %.1fs", req.video_id, len(tracklets), elapsed)

    return ProcessVideoResponse(
        video_id=req.video_id,
        camera_id=camera_id,
        tracklets=tracklets,
        total_detections=len(all_detections),
        processing_time_s=elapsed,
    )


# ---------------------------------------------------------------------------
# STREAMING endpoint — accepts video bytes directly (no disk download)
# ---------------------------------------------------------------------------

@router.post("/process/stream", response_model=ProcessVideoResponse)
async def process_video_stream(
    video_id: str = Form(...),
    camera_id: str | None = Form(None),
    source_filename: str | None = Form(None),
    sample_interval: int = Form(15),
    bev_max_dist: float = Form(1.5),
    video: UploadFile = File(...),
) -> ProcessVideoResponse:
    """
    Accept video bytes via multipart upload, write to a temp file,
    process, then delete the temp file.

    This lets ingest_service stream bytes directly from Google Drive
    to the GPU pipeline without caching the file on disk permanently.
    """
    import tempfile

    tmp_path: Path | None = None
    try:
        suffix = Path(source_filename or "video").suffix.lower() or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            while True:
                chunk = await video.read(1024 * 1024 * 50)  # 50 MB chunks
                if not chunk:
                    break
                tmp.write(chunk)

        logger.info("[stream] Received %s (%s), saved to %s", video_id, source_filename, tmp_path)

        # Delegate to the sync process handler (uses cv2 which is sync-only)
        import asyncio
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            _process_video_sync,
            str(tmp_path),
            video_id,
            camera_id,
            sample_interval,
            bev_max_dist,
        )
        return result

    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
                logger.info("[stream] Cleaned up temp file: %s", tmp_path)
            except OSError:
                pass


def _batch_extract_features(
    t_data: list,
    video_id: str,
) -> tuple[list, list, list, list, list, list]:
    """
    TRUE batch GPU inference: 1 call per model for ALL tracklets combined.
    DINOv2: stack all crops → 1 forward pass → split results.
    SigLIP: stack all crops → 1 forward pass → also returns per-attribute confidence.
    VideoMAE: stack all clips → 1 forward pass.
    A100 80GB can handle 150+ tracklets × 5 crops in one shot.

    Returns: (all_embeddings, all_attributes, all_attr_confs, all_actions, all_rep_crops)
      all_attr_confs[i]: dict with keys gender_conf, top_color_conf, shoes_conf, accessory_conf
      all_rep_crops[i]:  PIL Image (384×384) — representative crop for saving
    """
    import torch.nn.functional as F

    device = _get_device()
    dtype = torch.float16

    # ── Build crops for all tracklets ────────────────────────────────────────
    def _extract_crop(frame, bbox, size):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        max_dim = max(crop.shape[0], crop.shape[1])
        pad = cv2.copyMakeBorder(
            crop,
            (max_dim - crop.shape[0]) // 2, max_dim - crop.shape[0] - (max_dim - crop.shape[0]) // 2,
            (max_dim - crop.shape[1]) // 2, max_dim - crop.shape[1] - (max_dim - crop.shape[1]) // 2,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return cv2.resize(pad, (size, size), interpolation=cv2.INTER_LINEAR)

    # ── Representative crops (384×384) — built once, shared by SigLIP + storage ──
    all_rep_crops: list[Image.Image] = []
    for lt, t_idx, rep_bbox, t_frames in t_data:
        mid_idx = len(t_frames) // 2
        c = _extract_crop(t_frames[mid_idx], rep_bbox, 384) if t_frames else None
        all_rep_crops.append(
            Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) if c is not None
            else Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8))
        )

    # ── DINOv2 batch ──────────────────────────────────────────────────────────
    all_embeddings = []
    model_dino = get_model("dinov2")
    proc_dino = get_model("dinov2_processor")
    if model_dino and proc_dino:
        try:
            pil_crops, tracklet_slices = [], []
            for lt, t_idx, rep_bbox, t_frames in t_data:
                n = min(5, len(t_frames))
                indices = np.linspace(0, len(t_frames) - 1, n, dtype=int)
                start = len(pil_crops)
                for idx in indices:
                    c = _extract_crop(t_frames[idx], rep_bbox, 224)
                    if c is not None:
                        pil_crops.append(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                tracklet_slices.append((start, len(pil_crops)))

            if pil_crops:
                inputs = proc_dino(images=pil_crops, return_tensors="pt")
                inputs = {
                    k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
                    for k, v in inputs.items()
                }
                with torch.no_grad():
                    feats = model_dino(**inputs).pooler_output.float()  # [N, 1024]

                for start, end in tracklet_slices:
                    if end > start:
                        avg = feats[start:end].mean(0)
                        norm = avg.norm()
                        all_embeddings.append((avg / norm if norm > 0 else avg).tolist())
                    else:
                        all_embeddings.append(None)
            else:
                all_embeddings = [None] * len(t_data)
        except Exception as exc:
            logger.warning("[pipeline] DINOv2 batch failed: %s — falling back", exc)
            all_embeddings = [_generate_dinov2_embeddings(t[3], [t[2]] * len(t[3]), f"{video_id}_{t[1]}") for t in t_data]
    else:
        all_embeddings = [None] * len(t_data)

    # ── SigLIP image encoding — for text-image search embeddings only ────────
    img_feats = None
    model_sip = get_model("siglip2")
    proc_sip = get_model("siglip2_processor")
    if model_sip and proc_sip and t_data:
        try:
            img_inputs = proc_sip(images=all_rep_crops, return_tensors="pt", padding=True)
            img_inputs = {k: v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device)
                          for k, v in img_inputs.items()}
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
                img_feats = model_sip.get_image_features(**{k: v for k, v in img_inputs.items()
                                                             if k in ["pixel_values"]})
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)  # [N, D]
        except Exception as exc:
            logger.warning("[pipeline] SigLIP image encoding failed: %s", exc)
            img_feats = None

    # ── Qwen2-VL-7B-Instruct: open-vocabulary attribute captioning ────────────
    logger.info(
        "[vlm] captioning %d representative crops with batch_size=%d",
        len(all_rep_crops),
        VLM_BATCH_SIZE,
    )
    all_attributes: list[dict] = _caption_crops_vlm_batch(
        all_rep_crops,
        batch_size=VLM_BATCH_SIZE,
    )
    all_attr_confs: list[dict] = []
    for attrs in all_attributes:
        all_attr_confs.append({
            "gender_conf":       attrs.get("gender_conf"),
            "age_range_conf":    attrs.get("age_range_conf"),
            "top_color_conf":    attrs.get("upper_clothing_conf"),
            "bottom_color_conf": attrs.get("lower_clothing_conf"),
            "shoes_conf":        attrs.get("shoes_conf"),
            "accessory_conf":    max(
                attrs.get("bag_conf") or 0.0,
                attrs.get("hat_conf") or 0.0,
            ) or None,
            "hat_color_conf":    attrs.get("hat_conf"),
            "bag_type_conf":     attrs.get("bag_conf"),
            "mask_conf":         attrs.get("mask_conf"),
            "hair_style_conf":   attrs.get("hair_conf"),
            "hair_color_conf":   attrs.get("hair_conf"),
        })

    # ── VideoMAE TRUE BATCH: stack all tracklet clips → 1 forward pass ───────
    all_actions = []
    model_vmae = get_model("videomae")
    proc_vmae = get_model("videomae_processor")
    if model_vmae and proc_vmae and t_data:
        try:
            # Build 16-frame clip for every tracklet
            all_clips: list[list[np.ndarray]] = []
            for lt, t_idx, rep_bbox, t_frames in t_data:
                x1, y1, x2, y2 = (max(0, int(v)) for v in rep_bbox)
                n_f = len(t_frames)
                idx_list = np.linspace(0, n_f - 1, min(16, n_f), dtype=int)
                frames_224 = []
                for fi in idx_list:
                    f = t_frames[fi]
                    h, w = f.shape[:2]
                    x2c, y2c = min(w, x2), min(h, y2)
                    crop = f[y1:y2c, x1:x2c] if x2c > x1 and y2c > y1 else f
                    frames_224.append(cv2.resize(crop if crop.size > 0 else f,
                                                  (224, 224), interpolation=cv2.INTER_LINEAR))
                while len(frames_224) < 16:
                    frames_224.append(frames_224[-1] if frames_224 else np.zeros((224, 224, 3), dtype=np.uint8))
                all_clips.append(frames_224[:16])

            # Batch all clips: proc_vmae expects list-of-frames per video
            # Stack into [N, 16, H, W, C] then process
            vmae_dtype = torch.float16
            inputs = proc_vmae(all_clips, return_tensors="pt")
            inputs = {k: v.to(device=device, dtype=vmae_dtype) if v.is_floating_point() else v.to(device)
                       for k, v in inputs.items()}
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=vmae_dtype):
                outputs = model_vmae(**inputs)
            logits = outputs.logits.float()  # [N, num_classes]
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = probs.topk(1, dim=-1)
            top_idx = top_indices.squeeze(1).tolist()
            top_conf = top_probs.squeeze(1).tolist()
            id2label = getattr(model_vmae.config, "id2label", {})
            for idx, conf in zip(top_idx, top_conf):
                label = id2label.get(idx, "")
                action = _map_kinetics_to_tracex_action(label) if label else "unknown"
                all_actions.append((action, float(conf), label))
        except Exception as exc:
            logger.warning("[pipeline] VideoMAE true-batch failed: %s — fallback", exc)
            all_actions = [(_run_videomae_actions(t[3], t[2]), 0.0, "") for t in t_data]
    else:
        all_actions = [("unknown", 0.0, "")] * len(t_data)

    # Capture SigLIP2 image embeddings (already computed above as img_feats)
    # These are in the same embedding space as SigLIP2 text queries → usable for text search
    all_siglip_embeddings: list[list[float]] = []
    try:
        if img_feats is not None:
            all_siglip_embeddings = [img_feats[i].cpu().float().tolist() for i in range(len(t_data))]
        else:
            all_siglip_embeddings = [[]] * len(t_data)
    except Exception:
        all_siglip_embeddings = [[]] * len(t_data)

    logger.warning("[pipeline] batch features done: %d tracklets | DINOv2=%d | SigLIP=%d | VideoMAE=%d",
                   len(t_data), sum(1 for e in all_embeddings if e),
                   sum(1 for a in all_attributes if a.get("gender") != "unknown"),
                   sum(1 for a in all_actions if a != "unknown"))
    return all_embeddings, all_attributes, all_attr_confs, all_actions, all_rep_crops, all_siglip_embeddings


def _process_video_sync(
    video_path: str,
    video_id: str,
    camera_id: str | None,
    sample_interval: int,
    bev_max_dist: float,
    presampled_frames=None,
) -> ProcessVideoResponse:
    """
    Sync video processing pipeline using BodyPartAdaptiveTracker + 4fps sampling.
    presampled_frames: pre-decoded frames from background thread (skips Stage 1).
    """
    import time
    from .tracking_pipeline import (
        VideoFrameSampler, BodyPartAdaptiveTracker, TrackletQualityScorer,
        TrackletFragmentMerger, FrameDetection, _crop_from_bbox,
    )
    start = time.time()
    camera_id = camera_id or "Camera_0000"

    # Stage 1: Sample frames at 4fps (skip if pre-decoded externally)
    if presampled_frames is not None:
        sampled_frames = presampled_frames
    else:
        sampler = VideoFrameSampler(sample_fps=4)
        try:
            sampled_frames = sampler.sample(video_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot read video: {e}")

    if not sampled_frames:
        raise HTTPException(status_code=400, detail="No frames extracted from video")

    logger.warning("[pipeline] %s: %d frames sampled at 4fps", video_id, len(sampled_frames))

    # Stage 2: Batch detect — RT-DETR primary, GDINO fallback
    from .tracking_pipeline import _crop_from_bbox as _tcrop
    t_det_start = time.time()
    detector = "RT-DETR" if get_model("rtdetr") is not None else "GDINO-fallback"
    logger.info("[pipeline] %s: running %s detection on %d frames...", video_id, detector, len(sampled_frames))
    all_batch_dets = _detect_persons_batch([sf.image for sf in sampled_frames], threshold=0.25)
    logger.warning("[pipeline] %s: %s done in %.1fs", video_id, detector, time.time() - t_det_start)

    detections_by_frame: dict[int, list[FrameDetection]] = {}
    total_raw = 0
    for sf, raw_dets in zip(sampled_frames, all_batch_dets):
        frame_dets: list[FrameDetection] = []
        for d in raw_dets:
            bbox = tuple(int(x) for x in d["bbox"])
            crop = _tcrop(sf.image, bbox)
            frame_dets.append(FrameDetection(
                frame_index=sf.frame_index,
                timestamp_second=sf.timestamp_second,
                bbox=bbox,
                confidence=float(d["score"]),
                laplacian_score=sf.laplacian_score,
                crop_bgr=crop,
            ))
        if frame_dets:
            detections_by_frame[sf.frame_index] = frame_dets
            total_raw += len(frame_dets)

    logger.warning("[pipeline] %s: %d detections across %d frames", video_id, total_raw, len(detections_by_frame))

    import torch as _torch

    if not detections_by_frame:
        logger.warning("[pipeline] %s: no persons detected", video_id)
        return ProcessVideoResponse(
            video_id=video_id, camera_id=camera_id,
            tracklets=[], total_detections=0,
            processing_time_s=time.time() - start,
        )

    # Stage 3: Track with BodyPartAdaptiveTracker
    tracker = BodyPartAdaptiveTracker(
        track_thresh=0.30,
        low_thresh=0.10,
        new_track_threshold=0.30,
        min_track_frames=2,
        min_track_density=0.03,
    )
    local_tracklets = tracker.track(video_id, camera_id, detections_by_frame)
    logger.warning("[pipeline] %s: %d raw tracklets from tracker", video_id, len(local_tracklets))

    # Stage 4: Quality filter
    scorer = TrackletQualityScorer(
        min_confidence=0.25,
        min_frames=2,
        min_density=0.03,
        min_duration_s=0.25,
        min_laplacian=5.0,
    )
    quality_results = {t.track_id: scorer.score(t) for t in local_tracklets}
    accepted = [t for t in local_tracklets if quality_results[t.track_id].accepted]
    rejected_reasons = {}
    for t in local_tracklets:
        q = quality_results[t.track_id]
        if not q.accepted:
            r = q.rejection_reason or "unknown"
            rejected_reasons[r] = rejected_reasons.get(r, 0) + 1
    logger.warning("[pipeline] %s: %d accepted, %d rejected %s",
                   video_id, len(accepted), len(local_tracklets) - len(accepted), rejected_reasons)

    # Stage 5-7: TRUE batch feature extraction — 1 GPU call per model for ALL tracklets
    frame_lookup = {sf.frame_index: sf.image for sf in sampled_frames}

    t_data = []
    for t_idx, lt in enumerate(accepted):
        obs = lt.observations
        mid = obs[len(obs) // 2]
        rep_bbox_float = [float(x) for x in mid.bbox]
        t_frames = [frame_lookup[o.frame_index] for o in obs if o.frame_index in frame_lookup] or [sampled_frames[0].image]
        t_data.append((lt, t_idx, rep_bbox_float, t_frames))

    all_embeddings, all_attributes, all_attr_confs, all_actions, all_rep_crops, all_siglip_embeddings = _batch_extract_features(t_data, video_id)

    # Stage 8: Post-hoc fragment merging via embedding cosine similarity.
    # Short tracklets broken by occlusion are re-joined when appearance similarity
    # is high and the temporal gap is within the threshold.
    import numpy as _np

    _merge_embs = (
        all_siglip_embeddings
        if any(len(e) > 0 for e in all_siglip_embeddings)
        else all_embeddings
    )
    _merger = TrackletFragmentMerger(similarity_threshold=0.85, max_gap_seconds=60.0)
    _orig_accepted = list(accepted)
    accepted, _groups = _merger.merge(list(accepted), _merge_embs)

    _n_merged = sum(len(g) - 1 for g in _groups if len(g) > 1)
    if _n_merged > 0:
        logger.warning(
            "[pipeline] %s: fragment merger joined %d fragment(s) → %d tracklets",
            video_id, _n_merged, len(accepted),
        )

    def _pool_avg(vecs: list) -> list:
        valid = [v for v in vecs if v]
        if not valid:
            return []
        return _np.array(valid, dtype=_np.float32).mean(axis=0).tolist()

    def _richest(g: list[int]) -> int:
        """Return index of the fragment with most observations (best appearance data)."""
        return max(g, key=lambda i: len(_orig_accepted[i].observations))

    all_embeddings        = [_pool_avg([all_embeddings[i]        for i in g]) for g in _groups]
    all_siglip_embeddings = [_pool_avg([all_siglip_embeddings[i] for i in g]) for g in _groups]
    all_attributes        = [all_attributes[_richest(g)]  for g in _groups]
    all_attr_confs        = [all_attr_confs[_richest(g)]  for g in _groups]
    all_actions           = [all_actions[_richest(g)]     for g in _groups]
    all_rep_crops         = [all_rep_crops[_richest(g)]   for g in _groups]

    # Rebuild t_data aligned to merged accepted list
    t_data = []
    for new_idx, mt in enumerate(accepted):
        obs = mt.observations
        mid = obs[len(obs) // 2]
        rep_bbox_float = [float(x) for x in mid.bbox]
        t_frames = (
            [frame_lookup[o.frame_index] for o in obs if o.frame_index in frame_lookup]
            or [sampled_frames[0].image]
        )
        t_data.append((mt, new_idx, rep_bbox_float, t_frames))

    # Project representative bboxes to BEV coordinates
    cal_path = os.getenv("CAMERA_CALIBRATION_PATH")
    _bev_inputs = [{"bbox": rep_bbox_float} for _, _, rep_bbox_float, _ in t_data]
    _bev_inputs = _project_to_bev_single(_bev_inputs, camera_id, cal_path)

    _CROPS_DIR = Path("/workspace/storage/crops")
    _CROPS_DIR.mkdir(parents=True, exist_ok=True)

    tracklets: list[TrackletResult] = []
    for t_idx, (lt, _, rep_bbox_float, t_frames) in enumerate(t_data):
        obs = lt.observations
        attributes = all_attributes[t_idx]
        embedding = all_embeddings[t_idx]
        siglip_emb = all_siglip_embeddings[t_idx] if t_idx < len(all_siglip_embeddings) else []
        action_tuple = all_actions[t_idx]
        if isinstance(action_tuple, tuple) and len(action_tuple) >= 3:
            action, action_conf, kinetics_raw = str(action_tuple[0]), float(action_tuple[1]), str(action_tuple[2])
        elif isinstance(action_tuple, tuple):
            action, action_conf, kinetics_raw = str(action_tuple[0]), float(action_tuple[1]), ""
        else:
            action, action_conf, kinetics_raw = str(action_tuple), 0.0, ""
        summary = _build_appearance_summary(attributes)
        attr_conf = all_attr_confs[t_idx]

        # Save representative crop image
        crop_url = ""
        try:
            rep_crop = all_rep_crops[t_idx]
            crop_filename = f"{video_id}_{camera_id}_{t_idx}.jpg"
            crop_path = _CROPS_DIR / crop_filename
            rep_crop.save(str(crop_path), "JPEG", quality=85)
            crop_url = f"/static/crops/{crop_filename}"
        except Exception as exc:
            logger.warning("[pipeline] Failed to save crop for %s_%s_%d: %s", video_id, camera_id, t_idx, exc)

        quality = quality_results[lt.track_id]
        tracklets.append(TrackletResult(
            tracklet_id=f"{video_id}_{camera_id}_{t_idx}",
            video_id=video_id,
            camera_id=camera_id,
            track_id=t_idx,
            start_time=obs[0].timestamp_second,
            end_time=obs[-1].timestamp_second,
            quality_score=quality.average_confidence,
            gender=attributes.get("gender", "unknown"),
            age_range=attributes.get("age_range", "unknown"),
            top_color=attributes.get("top_color", "unknown"),
            bottom_color=attributes.get("bottom_color", "unknown"),
            shoes_color=attributes.get("shoes_color", "unknown"),
            hat_color=attributes.get("hat_color", "unknown"),
            bag_type=attributes.get("bag_type", "unknown"),
            is_wearing_mask=attributes.get("is_wearing_mask", "unknown"),
            hair_style=attributes.get("hair_style", "unknown"),
            hair_color=attributes.get("hair_color", "unknown"),
            appearance_summary=summary,
            crop_url=crop_url,
            upper_clothing_desc=attributes.get("upper_clothing_desc"),
            upper_clothing_color=attributes.get("upper_clothing_color"),
            upper_clothing_type=attributes.get("upper_clothing_type"),
            upper_clothing_conf=attributes.get("upper_clothing_conf"),
            lower_clothing_desc=attributes.get("lower_clothing_desc"),
            lower_clothing_color=attributes.get("lower_clothing_color"),
            lower_clothing_type=attributes.get("lower_clothing_type"),
            lower_clothing_conf=attributes.get("lower_clothing_conf"),
            shoes_desc=attributes.get("shoes_desc"),
            shoes_type=attributes.get("shoes_type"),
            bag_presence=attributes.get("bag_presence"),
            bag_desc=attributes.get("bag_desc"),
            bag_conf=attributes.get("bag_conf"),
            hat_presence=attributes.get("hat_presence"),
            hat_type=attributes.get("hat_type"),
            hat_desc=attributes.get("hat_desc"),
            hat_conf=attributes.get("hat_conf"),
            representative_bbox=[int(x) for x in rep_bbox_float],
            bev_x=_bev_inputs[t_idx].get("bev_x", 0.0),
            bev_y=_bev_inputs[t_idx].get("bev_y", 0.0),
            embedding_vector=embedding or [],
            siglip_embedding=siglip_emb or [],
            action=action,
            action_confidence=action_conf,
            kinetics_label=kinetics_raw,
            occlusion_score=0.0,
            gender_conf=attr_conf.get("gender_conf"),
            top_color_conf=attr_conf.get("top_color_conf"),
            bottom_color_conf=attr_conf.get("bottom_color_conf"),
            shoes_conf=attr_conf.get("shoes_conf"),
            accessory_conf=attr_conf.get("accessory_conf"),
            age_range_conf=attr_conf.get("age_range_conf"),
            hat_color_conf=attr_conf.get("hat_color_conf"),
            bag_type_conf=attr_conf.get("bag_type_conf"),
            mask_conf=attr_conf.get("mask_conf"),
            hair_style_conf=attr_conf.get("hair_style_conf"),
            hair_color_conf=attr_conf.get("hair_color_conf"),
            contributing_cameras=[camera_id],
            contributing_video_ids=[video_id],
        ))

    elapsed = time.time() - start
    logger.warning("[pipeline] %s: %d tracklets saved in %.1fs", video_id, len(tracklets), elapsed)

    return ProcessVideoResponse(
        video_id=video_id,
        camera_id=camera_id,
        tracklets=tracklets,
        total_detections=total_raw,
        processing_time_s=elapsed,
    )


# ---------------------------------------------------------------------------
# BATCH endpoint — cross-camera processing (STAGES 1-7 across all cameras)
# ---------------------------------------------------------------------------

@router.post("/batch/process", response_model=BatchProcessResponse)
def process_batch(req: BatchProcessRequest) -> BatchProcessResponse:
    """
    Batch cross-camera processing pipeline.

    Takes up to 100 videos from different cameras (same timestamp),
    runs per-video detection + BEV in parallel, then ONE MCBLT call
    across all cameras, then DINOv2 + Qwen2-VL + VideoMAE per unified tracklet.

    Returns unified cross-camera tracklets with global IDs.
    """
    start = time.time()
    n_videos = len(req.videos)

    if n_videos == 0:
        raise HTTPException(status_code=400, detail="Empty batch: no videos provided")

    batch_id = req.batch_id or f"batch_{int(time.time())}"

    logger.info(
        "[%s] Batch processing %d videos: %s",
        batch_id,
        n_videos,
        [v.camera_id or v.video_id for v in req.videos],
    )

    # ---- Stage 1-3: Parallel per-video detection + BEV projection ----
    per_video_results: list[dict] = []
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, n_videos)) as executor:
        futures = {
            executor.submit(_process_single_video, entry): entry.video_id
            for entry in req.videos
        }
        for future in as_completed(futures):
            result = future.result()
            per_video_results.append(result)
            if result.get("error"):
                errors.append(f"{result['video_id']}: {result['error']}")

    # Collect detections by camera_id for MCBLT
    detections_by_camera: dict[str, list[dict]] = {}
    camera_stats: dict[str, int] = {}
    all_frames: dict[str, list[np.ndarray]] = {}  # video_id -> frames
    all_fps: dict[str, float] = {}

    for result in per_video_results:
        cam_id = result.get("camera_id", "unknown")
        dets = result.get("detections", [])

        if cam_id not in detections_by_camera:
            detections_by_camera[cam_id] = []

        # Attach per-video frames for later embedding extraction
        video_id = result["video_id"]
        try:
            frames, fps = _video_to_frames(
                next(e.video_path for e in req.videos if e.video_id == video_id),
                max_frames=500,
            )
            all_frames[video_id] = frames
            all_fps[video_id] = fps
        except Exception:
            all_frames[video_id] = []
            all_fps[video_id] = 30.0

        detections_by_camera[cam_id].extend(dets)
        camera_stats[cam_id] = len(dets)

    total_detections = sum(len(d) for d in detections_by_camera.values())
    logger.info(
        "[%s] Detection complete: %d total detections across %d cameras. Stats: %s",
        batch_id, total_detections, len(detections_by_camera), camera_stats,
    )

    if not detections_by_camera or total_detections == 0:
        return BatchProcessResponse(
            batch_id=batch_id,
            n_videos=n_videos,
            n_cameras=len(detections_by_camera),
            total_detections=0,
            n_tracklets=0,
            tracklets=[],
            processing_time_s=time.time() - start,
            camera_stats=camera_stats,
        )

    # ---- Stage 4: ONE MCBLT call across all cameras ----
    max_dist = 1.5
    for entry in req.videos:
        if entry.bev_max_dist:
            max_dist = entry.bev_max_dist
            break

    groups = _mcblt_associate(detections_by_camera, max_dist=max_dist)
    logger.info("[%s] MCBLT formed %d cross-camera groups", batch_id, len(groups))

    # ---- Stages 5-7: Per unified tracklet → DINOv2 + Qwen2-VL + VideoMAE ----
    tracklets: list[TrackletResult] = []
    tracklet_id_prefix = f"{batch_id}_tracklet"

    for group_idx, group in enumerate(groups):
        if len(group) < 1:
            continue

        # Collect frames from all contributing videos
        contributing_vids = list(set(d.get("video_id", "") for d in group))
        contributing_cams = list(set(d.get("camera_id", "") for d in group))

        # Gather all relevant frames
        all_track_frames: list[np.ndarray] = []
        all_track_bboxes: list[list[float]] = []

        for vid in contributing_vids:
            frames = all_frames.get(vid, [])
            if not frames:
                continue
            for det in group:
                if det.get("video_id") != vid:
                    continue
                fidx = det.get("frame_idx", 0)
                if 0 <= fidx < len(frames):
                    all_track_frames.append(frames[fidx])
                    all_track_bboxes.append(det["bbox"])

        if not all_track_frames:
            continue

        # Representative detection: highest score across group
        rep_det = max(group, key=lambda d: d.get("score", 0))
        rep_bbox = rep_det["bbox"]
        rep_bev_x = rep_det.get("bev_x", 0.0)
        rep_bev_y = rep_det.get("bev_y", 0.0)
        rep_cam = rep_det.get("camera_id", "unknown")
        rep_vid = rep_det.get("video_id", contributing_vids[0] if contributing_vids else "unknown")

        # DINOv2 embedding
        embedding = _generate_dinov2_embeddings(
            all_track_frames, all_track_bboxes,
            f"{tracklet_id_prefix}_{group_idx}",
        )

        # VLM open-vocabulary attribute captioning
        mid_frame = all_track_frames[len(all_track_frames) // 2]
        rep_crop_cv = _extract_crop_for_vlm(mid_frame, rep_bbox)
        rep_crop_pil = (
            Image.fromarray(cv2.cvtColor(rep_crop_cv, cv2.COLOR_BGR2RGB))
            if rep_crop_cv is not None
            else Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8))
        )
        attributes = _caption_crop_vlm(rep_crop_pil)

        # VideoMAE action
        action = _run_videomae_actions(all_track_frames, rep_bbox)

        # Time range from group
        timestamps = [d.get("timestamp", 0) for d in group if d.get("timestamp") is not None]
        start_time = min(timestamps) if timestamps else 0.0
        end_time = max(timestamps) if timestamps else 0.0

        summary = _build_appearance_summary(attributes)
        global_tracklet_id = f"{tracklet_id_prefix}_{group_idx}"

        tracklets.append(TrackletResult(
            tracklet_id=global_tracklet_id,
            video_id=rep_vid,
            camera_id=rep_cam,
            track_id=group_idx,
            start_time=start_time,
            end_time=end_time,
            quality_score=float(rep_det.get("score", 0.5)),
            gender=attributes.get("gender", "unknown"),
            age_range=attributes.get("age_range", "unknown"),
            top_color=attributes.get("top_color", "unknown"),
            bottom_color=attributes.get("bottom_color", "unknown"),
            shoes_color=attributes.get("shoes_color", "unknown"),
            hat_color=attributes.get("hat_color", "unknown"),
            bag_type=attributes.get("bag_type", "unknown"),
            is_wearing_mask=attributes.get("is_wearing_mask", "unknown"),
            hair_style=attributes.get("hair_style", "unknown"),
            hair_color=attributes.get("hair_color", "unknown"),
            appearance_summary=summary,
            upper_clothing_desc=attributes.get("upper_clothing_desc"),
            upper_clothing_color=attributes.get("upper_clothing_color"),
            upper_clothing_type=attributes.get("upper_clothing_type"),
            upper_clothing_conf=attributes.get("upper_clothing_conf"),
            lower_clothing_desc=attributes.get("lower_clothing_desc"),
            lower_clothing_color=attributes.get("lower_clothing_color"),
            lower_clothing_type=attributes.get("lower_clothing_type"),
            lower_clothing_conf=attributes.get("lower_clothing_conf"),
            shoes_desc=attributes.get("shoes_desc"),
            shoes_type=attributes.get("shoes_type"),
            bag_presence=attributes.get("bag_presence"),
            bag_desc=attributes.get("bag_desc"),
            bag_conf=attributes.get("bag_conf"),
            hat_presence=attributes.get("hat_presence"),
            hat_type=attributes.get("hat_type"),
            hat_desc=attributes.get("hat_desc"),
            hat_conf=attributes.get("hat_conf"),
            gender_conf=attributes.get("gender_conf"),
            age_range_conf=attributes.get("age_range_conf"),
            mask_conf=attributes.get("mask_conf"),
            hair_style_conf=attributes.get("hair_conf"),
            hair_color_conf=attributes.get("hair_conf"),
            representative_bbox=[int(x) for x in rep_bbox],
            bev_x=rep_bev_x,
            bev_y=rep_bev_y,
            embedding_vector=embedding or [],
            action=action,
            occlusion_score=0.0,
            contributing_cameras=sorted(set(contributing_cams)),
            contributing_video_ids=sorted(contributing_vids),
        ))

    elapsed = time.time() - start
    logger.info(
        "[%s] Batch complete: %d tracklets (from %d detections, %d cameras) in %.1fs",
        batch_id, len(tracklets), total_detections, len(detections_by_camera), elapsed,
    )

    if errors:
        logger.warning("[%s] %d video(s) had errors: %s", batch_id, len(errors), errors)

    return BatchProcessResponse(
        batch_id=batch_id,
        n_videos=n_videos,
        n_cameras=len(detections_by_camera),
        total_detections=total_detections,
        n_tracklets=len(tracklets),
        tracklets=tracklets,
        processing_time_s=elapsed,
        camera_stats=camera_stats,
    )
