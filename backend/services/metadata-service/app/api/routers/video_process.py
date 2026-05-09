"""Video processing pipeline for metadata-service — SOTA 2026 AI.

Full 7-stage pipeline (per video):
  1. Sample frames (uniform)
  2. Grounding DINO 1.6 person detection
  3. BEVProjector (2D→3D via homography)
  4. MCBLT Hungarian cross-camera association
  5. DINOv2 ViT-L/14 appearance embedding (1024-dim)
  6. SigLIP 2 zero-shot attribute tagging
  7. VideoMAE V2 action classification

Batch pipeline (across cameras):
  - Stage 1-3: Run per video in parallel
  - Stage 4: ONE MCBLT call across all cameras
  - Stage 5-7: Per unified tracklet (cross-camera)
"""

from __future__ import annotations

import base64
import io
import logging
import os
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

DEFAULT_SAMPLE_INTERVAL = 15
DEFAULT_MIN_BBOX_AREA = 400
DEFAULT_BEV_MAX_DIST = 1.5
MAX_WORKERS = int(os.getenv("BATCH_MAX_WORKERS", "8"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


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
# Stage 2: Person Detection (Grounding DINO 1.6)
# ---------------------------------------------------------------------------

def _detect_persons(frame: np.ndarray, threshold: float = 0.3) -> list[dict]:
    """Detect persons in a frame using Grounding DINO 1.6."""
    model = get_model("gdino16")
    processor = get_model("gdino16_processor")
    if model is None or processor is None:
        logger.warning("Grounding DINO not available, returning empty detections")
        return []

    device = _get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil_img, text="person.", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    autocast_ctx = torch.autocast(device_type=device.type, dtype=dtype) if device.type == "cuda" else torch.no_grad()
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
    batch_size: int = 64,
) -> list[list[dict]] | None:
    """RT-DETR R50 person detection — primary detector when loaded.
    Returns None if not available (caller falls back to GDINO).
    batch_size=64 is safe on A100 80GB (256 causes OOM).
    """
    model = get_model("rtdetr")
    processor = get_model("rtdetr_processor")
    if model is None or processor is None:
        return None

    person_ids: set = get_model("rtdetr_person_ids") or {0, 1}
    device = _get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    all_dets: list[list[dict]] = []

    for i in range(0, len(frames), batch_size):
        batch = frames[i: i + batch_size]
        sizes = [(f.shape[0], f.shape[1]) for f in batch]
        pil_imgs = [Image.fromarray(f[:, :, ::-1]) for f in batch]  # BGR→RGB

        inputs = processor(images=pil_imgs, return_tensors="pt")
        inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
                  for k, v in inputs.items()}

        try:
            with torch.no_grad():
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
        return [[] for _ in frames]

    device = _get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    autocast_ctx = torch.autocast(device_type=device.type, dtype=dtype) if device.type == "cuda" else torch.no_grad()

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

    for frame_idx, frame in sampled:
        detections = _detect_persons(frame)
        for det in detections:
            det["frame_idx"] = frame_idx
            det["timestamp"] = frame_idx / fps if fps > 0 else 0
            det["video_id"] = entry.video_id
        all_detections.extend(detections)

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
    dtype = torch.float16 if device.type == "cuda" else torch.float32

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
        feats = model(**inputs).pooler_output.cpu().float()  # [N, 1024]

    avg = feats.mean(0).numpy()
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm
    return avg.tolist()


# ---------------------------------------------------------------------------
# Stage 6: SigLIP 2 Attribute Tagging
# ---------------------------------------------------------------------------

def _run_siglip2_attributes(
    frames: list[np.ndarray],
    bbox: list[float],
) -> dict[str, str]:
    """Zero-shot attribute tagging using SigLIP 2."""
    model = get_model("siglip2")
    processor = get_model("siglip2_processor")
    if model is None or processor is None:
        return _default_attributes()

    device = _get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

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
    }

    attributes: dict[str, str] = {}
    for attr_type, labels in label_groups.items():
        try:
            inputs = processor(
                text=labels, images=pil_crop,
                return_tensors="pt", padding=True
            )
            siglip_dtype = torch.float16 if device.type == "cuda" else torch.float32
            inputs = {k: v.to(device, dtype=siglip_dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}
            siglip_ctx = torch.autocast(device_type=device.type, dtype=siglip_dtype) if device.type == "cuda" else torch.no_grad()
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

        except Exception as exc:
            logger.warning("SigLIP2 attribute failed for %s: %s", attr_type, exc)

    for key in ["top_color", "bottom_color", "gender", "bag", "hat"]:
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
    dtype = torch.float16 if device.type == "cuda" else torch.float32

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
        vmae_ctx = torch.autocast(device_type=device.type, dtype=dtype) if device.type == "cuda" else torch.no_grad()
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

def _default_attributes() -> dict[str, str]:
    return {
        "top_color": "unknown",
        "bottom_color": "unknown",
        "gender": "unknown",
        "bag": "no_bag",
        "hat": "no_hat",
    }


def _build_appearance_summary(attrs: dict) -> str:
    parts = []
    gender = attrs.get("gender", "").strip()
    top = attrs.get("top_color", "").strip()
    bottom = attrs.get("bottom_color", "").strip()
    if gender:
        parts.append(gender)
    if top:
        parts.append(f"{top} shirt")
    if bottom:
        parts.append(f"{bottom} pants")
    return " ".join(parts) or "person"


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

    for frame_idx, frame in sampled:
        detections = _detect_persons(frame)
        for det in detections:
            det["frame_idx"] = frame_idx
            det["timestamp"] = frame_idx / fps if fps > 0 else 0
            det["video_id"] = req.video_id
        all_detections.extend(detections)

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
        attributes = _run_siglip2_attributes(tracklet_frames, rep_bbox)
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
            top_color=attributes.get("top_color", "unknown"),
            bottom_color=attributes.get("bottom_color", "unknown"),
            shoes_color="unknown",
            appearance_summary=summary,
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
) -> tuple[list, list, list, list, list]:
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
    dtype = torch.float16 if device.type == "cuda" else torch.float32

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
                    feats = model_dino(**inputs).pooler_output.cpu().float()  # [N, 1024]

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

    # ── SigLIP TRUE BATCH: 1 call per attribute type × ALL tracklets ────────
    all_attributes: list[dict] = []
    all_attr_confs: list[dict] = [{} for _ in t_data]
    img_feats = None  # SigLIP2 image embeddings, captured for storage
    model_sip = get_model("siglip2")
    proc_sip = get_model("siglip2_processor")
    if model_sip and proc_sip and t_data:
        try:
            label_groups = {
                "top_color": ["red shirt", "blue shirt", "green shirt", "white shirt",
                               "black shirt", "yellow shirt", "orange shirt",
                               "gray shirt", "brown shirt", "pink shirt"],
                "bottom_color": ["black pants", "blue jeans", "gray pants",
                                  "white pants", "brown pants", "beige pants", "dark pants"],
                "gender": ["male person", "female person"],
                "shoes_color": ["white shoes", "black shoes", "brown shoes",
                                 "gray shoes", "blue shoes", "red shoes"],
                "bag": ["person carrying bag", "person without bag"],
                "hat": ["person wearing hat", "person without hat"],
            }

            # Encode ALL crop images once using the pre-built 384px crops
            img_inputs = proc_sip(images=all_rep_crops, return_tensors="pt", padding=True)
            img_inputs = {k: v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device)
                          for k, v in img_inputs.items()}
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
                img_feats = model_sip.get_image_features(**{k: v for k, v in img_inputs.items()
                                                             if k in ["pixel_values"]})
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)  # [N, D]

            # For each attribute group: 1 text encode → softmax probability → label + confidence
            attr_votes: list[dict] = [{} for _ in t_data]
            attr_raw_confs: list[dict] = [{} for _ in t_data]
            for attr_name, labels in label_groups.items():
                txt_inputs = proc_sip(text=labels, return_tensors="pt", padding=True)
                txt_inputs = {k: v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device)
                               for k, v in txt_inputs.items()}
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
                    txt_feats = model_sip.get_text_features(**{k: v for k, v in txt_inputs.items()
                                                                if k in ["input_ids", "attention_mask"]})
                txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)  # [L, D]
                scores = (img_feats @ txt_feats.T).cpu().float()  # [N, L]
                # Softmax across labels gives per-class probability within this attribute group
                probs = F.softmax(scores, dim=1)  # [N, L]
                best_idx = probs.argmax(dim=1).tolist()
                best_conf = probs.max(dim=1).values.tolist()
                for i_t, (b_idx, conf) in enumerate(zip(best_idx, best_conf)):
                    lbl = labels[b_idx]
                    if attr_name == "bag":
                        val = "no_bag" if "without" in lbl else "carrying_bag"
                    elif attr_name == "hat":
                        val = "no_hat" if "without" in lbl else "wearing_hat"
                    else:
                        val = lbl.split()[0]
                    attr_votes[i_t][attr_name] = val
                    attr_raw_confs[i_t][attr_name] = float(conf)

            for i_t, av in enumerate(attr_votes):
                all_attributes.append({
                    "gender": av.get("gender", "unknown"),
                    "top_color": av.get("top_color", "unknown"),
                    "bottom_color": av.get("bottom_color", "unknown"),
                    "shoes_color": av.get("shoes_color", "unknown"),
                    "bag": av.get("bag", "unknown"),
                    "hat": av.get("hat", "unknown"),
                })
                rc = attr_raw_confs[i_t]
                bag_conf = rc.get("bag", 0.0)
                hat_conf = rc.get("hat", 0.0)
                all_attr_confs[i_t] = {
                    "gender_conf": rc.get("gender"),
                    "top_color_conf": rc.get("top_color"),
                    "shoes_conf": rc.get("shoes_color"),
                    "accessory_conf": float(max(bag_conf, hat_conf)) if (bag_conf or hat_conf) else None,
                }
        except Exception as exc:
            logger.warning("[pipeline] SigLIP true-batch failed: %s — fallback", exc)
            all_attributes = [_run_siglip2_attributes(t[3], t[2]) for t in t_data]
            all_attr_confs = [{} for _ in t_data]
    else:
        all_attributes = [_default_attributes()] * len(t_data)
        all_attr_confs = [{} for _ in t_data]

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
            vmae_dtype = torch.float16 if device.type == "cuda" else torch.float32
            inputs = proc_vmae(all_clips, return_tensors="pt")
            inputs = {k: v.to(device=device, dtype=vmae_dtype) if v.is_floating_point() else v.to(device)
                       for k, v in inputs.items()}
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=vmae_dtype):
                outputs = model_vmae(**inputs)
            logits = outputs.logits.cpu().float()  # [N, num_classes]
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = probs.topk(1, dim=-1)
            top_idx = top_indices.squeeze(1).tolist()
            top_conf = top_probs.squeeze(1).tolist()
            id2label = getattr(model_vmae.config, "id2label", {})
            for idx, conf in zip(top_idx, top_conf):
                label = id2label.get(idx, "")
                action = _map_kinetics_to_tracex_action(label) if label else "unknown"
                all_actions.append((action, float(conf)))
        except Exception as exc:
            logger.warning("[pipeline] VideoMAE true-batch failed: %s — fallback", exc)
            all_actions = [(_run_videomae_actions(t[3], t[2]), 0.0) for t in t_data]
    else:
        all_actions = [("unknown", 0.0)] * len(t_data)

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
    Sync video processing pipeline using HeadBoxTracker + 4fps sampling.
    presampled_frames: pre-decoded frames from background thread (skips Stage 1).
    """
    import time
    from .tracking_pipeline import (
        VideoFrameSampler, HeadBoxTracker, TrackletQualityScorer,
        FrameDetection, _crop_from_bbox,
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

    # Stage 2: Batch detect — RF-DETR primary, GDINO fallback
    from .tracking_pipeline import _crop_from_bbox as _tcrop
    t_det_start = time.time()
    detector = "RF-DETR" if get_model("rfdetr") is not None else "GDINO"
    logger.info("[pipeline] %s: running %s detection on %d frames...", video_id, detector, len(sampled_frames))
    all_batch_dets = _detect_persons_batch([sf.image for sf in sampled_frames], threshold=0.25)
    logger.warning("[pipeline] %s: GDINO done in %.1fs", video_id, time.time() - t_det_start)

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

    # Stage 3: Track with HeadBoxTracker (ByteTrack-style)
    tracker = HeadBoxTracker(
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

    _CROPS_DIR = Path("/workspace/storage/crops")
    _CROPS_DIR.mkdir(parents=True, exist_ok=True)

    tracklets: list[TrackletResult] = []
    for t_idx, (lt, _, rep_bbox_float, t_frames) in enumerate(t_data):
        obs = lt.observations
        attributes = all_attributes[t_idx]
        embedding = all_embeddings[t_idx]
        siglip_emb = all_siglip_embeddings[t_idx] if t_idx < len(all_siglip_embeddings) else []
        action_tuple = all_actions[t_idx]
        action = action_tuple[0] if isinstance(action_tuple, tuple) else str(action_tuple)
        action_conf = float(action_tuple[1]) if isinstance(action_tuple, tuple) else 0.0
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
            top_color=attributes.get("top_color", "unknown"),
            bottom_color=attributes.get("bottom_color", "unknown"),
            shoes_color=attributes.get("shoes_color", "unknown"),
            appearance_summary=summary,
            crop_url=crop_url,
            representative_bbox=[int(x) for x in rep_bbox_float],
            bev_x=0.0,
            bev_y=0.0,
            embedding_vector=embedding or [],
            siglip_embedding=siglip_emb or [],
            action=action,
            action_confidence=action_conf,
            occlusion_score=0.0,
            gender_conf=attr_conf.get("gender_conf"),
            top_color_conf=attr_conf.get("top_color_conf"),
            shoes_conf=attr_conf.get("shoes_conf"),
            accessory_conf=attr_conf.get("accessory_conf"),
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
    across all cameras, then DINOv2 + SigLIP2 + VideoMAE per unified tracklet.

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

    # ---- Stages 5-7: Per unified tracklet → DINOv2 + SigLIP2 + VideoMAE ----
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

        # SigLIP2 attributes
        attributes = _run_siglip2_attributes(all_track_frames, rep_bbox)

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
            top_color=attributes.get("top_color", "unknown"),
            bottom_color=attributes.get("bottom_color", "unknown"),
            shoes_color="unknown",
            appearance_summary=summary,
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
