"""Video processing pipeline for metadata-service — SOTA 2026 AI.

Full 7-stage pipeline (per video):
  1. Sample frames (uniform)
  2. Grounding DINO 1.6 person detection
  3. BEVProjector (2D→3D via homography)
  4. MCBLT Hungarian cross-camera association
  5. EVA-02 ViT-L/14 appearance embedding (1024-dim)
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
from fastapi import APIRouter, HTTPException
from PIL import Image

from ...services.model_warmup import get_model
from .video_process_schemas import (
    BatchProcessRequest,
    BatchProcessResponse,
    BatchVideoEntry,
    ProcessVideoRequest,
    ProcessVideoResponse,
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
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil_img, text="person.", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
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
# Stage 5: EVA-02 Appearance Embedding
# ---------------------------------------------------------------------------

def _generate_eva02_embeddings(
    frames: list[np.ndarray],
    bboxes: list[list[float]],
    tracklet_id: str,
) -> Optional[list[float]]:
    """Generate EVA-02 ViT-L/14 appearance embeddings (1024-dim)."""
    model = get_model("eva02")
    transform = get_model("eva02_transform")
    if model is None or transform is None:
        return None

    device = _get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if not frames or not bboxes:
        return None

    n = min(5, len(frames))
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)

    embeddings = []
    with torch.no_grad():
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
            pil_crop = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

            inp = transform(pil_crop).unsqueeze(0).to(device=device, dtype=dtype)
            feat = model(inp)
            embeddings.append(feat.cpu().float().squeeze().tolist())

    if not embeddings:
        return None

    avg = np.mean(embeddings, axis=0)
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
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
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
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
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
        if len(group) < 2:
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

        embedding = _generate_eva02_embeddings(
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
# BATCH endpoint — cross-camera processing (STAGES 1-7 across all cameras)
# ---------------------------------------------------------------------------

@router.post("/batch/process", response_model=BatchProcessResponse)
def process_batch(req: BatchProcessRequest) -> BatchProcessResponse:
    """
    Batch cross-camera processing pipeline.

    Takes up to 100 videos from different cameras (same timestamp),
    runs per-video detection + BEV in parallel, then ONE MCBLT call
    across all cameras, then EVA-02 + SigLIP2 + VideoMAE per unified tracklet.

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

    # ---- Stages 5-7: Per unified tracklet → EVA-02 + SigLIP2 + VideoMAE ----
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

        # EVA-02 embedding
        embedding = _generate_eva02_embeddings(
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
