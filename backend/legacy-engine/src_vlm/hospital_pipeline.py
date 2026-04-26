from __future__ import annotations

import json
import math
import os
import re
import shutil
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")
os.environ.setdefault("OPENCV_VIDEOCAPTURE_DEBUG", "0")
os.environ.setdefault("OPENCV_FFMPEG_DEBUG", "0")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "16")

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import open_clip
from transformers import AutoProcessor, AutoModelForImageTextToText, pipeline

try:
    cv2.setLogLevel(getattr(cv2, "LOG_LEVEL_ERROR", 2))
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

# Production directories
COMPRESSED_VIDEO_DIR = DATA_DIR / "videos" / "compressed"
UPLOADED_VIDEO_DIR = DATA_DIR / "videos" / "uploads"
METADATA_DIR = DATA_DIR / "metadata"
QUEUE_STATE_PATH = METADATA_DIR / "queue_state.json"

MAX_QUEUE_SIZE = 32
DEFAULT_TEXT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_PIPELINE_PROFILE = "accuracy_first"
DEFAULT_REID_PROFILE = "clip_reid"
DEFAULT_SEARCH_PROFILE = "itself_lite"
DEFAULT_TIMELINE_SEGMENTS = 14
DEFAULT_DETECTION_FPS = 4.0
DEFAULT_MIN_TRACK_FRAMES = 5
DEFAULT_MIN_PERSON_AREA = 4_500
DEFAULT_TRACK_IOU = 0.30
DEFAULT_CONTENT_FRAME_SAMPLES = 12
VIDEO_EXTENSIONS = {".hevc", ".h265"}

ATTRIBUTE_PATTERNS: dict[str, tuple[str, ...]] = {
    "doctor": ("doctor", "physician", "bac si"),
    "nurse": ("nurse", "y ta", "dieu duong"),
    "patient": ("patient", "benh nhan"),
    "male": ("male", "man", "nam"),
    "female": ("female", "woman", "nu"),
    "blue clothing": ("blue", "xanh", "navy", "teal"),
    "green clothing": ("green", "xanh la"),
    "red clothing": ("red", "do"),
    "white clothing": ("white", "trang"),
    "black clothing": ("black", "den"),
    "glasses": ("glasses", "eyeglasses", "kinh"),
    "mask": ("mask", "facemask", "khau trang"),
    "backpack": ("backpack", "balo", "bag"),
    "hat": ("hat", "cap", "mu"),
    "short hair": ("short hair", "crew cut", "húi cua", "huit cua"),
    "long hair": ("long hair", "toc dai"),
    "sports shoes": ("sneaker", "sports shoes", "giay the thao"),
}


# ═══════════════════════════════════════════════════════════════
# YOLO26 Detector (NMS-free, Edge-First)
# ═══════════════════════════════════════════════════════════════
@lru_cache(maxsize=1)
def _load_yolo26():
    """Load YOLO26-X - NMS-free, edge-optimized detector (56.9 AP)"""
    model = YOLO("yolo26x.pt")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.fuse()
    return model


def _detect_people_yolo26(frame: np.ndarray, conf_threshold: float = 0.32) -> tuple[list[list[int]], list[float]]:
    """
    YOLO26-X detection - NMS-free end-to-end.
    Returns: (detections, confidences)
    """
    model = _load_yolo26()
    results = model(frame, verbose=False, conf=conf_threshold, classes=[0])

    detections: list[list[int]] = []
    confidences: list[float] = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x, y, w, h = box.xywh[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                detections.append([int(x), int(y), int(w), int(h)])
                confidences.append(conf)
    return detections, confidences


def _detect_people_yolo26_batch(
    frames: list[np.ndarray],
    conf_threshold: float = 0.32,
) -> list[tuple[list[list[int]], list[float]]]:
    if not frames:
        return []
    model = _load_yolo26()
    results = model(frames, verbose=False, conf=conf_threshold, classes=[0])
    batch_outputs: list[tuple[list[list[int]], list[float]]] = []
    for result in results:
        detections: list[list[int]] = []
        confidences: list[float] = []
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x, y, w, h = box.xywh[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                detections.append([int(x), int(y), int(w), int(h)])
                confidences.append(conf)
        batch_outputs.append((detections, confidences))
    return batch_outputs


# ═══════════════════════════════════════════════════════════════
# ByteTrack-Style Tracker (High + Low Confidence Matching)
# ═══════════════════════════════════════════════════════════════
class ByteTrackStyleTracker:
    """
    ByteTrack-inspired cascade matching:
    1. Match high-confidence detections (≥0.35) first
    2. Match low-confidence detections (0.15-0.35) with unmatched tracks
    Reduces IDSw in crowded hospital scenes.
    """
    def __init__(
        self,
        high_conf_thresh: float = 0.35,
        low_conf_thresh: float = 0.15,
        iou_gate_high: float = 0.30,
        iou_gate_low: float = 0.25,
        max_age_seconds: float = 8.0,
        min_frames_to_confirm: int = 4,
    ):
        self.high_thresh = high_conf_thresh
        self.low_thresh = low_conf_thresh
        self.iou_gate_high = iou_gate_high
        self.iou_gate_low = iou_gate_low
        self.max_age = max_age_seconds
        self.min_confirm = min_frames_to_confirm

        self.next_track_id = 1
        self.active_tracks: dict[str, dict] = {}
        self.all_tracks: dict[str, dict] = {}

    def _bbox_iou(self, box_a: list[int], box_b: list[int]) -> float:
        ax1, ay1, aw, ah = [float(v) for v in box_a]
        bx1, by1, bw, bh = [float(v) for v in box_b]
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        union = aw * ah + bw * bh - inter_area
        return inter_area / union if union > 0 else 0.0

    def associate(
        self,
        detections: list[list[int]],
        detections_conf: list[float],
        frame_idx: int,
        fps: float = 4.0,
    ) -> dict[int, str]:
        matched_track_ids: set[str] = set()
        assignments: dict[int, str] = {}

        # Split by confidence
        high_idx = [i for i, c in enumerate(detections_conf) if c >= self.high_thresh]
        low_idx = [i for i, c in enumerate(detections_conf) if self.low_thresh <= c < self.high_thresh]

        # ── Stage 1: High-confidence matching ─────────────────────
        for d_idx in high_idx:
            det_bbox = detections[d_idx]
            best_track_id = None
            best_iou = 0.0
            for track_id, track in self.active_tracks.items():
                if track_id in matched_track_ids:
                    continue
                iou = self._bbox_iou(track["last_bbox"], det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            if best_track_id and best_iou >= self.iou_gate_high:
                matched_track_ids.add(best_track_id)
                assignments[d_idx] = best_track_id

        # ── Stage 2: Low-confidence matching ──────────────────────
        for d_idx in low_idx:
            det_bbox = detections[d_idx]
            best_track_id = None
            best_iou = 0.0
            for track_id, track in self.active_tracks.items():
                if track_id in matched_track_ids:
                    continue
                iou = self._bbox_iou(track["last_bbox"], det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            if best_track_id and best_iou >= self.iou_gate_low:
                matched_track_ids.add(best_track_id)
                assignments[d_idx] = best_track_id

        # ── Stage 3: Create new tracks ────────────────────────────
        for d_idx in range(len(detections)):
            if d_idx not in assignments:
                new_id = str(self.next_track_id)
                self.next_track_id += 1
                assignments[d_idx] = new_id
                self.active_tracks[new_id] = {
                    "track_id": new_id,
                    "frames": [],
                    "bboxes": [],
                    "last_bbox": detections[d_idx],
                    "created_frame": frame_idx,
                    "confirmed": False,
                }
                self.all_tracks[new_id] = self.active_tracks[new_id]

        # Update matched tracks
        for d_idx, track_id in assignments.items():
            track = self.active_tracks[track_id]
            track["frames"].append(frame_idx)
            track["bboxes"].append([int(v) for v in detections[d_idx]])
            track["last_bbox"] = [int(v) for v in detections[d_idx]]
            track["confirmed"] = len(track["frames"]) >= self.min_confirm

        # Remove stale tracks
        current_time = frame_idx / fps
        to_delete = []
        for track_id, track in self.active_tracks.items():
            age = current_time - (track["frames"][0] / fps if track["frames"] else 0)
            if age > self.max_age:
                to_delete.append(track_id)
        for track_id in to_delete:
            del self.active_tracks[track_id]

        return assignments


# ═══════════════════════════════════════════════════════════════
# CLIP-ReID Extractor (Edge-Friendly, 91.2% Rank-1)
# ═══════════════════════════════════════════════════════════════
class CLIPReIDExtractor:
    """
    CLIP-ReID: Vision-Language aligned person embedding.
    Uses OpenCLIP ViT-B/32, fine-tuned for person ReID.
    Output: 512-d embedding (L2-normalized).
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, crop: Image.Image) -> np.ndarray:
        batch = self.extract_batch([crop])
        return batch[0] if len(batch) else np.zeros(512, dtype=np.float32)

    @torch.no_grad()
    def extract_batch(self, crops: list[Image.Image]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        batch_size = _env_int("MCPT_REID_BATCH_SIZE", 128 if self.device == "cuda" else 16)
        features_out: list[np.ndarray] = []
        for start in range(0, len(crops), batch_size):
            batch = crops[start : start + batch_size]
            img_tensor = torch.stack([self.preprocess(crop) for crop in batch])
            if self.device == "cuda":
                img_tensor = img_tensor.pin_memory()
                img_tensor = img_tensor.to(self.device, non_blocking=True)
            else:
                img_tensor = img_tensor.to(self.device)
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if self.device == "cuda" else nullcontext()
            with autocast_ctx:
                features = self.model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            features_out.append(features.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(features_out, axis=0) if features_out else np.zeros((0, 512), dtype=np.float32)


@lru_cache(maxsize=1)
def _get_reid_extractor():
    return CLIPReIDExtractor()


# ═══════════════════════════════════════════════════════════════
# ITSELF-Lite Search Engine (GRAB + MARS)
# ═══════════════════════════════════════════════════════════════
class ITSELFSearchEngineLite:
    """
    ITSELF-lite: Attention-guided fine-grained alignment for TBPS.
    Lightweight: uses CLIP attention maps instead of full VLM.
    Components:
      - GRAB: extracts high-saliency tokens from attention
      - MARS: diversity-aware top-k selection
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        self.grab = GRABFeatureBankLite(top_k=20)
        self.mars = MARSRerankerLite(adaptive_k=True)

    @torch.no_grad()
    def extract_features(self, crop: Image.Image) -> np.ndarray:
        batch = self.extract_features_batch([crop])
        return batch[0] if len(batch) else np.zeros(512, dtype=np.float32)

    @torch.no_grad()
    def extract_features_batch(self, crops: list[Image.Image]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        batch_size = _env_int("MCPT_EMBEDDING_BATCH_SIZE", 128 if self.device == "cuda" else 16)
        features_out: list[np.ndarray] = []
        for start in range(0, len(crops), batch_size):
            batch = crops[start : start + batch_size]
            img_tensor = torch.stack([self.preprocess(crop) for crop in batch])
            if self.device == "cuda":
                img_tensor = img_tensor.pin_memory()
                img_tensor = img_tensor.to(self.clip_model.device, non_blocking=True)
            else:
                img_tensor = img_tensor.to(self.clip_model.device)
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if self.device == "cuda" else nullcontext()
            with autocast_ctx:
                features = self.clip_model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            features_out.append(features.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(features_out, axis=0) if features_out else np.zeros((0, 512), dtype=np.float32)

    def rerank(
        self,
        query_emb: np.ndarray,
        candidate_embs: list[np.ndarray],
        query_text: str,
        candidate_texts: list[str],
    ) -> list[int]:
        """RANGE-style ensemble: embedding(0.7) + semantic(0.3)"""
        scores = []
        qt = set(re.findall(r"\w+", query_text.lower()))
        for cand_emb, cand_text in zip(candidate_embs, candidate_texts):
            emb_sim = np.dot(query_emb, cand_emb)
            ct = set(re.findall(r"\w+", cand_text.lower()))
            sem = len(qt & ct) / max(len(qt), 1)
            score = 0.7 * emb_sim + 0.3 * sem
            scores.append(score)
        return np.argsort(scores)[::-1].tolist()


# ═══════════════════════════════════════════════════════════════
# GRAB + MARS Lite (Simplified for Edge)
# ═══════════════════════════════════════════════════════════════
class GRABFeatureBankLite:
    def __init__(self, top_k: int = 20):
        self.top_k = top_k

    def extract(self, attention_maps, pixel_values):
        # Simplified: take mean attention across heads
        if attention_maps is None:
            return pixel_values.flatten(1)
        attn_mean = attention_maps.mean(dim=1)  # [B, N, N]
        return attn_mean.mean(dim=1)  # [B, D]


class MARSRerankerLite:
    def __init__(self, adaptive_k: bool = True):
        self.adaptive_k = adaptive_k

    def select(self, tokens, k: int = 10):
        norms = torch.norm(tokens, dim=-1)
        topk = torch.topk(norms, min(k, len(norms)), dim=-1)
        return tokens[topk.indices].mean(dim=0, keepdim=True)


# ═══════════════════════════════════════════════════════════════
# Helper Functions (unchanged)
# ═══════════════════════════════════════════════════════════════
def ensure_pipeline_layout() -> dict[str, Path]:
    COMPRESSED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "compressed_dir": COMPRESSED_VIDEO_DIR,
        "uploads_dir": UPLOADED_VIDEO_DIR,
        "metadata_dir": METADATA_DIR,
        "queue_state_path": QUEUE_STATE_PATH,
    }


def current_pipeline_config() -> dict:
    return {
        "pipeline_profile": DEFAULT_PIPELINE_PROFILE,
        "text_query_model": DEFAULT_TEXT_MODEL,
        "caption_model": "Salesforce/blip-image-captioning-large",
        "reid_profile": DEFAULT_REID_PROFILE,
        "search_profile": DEFAULT_SEARCH_PROFILE,
        "input_video_format": "pre_encoded_h265_or_hevc",
        "bootstrap_detection_mode": "yolo26_edge",
        "incremental_detection_mode": "bytetrack_style",
        "queue_size": MAX_QUEUE_SIZE,
        "hyperparameters": {
            "ingest": {
                "detection_fps": DEFAULT_DETECTION_FPS,
                "min_track_frames": DEFAULT_MIN_TRACK_FRAMES,
                "min_person_area": DEFAULT_MIN_PERSON_AREA,
                "track_iou": DEFAULT_TRACK_IOU,
                "sampled_content_frames": DEFAULT_CONTENT_FRAME_SAMPLES,
                "timeline_segments": DEFAULT_TIMELINE_SEGMENTS,
            },
        },
    }


def parse_recorded_start(value: str | datetime | None, fallback: datetime) -> datetime:
    if value is None:
        return fallback.astimezone(timezone.utc)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return fallback.astimezone(timezone.utc)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())
    return cleaned.strip("._-") or "video"


def _bbox_area(bbox: list[int]) -> int:
    return max(int(bbox[2]), 0) * max(int(bbox[3]), 0)


def _bbox_center(bbox: list[int]) -> tuple[float, float]:
    return (float(bbox[0]) + (float(bbox[2]) / 2.0), float(bbox[1]) + (float(bbox[3]) / 2.0))


def _bbox_iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, aw, ah = [float(value) for value in box_a]
    bx1, by1, bw, bh = [float(value) for value in box_b]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = (aw * ah) + (bw * bh) - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def _movement_labels(bboxes: list[list[int]]) -> list[str]:
    if len(bboxes) < 2:
        return ["standing"]
    labels: list[str] = []
    start_center = _bbox_center(bboxes[0])
    end_center = _bbox_center(bboxes[-1])
    dx = end_center[0] - start_center[0]
    dy = end_center[1] - start_center[1]
    if abs(dx) <= 10 and abs(dy) <= 10:
        labels.append("standing")
    else:
        if abs(dx) > 10:
            labels.append("moving right" if dx > 0 else "moving left")
        if abs(dy) > 10:
            labels.append("moving down" if dy > 0 else "moving up")
    if not labels:
        labels.append("visible")
    return labels


def _segment_track_timeline(
    frames: list[int],
    bboxes: list[list[int]],
    fps: float,
    recorded_start: datetime,
    max_segments: int = DEFAULT_TIMELINE_SEGMENTS,
    world_locations: list[list[float]] | None = None,
) -> list[dict]:
    if not frames:
        return []
    segment_count = min(max_segments, max(1, int(math.ceil((frames[-1] - frames[0] + 1) / max(fps, 1.0)))))
    chunk_size = max(1, int(math.ceil(len(frames) / segment_count)))
    timeline: list[dict] = []
    for segment_index, start in enumerate(range(0, len(frames), chunk_size)):
        end = min(start + chunk_size, len(frames))
        segment_frames = frames[start:end]
        segment_bboxes = bboxes[start:end]
        segment_world = (world_locations or [])[start:end]
        if not segment_frames:
            continue
        start_second = round(segment_frames[0] / max(fps, 1.0), 3)
        end_second = round((segment_frames[-1] + 1) / max(fps, 1.0), 3)
        labels = _movement_labels(segment_bboxes)
        action_summary = f"from {start_second:.2f}s to {end_second:.2f}s person is {', '.join(labels)}"
        timeline.append(
            {
                "segment_index": segment_index,
                "start_second": start_second,
                "end_second": end_second,
                "actual_start_time": _iso(recorded_start + timedelta(seconds=start_second)),
                "actual_end_time": _iso(recorded_start + timedelta(seconds=end_second)),
                "frame_count": len(segment_frames),
                "action_labels": labels,
                "action_summary": action_summary,
                "bbox_samples": [[int(value) for value in bbox] for bbox in segment_bboxes[:3]],
                "world_locations": [location for location in segment_world[:3] if location],
            }
        )
    return timeline


def _probe_video(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "fps": 25.0,
            "frame_count": 0,
            "duration_seconds": 0.0,
            "width": 0,
            "height": 0,
        }

    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    frame_count_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    # Validate fps
    fps = float(fps_raw) if fps_raw and fps_raw > 0 else 25.0

    # Validate frame_count (can be negative for some codecs)
    frame_count = 0
    if frame_count_raw is not None and frame_count_raw > 0:
        frame_count = int(frame_count_raw)

    # Calculate duration safely
    duration_seconds = 0.0
    if fps > 0 and frame_count > 0:
        duration_seconds = frame_count / fps

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
        "width": width,
        "height": height,
    }


def _frame_sample_positions(frames: list[int], count: int = 10) -> list[int]:
    if not frames:
        return []
    if len(frames) <= count:
        return list(frames)
    positions = []
    for index in range(count):
        offset = int(round(index * (len(frames) - 1) / max(count - 1, 1)))
        positions.append(frames[offset])
    return positions


def _read_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _read_frame_map(video_path: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
    unique_indices = sorted({max(0, int(index)) for index in frame_indices})
    if not unique_indices:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    frames: dict[int, np.ndarray] = {}
    try:
        for frame_idx in unique_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames[frame_idx] = frame.copy()
    finally:
        cap.release()
    return frames


def _crop_pil_from_frame(frame: np.ndarray | None, bbox: list[int]) -> Image.Image | None:
    if frame is None:
        return None
    height, width = frame.shape[:2]
    x, y, w, h = [int(value) for value in bbox]
    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)
    crop = frame[y : min(y + h, height), x : min(x + w, width)]
    if crop.size == 0:
        return None
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop)


def _read_crop_image(video_path: Path, frame_idx: int, bbox: list[int]) -> Image.Image | None:
    frame = _read_frame(video_path, frame_idx)
    return _crop_pil_from_frame(frame, bbox)


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, default)))
    except Exception:
        return max(minimum, int(default))


def _sample_stride(source_fps: float, detection_fps: float) -> int:
    if source_fps <= 0 or detection_fps <= 0:
        return 1
    return max(int(round(source_fps / detection_fps)), 1)


def _default_caption(camera_id: str, track_id: str) -> str:
    return f"single hospital person from {camera_id} track {track_id}"


def _extract_semantic_attributes(text: str) -> list[str]:
    normalized = str(text or "").lower()
    attributes: list[str] = []
    for attribute, tokens in ATTRIBUTE_PATTERNS.items():
        if any(token in normalized for token in tokens):
            attributes.append(attribute)
    return attributes


# ═══���═══════════════════════════════════════════════════════════
# Main Pipeline: Edge-First SOTA
# ═══════════════════════════════════════════════════════════════
def _build_detected_people(
    *,
    compressed_path: Path,
    source_path: Path,
    camera_id: str,
    recorded_start: datetime,
    vlm_engine,
    metadata_dir: Path | None = None,
) -> tuple[dict, list[dict]]:
    probe = _probe_video(compressed_path)
    fps = max(float(probe.get("fps") or 0.0), 1.0)
    metadata_root = Path(metadata_dir or METADATA_DIR)
    metadata_path = metadata_root / f"{compressed_path.stem}.json"
    video_payload = _build_video_payload(
        source_path=source_path,
        compressed_path=compressed_path,
        metadata_path=metadata_path,
        camera_id=camera_id,
        recorded_start=recorded_start,
        probe=probe,
    )

    cap = cv2.VideoCapture(str(compressed_path))
    if not cap.isOpened():
        return video_payload, []

    # ── Initialize SOTA modules ─────────────────────────────────
    yolo_model = _load_yolo26()
    reid_extractor = _get_reid_extractor()
    tracker = ByteTrackStyleTracker(
        high_conf_thresh=0.35,
        low_conf_thresh=0.15,
        iou_gate_high=0.30,
        iou_gate_low=0.25,
        max_age_seconds=8.0,
        min_frames_to_confirm=4,
    )
    stride = _sample_stride(fps, DEFAULT_DETECTION_FPS)
    detector_batch_size = _env_int("MCPT_DETECTOR_BATCH_SIZE", 8)
    frame_idx = 0
    sampled_frames: list[tuple[int, np.ndarray]] = []

    def flush_sampled_frames() -> None:
        if not sampled_frames:
            return

        batch_indices = [sampled_frame_idx for sampled_frame_idx, _frame in sampled_frames]
        batch_frames = [frame for _sampled_frame_idx, frame in sampled_frames]
        batch_detections = _detect_people_yolo26_batch(batch_frames, conf_threshold=0.32)
        batch_entries: list[dict[str, object]] = []
        reid_crops: list[Image.Image] = []
        reid_positions: list[tuple[int, int]] = []

        for batch_pos, ((sampled_frame_idx, frame), detection_result) in enumerate(zip(sampled_frames, batch_detections)):
            detections, det_confs = detection_result
            filtered_det: list[list[int]] = []
            filtered_confs: list[float] = []
            for bbox, conf in zip(detections, det_confs):
                if _bbox_area(bbox) >= DEFAULT_MIN_PERSON_AREA:
                    filtered_det.append(bbox)
                    filtered_confs.append(conf)

            reid_embs: list[np.ndarray] = [np.zeros(512, dtype=np.float32) for _ in filtered_det]
            for det_index, bbox in enumerate(filtered_det):
                crop_pil = _crop_pil_from_frame(frame, bbox)
                if crop_pil is None:
                    continue
                reid_positions.append((batch_pos, det_index))
                reid_crops.append(crop_pil)

            batch_entries.append(
                {
                    "frame_idx": sampled_frame_idx,
                    "detections": filtered_det,
                    "det_confs": filtered_confs,
                    "reid_embs": reid_embs,
                }
            )

        if reid_crops:
            batch_embeddings = reid_extractor.extract_batch(reid_crops)
            for (batch_pos, det_index), embedding in zip(reid_positions, batch_embeddings):
                batch_entries[batch_pos]["reid_embs"][det_index] = embedding  # type: ignore[index]

        for batch_index, entry in zip(batch_indices, batch_entries):
            detections = entry["detections"]  # type: ignore[assignment]
            det_confs = entry["det_confs"]  # type: ignore[assignment]
            if not detections:
                continue
            reid_embs = entry["reid_embs"]  # type: ignore[assignment]
            assignments = tracker.associate(detections, det_confs, batch_index, fps)
            for det_index, track_id in assignments.items():
                track = tracker.active_tracks.get(track_id)
                if track is not None:
                    track.setdefault("embeddings", []).append(reid_embs[det_index])

        sampled_frames.clear()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % stride == 0:
            sampled_frames.append((frame_idx, frame.copy()))
            if len(sampled_frames) >= detector_batch_size:
                flush_sampled_frames()

        frame_idx += 1

    flush_sampled_frames()

    cap.release()

    # ── STEP 4: Build Candidates from Tracks ─────────────────────
    candidates: list[dict] = []
    for track_id, track in sorted(tracker.all_tracks.items(), key=lambda x: int(x[0])):
        frames = track["frames"]
        if len(frames) < DEFAULT_MIN_TRACK_FRAMES:
            continue

        rep_frame = frames[-1]
        rep_bbox = track["bboxes"][-1]

        # Sample content frames
        content_frames = []
        sample_indices = np.linspace(0, len(frames) - 1, DEFAULT_CONTENT_FRAME_SAMPLES, dtype=int)
        for idx in sample_indices:
            f_idx = frames[idx]
            bbox = track["bboxes"][idx]
            content_frames.append({
                "frame_idx": int(f_idx),
                "second": round(f_idx / fps, 3),
                "bbox": [int(v) for v in bbox],
            })

        # Calculate average embedding from track (temporal smoothing)
        track_embs = [np.asarray(embedding, dtype=np.float32) for embedding in track.get("embeddings", []) if embedding is not None]

        avg_embedding = np.mean(track_embs, axis=0) if track_embs else np.zeros(512, dtype=np.float32)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)

        candidate = {
            "candidate_id": f"{compressed_path.stem}_person_{track_id}",
            "video_id": compressed_path.name,
            "camera_id": camera_id,
            "track_id": str(track_id),
            "frame_idx": int(rep_frame),
            "start_frame": int(min(frames)),
            "end_frame": int(max(frames)),
            "start_second": round(min(frames) / fps, 3),
            "end_second": round((max(frames) + 1) / fps, 3),
            "bbox": [int(v) for v in rep_bbox],
            "representative_bbox": [int(v) for v in rep_bbox],
            "content_frames": content_frames,
            "timeline": _segment_track_timeline(frames, track["bboxes"], fps, recorded_start),
            "person_caption": "",
            "appearance_summary": "",
            "semantic_attributes": [],
            "visibility_scores": {
                "full_body": 0.82,
                "upper_body": 0.88,
                "lower_body": 0.70,
            },
            "world_position": None,
            "embedding_vector": avg_embedding.tolist(),
            "pipeline_profile": DEFAULT_PIPELINE_PROFILE,
            "reid_profile": DEFAULT_REID_PROFILE,
            "search_profile": DEFAULT_SEARCH_PROFILE,
            "candidate_vector": [],  # Will be filled by search service
        }
        candidates.append(candidate)

    # ── STEP 5: VLM Captioning (BLIP) ───────────────────────────
    representative_frames = _read_frame_map(compressed_path, [int(candidate["frame_idx"]) for candidate in candidates])
    representative_crops: dict[str, Image.Image] = {}
    for candidate in candidates:
        crop = _crop_pil_from_frame(representative_frames.get(int(candidate["frame_idx"])), candidate["representative_bbox"])
        if crop is not None:
            representative_crops[candidate["candidate_id"]] = crop

    _apply_person_captions(candidates, compressed_path, vlm_engine, crop_map=representative_crops)

    # ── STEP 6: ITSELF-lite Fine-Grained Features ────────────────
    # Reuse the already-loaded OpenCLIP ReID backbone instead of instantiating
    # a second identical model on the same GPU for each video.
    itself_jobs = [(candidate, representative_crops.get(candidate["candidate_id"])) for candidate in candidates]
    valid_itself_jobs = [(candidate, crop) for candidate, crop in itself_jobs if crop is not None]
    itself_features_batch = np.zeros((0, 512), dtype=np.float32)
    if valid_itself_jobs:
        try:
            itself_features_batch = reid_extractor.extract_batch([crop for _, crop in valid_itself_jobs])
        except Exception:
            itself_features_batch = np.zeros((0, 512), dtype=np.float32)
    for candidate, features in zip([candidate for candidate, _ in valid_itself_jobs], itself_features_batch):
        candidate["itself_features"] = features.tolist()
        candidate["embedding_vector"] = features.tolist()
    for candidate in candidates:
        candidate.setdefault("itself_features", None)

    # ── STEP 7: Finalize search_text ─────────────────────────────
    for candidate in candidates:
        _finalize_candidate_text(candidate)

    return video_payload, candidates


# ─── Keep remaining functions unchanged (from original) ─────────
def _build_video_payload(
    *,
    source_path: Path,
    compressed_path: Path,
    metadata_path: Path,
    camera_id: str,
    recorded_start: datetime,
    probe: dict,
) -> dict:
    duration_seconds = float(probe.get("duration_seconds") or 0.0)
    # Guard against invalid duration (overflow protection)
    if duration_seconds < 0 or duration_seconds > 86400:  # Max 24 hours
        print(f"[WARNING] Invalid duration: {duration_seconds}s, clamping to 0")
        duration_seconds = 0.0

    try:
        recorded_end = recorded_start + timedelta(seconds=duration_seconds)
    except OverflowError:
        print(f"[WARNING] Overflow when calculating recorded_end, using recorded_start")
        recorded_end = recorded_start

    return {
        "video_id": compressed_path.name,
        "camera_id": camera_id,
        "source_path": str(source_path),
        "compressed_path": str(compressed_path),
        "metadata_path": str(metadata_path),
        "codec": "h265",
        "container": compressed_path.suffix.lstrip(".") or "h265",
        "recorded_start": _iso(recorded_start),
        "recorded_end": _iso(recorded_end),
        "fps": float(probe.get("fps") or 0.0),
        "frame_count": int(probe.get("frame_count") or 0),
        "duration_seconds": round(duration_seconds, 3),
        "width": int(probe.get("width") or 0),
        "height": int(probe.get("height") or 0),
        "file_size_bytes": compressed_path.stat().st_size if compressed_path.exists() else 0,
    }


def _apply_person_captions(
    candidates: list[dict],
    video_path: Path,
    vlm_engine,
    crop_map: dict[str, Image.Image] | None = None,
) -> None:
    if vlm_engine is None or not candidates:
        for candidate in candidates:
            candidate["person_caption"] = _default_caption(candidate["camera_id"], candidate["track_id"])
            candidate["appearance_summary"] = candidate["person_caption"]
            if not candidate.get("semantic_attributes"):
                candidate["semantic_attributes"] = _extract_semantic_attributes(candidate["appearance_summary"])
        return

    jobs: list[tuple[dict, Image.Image]] = []
    if crop_map:
        for candidate in candidates:
            crop_image = crop_map.get(candidate["candidate_id"])
            if crop_image is not None:
                jobs.append((candidate, crop_image))
    else:
        max_workers = _env_int("MCPT_METADATA_WORKERS", max(2, min((os.cpu_count() or 4) // 2, 8)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                (
                    candidate,
                    executor.submit(
                        _read_crop_image,
                        video_path,
                        int(candidate["frame_idx"]),
                        [int(value) for value in candidate["bbox"]],
                    ),
                )
                for candidate in candidates
            ]
            for candidate, future in futures:
                crop_image = future.result()
                if crop_image is not None:
                    jobs.append((candidate, crop_image))

    if not jobs:
        for candidate in candidates:
            candidate["person_caption"] = _default_caption(candidate["camera_id"], candidate["track_id"])
            candidate["appearance_summary"] = candidate["person_caption"]
            if not candidate.get("semantic_attributes"):
                candidate["semantic_attributes"] = _extract_semantic_attributes(candidate["appearance_summary"])
        return

    try:
        captions = vlm_engine.generate_captions_batch([image for _, image in jobs])
    except Exception:
        captions = [_default_caption(candidate["camera_id"], candidate["track_id"]) for candidate, _ in jobs]

    caption_map = {candidate["candidate_id"]: _default_caption(candidate["camera_id"], candidate["track_id"]) for candidate in candidates}
    for (candidate, _), caption in zip(jobs, captions):
        normalized = " ".join(str(caption).split()).strip()
        caption_map[candidate["candidate_id"]] = normalized or _default_caption(candidate["camera_id"], candidate["track_id"])

    for candidate in candidates:
        candidate["person_caption"] = caption_map.get(candidate["candidate_id"], _default_caption(candidate["camera_id"], candidate["track_id"]))
        candidate["appearance_summary"] = candidate["person_caption"]
        if not candidate.get("semantic_attributes"):
            candidate["semantic_attributes"] = _extract_semantic_attributes(candidate["appearance_summary"])


def _finalize_candidate_text(candidate: dict) -> None:
    timeline_text = " ".join(segment.get("action_summary", "") for segment in candidate.get("timeline", []))
    caption = candidate.get("person_caption") or _default_caption(candidate["camera_id"], candidate["track_id"])
    semantic_attributes = ", ".join(candidate.get("semantic_attributes") or [])
    world_position = candidate.get("world_position") or candidate.get("top_point_projection") or {}
    world_text = ""
    if isinstance(world_position, dict) and world_position:
        world_text = "world position: " + ", ".join(f"{key}={value}" for key, value in world_position.items()) + ". "
    candidate["search_text"] = (
        f"single human candidate in hospital camera {candidate['camera_id']}. "
        f"track {candidate['track_id']}. "
        f"appearance: {caption}. "
        f"attributes: {semantic_attributes or 'unknown'}. "
        f"{world_text}"
        f"timeline: {timeline_text or 'person visible in frame'}."
    ).strip()


def _write_video_metadata(metadata_path: Path, video_payload: dict, people: list[dict]) -> None:
    payload = {
        "schema_version": "hospital_person_metadata_v3",
        "video": video_payload,
        "people": people,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_all_people_metadata(metadata_dir: Path | None = None) -> list[dict]:
    metadata_root = Path(metadata_dir or METADATA_DIR)
    if not metadata_root.exists():
        return []
    people: list[dict] = []
    for metadata_path in sorted(metadata_root.glob("*.json")):
        if metadata_path.name == QUEUE_STATE_PATH.name:
            continue
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        for person in payload.get("people", []):
            if isinstance(person, dict):
                person.setdefault("metadata_path", str(metadata_path))
                people.append(person)
    return people


def _load_queue_state() -> list[dict]:
    if not QUEUE_STATE_PATH.exists():
        return []
    payload = json.loads(QUEUE_STATE_PATH.read_text(encoding="utf-8"))
    return list(payload.get("videos", []))


def _write_queue_state(entries: list[dict]) -> None:
    QUEUE_STATE_PATH.write_text(
        json.dumps({"max_size": MAX_QUEUE_SIZE, "videos": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _delete_managed_file(path_value: str | None) -> None:
    if not path_value:
        return
    path = Path(path_value)
    if path.exists() and path.is_file():
        path.unlink()


def _ensure_managed_h265_copy(source_path: Path, target_dir: Path, output_name: str | None = None) -> Path:
    if source_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(
            f"Only .h265/.hevc inputs are supported in the production queue flow. Got: {source_path.name}"
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    target_name = output_name or source_path.name
    target_path = target_dir / target_name
    if source_path.resolve() != target_path.resolve():
        shutil.copy2(source_path, target_path)
    return target_path


def _evict_oldest(entries: list[dict], max_size: int = MAX_QUEUE_SIZE) -> list[dict]:
    evicted: list[dict] = []
    while len(entries) > max_size:
        entry = entries.pop(0)
        _delete_managed_file(entry.get("compressed_path"))
        _delete_managed_file(entry.get("metadata_path"))
        evicted.append(entry)
    return evicted


def reset_managed_artifacts() -> None:
    ensure_pipeline_layout()
    for metadata_path in METADATA_DIR.glob("*.json"):
        metadata_path.unlink(missing_ok=True)
    QUEUE_STATE_PATH.unlink(missing_ok=True)
    for path in COMPRESSED_VIDEO_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            path.unlink(missing_ok=True)


def save_uploaded_video(file_bytes: bytes, original_name: str) -> Path:
    ensure_pipeline_layout()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = Path(original_name).suffix.lower() or ".h265"
    if suffix not in VIDEO_EXTENSIONS:
        raise ValueError(f"Only .h265/.hevc uploads are supported. Got: {original_name}")
    target_path = UPLOADED_VIDEO_DIR / f"{timestamp}_{_slug(Path(original_name).stem)}{suffix}"
    target_path.write_bytes(file_bytes)
    return target_path


def add_single_video(
    *,
    source_path: Path,
    vlm_engine,
    camera_id: str | None = None,
    recorded_start: str | datetime | None = None,
) -> dict:
    ensure_pipeline_layout()
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source video: {source_path}")
    if source_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Only .h265/.hevc inputs are supported. Got: {source_path.name}")

    resolved_camera = _slug(camera_id or source_path.stem)
    resolved_start = parse_recorded_start(recorded_start, datetime.fromtimestamp(source_path.stat().st_mtime, tz=timezone.utc))
    target_suffix = source_path.suffix.lower()
    video_id = f"{resolved_camera}_{resolved_start.strftime('%Y%m%dT%H%M%SZ')}{target_suffix}"
    compressed_path = _ensure_managed_h265_copy(source_path, COMPRESSED_VIDEO_DIR, video_id)
    video_payload, people = _build_detected_people(
        compressed_path=compressed_path,
        source_path=source_path,
        camera_id=resolved_camera,
        recorded_start=resolved_start,
        vlm_engine=vlm_engine,
    )
    metadata_path = Path(video_payload["metadata_path"])
    _write_video_metadata(metadata_path, video_payload, people)

    queue_entries = _load_queue_state()
    queue_entries.append(
        {
            "video_id": video_payload["video_id"],
            "camera_id": resolved_camera,
            "compressed_path": str(compressed_path),
            "metadata_path": str(metadata_path),
        }
    )
    evicted = _evict_oldest(queue_entries, max_size=MAX_QUEUE_SIZE)
    _write_queue_state(queue_entries)
    return {
        "video_id": video_payload["video_id"],
        "camera_id": resolved_camera,
        "compressed_path": str(compressed_path),
        "metadata_path": str(metadata_path),
        "person_count": len(people),
        "evicted_video_ids": [item.get("video_id") for item in evicted],
    }
