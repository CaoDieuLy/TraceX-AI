from __future__ import annotations

import json
import math
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import cv2
from PIL import Image

from .video_ingestion import compress_video


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
NVIDIA_HOSPITAL_ROOT = DATA_DIR / "NVIDIA_SmartSpaces" / "MTMC_Tracking_2025" / "val" / "Hospital_000"
NVIDIA_VIDEO_DIR = NVIDIA_HOSPITAL_ROOT / "videos"
NVIDIA_GROUND_TRUTH_PATH = NVIDIA_HOSPITAL_ROOT / "ground_truth.json"
COMPRESSED_VIDEO_DIR = DATA_DIR / "videos" / "compressed"
UPLOADED_VIDEO_DIR = DATA_DIR / "videos" / "uploads"
METADATA_DIR = DATA_DIR / "metadata"
QUEUE_STATE_PATH = METADATA_DIR / "queue_state.json"

MAX_QUEUE_SIZE = 32
DEFAULT_TEXT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_PIPELINE_PROFILE = "accuracy_first"
DEFAULT_REID_PROFILE = "solider_kpr"
DEFAULT_SEARCH_PROFILE = "itself_grab_mars"
DEFAULT_TIMELINE_SEGMENTS = 14
DEFAULT_DETECTION_FPS = 4.0
DEFAULT_MIN_TRACK_FRAMES = 5
DEFAULT_MIN_PERSON_AREA = 4_500
DEFAULT_TRACK_IOU = 0.20
DEFAULT_CONTENT_FRAME_SAMPLES = 12
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".hevc", ".h265"}

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
        "bootstrap_detection_mode": "nvidia_ground_truth_per_person",
        "incremental_detection_mode": "opencv_hog_tracking_fallback",
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


def list_nvidia_hospital_videos(limit: int = 31) -> list[Path]:
    if not NVIDIA_VIDEO_DIR.exists():
        return []
    videos = sorted(path for path in NVIDIA_VIDEO_DIR.glob("Camera_*.mp4") if re.fullmatch(r"Camera_\d{2}", path.stem))
    return videos[:limit]


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
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": (frame_count / fps) if fps > 0 else 0.0,
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


def _read_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _read_crop_image(video_path: Path, frame_idx: int, bbox: list[int]) -> Image.Image | None:
    frame = _read_frame(video_path, frame_idx)
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


@lru_cache(maxsize=1)
def _hog_detector():
    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return detector


def _detect_people_hog(frame) -> list[list[int]]:
    detector = _hog_detector()
    rects, _weights = detector.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )
    detections = []
    for x, y, w, h in rects:
        detections.append([int(x), int(y), int(w), int(h)])
    return detections


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


def _load_nvidia_ground_truth() -> dict:
    if not NVIDIA_GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(f"Missing NVIDIA ground truth at {NVIDIA_GROUND_TRUTH_PATH}")
    with NVIDIA_GROUND_TRUTH_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Unexpected NVIDIA ground truth format.")
    return payload


def _build_video_payload(
    *,
    source_path: Path,
    compressed_path: Path,
    metadata_path: Path,
    camera_id: str,
    recorded_start: datetime,
    probe: dict,
) -> dict:
    recorded_end = recorded_start + timedelta(seconds=float(probe.get("duration_seconds") or 0.0))
    return {
        "video_id": compressed_path.name,
        "camera_id": camera_id,
        "source_path": str(source_path),
        "compressed_path": str(compressed_path),
        "metadata_path": str(metadata_path),
        "codec": "h265",
        "container": compressed_path.suffix.lstrip(".") or "mp4",
        "recorded_start": _iso(recorded_start),
        "recorded_end": _iso(recorded_end),
        "fps": float(probe.get("fps") or 0.0),
        "frame_count": int(probe.get("frame_count") or 0),
        "duration_seconds": round(float(probe.get("duration_seconds") or 0.0), 3),
        "width": int(probe.get("width") or 0),
        "height": int(probe.get("height") or 0),
        "file_size_bytes": compressed_path.stat().st_size if compressed_path.exists() else 0,
    }


def _apply_person_captions(candidates: list[dict], video_path: Path, vlm_engine) -> None:
    if vlm_engine is None or not candidates:
        for candidate in candidates:
            candidate["person_caption"] = _default_caption(candidate["camera_id"], candidate["track_id"])
            candidate["appearance_summary"] = candidate["person_caption"]
            if not candidate.get("semantic_attributes"):
                candidate["semantic_attributes"] = _extract_semantic_attributes(candidate["appearance_summary"])
        return

    jobs: list[tuple[dict, Image.Image]] = []
    max_workers = max(2, min((os.cpu_count() or 4) // 2, 8))
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


def _build_gt_people(
    *,
    compressed_path: Path,
    source_path: Path,
    camera_id: str,
    recorded_start: datetime,
    ground_truth: dict,
    vlm_engine,
) -> tuple[dict, list[dict]]:
    probe = _probe_video(compressed_path)
    fps = max(float(probe.get("fps") or 0.0), 1.0)
    grouped: dict[str, dict] = {}
    for raw_frame_idx, entries in ground_truth.items():
        try:
            frame_idx = int(raw_frame_idx)
        except Exception:
            continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            visible = entry.get("2d bounding box visible") or {}
            bbox = visible.get(camera_id)
            if not bbox:
                continue
            object_id = str(entry.get("object id"))
            record = grouped.setdefault(
                object_id,
                {
                    "frames": [],
                    "bboxes": [],
                    "world_locations": [],
                },
            )
            record["frames"].append(frame_idx)
            record["bboxes"].append([int(value) for value in bbox])
            record["world_locations"].append(entry.get("3d location") or [])

    metadata_path = METADATA_DIR / f"{camera_id}.json"
    video_payload = _build_video_payload(
        source_path=source_path,
        compressed_path=compressed_path,
        metadata_path=metadata_path,
        camera_id=camera_id,
        recorded_start=recorded_start,
        probe=probe,
    )

    candidates: list[dict] = []
    for object_id, payload in sorted(grouped.items(), key=lambda item: int(item[0])):
        frames = payload["frames"]
        bboxes = payload["bboxes"]
        if not frames:
            continue
        largest_index = max(range(len(bboxes)), key=lambda index: _bbox_area(bboxes[index]))
        representative_frame = int(frames[largest_index])
        representative_bbox = [int(value) for value in bboxes[largest_index]]
        start_second = round(min(frames) / fps, 3)
        end_second = round((max(frames) + 1) / fps, 3)
        content_frames = []
        for sample_frame in _frame_sample_positions(frames, count=DEFAULT_CONTENT_FRAME_SAMPLES):
            sample_index = frames.index(sample_frame)
            content_frames.append(
                {
                    "frame_idx": int(sample_frame),
                    "second": round(sample_frame / fps, 3),
                    "actual_time": _iso(recorded_start + timedelta(seconds=(sample_frame / fps))),
                    "bbox": [int(value) for value in bboxes[sample_index]],
                    "world_location": payload["world_locations"][sample_index],
                }
            )
        candidate = {
            "candidate_id": f"{camera_id}_person_{object_id}",
            "human_key": str(object_id),
            "global_person_id": str(object_id),
            "video_id": compressed_path.name,
            "camera_id": camera_id,
            "track_id": str(object_id),
            "candidate_type": "person_track",
            "source_mode": "nvidia_ground_truth",
            "frame_idx": representative_frame,
            "start_frame": int(min(frames)),
            "end_frame": int(max(frames)),
            "start_second": start_second,
            "end_second": end_second,
            "actual_start_time": _iso(recorded_start + timedelta(seconds=start_second)),
            "actual_end_time": _iso(recorded_start + timedelta(seconds=end_second)),
            "bbox": representative_bbox,
            "representative_bbox": representative_bbox,
            "content_frames": content_frames,
            "timeline": _segment_track_timeline(
                frames=frames,
                bboxes=bboxes,
                fps=fps,
                recorded_start=recorded_start,
                max_segments=DEFAULT_TIMELINE_SEGMENTS,
                world_locations=payload["world_locations"],
            ),
            "person_caption": "",
            "appearance_summary": "",
            "semantic_attributes": [],
            "visibility_scores": {
                "full_body": 1.0,
                "upper_body": 1.0,
                "lower_body": 1.0,
            },
            "world_position": {
                "x": payload["world_locations"][largest_index][0] if payload["world_locations"][largest_index] else None,
                "y": payload["world_locations"][largest_index][1] if payload["world_locations"][largest_index] else None,
                "z": payload["world_locations"][largest_index][2] if payload["world_locations"][largest_index] else None,
            },
            "pipeline_profile": DEFAULT_PIPELINE_PROFILE,
            "reid_profile": DEFAULT_REID_PROFILE,
            "search_profile": DEFAULT_SEARCH_PROFILE,
            "query_kind": "human",
            "vector_model": DEFAULT_TEXT_MODEL,
            "candidate_vector": [],
        }
        candidates.append(candidate)

    _apply_person_captions(candidates, compressed_path, vlm_engine)
    for candidate in candidates:
        _finalize_candidate_text(candidate)
    return video_payload, candidates


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
    frame_count = int(probe.get("frame_count") or 0)
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

    stride = _sample_stride(fps, DEFAULT_DETECTION_FPS)
    next_track_id = 1
    active_tracks: dict[str, dict] = {}
    all_tracks: dict[str, dict] = {}

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        detections = [bbox for bbox in _detect_people_hog(frame) if _bbox_area(bbox) >= DEFAULT_MIN_PERSON_AREA]
        matched_track_ids: set[str] = set()
        for bbox in sorted(detections, key=_bbox_area, reverse=True):
            best_track_id = None
            best_iou = 0.0
            for track_id, track in active_tracks.items():
                if track_id in matched_track_ids:
                    continue
                overlap = _bbox_iou(track["last_bbox"], bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_track_id = track_id
            if best_track_id and best_iou >= DEFAULT_TRACK_IOU:
                track = active_tracks[best_track_id]
            else:
                best_track_id = str(next_track_id)
                next_track_id += 1
                track = {
                    "track_id": best_track_id,
                    "frames": [],
                    "bboxes": [],
                    "largest_bbox": bbox,
                    "largest_area": 0,
                    "last_bbox": bbox,
                }
                active_tracks[best_track_id] = track
                all_tracks[best_track_id] = track
            matched_track_ids.add(best_track_id)
            track["frames"].append(frame_idx)
            track["bboxes"].append([int(value) for value in bbox])
            track["last_bbox"] = [int(value) for value in bbox]
            if _bbox_area(bbox) >= int(track.get("largest_area", 0)):
                track["largest_bbox"] = [int(value) for value in bbox]
                track["largest_area"] = _bbox_area(bbox)
        active_tracks = {track_id: track for track_id, track in active_tracks.items() if track_id in matched_track_ids}
        frame_idx += 1

    cap.release()

    candidates: list[dict] = []
    for track_id, track in sorted(all_tracks.items(), key=lambda item: int(item[0])):
        frames = track["frames"]
        bboxes = track["bboxes"]
        if len(frames) < DEFAULT_MIN_TRACK_FRAMES:
            continue
        representative_frame = int(frames[-1])
        representative_bbox = [int(value) for value in track["largest_bbox"]]
        start_second = round(min(frames) / fps, 3)
        end_second = round((max(frames) + 1) / fps, 3)
        content_frames = []
        for sample_frame in _frame_sample_positions(frames, count=DEFAULT_CONTENT_FRAME_SAMPLES):
            sample_index = frames.index(sample_frame)
            content_frames.append(
                {
                    "frame_idx": int(sample_frame),
                    "second": round(sample_frame / fps, 3),
                    "actual_time": _iso(recorded_start + timedelta(seconds=(sample_frame / fps))),
                    "bbox": [int(value) for value in bboxes[sample_index]],
                }
            )
        candidate = {
            "candidate_id": f"{compressed_path.stem}_person_{track_id}",
            "human_key": f"{compressed_path.stem}_person_{track_id}",
            "global_person_id": None,
            "video_id": compressed_path.name,
            "camera_id": camera_id,
            "track_id": str(track_id),
            "candidate_type": "person_track",
            "source_mode": "hog_tracking",
            "frame_idx": representative_frame,
            "start_frame": int(min(frames)),
            "end_frame": int(max(frames)),
            "start_second": start_second,
            "end_second": end_second,
            "actual_start_time": _iso(recorded_start + timedelta(seconds=start_second)),
            "actual_end_time": _iso(recorded_start + timedelta(seconds=end_second)),
            "bbox": representative_bbox,
            "representative_bbox": representative_bbox,
            "content_frames": content_frames,
            "timeline": _segment_track_timeline(
                frames=frames,
                bboxes=bboxes,
                fps=fps,
                recorded_start=recorded_start,
                max_segments=DEFAULT_TIMELINE_SEGMENTS,
            ),
            "person_caption": "",
            "appearance_summary": "",
            "semantic_attributes": [],
            "visibility_scores": {
                "full_body": 0.82,
                "upper_body": 0.88,
                "lower_body": 0.70,
            },
            "world_position": None,
            "pipeline_profile": DEFAULT_PIPELINE_PROFILE,
            "reid_profile": DEFAULT_REID_PROFILE,
            "search_profile": DEFAULT_SEARCH_PROFILE,
            "query_kind": "human",
            "vector_model": DEFAULT_TEXT_MODEL,
            "candidate_vector": [],
        }
        candidates.append(candidate)

    _apply_person_captions(candidates, compressed_path, vlm_engine)
    for candidate in candidates:
        _finalize_candidate_text(candidate)
    return video_payload, candidates


def _write_video_metadata(metadata_path: Path, video_payload: dict, people: list[dict]) -> None:
    payload = {
        "schema_version": "hospital_person_metadata_v2",
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


def bootstrap_nvidia_hospital_dataset(vlm_engine, limit: int = 31) -> dict:
    ensure_pipeline_layout()
    reset_managed_artifacts()
    ground_truth = _load_nvidia_ground_truth()
    source_videos = list_nvidia_hospital_videos(limit=limit)
    base_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    queue_entries: list[dict] = []
    processed: list[dict] = []

    for index, source_path in enumerate(source_videos):
        camera_id = source_path.stem
        compressed_path = Path(
            compress_video(
                str(source_path),
                str(COMPRESSED_VIDEO_DIR),
                use_h265=True,
                output_filename=source_path.name,
            )
        )
        video_payload, people = _build_gt_people(
            compressed_path=compressed_path,
            source_path=source_path,
            camera_id=camera_id,
            recorded_start=base_start + timedelta(seconds=index * 10),
            ground_truth=ground_truth,
            vlm_engine=vlm_engine,
        )
        metadata_path = Path(video_payload["metadata_path"])
        _write_video_metadata(metadata_path, video_payload, people)
        queue_entries.append(
            {
                "video_id": video_payload["video_id"],
                "camera_id": camera_id,
                "compressed_path": str(compressed_path),
                "metadata_path": str(metadata_path),
            }
        )
        processed.append(
            {
                "video_id": video_payload["video_id"],
                "camera_id": camera_id,
                "person_count": len(people),
            }
        )

    _write_queue_state(queue_entries)
    return {
        "processed_videos": len(processed),
        "queue_size": len(queue_entries),
        "people_indexed": sum(item["person_count"] for item in processed),
        "videos": processed,
    }


def save_uploaded_video(file_bytes: bytes, original_name: str) -> Path:
    ensure_pipeline_layout()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = Path(original_name).suffix or ".mp4"
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

    resolved_camera = _slug(camera_id or source_path.stem)
    resolved_start = parse_recorded_start(recorded_start, datetime.fromtimestamp(source_path.stat().st_mtime, tz=timezone.utc))
    video_id = f"{resolved_camera}_{resolved_start.strftime('%Y%m%dT%H%M%SZ')}.mp4"
    compressed_path = Path(
        compress_video(
            str(source_path),
            str(COMPRESSED_VIDEO_DIR),
            use_h265=True,
            output_filename=video_id,
        )
    )
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
