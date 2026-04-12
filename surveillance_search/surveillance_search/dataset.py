from __future__ import annotations

import csv
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

from .config import DEFAULT_DATASET_TYPE, DEFAULT_FPS
from .enrichment import apply_enrichment
from .models import Moment

TOKEN_RE = re.compile(r"[a-z0-9]+")
PERSONPATH22_VISIBLE_PATTERNS = ("anno_visible*.json", "anno_visible*/*.json")
PERSONPATH22_AMODAL_PATTERNS = ("anno_amodal*.json", "anno_amodal*/*.json")
DEFAULT_PERSON_KEYWORDS = ("person", "pedestrian", "human", "surveillance", "track")
PERSONPATH22_HTTP_ROOT = "https://tracking-dataset-eccv-2022.s3.amazonaws.com/dataset"
PERSONPATH22_ANNOTATION_OBJECTS = (
    ("annotation/anno_visible.zip", "annotations"),
    ("annotation/anno_amodal.zip", "annotations"),
    ("annotation/splits.json", "annotations"),
)
PERSONPATH22_VIDEO_OBJECTS = (
    ("raw_data/videos.zip", "raw_data"),
)
PERSONPATH22_CANONICAL_VIDEO_DIRS = ("raw_data", "videos", "video")
PERSONPATH22_SPLITS = ("train", "val", "test")


def _normalize_dataset_type(dataset_type: str) -> str:
    normalized = str(dataset_type or DEFAULT_DATASET_TYPE).strip().lower()
    if normalized != "personpath22":
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    return normalized


def download_from_huggingface(
    dataset_root: Path,
    repo_id: str,
    locations: list[str],
    splits: list[str],
    include_videos: bool = False,
    include_docs: bool = False,
    dataset_type: str = DEFAULT_DATASET_TYPE,
) -> Path:
    _normalize_dataset_type(dataset_type)
    return download_personpath22_public(
        dataset_root=dataset_root,
        include_videos=include_videos,
    )


def download_personpath22_public(
    dataset_root: Path,
    include_videos: bool = False,
    force: bool = False,
) -> Path:
    dataset_root.mkdir(parents=True, exist_ok=True)
    objects = list(PERSONPATH22_ANNOTATION_OBJECTS)
    if include_videos:
        objects.extend(PERSONPATH22_VIDEO_OBJECTS)

    for object_key, relative_dir in objects:
        destination_dir = dataset_root / relative_dir
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / Path(object_key).name
        if destination_path.exists() and not force:
            if destination_path.suffix.lower() == ".zip":
                if _personpath22_archive_already_materialized(dataset_root, destination_path.name):
                    if not _is_valid_zip_archive(destination_path):
                        destination_path.unlink(missing_ok=True)
                    continue
                if not _is_valid_zip_archive(destination_path):
                    destination_path.unlink(missing_ok=True)
            _materialize_personpath22_archives(dataset_root, include_videos=include_videos)
            if destination_path.exists():
                continue
        url = f"{PERSONPATH22_HTTP_ROOT}/{object_key}"
        _download_url_to_path(url, destination_path)
        if destination_path.suffix.lower() == ".zip" and not _is_valid_zip_archive(destination_path):
            destination_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded archive is not a valid zip file: {destination_path}. "
                "The download was likely interrupted. Rerun the command to fetch it again."
            )
        _materialize_personpath22_archives(dataset_root, include_videos=include_videos)
    return dataset_root


def _download_url_to_path(url: str, destination_path: Path) -> None:
    temp_path = destination_path.with_name(f"{destination_path.name}.partial")
    temp_path.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url) as response, temp_path.open("wb") as handle:
            expected_length = response.headers.get("Content-Length")
            shutil.copyfileobj(response, handle, length=1024 * 1024 * 8)
        if expected_length:
            actual_length = temp_path.stat().st_size
            if actual_length != int(expected_length):
                raise RuntimeError(
                    f"Download was incomplete for {destination_path.name}: expected {expected_length} bytes, "
                    f"received {actual_length} bytes."
                )
        temp_path.replace(destination_path)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _is_valid_zip_archive(zip_path: Path) -> bool:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.infolist()
        return True
    except (FileNotFoundError, zipfile.BadZipFile, OSError):
        return False


def _personpath22_archive_already_materialized(dataset_root: Path, archive_name: str) -> bool:
    annotations_dir = dataset_root / "annotations"
    raw_data_dir = dataset_root / "raw_data"
    if archive_name == "anno_visible.zip":
        return (annotations_dir / "anno_visible_2022").exists()
    if archive_name == "anno_amodal.zip":
        return (annotations_dir / "anno_amodal_2022").exists()
    if archive_name == "videos.zip":
        return any(raw_data_dir.rglob("*.mp4"))
    return False


def _extract_zip_archive(zip_path: Path) -> None:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(zip_path.parent)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"Archive is not a valid zip file: {zip_path}. Delete it and rerun the download command."
        ) from exc
    zip_path.unlink(missing_ok=True)


def _materialize_personpath22_archives(dataset_root: Path, include_videos: bool) -> None:
    annotations_dir = dataset_root / "annotations"
    for archive_name, extracted_dir_name in (
        ("anno_visible.zip", "anno_visible_2022"),
        ("anno_amodal.zip", "anno_amodal_2022"),
    ):
        archive_path = annotations_dir / archive_name
        extracted_dir = annotations_dir / extracted_dir_name
        if archive_path.exists() and not extracted_dir.exists():
            _extract_zip_archive(archive_path)

    if include_videos:
        raw_data_dir = dataset_root / "raw_data"
        video_archive = raw_data_dir / "videos.zip"
        if video_archive.exists() and not any(raw_data_dir.rglob("*.mp4")):
            _extract_zip_archive(video_archive)


def _matches_filters(value: str, wanted_values: set[str]) -> bool:
    return not wanted_values or value in wanted_values


def iter_label_files(
    dataset_root: Path,
    locations: list[str] | None = None,
    splits: list[str] | None = None,
    dataset_type: str = DEFAULT_DATASET_TYPE,
) -> Iterable[Path]:
    _normalize_dataset_type(dataset_type)
    annotation_files = list(_iter_personpath22_annotation_files(dataset_root))
    if annotation_files:
        for annotation_path in annotation_files:
            yield annotation_path
        return

    for gt_path in _iter_personpath22_gt_files(dataset_root):
        yield gt_path


def _iter_personpath22_annotation_roots(dataset_root: Path) -> Iterable[Path]:
    annotations_root = dataset_root / "annotations"
    if annotations_root.is_dir():
        yield annotations_root
    if dataset_root.name == "annotations":
        yield dataset_root


def _iter_personpath22_gt_files(dataset_root: Path) -> Iterable[Path]:
    for split in PERSONPATH22_SPLITS:
        split_root = dataset_root / split
        if not split_root.is_dir():
            continue
        for gt_path in sorted(split_root.rglob("gt.txt")):
            if gt_path.is_file():
                yield gt_path


def _iter_personpath22_annotation_files(dataset_root: Path) -> Iterable[Path]:
    roots = list(_iter_personpath22_annotation_roots(dataset_root))
    if not roots:
        return
    for patterns in (PERSONPATH22_VISIBLE_PATTERNS, PERSONPATH22_AMODAL_PATTERNS):
        matches: list[Path] = []
        for root in roots:
            for pattern in patterns:
                matches.extend(sorted(root.rglob(pattern)))
        if matches:
            deduped = sorted({path for path in matches if path.is_file() and path.suffix.lower() == ".json"})
            for path in deduped:
                yield path
            return


def _iter_personpath22_video_files(dataset_root: Path) -> Iterable[Path]:
    yielded: set[Path] = set()
    for directory_name in PERSONPATH22_CANONICAL_VIDEO_DIRS:
        video_root = dataset_root / directory_name
        if not video_root.is_dir():
            continue
        for video_path in sorted(video_root.rglob("*.mp4")):
            if video_path.is_file() and video_path not in yielded:
                yielded.add(video_path)
                yield video_path

    for split in PERSONPATH22_SPLITS:
        split_root = dataset_root / split
        if not split_root.is_dir():
            continue
        for video_path in sorted(split_root.rglob("*.mp4")):
            if video_path.is_file() and video_path not in yielded:
                yielded.add(video_path)
                yield video_path


def iter_dataset_source_files(
    dataset_root: Path,
    dataset_type: str = DEFAULT_DATASET_TYPE,
    locations: list[str] | None = None,
    splits: list[str] | None = None,
) -> Iterable[Path]:
    _normalize_dataset_type(dataset_type)
    yielded: set[Path] = set()
    for annotation_path in iter_label_files(dataset_root, dataset_type=dataset_type):
        if annotation_path not in yielded:
            yielded.add(annotation_path)
            yield annotation_path

    wanted_locations = set(locations or [])
    for video_path in _iter_personpath22_video_files(dataset_root):
        stem = video_path.stem
        if wanted_locations and stem not in wanted_locations and video_path.parent.name not in wanted_locations:
            continue
        if video_path not in yielded:
            yielded.add(video_path)
            yield video_path


def _bbox_xywh_to_xyxy(raw_bbox: list[float] | tuple[float, ...]) -> list[int]:
    if len(raw_bbox) < 4:
        return [0, 0, 0, 0]
    left = int(round(float(raw_bbox[0])))
    top = int(round(float(raw_bbox[1])))
    width = float(raw_bbox[2])
    height = float(raw_bbox[3])
    return [
        left,
        top,
        int(round(left + width)),
        int(round(top + height)),
    ]


def _load_personpath22_annotation(annotation_path: Path) -> dict:
    try:
        with annotation_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse PersonPath22 annotation file: {annotation_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported PersonPath22 annotation payload in {annotation_path}")
    return payload


def _normalize_video_name(raw_value: object) -> str:
    value = str(raw_value or "").strip()
    if not value:
        return ""
    return Path(value).name


def _resolve_personpath22_video_path(
    dataset_root: Path,
    video_name: str,
    file_name: str | None = None,
) -> Path:
    candidates: list[Path] = []
    raw_video_name = _normalize_video_name(video_name or file_name or "")
    file_candidate = _normalize_video_name(file_name or "")
    base_names = [name for name in {raw_video_name, file_candidate} if name]

    for name in list(base_names):
        if not Path(name).suffix:
            base_names.append(f"{name}.mp4")

    for name in base_names:
        path_obj = Path(name)
        if path_obj.is_absolute():
            candidates.append(path_obj)
        else:
            candidates.extend(
                [
                    dataset_root / name,
                    dataset_root / "raw_data" / name,
                    dataset_root / "videos" / name,
                    dataset_root / "video" / name,
                ]
            )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if base_names:
        return dataset_root / "raw_data" / base_names[-1]
    return dataset_root / "raw_data" / "unknown.mp4"


def _extract_numeric_suffix(value: str | None, fallback: int) -> int:
    if not value:
        return fallback
    matches = re.findall(r"(\d+)", str(value))
    if not matches:
        return fallback
    return int(matches[-1])


def _frame_index_from_image(image_payload: dict | None, fallback: int) -> int:
    if not image_payload:
        return fallback

    for key in ("frame_id", "frame_index", "frame_idx", "index"):
        value = image_payload.get(key)
        if value is None:
            continue
        return max(int(value), 0)

    if "frame_num" in image_payload:
        return max(int(image_payload["frame_num"]) - 1, 0)

    return _extract_numeric_suffix(str(image_payload.get("file_name", "")), fallback)


def _personpath22_split_from_path(annotation_path: Path) -> str:
    split_names = {"train", "val", "valid", "validation", "test"}
    for part in annotation_path.parts:
        lowered = part.lower()
        if lowered in split_names:
            return "val" if lowered in {"valid", "validation"} else lowered
    return "train"


def _personpath22_split(video_payload: dict, annotation_path: Path, payload: dict) -> str:
    for source in (video_payload, payload):
        for key in ("split", "subset", "set", "partition"):
            value = source.get(key) if isinstance(source, dict) else None
            if value:
                return str(value).strip().lower()
    return _personpath22_split_from_path(annotation_path)


def _personpath22_location(video_payload: dict, video_name: str) -> str:
    for key in ("location", "camera", "scene", "source"):
        value = video_payload.get(key)
        if value:
            return str(value).strip()
    return Path(video_name).stem or "personpath22"


def _personpath22_visibility_score(raw_annotation: dict) -> float | None:
    for key in ("visibility", "vis", "visible_ratio"):
        value = raw_annotation.get(key)
        if value is None:
            continue
        score = float(value)
        if score > 1.0 and score <= 100.0:
            score = score / 100.0
        return max(0.0, min(score, 1.0))
    return None


def _personpath22_fps(video_payload: dict, fallback: float) -> float:
    for key in ("fps", "frame_rate"):
        value = video_payload.get(key)
        if value is None:
            continue
        return float(value)
    return float(fallback)


def _personpath22_caption_from_visibility(visibility: float | None) -> str:
    if visibility is None:
        return "person"
    if visibility < 0.35:
        return "heavily occluded person"
    if visibility < 0.7:
        return "partially occluded person"
    return "person"


def _personpath22_keywords(
    location: str,
    split: str,
    track_id: str,
    video_name: str,
    captions: list[str],
) -> list[str]:
    text = " ".join([location, split, track_id, Path(video_name).stem, *DEFAULT_PERSON_KEYWORDS, *captions]).lower()
    return sorted(set(TOKEN_RE.findall(text)))


def _personpath22_text(
    location: str,
    split: str,
    track_id: str,
    video_name: str,
    captions: list[str],
) -> str:
    caption_text = ", ".join(captions) if captions else "person"
    return (
        f"{caption_text}. "
        f"person pedestrian track in surveillance video {Path(video_name).stem}. "
        f"camera location {location}. "
        f"dataset split {split}. "
        f"tracked person {track_id}."
    )


def _person_category_ids(payload: dict) -> set[int]:
    categories = payload.get("categories")
    if not isinstance(categories, list):
        return set()

    person_ids: set[int] = set()
    for category in categories:
        if not isinstance(category, dict):
            continue
        name = str(category.get("name", "")).strip().lower()
        if name in {"person", "pedestrian", "people", "human"}:
            try:
                person_ids.add(int(category["id"]))
            except Exception:
                continue
    return person_ids


def _frame_index_from_personpath22_entity(entity: dict, effective_fps: float, fallback: int) -> int:
    blob = entity.get("blob")
    if isinstance(blob, dict):
        for key in ("frame_idx", "frame_index", "index"):
            value = blob.get(key)
            if value is None:
                continue
            return max(int(value), 0)
        if "frame_num" in blob:
            return max(int(blob["frame_num"]) - 1, 0)

    time_value = entity.get("time")
    if time_value is not None and effective_fps > 0:
        seconds = float(time_value) / 1000.0
        return max(int(round(seconds * effective_fps)), 0)

    return fallback


def _build_moments_from_personpath22_entity_json(
    annotation_path: Path,
    dataset_root: Path,
    payload: dict,
    fps: float,
    group_by_track: bool,
) -> list[Moment]:
    entities = payload.get("entities", [])
    metadata = payload.get("metadata", {})
    if not isinstance(entities, list) or not isinstance(metadata, dict):
        raise ValueError(f"Unsupported PersonPath22 entity schema in {annotation_path}")

    raw_video_name = _normalize_video_name(metadata.get("data_path") or annotation_path.stem)
    video_payload = dict(metadata)
    location = _personpath22_location(video_payload, raw_video_name)
    split = _personpath22_split(video_payload, annotation_path, payload)
    effective_fps = _personpath22_fps(video_payload, fps)
    video_path = _resolve_personpath22_video_path(
        dataset_root=dataset_root,
        video_name=raw_video_name,
        file_name=metadata.get("data_path"),
    )
    grouped: dict[str, dict] = {}

    for annotation_index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            continue

        frame_idx = _frame_index_from_personpath22_entity(entity, effective_fps, annotation_index)
        track_value = entity.get("id")
        if track_value is None:
            track_value = f"frame-{frame_idx}-ann-{annotation_index}"
        track_key = str(track_value)

        if group_by_track:
            moment_id = f"{location}:{split}:track:{track_key}"
        else:
            moment_id = f"{location}:{split}:frame-{frame_idx}:track:{track_key}"

        bbox = _bbox_xywh_to_xyxy(entity.get("bb") or [0, 0, 0, 0])
        visibility = _personpath22_visibility_score(entity)
        caption = _personpath22_caption_from_visibility(visibility)

        if moment_id not in grouped:
            grouped[moment_id] = {
                "id": moment_id,
                "location": location,
                "split": split,
                "fps": effective_fps,
                "video_name": raw_video_name or video_path.name,
                "video_path": str(video_path),
                "label_path": str(annotation_path),
                "track_id": track_key,
                "start_frame": frame_idx,
                "end_frame": frame_idx,
                "captions": {caption},
                "representative_bbox": bbox,
                "sample_frames": [frame_idx],
            }
            continue

        entry = grouped[moment_id]
        entry["end_frame"] = max(entry["end_frame"], frame_idx)
        entry["captions"].add(caption)
        if len(entry["sample_frames"]) < 8 and frame_idx not in entry["sample_frames"]:
            entry["sample_frames"].append(frame_idx)

    moments: list[Moment] = []
    for payload_entry in grouped.values():
        captions = sorted(payload_entry["captions"])
        start_second = payload_entry["start_frame"] / effective_fps
        end_second = (payload_entry["end_frame"] + 1) / effective_fps
        moments.append(
            Moment(
                id=payload_entry["id"],
                location=payload_entry["location"],
                split=payload_entry["split"],
                video_path=payload_entry["video_path"],
                label_path=payload_entry["label_path"],
                track_id=payload_entry["track_id"],
                fps=effective_fps,
                start_frame=payload_entry["start_frame"],
                end_frame=payload_entry["end_frame"],
                start_second=round(start_second, 3),
                end_second=round(end_second, 3),
                captions=captions,
                representative_bbox=payload_entry["representative_bbox"],
                sample_frames=sorted(payload_entry["sample_frames"]),
                text=_personpath22_text(
                    location=payload_entry["location"],
                    split=payload_entry["split"],
                    track_id=payload_entry["track_id"],
                    video_name=payload_entry["video_name"],
                    captions=captions,
                ),
                keywords=_personpath22_keywords(
                    location=payload_entry["location"],
                    split=payload_entry["split"],
                    track_id=payload_entry["track_id"],
                    video_name=payload_entry["video_name"],
                    captions=captions,
                ),
            )
        )

    return sorted(moments, key=lambda item: (item.location, item.split, item.start_frame, item.id))


def build_moments_from_personpath22_json(
    annotation_path: Path,
    dataset_root: Path,
    fps: float = DEFAULT_FPS,
    group_by_track: bool = True,
) -> list[Moment]:
    payload = _load_personpath22_annotation(annotation_path)
    if isinstance(payload.get("entities"), list) and isinstance(payload.get("metadata"), dict):
        return _build_moments_from_personpath22_entity_json(
            annotation_path=annotation_path,
            dataset_root=dataset_root,
            payload=payload,
            fps=fps,
            group_by_track=group_by_track,
        )

    videos = payload.get("videos", [])
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    if not isinstance(videos, list) or not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError(f"Unsupported PersonPath22 schema in {annotation_path}")

    video_by_id = {
        item.get("id"): item
        for item in videos
        if isinstance(item, dict) and item.get("id") is not None
    }
    image_by_id = {
        item.get("id"): item
        for item in images
        if isinstance(item, dict) and item.get("id") is not None
    }
    person_category_ids = _person_category_ids(payload)
    grouped: dict[str, dict] = {}

    for annotation_index, raw_annotation in enumerate(annotations):
        if not isinstance(raw_annotation, dict):
            continue

        category_id = raw_annotation.get("category_id")
        if person_category_ids and category_id is not None:
            try:
                if int(category_id) not in person_category_ids:
                    continue
            except Exception:
                continue

        image_payload = image_by_id.get(raw_annotation.get("image_id"))
        video_id = raw_annotation.get("video_id")
        if video_id is None and image_payload is not None:
            video_id = image_payload.get("video_id")
        video_payload = video_by_id.get(video_id, {}) if video_id is not None else {}

        raw_video_name = _normalize_video_name(
            video_payload.get("file_name")
            or video_payload.get("name")
            or video_payload.get("video")
            or ((image_payload or {}).get("video_name"))
            or ((image_payload or {}).get("file_name"))
        )
        location = _personpath22_location(video_payload, raw_video_name)
        split = _personpath22_split(video_payload, annotation_path, payload)
        frame_idx = _frame_index_from_image(image_payload, annotation_index)

        track_value = (
            raw_annotation.get("track_id")
            or raw_annotation.get("instance_id")
            or raw_annotation.get("person_id")
            or raw_annotation.get("tracklet_id")
            or raw_annotation.get("id")
        )
        if track_value is None:
            track_value = f"frame-{frame_idx}-ann-{annotation_index}"
        track_key = str(track_value)

        if group_by_track:
            moment_id = f"{location}:{split}:track:{track_key}"
        else:
            moment_id = f"{location}:{split}:frame-{frame_idx}:track:{track_key}"

        visibility = _personpath22_visibility_score(raw_annotation)
        caption = _personpath22_caption_from_visibility(visibility)
        bbox = _bbox_xywh_to_xyxy(raw_annotation.get("bbox") or [0, 0, 0, 0])
        video_path = _resolve_personpath22_video_path(
            dataset_root=dataset_root,
            video_name=raw_video_name,
            file_name=video_payload.get("file_name"),
        )

        if moment_id not in grouped:
            grouped[moment_id] = {
                "id": moment_id,
                "location": location,
                "split": split,
                "fps": _personpath22_fps(video_payload, fps),
                "video_name": raw_video_name or video_path.name,
                "video_path": str(video_path),
                "label_path": str(annotation_path),
                "track_id": track_key,
                "start_frame": frame_idx,
                "end_frame": frame_idx,
                "captions": {caption},
                "representative_bbox": bbox,
                "sample_frames": [frame_idx],
            }
            continue

        entry = grouped[moment_id]
        entry["end_frame"] = max(entry["end_frame"], frame_idx)
        entry["captions"].add(caption)
        if len(entry["sample_frames"]) < 8 and frame_idx not in entry["sample_frames"]:
            entry["sample_frames"].append(frame_idx)

    moments: list[Moment] = []
    for payload_entry in grouped.values():
        captions = sorted(payload_entry["captions"])
        effective_fps = float(payload_entry["fps"])
        start_second = payload_entry["start_frame"] / effective_fps
        end_second = (payload_entry["end_frame"] + 1) / effective_fps
        moments.append(
            Moment(
                id=payload_entry["id"],
                location=payload_entry["location"],
                split=payload_entry["split"],
                video_path=payload_entry["video_path"],
                label_path=payload_entry["label_path"],
                track_id=payload_entry["track_id"],
                fps=effective_fps,
                start_frame=payload_entry["start_frame"],
                end_frame=payload_entry["end_frame"],
                start_second=round(start_second, 3),
                end_second=round(end_second, 3),
                captions=captions,
                representative_bbox=payload_entry["representative_bbox"],
                sample_frames=sorted(payload_entry["sample_frames"]),
                text=_personpath22_text(
                    location=payload_entry["location"],
                    split=payload_entry["split"],
                    track_id=payload_entry["track_id"],
                    video_name=payload_entry["video_name"],
                    captions=captions,
                ),
                keywords=_personpath22_keywords(
                    location=payload_entry["location"],
                    split=payload_entry["split"],
                    track_id=payload_entry["track_id"],
                    video_name=payload_entry["video_name"],
                    captions=captions,
                ),
            )
        )

    return sorted(moments, key=lambda item: (item.location, item.split, item.start_frame, item.id))


def _split_from_gt_path(gt_path: Path) -> str:
    split_names = {"train", "val", "test"}
    for part in gt_path.parts:
        lowered = part.lower()
        if lowered in split_names:
            return lowered
    return "train"


def build_moments_from_personpath22_gt(
    gt_path: Path,
    dataset_root: Path,
    fps: float = DEFAULT_FPS,
    group_by_track: bool = True,
) -> list[Moment]:
    video_dir = gt_path.parent.parent
    video_name = video_dir.name
    location = Path(video_name).stem or "personpath22"
    split = _split_from_gt_path(gt_path)
    video_path = _resolve_personpath22_video_path(dataset_root, video_name)
    grouped: dict[str, dict] = {}

    with gt_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row_index, row in enumerate(reader):
            if len(row) < 6:
                continue

            frame_idx = max(int(float(row[0])) - 1, 0)
            track_key = str(int(float(row[1])))
            left = float(row[2])
            top = float(row[3])
            width = float(row[4])
            height = float(row[5])
            bbox = _bbox_xywh_to_xyxy([left, top, width, height])
            moment_id = (
                f"{location}:{split}:track:{track_key}"
                if group_by_track
                else f"{location}:{split}:frame-{frame_idx}:track:{track_key}:{row_index}"
            )

            if moment_id not in grouped:
                grouped[moment_id] = {
                    "id": moment_id,
                    "location": location,
                    "split": split,
                    "video_name": video_name,
                    "video_path": str(video_path),
                    "label_path": str(gt_path),
                    "track_id": track_key,
                    "start_frame": frame_idx,
                    "end_frame": frame_idx,
                    "captions": {"person"},
                    "representative_bbox": bbox,
                    "sample_frames": [frame_idx],
                }
                continue

            entry = grouped[moment_id]
            entry["end_frame"] = max(entry["end_frame"], frame_idx)
            if len(entry["sample_frames"]) < 8 and frame_idx not in entry["sample_frames"]:
                entry["sample_frames"].append(frame_idx)

    moments: list[Moment] = []
    for payload_entry in grouped.values():
        captions = sorted(payload_entry["captions"])
        start_second = payload_entry["start_frame"] / fps
        end_second = (payload_entry["end_frame"] + 1) / fps
        moments.append(
            Moment(
                id=payload_entry["id"],
                location=payload_entry["location"],
                split=payload_entry["split"],
                video_path=payload_entry["video_path"],
                label_path=payload_entry["label_path"],
                track_id=payload_entry["track_id"],
                fps=float(fps),
                start_frame=payload_entry["start_frame"],
                end_frame=payload_entry["end_frame"],
                start_second=round(start_second, 3),
                end_second=round(end_second, 3),
                captions=captions,
                representative_bbox=payload_entry["representative_bbox"],
                sample_frames=sorted(payload_entry["sample_frames"]),
                text=_personpath22_text(
                    location=payload_entry["location"],
                    split=payload_entry["split"],
                    track_id=payload_entry["track_id"],
                    video_name=payload_entry["video_name"],
                    captions=captions,
                ),
                keywords=_personpath22_keywords(
                    location=payload_entry["location"],
                    split=payload_entry["split"],
                    track_id=payload_entry["track_id"],
                    video_name=payload_entry["video_name"],
                    captions=captions,
                ),
            )
        )

    return sorted(moments, key=lambda item: (item.location, item.split, item.start_frame, item.id))


def _moment_matches_filters(moment: Moment, locations: set[str], splits: set[str]) -> bool:
    if splits and moment.split not in splits:
        return False
    if not locations:
        return True
    video_stem = Path(moment.video_path).stem
    return moment.location in locations or video_stem in locations


def collect_moments(
    dataset_root: Path,
    fps: float = DEFAULT_FPS,
    group_by_track: bool = True,
    locations: list[str] | None = None,
    splits: list[str] | None = None,
    dataset_type: str = DEFAULT_DATASET_TYPE,
) -> list[Moment]:
    dataset_type = _normalize_dataset_type(dataset_type)
    wanted_locations = set(locations or [])
    wanted_splits = set(splits or [])
    moments = []
    for annotation_path in iter_label_files(dataset_root, dataset_type=dataset_type):
        if annotation_path.suffix.lower() == ".json":
            source_moments = build_moments_from_personpath22_json(
                annotation_path,
                dataset_root=dataset_root,
                fps=fps,
                group_by_track=group_by_track,
            )
        else:
            source_moments = build_moments_from_personpath22_gt(
                annotation_path,
                dataset_root=dataset_root,
                fps=fps,
                group_by_track=group_by_track,
            )
        moments.extend(
            moment for moment in source_moments if _moment_matches_filters(moment, wanted_locations, wanted_splits)
        )
    return apply_enrichment(moments, dataset_root)
