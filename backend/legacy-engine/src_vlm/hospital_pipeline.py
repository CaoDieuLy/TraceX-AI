from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


STRICT_PIPELINE_NAME = "fixed_strict_pipeline"


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip()).strip("-._") or "camera"


def _runtime_removed() -> RuntimeError:
    return RuntimeError(
        f"Local legacy video analysis has been removed. Use the strict {STRICT_PIPELINE_NAME} "
        "runtime through tracking-service instead."
    )


def _build_detected_people(
    *,
    compressed_path: Path,
    source_path: Path,
    camera_id: str,
    recorded_start: datetime,
    vlm_engine: object | None = None,
    metadata_dir: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raise _runtime_removed()


def _write_video_metadata(metadata_path: Path, video_payload: dict, people: list[dict]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps({"schema_version": "strict_pipeline_metadata_v1", "video": video_payload, "people": people}, indent=2),
        encoding="utf-8",
    )


def current_pipeline_config() -> dict[str, str]:
    return {"pipeline": STRICT_PIPELINE_NAME, "runtime": "external_strict_runtime_required"}


def ensure_pipeline_layout() -> None:
    raise _runtime_removed()


def save_uploaded_video(file_bytes: bytes, original_name: str) -> Path:
    raise _runtime_removed()


def add_single_video(*args: object, **kwargs: object) -> dict[str, Any]:
    raise _runtime_removed()


def bootstrap_nvidia_hospital_dataset(*args: object, **kwargs: object) -> dict[str, Any]:
    raise _runtime_removed()


def load_all_people_metadata(metadata_dir: Path | None = None) -> list[dict[str, Any]]:
    return []
