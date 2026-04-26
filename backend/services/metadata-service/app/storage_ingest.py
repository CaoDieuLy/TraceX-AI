from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import re
from typing import Iterable


STORAGE_VIDEO_PATTERN = re.compile(
    r"^(?P<camera_id>cam_\d{2,})_"
    r"(?P<recorded_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<hour>\d{2})-(?P<minute>\d{2})"
    r"(?:-(?P<second>\d{2}))?"
    r"(?P<suffix>\.mp4)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StorageVideoIdentity:
    """Parsed identity from storage/<camera>/<date>/<filename>.mp4."""

    camera_id: str
    recorded_date: date
    recorded_at: datetime
    source_filename: str

    @classmethod
    def parse(cls, source_path: Path) -> "StorageVideoIdentity | None":
        match = STORAGE_VIDEO_PATTERN.fullmatch(source_path.name)
        if not match:
            return None

        try:
            recorded_date = date.fromisoformat(match.group("recorded_date"))
            recorded_at = datetime(
                recorded_date.year,
                recorded_date.month,
                recorded_date.day,
                int(match.group("hour")),
                int(match.group("minute")),
                int(match.group("second") or "0"),
            )
        except ValueError:
            return None

        return cls(
            camera_id=match.group("camera_id").lower(),
            recorded_date=recorded_date,
            recorded_at=recorded_at,
            source_filename=source_path.name,
        )


@dataclass(frozen=True)
class StorageVideoItem:
    """Input contract for one storage video that can be sent to ingestion."""

    source_path: Path
    relative_path: str
    camera_id: str
    recorded_at: datetime
    size_bytes: int
    modified_ns: int
    fingerprint: str

    @property
    def source_filename(self) -> str:
        return self.source_path.name

    @property
    def output_basename(self) -> str:
        return self.source_path.name

    def marker_payload(self) -> dict:
        payload = asdict(self)
        payload["source_path"] = str(self.source_path)
        payload["recorded_at"] = self.recorded_at.isoformat()
        return payload


class StorageIngestRegistry:
    """Persistent processed-file registry for storage scanner idempotency."""

    def __init__(self, marker_dir: Path) -> None:
        self.marker_dir = marker_dir

    def is_processed(self, item: StorageVideoItem) -> bool:
        marker_path = self._marker_path(item)
        if not marker_path.exists():
            return False
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return (
            str(marker.get("relative_path") or "") == item.relative_path
            and int(marker.get("size_bytes") or -1) == item.size_bytes
            and int(marker.get("modified_ns") or -1) == item.modified_ns
        )

    def mark_processed(self, item: StorageVideoItem, result: dict) -> Path:
        self.marker_dir.mkdir(parents=True, exist_ok=True)
        marker_path = self._marker_path(item)
        payload = item.marker_payload()
        payload["processed_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        payload["result_video_id"] = str((result.get("video") or {}).get("video_id") or "")
        payload["person_count"] = int(result.get("person_count") or 0)
        marker_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return marker_path

    def _marker_path(self, item: StorageVideoItem) -> Path:
        return self.marker_dir / f"{item.fingerprint}.json"


class StorageVideoScanner:
    """
    Finds new H.265-in-MP4 clips after move.py has organized temp/ into storage/.

    The scanner only accepts .mp4 files in the expected camera/date layout.
    Non-.mp4 queue folders are intentionally ignored by this module.
    """

    def __init__(
        self,
        *,
        storage_root: Path,
        registry: StorageIngestRegistry,
        min_file_age_seconds: int = 2,
    ) -> None:
        self.storage_root = storage_root
        self.registry = registry
        self.min_file_age_seconds = max(0, int(min_file_age_seconds))

    def list_pending(self, *, limit: int | None = None) -> list[StorageVideoItem]:
        pending: list[StorageVideoItem] = []
        for item in self._iter_storage_items():
            if self._is_too_new(item):
                continue
            if self.registry.is_processed(item):
                continue
            pending.append(item)
            if limit is not None and len(pending) >= max(0, int(limit)):
                break
        return pending

    def _iter_storage_items(self) -> Iterable[StorageVideoItem]:
        if not self.storage_root.exists() or not self.storage_root.is_dir():
            return []

        candidates = sorted(self.storage_root.glob("cam_*/*/*.mp4"))
        items: list[StorageVideoItem] = []
        for source_path in candidates:
            if not source_path.is_file() or source_path.name.startswith("."):
                continue
            identity = StorageVideoIdentity.parse(source_path)
            if identity is None:
                continue
            try:
                relative_path = source_path.relative_to(self.storage_root).as_posix()
                relative_parts = source_path.relative_to(self.storage_root).parts
                stat = source_path.stat()
            except OSError:
                continue
            if len(relative_parts) < 3:
                continue
            if relative_parts[0].lower() != identity.camera_id:
                continue
            if relative_parts[1] != identity.recorded_date.isoformat():
                continue
            items.append(
                StorageVideoItem(
                    source_path=source_path,
                    relative_path=relative_path,
                    camera_id=identity.camera_id,
                    recorded_at=identity.recorded_at,
                    size_bytes=int(stat.st_size),
                    modified_ns=int(stat.st_mtime_ns),
                    fingerprint=self._fingerprint(relative_path),
                )
            )
        return items

    def _is_too_new(self, item: StorageVideoItem) -> bool:
        if self.min_file_age_seconds <= 0:
            return False
        modified_seconds = item.modified_ns / 1_000_000_000
        return (time.time() - modified_seconds) < self.min_file_age_seconds

    @staticmethod
    def _fingerprint(relative_path: str) -> str:
        return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()
