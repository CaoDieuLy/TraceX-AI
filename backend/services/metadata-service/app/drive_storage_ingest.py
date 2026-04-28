from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .storage_ingest import StorageIngestRegistry, StorageVideoIdentity, StorageVideoItem


DRIVE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


@dataclass(frozen=True)
class DriveFolderRef:
    id: str
    name: str


class DriveStorageVideoScanner:
    """
    Finds pending storage videos directly from Google Drive under VinUni/Storage.

    Expected Drive layout:
      VinUni/
        Storage/
          cam_01/
            2026-04-28/
              cam_01_2026-04-28_11-00.mp4
    """

    def __init__(
        self,
        *,
        drive_service,
        source_storage_root_id: str,
        registry: StorageIngestRegistry,
        min_file_age_seconds: int = 2,
    ) -> None:
        self.drive_service = drive_service
        self.source_storage_root_id = str(source_storage_root_id or "").strip()
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
        if not self.source_storage_root_id:
            return []

        items: list[StorageVideoItem] = []
        for camera_folder in self._list_folders(self.source_storage_root_id):
            if not camera_folder.name.lower().startswith("cam_"):
                continue
            for date_folder in self._list_folders(camera_folder.id):
                date_name = date_folder.name
                for file_row in self._list_mp4_files(date_folder.id):
                    identity = StorageVideoIdentity.parse(Path(str(file_row.get("name") or "")))
                    if identity is None:
                        continue
                    if identity.camera_id != camera_folder.name.lower():
                        continue
                    if date_name != identity.recorded_date.isoformat():
                        continue
                    relative_path = f"{camera_folder.name}/{date_name}/{identity.source_filename}"
                    modified_at = self._parse_drive_modified_time(str(file_row.get("modifiedTime") or ""))
                    items.append(
                        StorageVideoItem(
                            source_path=None,
                            source_drive_file_id=str(file_row["id"]),
                            relative_path=relative_path,
                            source_filename=identity.source_filename,
                            camera_id=identity.camera_id,
                            recorded_at=identity.recorded_at,
                            size_bytes=int(file_row.get("size") or 0),
                            modified_ns=int(modified_at.timestamp() * 1_000_000_000),
                            fingerprint=self._fingerprint(relative_path, str(file_row["id"])),
                        )
                    )
        return sorted(items, key=lambda item: item.relative_path)

    def _list_folders(self, parent_id: str) -> list[DriveFolderRef]:
        rows = self._query_children(
            parent_id,
            mime_type=DRIVE_FOLDER_MIME_TYPE,
            fields="files(id, name)",
        )
        return [DriveFolderRef(id=str(row["id"]), name=str(row["name"])) for row in rows]

    def _list_mp4_files(self, parent_id: str) -> list[dict]:
        rows = self._query_children(
            parent_id,
            mime_type=None,
            fields="files(id, name, size, modifiedTime, mimeType)",
        )
        return [row for row in rows if str(row.get("name") or "").lower().endswith(".mp4")]

    def _query_children(self, parent_id: str, *, mime_type: str | None, fields: str) -> list[dict]:
        query = [f"'{parent_id}' in parents", "trashed = false"]
        if mime_type:
            query.append(f"mimeType = '{mime_type}'")
        response = self.drive_service.files().list(
            q=" and ".join(query),
            spaces="drive",
            fields=fields,
            pageSize=1000,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        return list(response.get("files") or [])

    def _is_too_new(self, item: StorageVideoItem) -> bool:
        if self.min_file_age_seconds <= 0:
            return False
        modified_seconds = item.modified_ns / 1_000_000_000
        return (datetime.now(tz=timezone.utc).timestamp() - modified_seconds) < self.min_file_age_seconds

    @staticmethod
    def _parse_drive_modified_time(raw_value: str) -> datetime:
        value = raw_value.strip()
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).astimezone(timezone.utc)

    @staticmethod
    def _fingerprint(relative_path: str, file_id: str) -> str:
        import hashlib

        return hashlib.sha1(f"{relative_path}|{file_id}".encode("utf-8")).hexdigest()
