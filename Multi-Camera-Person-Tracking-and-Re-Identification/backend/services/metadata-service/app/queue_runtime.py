from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileDownload, MediaFileUpload
from sqlalchemy.orm import Session

from .config import settings
from .service import (
    delete_queue_video_asset,
    get_queue_video_rows,
    upsert_person_candidates,
    upsert_queue_video_asset,
)


LOGGER = logging.getLogger(__name__)
VIDEO_EXTENSIONS = {".hevc", ".h265"}
DRIVE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
DRIVE_SCOPES = ("https://www.googleapis.com/auth/drive",)
BOOTSTRAP_CAMERA_PATTERN = re.compile(r"Camera_\d{2}")


class QueueSyncService:
    def __init__(self) -> None:
        self.local_root = Path(settings.queue_local_root)
        self.local_queue_dir = self.local_root / "local" / settings.google_drive_queue_folder_name
        self.local_queue_video_dir = self.local_queue_dir / settings.google_drive_h265_folder_name
        self.local_queue_metadata_dir = self.local_queue_dir / settings.google_drive_metadata_folder_name
        self.local_import_dir = self.local_root / "local" / settings.google_drive_import_folder_name
        self.local_download_dir = self.local_root / "tmp" / "downloads"
        self._drive_service = None
        self._drive_layout: dict[str, str] | None = None

    def ensure_local_layout(self) -> None:
        self.local_queue_video_dir.mkdir(parents=True, exist_ok=True)
        self.local_queue_metadata_dir.mkdir(parents=True, exist_ok=True)
        self.local_import_dir.mkdir(parents=True, exist_ok=True)
        self.local_download_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip()).strip("._-") or "video"

    @staticmethod
    def _drive_view_link(file_id: str) -> str:
        return f"https://drive.google.com/file/d/{file_id}/view"

    @staticmethod
    def _drive_download_link(file_id: str) -> str:
        return f"https://drive.google.com/uc?id={file_id}&export=download"

    def _build_drive_service(self):
        if not settings.google_drive_enabled:
            raise RuntimeError("Google Drive sync is disabled. Set GOOGLE_DRIVE_ENABLED=true to use queue sync.")
        if self._drive_service is not None:
            return self._drive_service

        credentials_path = Path(settings.google_drive_credentials_file).expanduser()
        if not credentials_path.exists():
            raise FileNotFoundError(
                f"Missing Google Drive credentials file: {credentials_path}. "
                "Set GOOGLE_DRIVE_CREDENTIALS_FILE to a valid service-account JSON."
            )
        credentials = service_account.Credentials.from_service_account_file(
            str(credentials_path),
            scopes=list(DRIVE_SCOPES),
        )
        self._drive_service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        return self._drive_service

    def _query_single(self, query: str, fields: str = "files(id, name)") -> list[dict]:
        service = self._build_drive_service()
        response = service.files().list(q=query, spaces="drive", fields=f"files({fields})", pageSize=200).execute()
        return list(response.get("files") or [])

    def _find_child_folder(self, parent_id: str, name: str) -> dict | None:
        escaped_name = name.replace("'", "\\'")
        query = (
            f"'{parent_id}' in parents and trashed = false and "
            f"mimeType = '{DRIVE_FOLDER_MIME_TYPE}' and name = '{escaped_name}'"
        )
        rows = self._query_single(query, fields="id, name")
        return rows[0] if rows else None

    def _ensure_folder(self, parent_id: str, name: str) -> str:
        existing = self._find_child_folder(parent_id, name)
        if existing:
            return str(existing["id"])
        service = self._build_drive_service()
        created = service.files().create(
            body={"name": name, "mimeType": DRIVE_FOLDER_MIME_TYPE, "parents": [parent_id]},
            fields="id",
        ).execute()
        return str(created["id"])

    def ensure_drive_layout(self) -> dict[str, str]:
        if self._drive_layout is not None:
            return self._drive_layout

        vinuni_folder_id = (settings.google_drive_vinuni_folder_id or "").strip()
        root_folder_id = (settings.google_drive_root_folder_id or "").strip()
        if not vinuni_folder_id and not root_folder_id:
            raise RuntimeError(
                "Set GOOGLE_DRIVE_VINUNI_FOLDER_ID to the VinUni folder id, "
                "or GOOGLE_DRIVE_ROOT_FOLDER_ID so the service can create VinUni under that folder."
            )

        vinuni_id = vinuni_folder_id or self._ensure_folder(root_folder_id, settings.google_drive_vinuni_folder_name)
        queue_id = self._ensure_folder(vinuni_id, settings.google_drive_queue_folder_name)
        import_id = self._ensure_folder(vinuni_id, settings.google_drive_import_folder_name)
        queue_h265_id = self._ensure_folder(queue_id, settings.google_drive_h265_folder_name)
        queue_metadata_id = self._ensure_folder(queue_id, settings.google_drive_metadata_folder_name)
        self._drive_layout = {
            "vinuni_id": vinuni_id,
            "queue_id": queue_id,
            "import_id": import_id,
            "queue_h265_id": queue_h265_id,
            "queue_metadata_id": queue_metadata_id,
        }
        return self._drive_layout

    def _ensure_public_read(self, file_id: str) -> None:
        if not settings.google_drive_make_public:
            return
        service = self._build_drive_service()
        try:
            service.permissions().create(fileId=file_id, body={"type": "anyone", "role": "reader"}, fields="id").execute()
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Could not set public Drive permission for %s: %s", file_id, exc)

    def _delete_drive_file(self, file_id: str | None) -> None:
        if not file_id:
            return
        service = self._build_drive_service()
        try:
            service.files().delete(fileId=file_id).execute()
        except Exception as exc:
            LOGGER.warning("Failed to delete Drive file %s: %s", file_id, exc)

    def _delete_local_file(self, path_value: str | Path | None) -> None:
        if not path_value:
            return
        path = Path(path_value)
        if path.exists() and path.is_file():
            path.unlink(missing_ok=True)

    def _clear_local_directory(self, target_dir: Path) -> None:
        if not target_dir.exists():
            return
        for path in target_dir.iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)

    def _delete_drive_folder_children(self, folder_id: str) -> None:
        query = f"'{folder_id}' in parents and trashed = false"
        for row in self._query_single(query, fields="id, name"):
            self._delete_drive_file(str(row.get("id") or ""))

    def _upload_to_drive(self, local_path: Path, parent_id: str, name: str, mime_type: str) -> dict[str, str]:
        service = self._build_drive_service()
        media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=False)
        created = service.files().create(
            body={"name": name, "parents": [parent_id]},
            media_body=media,
            fields="id, name, webViewLink, webContentLink",
        ).execute()
        file_id = str(created["id"])
        self._ensure_public_read(file_id)
        return {
            "file_id": file_id,
            "view_link": str(created.get("webViewLink") or self._drive_view_link(file_id)),
            "download_link": str(created.get("webContentLink") or self._drive_download_link(file_id)),
        }

    def _download_drive_file(self, file_id: str, target_path: Path) -> None:
        service = self._build_drive_service()
        request = service.files().get_media(fileId=file_id)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as handle:
            downloader = MediaFileDownload(handle, request)
            done = False
            while not done:
                _status, done = downloader.next_chunk()

    def _request_tracking_processing(
        self,
        *,
        source_path: Path | None = None,
        source_drive_file_id: str | None = None,
        source_filename: str | None = None,
        camera_id: str | None,
        recorded_start: datetime | None,
        output_basename: str | None,
        source_mode: str,
        upload_outputs_to_drive: bool = False,
        destination_video_folder_id: str | None = None,
        destination_metadata_folder_id: str | None = None,
    ) -> dict:
        payload = {
            "source_path": str(source_path) if source_path is not None else None,
            "source_drive_file_id": source_drive_file_id,
            "source_filename": source_filename,
            "camera_id": camera_id,
            "recorded_start": recorded_start.isoformat() if recorded_start else None,
            "output_video_dir": str(self.local_queue_video_dir) if not upload_outputs_to_drive else None,
            "output_metadata_dir": str(self.local_queue_metadata_dir) if not upload_outputs_to_drive else None,
            "output_basename": output_basename,
            "destination_video_folder_id": destination_video_folder_id,
            "destination_metadata_folder_id": destination_metadata_folder_id,
            "upload_outputs_to_drive": upload_outputs_to_drive,
            "metadata": {
                "source_mode": source_mode,
                "pipeline_profile": "accuracy_first",
            },
        }
        endpoint = f"{settings.tracking_service_url.rstrip('/')}/api/v1/ingestion/process"
        with httpx.Client(timeout=1800.0) as client:
            response = client.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()

    def list_import_files(self) -> list[dict]:
        layout = self.ensure_drive_layout()
        query = f"'{layout['import_id']}' in parents and trashed = false and mimeType != '{DRIVE_FOLDER_MIME_TYPE}'"
        rows = self._query_single(query, fields="id, name, mimeType, createdTime")
        supported = [row for row in rows if Path(str(row.get("name") or "")).suffix.lower() in VIDEO_EXTENSIONS]
        supported.sort(key=lambda item: str(item.get("createdTime") or ""))
        return supported

    def _clear_queue_state(self, session: Session, clear_remote: bool) -> None:
        rows = get_queue_video_rows(session)
        if clear_remote:
            layout = self.ensure_drive_layout()
            self._delete_drive_folder_children(layout["queue_h265_id"])
            self._delete_drive_folder_children(layout["queue_metadata_id"])
        for row in rows:
            self._delete_local_file(row.local_video_path)
            self._delete_local_file(row.local_metadata_path)
            delete_queue_video_asset(session, row.video_id)
        self._clear_local_directory(self.local_queue_video_dir)
        self._clear_local_directory(self.local_queue_metadata_dir)
        self._clear_local_directory(self.local_download_dir)
        session.commit()

    def _append_processed_result(self, session: Session, result: dict, source_filename: str | None, source_mode: str) -> list[str]:
        current_rows = get_queue_video_rows(session)
        video = result["video"]
        people = result.get("people") or []
        compressed_path = Path(result["compressed_path"])
        metadata_path = Path(result["metadata_path"])
        layout = self.ensure_drive_layout()

        for row in list(current_rows):
            if row.video_id == str(video["video_id"]):
                self._delete_drive_file(row.drive_video_file_id)
                self._delete_drive_file(row.drive_metadata_file_id)
                self._delete_local_file(row.local_video_path)
                self._delete_local_file(row.local_metadata_path)
                delete_queue_video_asset(session, row.video_id)
                current_rows.remove(row)

        evicted_video_ids: list[str] = []
        while len(current_rows) >= settings.queue_max_size:
            oldest = current_rows.pop(0)
            self._delete_drive_file(oldest.drive_video_file_id)
            self._delete_drive_file(oldest.drive_metadata_file_id)
            self._delete_local_file(oldest.local_video_path)
            self._delete_local_file(oldest.local_metadata_path)
            delete_queue_video_asset(session, oldest.video_id)
            evicted_video_ids.append(oldest.video_id)

        for index, row in enumerate(current_rows):
            row.queue_position = index
            session.add(row)

        uploaded_video = self._upload_to_drive(compressed_path, layout["queue_h265_id"], compressed_path.name, "video/h265")
        uploaded_metadata = self._upload_to_drive(metadata_path, layout["queue_metadata_id"], metadata_path.name, "application/json")

        upsert_queue_video_asset(
            session,
            video_id=str(video["video_id"]),
            camera_id=video.get("camera_id"),
            title=str(video.get("camera_id") or video.get("video_id") or compressed_path.stem),
            queue_position=len(current_rows),
            available_link_video=uploaded_video["view_link"],
            available_link_metadata=uploaded_metadata["view_link"],
            storage_backend="google_drive",
            source_filename=source_filename,
            source_mode=source_mode,
            drive_video_file_id=uploaded_video["file_id"],
            drive_metadata_file_id=uploaded_metadata["file_id"],
            local_video_path=str(compressed_path),
            local_metadata_path=str(metadata_path),
            raw_video_metadata=video,
        )
        upsert_person_candidates(session, people, str(metadata_path))
        session.commit()
        return evicted_video_ids

    def bootstrap_from_source_dir(
        self,
        session: Session,
        *,
        source_dir: str | Path,
        limit: int = 31,
        reset_remote_queue: bool = True,
        delete_source_after_import: bool = False,
    ) -> dict:
        self.ensure_local_layout()
        self.ensure_drive_layout()
        if reset_remote_queue:
            self._clear_queue_state(session, clear_remote=True)

        source_root = Path(source_dir).expanduser()
        if not source_root.exists():
            raise FileNotFoundError(f"Missing bootstrap source dir: {source_root}")
        if not source_root.is_dir():
            raise NotADirectoryError(f"Bootstrap source path must be a directory: {source_root}")

        source_videos = sorted(
            path
            for path in source_root.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS and BOOTSTRAP_CAMERA_PATTERN.fullmatch(path.stem)
        )[:limit]

        processed_videos = 0
        people_indexed = 0
        evicted_video_ids: list[str] = []
        for source_path in source_videos:
            result = self._request_tracking_processing(
                source_path=source_path,
                camera_id=source_path.stem,
                recorded_start=None,
                output_basename=f"{source_path.stem}{source_path.suffix}",
                source_mode="bootstrap_dataset",
            )
            evicted_video_ids.extend(
                self._append_processed_result(
                    session,
                    result,
                    source_filename=source_path.name,
                    source_mode="bootstrap_dataset",
                )
            )
            if delete_source_after_import:
                source_path.unlink(missing_ok=True)
            processed_videos += 1
            people_indexed += int(result.get("person_count") or 0)

        return {
            "processed_videos": processed_videos,
            "queue_size": len(get_queue_video_rows(session)),
            "people_indexed": people_indexed,
            "evicted_video_ids": evicted_video_ids,
        }

    def process_import_queue(self, session: Session) -> dict:
        self.ensure_local_layout()
        self.ensure_drive_layout()

        processed_videos = 0
        imported_source_files: list[str] = []
        evicted_video_ids: list[str] = []

        for import_file in self.list_import_files():
            original_name = str(import_file.get("name") or "imported_video.h265")
            source_file_id = str(import_file["id"])
            local_source_path = self.local_download_dir / original_name
            self._download_drive_file(source_file_id, local_source_path)

            recorded_start = datetime.now(timezone.utc).replace(microsecond=0)
            camera_id = self._slug(Path(original_name).stem)
            original_suffix = Path(original_name).suffix.lower()
            output_basename = f"{camera_id}_{recorded_start.strftime('%Y%m%dT%H%M%SZ')}{original_suffix}"
            result = self._request_tracking_processing(
                source_path=local_source_path,
                camera_id=camera_id,
                recorded_start=recorded_start,
                output_basename=output_basename,
                source_mode="google_drive_import",
            )
            evicted_video_ids.extend(
                self._append_processed_result(
                    session,
                    result,
                    source_filename=original_name,
                    source_mode="google_drive_import",
                )
            )
            self._delete_drive_file(source_file_id)
            local_source_path.unlink(missing_ok=True)
            imported_source_files.append(original_name)
            processed_videos += 1

        return {
            "processed_videos": processed_videos,
            "imported_source_files": imported_source_files,
            "evicted_video_ids": evicted_video_ids,
        }


def run_queue_sync_forever(session_factory, poll_interval_seconds: int | None = None) -> None:
    import time

    service = QueueSyncService()
    interval = max(int(poll_interval_seconds or settings.queue_poll_interval_seconds), 5)
    LOGGER.info("Starting Google Drive queue worker with %ss polling interval", interval)

    while True:
        session = session_factory()
        try:
            result = service.process_import_queue(session)
            if result["processed_videos"]:
                LOGGER.info("Processed %s new import file(s)", result["processed_videos"])
        except Exception:
            LOGGER.exception("Queue sync iteration failed")
            session.rollback()
        finally:
            session.close()
        time.sleep(interval)
