from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import sys
from urllib.parse import urlparse

import httpx
from googleapiclient.http import MediaFileUpload
from sqlalchemy.orm import Session

from .config import settings
from .post_move_ingestion import (
    STORAGE_INGEST_SOURCE_MODE,
    PostMoveIngestionPolicy,
    StorageTrackingRequestFactory,
    StorageTrackingResultAssembler,
)
from .drive_storage_ingest import DriveStorageVideoScanner
from .service import (
    delete_queue_video_asset,
    get_queue_video_rows,
    upsert_person_candidates,
    upsert_queue_video_asset,
)
from .storage_ingest import StorageIngestRegistry, StorageVideoItem, StorageVideoScanner


LOGGER = logging.getLogger(__name__)
DRIVE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"

_HERE = Path(__file__).resolve()
for candidate in [Path(os.getenv("A20_ROOT", "")).expanduser() if os.getenv("A20_ROOT", "").strip() else None, Path("/workspace/a20-root"), *_HERE.parents]:
    if candidate and (candidate / "shared_secret_runtime.py").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from shared_secret_runtime import build_google_drive_oauth_service  # noqa: E402


class QueueSyncService:
    def __init__(self) -> None:
        self.local_root = Path(settings.queue_local_root)
        self.local_queue_dir = self.local_root / "local" / settings.google_drive_queue_folder_name
        self.local_queue_video_dir = self.local_queue_dir / settings.queue_video_folder_name
        self.local_queue_metadata_dir = self.local_queue_dir / settings.google_drive_metadata_folder_name
        self.storage_ingest_root = Path(settings.storage_ingest_root)
        self.storage_processed_dir = self.local_queue_dir / settings.storage_ingest_processed_dir_name
        self.post_move_policy = PostMoveIngestionPolicy()
        self.storage_request_factory = StorageTrackingRequestFactory(self.post_move_policy)
        self.storage_result_assembler = StorageTrackingResultAssembler(self.local_queue_metadata_dir)
        self._drive_service = None
        self._drive_layout: dict[str, str] | None = None

    @staticmethod
    def _parallel_jobs(count: int, default: int) -> int:
        return max(1, min(count, int(default)))

    def ensure_local_layout(self) -> None:
        self.local_queue_video_dir.mkdir(parents=True, exist_ok=True)
        self.local_queue_metadata_dir.mkdir(parents=True, exist_ok=True)
        self.storage_processed_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _video_media_type(path_or_name: str | Path) -> str:
        suffix = Path(str(path_or_name)).suffix.lower()
        if suffix == ".mp4":
            return "video/mp4"
        return "application/octet-stream"

    @staticmethod
    def _drive_view_link(file_id: str) -> str:
        return f"https://drive.google.com/file/d/{file_id}/view"

    @staticmethod
    def _drive_download_link(file_id: str) -> str:
        return f"https://drive.google.com/uc?id={file_id}&export=download"

    @staticmethod
    def _is_remote_endpoint(endpoint_root: str) -> bool:
        host = (urlparse(endpoint_root).hostname or "").strip().lower()
        return host not in {"", "127.0.0.1", "localhost", "tracking-service"}

    @staticmethod
    def _build_tracking_headers() -> dict[str, str]:
        token = str(settings.lightning_api_token or "").strip()
        if not token:
            return {}
        header_name = str(settings.lightning_api_auth_header or "Authorization").strip() or "Authorization"
        auth_prefix = str(settings.lightning_api_auth_prefix or "")
        if auth_prefix and not auth_prefix.endswith(" "):
            auth_prefix = f"{auth_prefix} "
        return {header_name: f"{auth_prefix}{token}".strip()}

    def _build_drive_service(self):
        if not settings.google_drive_enabled:
            raise RuntimeError("Google Drive sync is disabled. Set GOOGLE_DRIVE_ENABLED=true to use queue sync.")
        if self._drive_service is not None:
            return self._drive_service

        self._drive_service = build_google_drive_oauth_service()
        return self._drive_service

    def _query_single(self, query: str, fields: str = "files(id, name)") -> list[dict]:
        service = self._build_drive_service()
        response = service.files().list(
            q=query,
            spaces="drive",
            fields=f"files({fields})",
            pageSize=200,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        return list(response.get("files") or [])

    def _find_child_folder(self, parent_id: str, name: str) -> dict | None:
        escaped_name = name.replace("'", "\\'")
        query = (
            f"'{parent_id}' in parents and trashed = false and "
            f"mimeType = '{DRIVE_FOLDER_MIME_TYPE}' and name = '{escaped_name}'"
        )
        rows = self._query_single(query, fields="id, name")
        return rows[0] if rows else None

    def resolve_drive_source_storage_folder_id(self) -> str:
        configured_id = str(settings.google_drive_source_storage_folder_id or "").strip()
        if configured_id:
            return configured_id

        vinuni_folder_id = (settings.google_drive_vinuni_folder_id or "").strip()
        root_folder_id = (settings.google_drive_root_folder_id or "").strip()
        if not vinuni_folder_id and not root_folder_id:
            raise RuntimeError(
                "Set GOOGLE_DRIVE_SOURCE_STORAGE_FOLDER_ID or GOOGLE_DRIVE_VINUNI_FOLDER_ID/GOOGLE_DRIVE_ROOT_FOLDER_ID "
                "to enable Drive-backed storage ingestion."
            )

        vinuni_id = vinuni_folder_id or self._ensure_folder(root_folder_id, settings.google_drive_vinuni_folder_name)
        folder = self._find_child_folder(vinuni_id, settings.google_drive_source_storage_folder_name)
        if folder is None:
            raise RuntimeError(
                f"Google Drive source storage folder '{settings.google_drive_source_storage_folder_name}' was not found under VinUni."
            )
        return str(folder["id"])

    def _ensure_folder(self, parent_id: str, name: str) -> str:
        existing = self._find_child_folder(parent_id, name)
        if existing:
            return str(existing["id"])
        service = self._build_drive_service()
        created = service.files().create(
            body={"name": name, "mimeType": DRIVE_FOLDER_MIME_TYPE, "parents": [parent_id]},
            fields="id",
            supportsAllDrives=True,
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
        queue_video_id = self._ensure_folder(queue_id, settings.queue_video_folder_name)
        existing_metadata_folder = self._find_child_folder(queue_id, settings.google_drive_metadata_folder_name)
        self._drive_layout = {
            "vinuni_id": vinuni_id,
            "queue_id": queue_id,
            "queue_video_id": queue_video_id,
            "queue_metadata_id": str(existing_metadata_folder["id"]) if existing_metadata_folder else "",
        }
        return self._drive_layout

    def _ensure_public_read(self, file_id: str) -> None:
        if not settings.google_drive_make_public:
            return
        service = self._build_drive_service()
        try:
            service.permissions().create(
                fileId=file_id,
                body={"type": "anyone", "role": "reader"},
                fields="id",
                supportsAllDrives=True,
            ).execute()
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Could not set public Drive permission for %s: %s", file_id, exc)

    def _delete_drive_file(self, file_id: str | None) -> None:
        if not file_id:
            return
        service = self._build_drive_service()
        try:
            service.files().delete(fileId=file_id, supportsAllDrives=True).execute()
        except Exception as exc:
            LOGGER.warning("Failed to delete Drive file %s: %s", file_id, exc)

    def _delete_local_file(self, path_value: str | Path | None) -> None:
        if not path_value:
            return
        path = Path(path_value)
        if path.exists() and path.is_file():
            path.unlink(missing_ok=True)

    def _delete_queue_owned_video_file(self, row) -> None:
        if str(getattr(row, "source_mode", "") or "") == STORAGE_INGEST_SOURCE_MODE:
            return
        self._delete_local_file(getattr(row, "local_video_path", None))

    def _upload_to_drive(self, local_path: Path, parent_id: str, name: str, mime_type: str) -> dict[str, str]:
        service = self._build_drive_service()
        media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)
        created = service.files().create(
            body={"name": name, "parents": [parent_id]},
            media_body=media,
            fields="id, name, webViewLink, webContentLink",
            supportsAllDrives=True,
        ).execute()
        file_id = str(created["id"])
        self._ensure_public_read(file_id)
        return {
            "file_id": file_id,
            "view_link": str(created.get("webViewLink") or self._drive_view_link(file_id)),
            "download_link": str(created.get("webContentLink") or self._drive_download_link(file_id)),
        }

    def _request_tracking_processing(
        self,
        *,
        source_path: Path | None = None,
        source_drive_file_id: str | None = None,
        source_url: str | None = None,
        source_filename: str | None = None,
        camera_id: str | None,
        recorded_start: datetime | None,
        output_basename: str | None,
        source_mode: str,
        upload_outputs_to_drive: bool = False,
        destination_video_folder_id: str | None = None,
        destination_metadata_folder_id: str | None = None,
        metadata_extra: dict | None = None,
    ) -> dict:
        endpoint_root = settings.tracking_service_url.rstrip("/")

        if source_path is not None and self._is_remote_endpoint(endpoint_root):
            return self._request_tracking_processing_upload(
                source_path=source_path,
                source_filename=source_filename or source_path.name,
                camera_id=camera_id,
                recorded_start=recorded_start,
                output_basename=output_basename,
                source_mode=source_mode,
                metadata_extra=metadata_extra,
            )

        ingestion_metadata = {
            "source_mode": source_mode,
        }
        if isinstance(metadata_extra, dict):
            ingestion_metadata.update(metadata_extra)

        payload = {
            "source_path": str(source_path) if source_path is not None else None,
            "source_drive_file_id": source_drive_file_id,
            "source_url": source_url,
            "source_filename": source_filename,
            "camera_id": camera_id,
            "recorded_start": recorded_start.isoformat() if recorded_start else None,
            "output_video_dir": str(self.local_queue_video_dir) if not upload_outputs_to_drive else None,
            "output_metadata_dir": str(self.local_queue_metadata_dir) if not upload_outputs_to_drive else None,
            "output_basename": output_basename,
            "destination_video_folder_id": destination_video_folder_id,
            "destination_metadata_folder_id": destination_metadata_folder_id,
            "upload_outputs_to_drive": upload_outputs_to_drive,
            "metadata": ingestion_metadata,
        }
        endpoint = f"{endpoint_root}/api/v1/ingestion/process"
        headers = self._build_tracking_headers()
        with httpx.Client(timeout=float(settings.tracking_request_timeout_seconds)) as client:
            response = client.post(endpoint, json=payload, headers=headers or None)
            response.raise_for_status()
            return response.json()

    def _request_tracking_processing_upload(
        self,
        *,
        source_path: Path,
        source_filename: str,
        camera_id: str | None,
        recorded_start: datetime | None,
        output_basename: str | None,
        source_mode: str,
        metadata_extra: dict | None = None,
    ) -> dict:
        endpoint_root = settings.tracking_service_url.rstrip("/")
        endpoint = f"{endpoint_root}/api/v1/ingestion/upload"
        headers = self._build_tracking_headers()

        ingestion_metadata = {
            "source_mode": source_mode,
        }
        if isinstance(metadata_extra, dict):
            ingestion_metadata.update(metadata_extra)

        data = {
            "source_filename": source_filename,
            "camera_id": camera_id or "",
            "recorded_start": recorded_start.isoformat() if recorded_start else "",
            "output_basename": output_basename or "",
            "metadata": json.dumps(ingestion_metadata),
        }
        with source_path.open("rb") as handle:
            files = {"file": (source_filename, handle, self._video_media_type(source_filename))}
            with httpx.Client(timeout=float(settings.tracking_request_timeout_seconds)) as client:
                response = client.post(endpoint, data=data, files=files, headers=headers or None)
                response.raise_for_status()
                return response.json()

    @staticmethod
    def _drive_public_download_url(file_id: str) -> str:
        """Return the public direct-download URL for a Google Drive file.

        Works when the file is shared as 'Anyone with the link can view'.
        The query parameter confirm=t bypasses the large-file virus-scan page.
        """
        return f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t"

    def _process_storage_video_item(self, item: StorageVideoItem) -> dict:
        """Send one storage/camera/date MP4 to tracking and persist local queue artifacts."""

        task = self.storage_request_factory.build(item)

        # For Drive-sourced items: pass the public download URL so the tracking
        # service (LightningAI) can download directly from Drive without
        # requiring OAuth credentials on that host.
        source_url: str | None = None
        if task.source_path is None and task.source_drive_file_id:
            source_url = self._drive_public_download_url(task.source_drive_file_id)

        result = self._request_tracking_processing(
            source_path=task.source_path,
            source_drive_file_id=task.source_drive_file_id,
            source_url=source_url,
            source_filename=task.source_filename,
            camera_id=task.camera_id,
            recorded_start=task.recorded_at,
            output_basename=task.output_basename,
            source_mode=task.source_mode,
            metadata_extra=task.metadata,
        )
        processed = self.storage_result_assembler.assemble(item, result, task.source_mode)
        return {
            "result": processed.result,
            "source_filename": processed.source_filename,
            "source_mode": processed.source_mode,
            "source_item": processed.source_item,
        }

    def _append_processed_result(
        self,
        session: Session,
        result: dict,
        source_filename: str | None,
        source_mode: str,
        *,
        publish_to_drive: bool | None = None,
    ) -> list[str]:
        current_rows = get_queue_video_rows(session)
        video = result["video"]
        people = result.get("people") or []
        compressed_path_value = result.get("compressed_path")
        metadata_path_value = result.get("metadata_path")

        for row in list(current_rows):
            if row.video_id == str(video["video_id"]):
                self._delete_drive_file(row.drive_video_file_id)
                self._delete_drive_file(row.drive_metadata_file_id)
                self._delete_queue_owned_video_file(row)
                self._delete_local_file(row.local_metadata_path)
                delete_queue_video_asset(session, row.video_id)
                current_rows.remove(row)

        evicted_video_ids: list[str] = []
        while len(current_rows) >= settings.queue_max_size:
            oldest = current_rows.pop(0)
            self._delete_drive_file(oldest.drive_video_file_id)
            self._delete_drive_file(oldest.drive_metadata_file_id)
            self._delete_queue_owned_video_file(oldest)
            self._delete_local_file(oldest.local_metadata_path)
            delete_queue_video_asset(session, oldest.video_id)
            evicted_video_ids.append(oldest.video_id)

        for index, row in enumerate(current_rows):
            row.queue_position = index
            session.add(row)

        drive_video_file_id = str(result.get("drive_video_file_id") or "").strip() or None
        drive_metadata_file_id = None
        drive_video_link = str(result.get("drive_video_link") or "").strip() or None
        compressed_name = Path(str(compressed_path_value or source_filename or "queue-video.mp4")).stem
        should_publish_to_drive = bool(settings.google_drive_enabled) if publish_to_drive is None else bool(publish_to_drive)
        if drive_video_file_id and drive_video_link:
            uploaded_video = {"file_id": drive_video_file_id, "view_link": drive_video_link}
            storage_backend = "google_drive_local_metadata"
        elif should_publish_to_drive:
            layout = self.ensure_drive_layout()
            if not compressed_path_value or not metadata_path_value:
                raise RuntimeError(
                    "Tracking result missing both Drive output links and local output paths."
                )
            compressed_path = Path(str(compressed_path_value))
            uploaded_video = self._upload_to_drive(
                compressed_path,
                layout["queue_video_id"],
                compressed_path.name,
                self._video_media_type(compressed_path.name),
            )
            storage_backend = "google_drive_local_metadata"
        else:
            if not compressed_path_value or not metadata_path_value:
                raise RuntimeError(
                    "Tracking result missing both Drive output links and local output paths."
                )
            uploaded_video = {
                "file_id": None,
                "view_link": f"/api/v1/queue/videos/{video['video_id']}/file",
            }
            storage_backend = "local_queue_storage"
        metadata_link = f"/api/v1/queue/videos/{video['video_id']}/metadata"

        upsert_queue_video_asset(
            session,
            video_id=str(video["video_id"]),
            camera_id=video.get("camera_id"),
            title=str(video.get("camera_id") or video.get("video_id") or compressed_name),
            queue_position=len(current_rows),
            available_link_video=uploaded_video["view_link"],
            available_link_metadata=metadata_link,
            storage_backend=storage_backend,
            source_filename=source_filename,
            source_mode=source_mode,
            drive_video_file_id=uploaded_video.get("file_id"),
            drive_metadata_file_id=drive_metadata_file_id,
            local_video_path=str(compressed_path_value or ""),
            local_metadata_path=str(metadata_path_value or ""),
            raw_video_metadata=video,
        )
        upsert_person_candidates(session, people, str(metadata_path_value or ""))
        session.commit()
        return evicted_video_ids

    def process_storage_queue(self, session: Session) -> dict:
        self.ensure_local_layout()
        if not settings.storage_ingest_enabled:
            return {
                "processed_videos": 0,
                "imported_source_files": [],
                "evicted_video_ids": [],
            }

        registry = StorageIngestRegistry(self.storage_processed_dir)
        if str(settings.storage_ingest_source_backend or "filesystem").strip().lower() == "google_drive":
            scanner = DriveStorageVideoScanner(
                drive_service=self._build_drive_service(),
                source_storage_root_id=self.resolve_drive_source_storage_folder_id(),
                registry=registry,
                min_file_age_seconds=settings.storage_ingest_min_file_age_seconds,
            )
        else:
            scanner = StorageVideoScanner(
                storage_root=self.storage_ingest_root,
                registry=registry,
                min_file_age_seconds=settings.storage_ingest_min_file_age_seconds,
            )
        storage_items = scanner.list_pending(limit=settings.storage_ingest_batch_size)
        parallel_jobs = self._parallel_jobs(len(storage_items), settings.queue_parallel_jobs)

        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            for item in executor.map(self._process_storage_video_item, storage_items):
                results.append(item)

        processed_videos = 0
        imported_source_files: list[str] = []
        evicted_video_ids: list[str] = []

        for item in results:
            result = item["result"]
            evicted_video_ids.extend(
                self._append_processed_result(
                    session,
                    result,
                    source_filename=item["source_filename"],
                    source_mode=item["source_mode"],
                    publish_to_drive=bool(settings.google_drive_enabled),
                )
            )
            registry.mark_processed(item["source_item"], result)
            imported_source_files.append(str(item["source_item"].relative_path))
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
    LOGGER.info(
        "Starting queue worker with %ss polling interval; storage_ingest=%s root=%s",
        interval,
        settings.storage_ingest_enabled,
        service.storage_ingest_root,
    )

    while True:
        session = session_factory()
        try:
            result = service.process_storage_queue(session)
            if result["processed_videos"]:
                LOGGER.info("Processed %s new storage video(s)", result["processed_videos"])
        except Exception:
            LOGGER.exception("Queue sync iteration failed")
            session.rollback()
        finally:
            session.close()
        time.sleep(interval)
