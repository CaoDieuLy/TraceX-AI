"""Video Queue Manager with FIFO logic and PostgreSQL backend."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import settings
from .database import get_queue_db, VideoNotFoundError
from .video_converter import convert_to_h265, validate_video_file

logger = logging.getLogger(__name__)


class QueueManagerError(Exception):
    """Base exception for queue manager errors."""
    pass


class VideoAlreadyInQueueError(QueueManagerError):
    """Raised when video already exists in queue."""
    pass


class VideoQueueManager:
    """
    Manages video queue with FIFO logic:
    - Import_New folder: users upload MP4 here
    - Auto-convert to H.265 + generate metadata
    - Move to Queue folder (.h265 + Metadata subfolders)
    - Auto-cleanup oldest videos when queue exceeds MAX_QUEUE_SIZE
    - Sync with PostgreSQL
    """

    def __init__(
        self,
        import_new_dir: Path | str | None = None,
        queue_dir: Path | str | None = None,
        google_drive_enabled: bool | None = None,
    ) -> None:
        self.import_new_dir = Path(import_new_dir or self._default_import_new_dir())
        self.queue_dir = Path(queue_dir or self._default_queue_dir())
        self.queue_h265_dir = self.queue_dir / settings.gdrive_queue_h265_subfolder
        self.queue_metadata_dir = self.queue_dir / settings.gdrive_queue_metadata_subfolder

        self.google_drive_enabled = (
            google_drive_enabled
            if google_drive_enabled is not None
            else settings.google_drive_enabled
        )

        self._ensure_directories()

    def _default_import_new_dir(self) -> Path:
        """Default Import_New directory (local or mounted GDrive)."""
        # Prefer mounted Google Drive path if available
        gdrive_mount = Path.home() / "gdrive"
        if gdrive_mount.exists():
            return gdrive_mount / settings.gdrive_base_folder / settings.gdrive_import_new_folder
        return Path("./Import_New")

    def _default_queue_dir(self) -> Path:
        """Default Queue directory (local or mounted GDrive)."""
        gdrive_mount = Path.home() / "gdrive"
        if gdrive_mount.exists():
            return gdrive_mount / settings.gdrive_base_folder / settings.gdrive_queue_folder
        return Path("./Queue")

    def _ensure_directories(self) -> None:
        """Create required directories if not exist."""
        self.import_new_dir.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.queue_h265_dir.mkdir(parents=True, exist_ok=True)
        self.queue_metadata_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Queue directories initialized:\n"
            f"  Import_New: {self.import_new_dir}\n"
            f"  Queue/.h265: {self.queue_h265_dir}\n"
            f"  Queue/Metadata: {self.queue_metadata_dir}"
        )

    # ==================== Public API ====================

    def enqueue_new_uploads(self) -> Dict[str, Any]:
        """
        Scan Import_New folder, convert MP4 → H.265 + metadata,
        move to Queue, and add to PostgreSQL.

        Returns:
            Dict with stats: processed, failed, skipped
        """
        stats = {"processed": 0, "failed": 0, "skipped": 0, "details": []}

        mp4_files = list(self.import_new_dir.glob("*.mp4"))
        logger.info(f"Found {len(mp4_files)} MP4 files in Import_New")

        for mp4_path in mp4_files:
            try:
                result = self._process_single_video(mp4_path)
                stats["details"].append(result)
                if result["status"] == "processed":
                    stats["processed"] += 1
                elif result["status"] == "skipped":
                    stats["skipped"] += 1
                else:
                    stats["failed"] += 1
            except Exception as e:
                logger.error(f"Failed to process {mp4_path.name}: {e}", exc_info=True)
                stats["failed"] += 1
                stats["details"].append({"file": mp4_path.name, "status": "error", "error": str(e)})

        return stats

    def _process_single_video(self, mp4_path: Path) -> Dict[str, Any]:
        """Process a single MP4 file from Import_New."""
        stem = mp4_path.stem
        h265_filename = f"{stem}.h265.mp4"
        metadata_filename = f"{stem}.json"

        h265_dest = self.queue_h265_dir / h265_filename
        metadata_dest = self.queue_metadata_dir / metadata_filename

        # Check if already exists in queue
        db = get_queue_db()
        existing = db.get_video_by_filename(h265_filename)
        if existing:
            logger.warning(f"Video already in queue: {h265_filename}, skipping")
            return {"file": mp4_path.name, "status": "skipped", "reason": "Already in queue"}

        try:
            # Step 1: Convert MP4 → H.265
            logger.info(f"Converting {mp4_path.name} → H.265")
            h265_path = convert_to_h265(
                mp4_path,
                output_path=h265_dest,
                crf=settings.ffmpeg_crf,
                preset=settings.ffmpeg_preset,
                audio_codec=settings.ffmpeg_audio_codec,
                overwrite=settings.ffmpeg_overwrite_output,
            )

            # Step 2: Generate metadata (placeholder – will be implemented later)
            metadata = self._generate_metadata_for_video(h265_path, stem)

            # Save metadata JSON
            import json
            with open(metadata_dest, "w") as f:
                json.dump(metadata, f, indent=2)

            # Step 3: Get Google Drive links (if enabled)
            drive_link_h265 = ""
            drive_link_metadata = ""
            if self.google_drive_enabled:
                drive_link_h265 = self._get_drive_link(h265_path)
                drive_link_metadata = self._get_drive_link(metadata_dest)

            # Step 4: Add to PostgreSQL queue
            video_record = db.enqueue_video(
                filename=h265_filename,
                drive_link_h265=drive_link_h265 or str(h265_dest),
                drive_link_metadata=drive_link_metadata or str(metadata_dest),
                file_size_bytes=h265_path.stat().st_size if h265_path.exists() else None,
                camera_id=stem,  # using filename stem as camera_id for now
                notes=f"Processed from {mp4_path.name}",
            )

            # Step 5: Delete original MP4 (FIFO: remove old input)
            mp4_path.unlink()
            logger.info(f"Deleted original MP4: {mp4_path}")

            return {
                "file": mp4_path.name,
                "status": "processed",
                "video_id": video_record["id"],
                "h265_file": h265_filename,
            }

        except Exception as e:
            logger.error(f"Error processing {mp4_path.name}: {e}", exc_info=True)
            # Cleanup partial files
            for f in [h265_dest, metadata_dest]:
                if f.exists():
                    f.unlink()
            raise

    def _generate_metadata_for_video(self, h265_path: Path, video_id: str) -> Dict[str, Any]:
        """
        Generate metadata by running detection + embedding models.
        Placeholder – cần integrate với code hiện tại của anh.
        """
        # TODO: Call detection + embedding model (GPU từ Lightning AI hoặc local)
        # Hiện tại return mock metadata
        logger.info(f"Generating metadata for {h265_path.name} (placeholder)")

        # Anh cần integrate code detection/embedding vào đây.
        # Ví dụ: gọi LightningAIClient hoặc local model.
        return {
            "video_id": video_id,
            "h265_path": str(h265_path),
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "detection": {
                "model": "YOLOv8x",
                "count": 0,  # placeholder
                "confidence_threshold": 0.5,
            },
            "embedding": {
                "model": "CLIP-RN50",
                "dimension": 512,
            },
            "note": "Placeholder metadata – replace with real detection+embedding",
        }

    def _get_drive_link(self, file_path: Path) -> str:
        """Get Google Drive shareable link for a file."""
        if not self.google_drive_enabled:
            return str(file_path)

        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload

            credentials_path = Path(settings.google_drive_credentials_file).expanduser()
            if not credentials_path.exists():
                logger.warning(f"Google Drive credentials not found: {credentials_path}")
                return str(file_path)

            credentials = service_account.Credentials.from_service_account_file(
                str(credentials_path),
                scopes=["https://www.googleapis.com/auth/drive"],
            )
            service = build("drive", "v3", credentials=credentials, cache_discovery=False)

            # Upload file nếu chưa có trên Drive
            parent_folder_id = self._get_or_create_folder_path("VinUni/Queue")

            media = MediaFileUpload(str(file_path), resumable=False)
            file_metadata = {
                "name": file_path.name,
                "parents": [parent_folder_id],
            }

            created = service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id, webViewLink",
            ).execute()

            file_id = created["id"]

            # Make public readable
            if settings.google_drive_make_public:
                try:
                    service.permissions().create(
                        fileId=file_id,
                        body={"type": "anyone", "role": "reader"},
                        fields="id",
                    ).execute()
                except Exception as e:
                    logger.warning(f"Could not make file public: {e}")

            return created.get("webViewLink", f"https://drive.google.com/file/d/{file_id}/view")

        except Exception as e:
            logger.error(f"Failed to get Drive link for {file_path}: {e}")
            return str(file_path)

    def _get_or_create_folder_path(self, path: str) -> str:
        """Get or create folder by path (e.g., 'VinUni/Queue/.h265')."""
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            credentials_path = Path(settings.google_drive_credentials_file).expanduser()
            credentials = service_account.Credentials.from_service_account_file(
                str(credentials_path),
                scopes=["https://www.googleapis.com/auth/drive"],
            )
            service = build("drive", "v3", credentials=credentials, cache_discovery=False)

            parts = path.strip("/").split("/")
            parent_id = None  # root

            for folder_name in parts:
                # Tìm folder con
                query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
                if parent_id:
                    query += f" and '{parent_id}' in parents"

                results = service.files().list(
                    q=query,
                    fields="files(id, name)",
                    pageSize=1,
                ).execute()

                files = results.get("files", [])
                if files:
                    folder_id = files[0]["id"]
                else:
                    # Tạo mới
                    file_metadata = {
                        "name": folder_name,
                        "mimeType": "application/vnd.google-apps.folder",
                    }
                    if parent_id:
                        file_metadata["parents"] = [parent_id]

                    folder = service.files().create(body=file_metadata, fields="id").execute()
                    folder_id = folder["id"]

                parent_id = folder_id

            return parent_id  # ID của folder cuối cùng

        except Exception as e:
            logger.error(f"Failed to create GDrive folder path '{path}': {e}")
            return ""

    # ==================== Queue Inspection ====================

    def list_queue(self, status: str | None = None) -> List[Dict[str, Any]]:
        """List all videos in queue."""
        db = get_queue_db()
        return db.list_videos(status=status)

    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue statistics."""
        db = get_queue_db()
        return db.get_queue_status()

    # ==================== Maintenance ====================

    def run_fifo_cleanup(self, max_keep: int | None = None) -> int:
        """
        Run FIFO cleanup: remove oldest videos when queue exceeds limit.

        Args:
            max_keep: Max videos to keep (default from settings)

        Returns:
            Number of videos removed
        """
        max_keep = max_keep or settings.max_queue_size
        db = get_queue_db()
        return db.cleanup_old_videos(max_keep)


# ==================== Singleton ====================

_queue_manager: VideoQueueManager | None = None


def get_queue_manager() -> VideoQueueManager:
    """Get or create VideoQueueManager singleton."""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = VideoQueueManager()
    return _queue_manager
