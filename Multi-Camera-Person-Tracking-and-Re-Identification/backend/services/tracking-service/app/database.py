"""PostgreSQL database client for video queue management."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor

from .config import settings

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database errors."""
    pass


class VideoNotFoundError(DatabaseError):
    """Raised when video is not found in database."""
    pass


class VideoQueueDB:
    """PostgreSQL client for video queue management."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        """Initialize database connection parameters."""
        self.host = host or settings.postgres_host
        self.port = port or settings.postgres_port
        self.database = database or settings.postgres_database
        self.user = user or settings.postgres_user
        self.password = password or settings.postgres_password

        if not all([self.host, self.database, self.user]):
            raise ValueError("Missing required database connection parameters")

        self._conn: Optional[psycopg2.extensions.connection] = None

    @contextmanager
    def get_connection(self):
        """Context manager for database connection."""
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                cursor_factory=RealDictCursor,
            )
            yield conn
            conn.commit()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise DatabaseError(f"Database operation failed: {e}") from e
        finally:
            if conn:
                conn.close()

    @contextmanager
    def get_cursor(self):
        """Context manager for database cursor."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            try:
                yield cur
            finally:
                cur.close()

    # ==================== Video CRUD ====================

    def add_video(
        self,
        filename: str,
        drive_link_h265: str,
        drive_link_metadata: str,
        drive_file_id_h265: str | None = None,
        drive_file_id_metadata: str | None = None,
        file_size_bytes: int | None = None,
        duration_seconds: float | None = None,
        resolution: str | None = None,
        camera_id: str | None = None,
        notes: str | None = None,
        status: str = "available",
    ) -> Dict[str, Any]:
        """
        Add a new video to the queue.

        Returns:
            Dictionary with inserted video record
        """
        with self.get_cursor() as cur:
            cur.execute("""
                INSERT INTO videos (
                    filename, drive_link_h265, drive_link_metadata,
                    drive_file_id_h265, drive_file_id_metadata,
                    file_size_bytes, duration_seconds, resolution,
                    camera_id, notes, status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (filename) DO UPDATE SET
                    drive_link_h265 = EXCLUDED.drive_link_h265,
                    drive_link_metadata = EXCLUDED.drive_link_metadata,
                    drive_file_id_h265 = COALESCE(EXCLUDED.drive_file_id_h265, videos.drive_file_id_h265),
                    drive_file_id_metadata = COALESCE(EXCLUDED.drive_file_id_metadata, videos.drive_file_id_metadata),
                    file_size_bytes = COALESCE(EXCLUDED.file_size_bytes, videos.file_size_bytes),
                    duration_seconds = COALESCE(EXCLUDED.duration_seconds, videos.duration_seconds),
                    resolution = COALESCE(EXCLUDED.resolution, videos.resolution),
                    camera_id = COALESCE(EXCLUDED.camera_id, videos.camera_id),
                    notes = COALESCE(EXCLUDED.notes, videos.notes),
                    status = EXCLUDED.status,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING *
            """, (
                filename, drive_link_h265, drive_link_metadata,
                drive_file_id_h265, drive_file_id_metadata,
                file_size_bytes, duration_seconds, resolution,
                camera_id, notes, status
            ))
            result = dict(cur.fetchone())

        logger.info(f"Added/updated video in queue: {filename} (id={result['id']})")
        return result

    def get_video_by_id(self, video_id: int) -> Optional[Dict[str, Any]]:
        """Get video by ID."""
        with self.get_cursor() as cur:
            cur.execute("SELECT * FROM videos WHERE id = %s", (video_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_video_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get video by filename."""
        with self.get_cursor() as cur:
            cur.execute("SELECT * FROM videos WHERE filename = %s", (filename,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_next_video(self) -> Optional[Dict[str, Any]]:
        """
        Get next video for processing (FIFO: oldest available first).
        Uses SELECT ... FOR UPDATE SKIP LOCKED for concurrent access.
        """
        with self.get_cursor() as cur:
            cur.execute("SELECT * FROM get_next_video_for_processing()")
            row = cur.fetchone()
            return dict(row) if row else None

    def list_videos(
        self,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List videos with optional filters."""
        query = "SELECT * FROM videos"
        params: List[Any] = []

        if status:
            query += " WHERE status = %s"
            params.append(status)

        query += " ORDER BY uploaded_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        with self.get_cursor() as cur:
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]

    def update_video_status(self, video_id: int, status: str) -> None:
        """Update video status."""
        with self.get_cursor() as cur:
            cur.execute(
                "UPDATE videos SET status = %s WHERE id = %s",
                (status, video_id)
            )
        logger.info(f"Updated video id={video_id} status={status}")

    def update_video_processed(self, video_id: int) -> None:
        """Mark video as processed (set processed_at timestamp)."""
        with self.get_cursor() as cur:
            cur.execute(
                "UPDATE videos SET processed_at = CURRENT_TIMESTAMP WHERE id = %s",
                (video_id,)
            )

    def delete_video(self, video_id: int) -> bool:
        """Delete video record (FIFO cleanup)."""
        with self.get_cursor() as cur:
            cur.execute("DELETE FROM videos WHERE id = %s RETURNING id", (video_id,))
            deleted = cur.fetchone()
            if deleted:
                logger.info(f"Deleted video id={video_id} (FIFO cleanup)")
                return True
            return False

    # ==================== Queue Operations ====================

    def enqueue_video(
        self,
        filename: str,
        drive_link_h265: str,
        drive_link_metadata: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Add video to queue (FIFO).

        Automatically triggers cleanup if queue exceeds MAX_QUEUE_SIZE.
        """
        video = self.add_video(filename, drive_link_h265, drive_link_metadata, **kwargs)

        # Log action
        self._log_queue_action(video["id"], "enqueue", reason="New video added")

        # FIFO cleanup: giữ tối đa MAX_QUEUE_SIZE video available
        max_size = getattr(settings, 'max_queue_size', 100)
        self.cleanup_old_videos(max_size)

        return video

    def dequeue_video(self) -> Optional[Dict[str, Any]]:
        """
        Get next video from queue and mark as 'processing'.
        Returns None if queue empty.
        """
        video = self.get_next_video()
        if not video:
            return None

        self.update_video_status(video["id"], "processing")
        self._log_queue_action(video["id"], "dequeue", reason="Processing started")
        return video

    def complete_video(self, video_id: int) -> None:
        """Mark video processing as completed."""
        self.update_video_status(video_id, "available")  # hoặc 'completed' nếu muốn
        self.update_video_processed(video_id)
        self._log_queue_action(video_id, "dequeue", reason="Processing completed")

    def cleanup_old_videos(self, max_keep: int = 100) -> int:
        """
        Remove oldest videos from queue (FIFO).

        Args:
            max_keep: Số video tối đa giữ lại trong queue

        Returns:
            Số record bị xóa
        """
        with self.get_cursor() as cur:
            cur.execute("SELECT cleanup_old_videos(%s)", (max_keep,))
            deleted_count = cur.fetchone()[0]

        if deleted_count > 0:
            logger.info(f"FIFO cleanup: removed {deleted_count} old videos (max_keep={max_keep})")

        return deleted_count

    # ==================== Stats & Monitoring ====================

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue statistics."""
        with self.get_cursor() as cur:
            cur.execute("SELECT * FROM v_queue_status")
            rows = cur.fetchall()

        status_counts = {}
        oldest_uploaded = None
        newest_uploaded = None

        for row in rows:
            status_counts[row["status"]] = row["count"]
            if row["oldest_uploaded"] and (oldest_uploaded is None or row["oldest_uploaded"] < oldest_uploaded):
                oldest_uploaded = row["oldest_uploaded"]
            if row["newest_uploaded"] and (newest_uploaded is None or row["newest_uploaded"] > newest_uploaded):
                newest_uploaded = row["newest_uploaded"]

        return {
            "status_counts": status_counts,
            "total": sum(status_counts.values()),
            "oldest_uploaded": oldest_uploaded,
            "newest_uploaded": newest_uploaded,
        }

    def count_available(self) -> int:
        """Count available videos."""
        with self.get_cursor() as cur:
            cur.execute("SELECT count_available_videos()")
            return cur.fetchone()[0]

    # ==================== Internal Helpers ====================

    def _log_queue_action(
        self,
        video_id: int,
        action: str,
        reason: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Log queue action to queue_log table."""
        with self.get_cursor() as cur:
            cur.execute("""
                INSERT INTO queue_log (video_id, action, reason, metadata)
                VALUES (%s, %s, %s, %s)
            """, (video_id, action, reason, psycopg2.extras.Json(metadata) if metadata else None))


# ==================== Singleton Instance ====================

_queue_db: VideoQueueDB | None = None


def get_queue_db() -> VideoQueueDB:
    """Get or create VideoQueueDB singleton."""
    global _queue_db
    if _queue_db is None:
        _queue_db = VideoQueueDB()
    return _queue_db
