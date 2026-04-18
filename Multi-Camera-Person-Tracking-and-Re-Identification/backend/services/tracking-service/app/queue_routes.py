from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from .queue_manager import get_queue_manager, VideoQueueManager
from .database import get_queue_db, VideoNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/queue", tags=["queue"])


# ==================== Schemas ====================

class EnqueueResponse(BaseModel):
    status: str
    message: str
    processed: int
    failed: int
    skipped: int
    details: list


class QueueStatusResponse(BaseModel):
    status_counts: dict
    total: int
    oldest_uploaded: Optional[str] = None
    newest_uploaded: Optional[str] = None


class VideoItem(BaseModel):
    id: int
    filename: str
    drive_link_h265: str
    drive_link_metadata: str
    status: str
    uploaded_at: str


class VideoListResponse(BaseModel):
    videos: list[VideoItem]
    total: int


# ==================== Endpoints ====================

@router.post("/enqueue-import-new", response_model=EnqueueResponse)
async def enqueue_import_new(background_tasks: BackgroundTasks):
    """
    Scan Import_New folder, convert all MP4 → H.265 + metadata,
    move to Queue, and add to PostgreSQL database.
    """
    try:
        manager = get_queue_manager()
        result = manager.enqueue_new_uploads()
        return EnqueueResponse(
            status="completed",
            message=f"Processed {result['processed']} videos, {result['failed']} failed, {result['skipped']} skipped",
            **result
        )
    except Exception as e:
        logger.error(f"Enqueue failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=QueueStatusResponse)
async def get_queue_status():
    """Get current queue statistics."""
    db = get_queue_db()
    status = db.get_queue_status()
    return QueueStatusResponse(**status)


@router.get("/videos", response_model=VideoListResponse)
async def list_videos(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """List videos in queue with optional filter."""
    db = get_queue_db()
    videos = db.list_videos(status=status, limit=limit, offset=offset)
    total = db.count_available() if status is None else len(videos)
    return VideoListResponse(
        videos=[VideoItem(**v) for v in videos],
        total=total
    )


@router.post("/videos/{video_id}/dequeue")
async def dequeue_video(video_id: int):
    """Get next video for processing (FIFO)."""
    db = get_queue_db()
    video = db.get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    db.update_video_status(video_id, "processing")
    return {
        "status": "dequeued",
        "video_id": video_id,
        "filename": video["filename"],
        "drive_link_h265": video["drive_link_h265"],
        "drive_link_metadata": video["drive_link_metadata"],
    }


@router.post("/videos/{video_id}/complete")
async def complete_video(video_id: int):
    """Mark video processing as completed."""
    db = get_queue_db()
    try:
        db.complete_video(video_id)
        return {"status": "completed", "video_id": video_id}
    except VideoNotFoundError:
        raise HTTPException(status_code=404, detail="Video not found")


@router.delete("/videos/{video_id}")
async def delete_video(video_id: int):
    """Delete video from queue (FIFO cleanup)."""
    db = get_queue_db()
    deleted = db.delete_video(video_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Video not found")
    return {"status": "deleted", "video_id": video_id}


@router.post("/cleanup")
async def cleanup_queue(max_keep: int = 100):
    """
    Run FIFO cleanup: remove oldest videos when queue exceeds max_keep.
    """
    db = get_queue_db()
    deleted = db.cleanup_old_videos(max_keep)
    return {
        "status": "cleanup_completed",
        "max_keep": max_keep,
        "deleted_count": deleted,
    }


@router.post("/batch-convert")
async def batch_convert_existing_videos(
    source_dir: str,
    background_tasks: BackgroundTasks
):
    """
    Batch convert all MP4 videos in source directory to H.265 + metadata,
    then move to Queue folder.

    Args:
        source_dir: Path to directory containing MP4 files
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise HTTPException(status_code=404, detail=f"Source directory not found: {source_dir}")

    mp4_files = list(source_path.glob("*.mp4"))
    if not mp4_files:
        return {"status": "no_files", "message": "No MP4 files found"}

    # Background task to avoid blocking
    def _batch_convert():
        manager = get_queue_manager()
        for mp4 in mp4_files:
            try:
                # Copy to Import_New first, then process
                dest = manager.import_new_dir / mp4.name
                shutil.copy2(mp4, dest)
                logger.info(f"Copied {mp4} → {dest}")
            except Exception as e:
                logger.error(f"Failed to copy {mp4}: {e}")

        # Process all files in Import_New
        manager.enqueue_new_uploads()

    background_tasks.add_task(_batch_convert)

    return {
        "status": "started",
        "message": f"Batch converting {len(mp4_files)} videos",
        "source_dir": source_dir,
        "file_count": len(mp4_files),
    }
