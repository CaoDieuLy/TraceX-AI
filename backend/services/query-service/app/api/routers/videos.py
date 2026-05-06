"""Video query endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from shared import SessionLocal
from ..services.candidate_query import get_queue_video, list_queue_videos

router = APIRouter()


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/{video_id}")
def get_video(video_id: str, db: Session = Depends(get_db)):
    """Get video information."""
    video = get_queue_video(db, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")
    
    return {
        "video_id": video.video_id,
        "camera_id": video.camera_id,
        "title": video.title,
        "queue_position": video.queue_position,
        "storage_backend": video.storage_backend,
        "available_link_video": video.available_link_video,
        "available_link_metadata": video.available_link_metadata,
        "source_filename": video.source_filename,
        "created_at": video.created_at,
    }


@router.get("/")
def list_videos(
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """List all queue videos."""
    videos = list_queue_videos(db)
    
    # Apply pagination
    total = len(videos)
    videos = videos[offset:offset + limit]
    
    return {
        "videos": videos,
        "total": total,
        "limit": limit,
        "offset": offset,
    }
