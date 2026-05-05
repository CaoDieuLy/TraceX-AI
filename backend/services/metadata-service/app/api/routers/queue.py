from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from ..core.dependencies import get_current_user
from ..core.models import User
from ..core.schemas import QueueProcessResponse, QueueVideoListResponse
from ..services.queue_service import (
    QueueSyncService,
    list_queue_videos,
    load_queue_video_file_path,
    load_queue_video_metadata,
)

router = APIRouter(prefix="/queue", tags=["queue"])


@router.get("/videos", response_model=QueueVideoListResponse)
def queue_videos(
    session: Session = Depends(get_current_user.__self__.__class__),
    current_user: User = Depends(get_current_user),
) -> dict:
    from ..database import get_session
    session_gen = get_session()
    session = next(session_gen)
    try:
        items = list_queue_videos(session)
        return {"count": len(items), "items": items}
    finally:
        session.close()


@router.get("/videos/{video_id}/metadata")
def queue_video_metadata(
    video_id: str,
    session: Session = Depends(get_current_user.__self__.__class__),
    current_user: User = Depends(get_current_user),
) -> dict:
    from ..database import get_session
    session_gen = get_session()
    session = next(session_gen)
    try:
        try:
            return load_queue_video_metadata(session, video_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        session.close()


@router.get("/videos/{video_id}/file")
def queue_video_file(
    video_id: str,
    session: Session = Depends(get_current_user.__self__.__class__),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    from ..database import get_session
    session_gen = get_session()
    session = next(session_gen)
    try:
        try:
            video_path = load_queue_video_file_path(session, video_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        media_type = "video/mp4" if video_path.suffix.lower() == ".mp4" else "application/octet-stream"
        return FileResponse(video_path, media_type=media_type, filename=video_path.name)
    finally:
        session.close()


@router.post("/process-storage", response_model=QueueProcessResponse)
def process_storage(
    session: Session = Depends(get_current_user.__self__.__class__),
) -> dict:
    from ..database import get_session
    session_gen = get_session()
    session = next(session_gen)
    try:
        try:
            return QueueSyncService().process_storage_queue(session)
        except Exception as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        session.close()
