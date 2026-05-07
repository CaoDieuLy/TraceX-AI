from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ...core.dependencies import get_current_user
from shared.models import User
from ...core.schemas import (
    VideoListResponse,
    VideoQueryCreateRequest,
    VideoQueryListResponse,
    VideoQueryResponse,
    VideoQueryUpdateRequest,
    VideoResponse,
)
from ...database import get_session
from ...services.video_service import (
    create_video_asset,
    create_video_query,
    get_video_by_public_id,
    get_video_query,
    list_video_queries,
    list_videos,
    query_to_payload,
    save_uploaded_video_bytes,
    update_video_query,
    video_to_payload,
)

router = APIRouter(prefix="/videos", tags=["videos"])


@router.post("", response_model=VideoResponse, status_code=status.HTTP_201_CREATED)
async def create_video(
    title: str = Form(...),
    description: str | None = Form(default=None),
    storage_url: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> dict:
    try:
        if file is None and not (storage_url or "").strip():
            raise HTTPException(status_code=400, detail="Provide either a file upload or a storage_url")
        if file is not None:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            storage_path = save_uploaded_video_bytes(file.filename or "video.bin", content)
            storage_backend = "local_volume"
            source_filename = file.filename
            content_type = file.content_type
        else:
            storage_path = str(storage_url).strip()
            storage_backend = "external_reference"
            source_filename = None
            content_type = None
        video = create_video_asset(
            session=session,
            user=current_user,
            title=title,
            description=description,
            storage_path=storage_path,
            storage_backend=storage_backend,
            source_filename=source_filename,
            content_type=content_type,
        )
        return video_to_payload(video)
    finally:
        session.close()


@router.get("", response_model=VideoListResponse)
def videos(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> dict:
    try:
        items = list_videos(session, current_user)
        return {"count": len(items), "items": items}
    finally:
        session.close()


@router.get("/{video_id}", response_model=VideoResponse)
def video_detail(
    video_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> dict:
    try:
        video = get_video_by_public_id(session, current_user, video_id)
        if video is None:
            raise HTTPException(status_code=404, detail="Video not found")
        return video_to_payload(video)
    finally:
        session.close()


@router.post("/queries", response_model=VideoQueryResponse, status_code=status.HTTP_201_CREATED)
def create_query(
    payload: VideoQueryCreateRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> dict:
    try:
        video = get_video_by_public_id(session, current_user, payload.video_id)
        if video is None:
            raise HTTPException(status_code=404, detail="Video not found")
        query = create_video_query(session, current_user, video, payload.query_text)
        return query_to_payload(query)
    finally:
        session.close()


@router.get("/queries", response_model=VideoQueryListResponse)
def queries(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> dict:
    try:
        items = list_video_queries(session, current_user)
        return {"count": len(items), "items": items}
    finally:
        session.close()


@router.patch("/queries/{query_id}", response_model=VideoQueryResponse)
def patch_query(
    query_id: str,
    payload: VideoQueryUpdateRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> dict:
    try:
        query = get_video_query(session, current_user, query_id)
        if query is None:
            raise HTTPException(status_code=404, detail="Query not found")
        updated = update_video_query(
            session,
            query,
            status=payload.status,
            ai_job_id=payload.ai_job_id,
        )
        return query_to_payload(updated)
    finally:
        session.close()
