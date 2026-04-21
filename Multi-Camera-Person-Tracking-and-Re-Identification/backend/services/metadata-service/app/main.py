from __future__ import annotations

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from .auth import create_access_token
from .database import Base, engine, get_session
from .deps import get_current_user
from .models import User
from .queue_runtime import QueueSyncService
from .schemas import (
    AuthTokenResponse,
    CandidateListResponse,
    CandidateResponse,
    ImportResponse,
    OverviewResponse,
    QueueBootstrapRequest,
    QueueBootstrapResponse,
    QueueProcessResponse,
    QueueVideoListResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
    VideoListResponse,
    VideoQueryCreateRequest,
    VideoQueryListResponse,
    VideoQueryResponse,
    VideoQueryUpdateRequest,
    VideoResponse,
)
from .service import (
    authenticate_user,
    create_user,
    create_video_asset,
    create_video_query,
    get_candidate,
    get_overview,
    get_video_by_public_id,
    get_video_query,
    import_legacy_metadata,
    list_queue_videos,
    list_video_queries,
    list_videos,
    save_uploaded_video_bytes,
    search_candidates,
    update_video_query,
    video_to_payload,
    query_to_payload,
)

app = FastAPI(title="MCPT Metadata Service", version="2.0.0")


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok", "service": "metadata-service"}


@app.post("/api/v1/auth/register", response_model=AuthTokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: UserRegisterRequest, session: Session = Depends(get_session)) -> dict:
    try:
        user = create_user(session, email=payload.email, full_name=payload.full_name, password=payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    token = create_access_token(user.email)
    return {"access_token": token, "user": user}


@app.post("/api/v1/auth/login", response_model=AuthTokenResponse)
def login(payload: UserLoginRequest, session: Session = Depends(get_session)) -> dict:
    user = authenticate_user(session, email=payload.email, password=payload.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    token = create_access_token(user.email)
    return {"access_token": token, "user": user}


@app.get("/api/v1/auth/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)) -> User:
    return current_user


@app.get("/api/v1/overview", response_model=OverviewResponse)
def overview(session: Session = Depends(get_session)) -> dict:
    return get_overview(session)


@app.post("/api/v1/videos", response_model=VideoResponse, status_code=status.HTTP_201_CREATED)
async def create_video(
    title: str = Form(...),
    description: str | None = Form(default=None),
    storage_url: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> dict:
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


@app.get("/api/v1/videos", response_model=VideoListResponse)
def videos(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)) -> dict:
    items = list_videos(session, current_user)
    return {"count": len(items), "items": items}


@app.get("/api/v1/videos/{video_id}", response_model=VideoResponse)
def video_detail(video_id: str, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)) -> dict:
    video = get_video_by_public_id(session, current_user, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return video_to_payload(video)


@app.post("/api/v1/video-queries", response_model=VideoQueryResponse, status_code=status.HTTP_201_CREATED)
def create_query(
    payload: VideoQueryCreateRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> dict:
    video = get_video_by_public_id(session, current_user, payload.video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    query = create_video_query(session, current_user, video, payload.query_text)
    return query_to_payload(query)


@app.get("/api/v1/video-queries", response_model=VideoQueryListResponse)
def queries(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)) -> dict:
    items = list_video_queries(session, current_user)
    return {"count": len(items), "items": items}


@app.patch("/api/v1/video-queries/{query_id}", response_model=VideoQueryResponse)
def patch_query(
    query_id: str,
    payload: VideoQueryUpdateRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> dict:
    query = get_video_query(session, current_user, query_id)
    if query is None:
        raise HTTPException(status_code=404, detail="Query not found")
    updated = update_video_query(
        session,
        query,
        status=payload.status,
        ai_job_id=payload.ai_job_id,
        ai_response=payload.ai_response,
    )
    return query_to_payload(updated)


@app.get("/api/v1/candidates", response_model=CandidateListResponse)
def list_candidates(
    query: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> dict:
    items = search_candidates(session=session, query=query, limit=limit)
    return {"count": len(items), "items": items}


@app.get("/api/v1/candidates/{candidate_id}", response_model=CandidateResponse)
def candidate_detail(candidate_id: str, session: Session = Depends(get_session)) -> dict:
    candidate = get_candidate(session, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate


@app.post("/api/v1/candidates/import-legacy", response_model=ImportResponse)
def import_candidates(session: Session = Depends(get_session)) -> dict:
    return import_legacy_metadata(session)


@app.get("/api/v1/queue/videos", response_model=QueueVideoListResponse)
def queue_videos(session: Session = Depends(get_session)) -> dict:
    items = list_queue_videos(session)
    return {"count": len(items), "items": items}


@app.post("/api/v1/queue/bootstrap", response_model=QueueBootstrapResponse)
def bootstrap_queue(payload: QueueBootstrapRequest, session: Session = Depends(get_session)) -> dict:
    try:
        return QueueSyncService().bootstrap_from_source_dir(
            session,
            source_dir=payload.source_dir,
            limit=payload.limit,
            reset_remote_queue=payload.reset_remote_queue,
            delete_source_after_import=payload.delete_source_after_import,
        )
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/queue/process-imports", response_model=QueueProcessResponse)
def process_imports(session: Session = Depends(get_session)) -> dict:
    try:
        return QueueSyncService().process_import_queue(session)
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
