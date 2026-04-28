from __future__ import annotations

import json
import logging
from sqlalchemy import text

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from .auth import create_access_token
from .config import settings
from .database import Base, SessionLocal, engine, get_session
from .dependencies import get_current_user, require_admin
from .models import User
from .queue_runtime import QueueSyncService
from .schemas import (
    AuthTokenResponse,
    AdminUserCreateRequest,
    AdminUserPatchRequest,
    CandidateListResponse,
    CandidateSearchRequest,
    CandidateSearchResponse,
    CandidateTrackRequest,
    CandidateTrackResponse,
    CandidateResponse,
    OverviewResponse,
    QueueProcessResponse,
    QueueVideoListResponse,
    UserLoginRequest,
    UserListResponse,
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
    build_candidate_preview_image,
    build_tracking_video,
    create_user,
    create_video_asset,
    create_video_query,
    get_candidate,
    get_overview,
    get_video_by_public_id,
    get_video_query,
    list_users,
    list_queue_videos,
    list_video_queries,
    list_videos,
    load_queue_video_metadata,
    load_queue_video_file_path,
    rank_candidates,
    resolve_tracking_artifact_paths,
    save_uploaded_video_bytes,
    search_candidates,
    sync_local_queue_state,
    update_user_access,
    get_user_by_id,
    ensure_bootstrap_admin,
    normalize_role,
    role_rank,
    update_video_query,
    video_to_payload,
    query_to_payload,
)

app = FastAPI(title="MCPT Metadata Service", version="2.0.0")
LOGGER = logging.getLogger(__name__)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        # Keep auth/schema compatibility changes even if queue sync fails later.
        session.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(20) NOT NULL DEFAULT 'USER'"))
        session.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE"))
        session.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login TIMESTAMPTZ NULL"))
        session.execute(text("UPDATE users SET role = 'USER' WHERE role IS NULL"))
        session.execute(text("UPDATE users SET is_active = TRUE WHERE is_active IS NULL"))
        session.commit()
        ensure_bootstrap_admin(
            session,
            email=settings.bootstrap_admin_email,
            password=settings.bootstrap_admin_password,
            full_name=settings.bootstrap_admin_full_name,
        )
        session.commit()
        sync_local_queue_state(session, only_if_empty=True)
        session.commit()
    except Exception:
        session.rollback()
        LOGGER.exception("Metadata service startup initialization failed")
    finally:
        session.close()


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok", "service": "metadata-service"}


@app.post("/api/v1/auth/register")
def register() -> dict:
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Public registration is disabled")


@app.post("/api/v1/auth/login", response_model=AuthTokenResponse)
def login(payload: UserLoginRequest, session: Session = Depends(get_session)) -> dict:
    user = authenticate_user(session, identifier=payload.identifier, password=payload.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username, email, or password")
    token = create_access_token(user_id=user.id, email=user.email, role=user.role)
    return {"access_token": token, "user": user}


@app.post("/auth/login", response_model=AuthTokenResponse)
def auth_login(payload: UserLoginRequest, session: Session = Depends(get_session)) -> dict:
    user = authenticate_user(session, identifier=payload.identifier, password=payload.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username, email, or password")
    token = create_access_token(user_id=user.id, email=user.email, role=user.role)
    return {"access_token": token, "user": user}


@app.get("/api/v1/auth/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)) -> User:
    return current_user


@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def admin_create_user(
    payload: AdminUserCreateRequest,
    session: Session = Depends(get_session),
    _admin: User = Depends(require_admin),
) -> User:
    try:
        creator_rank = role_rank(_admin.role)
        requested_role = normalize_role(payload.role)
        requested_rank = role_rank(requested_role)
        if requested_rank >= creator_rank:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You cannot create account with equal or higher role.",
            )
        if requested_role == "ADMIN" and normalize_role(_admin.role) != "SUPER_ADMIN":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only SUPER_ADMIN can create ADMIN accounts.",
            )
        return create_user(
            session,
            email=payload.email,
            full_name=payload.full_name,
            password=payload.password,
            role=requested_role,
            is_active=payload.is_active,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/users", response_model=UserListResponse)
def admin_list_users(session: Session = Depends(get_session), _admin: User = Depends(require_admin)) -> dict:
    items = list_users(session)
    return {"count": len(items), "items": items}


@app.patch("/users/{user_id}", response_model=UserResponse)
def admin_patch_user(
    user_id: int,
    payload: AdminUserPatchRequest,
    session: Session = Depends(get_session),
    _admin: User = Depends(require_admin),
) -> User:
    target = get_user_by_id(session, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    if payload.role is None and payload.is_active is None:
        return target
    if target.id == _admin.id and (payload.role is not None or payload.is_active is not None):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot change your own role or active status.",
        )
    actor_role = normalize_role(_admin.role)
    actor_rank = role_rank(actor_role)
    target_role = normalize_role(target.role)
    target_rank = role_rank(target_role)
    if target_rank >= actor_rank:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot modify users with equal or higher role.",
        )
    if payload.role is not None:
        next_role = normalize_role(payload.role)
        next_rank = role_rank(next_role)
        if next_rank >= actor_rank:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You cannot assign equal or higher role.",
            )
        if next_role == "ADMIN" and actor_role != "SUPER_ADMIN":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only SUPER_ADMIN can modify ADMIN accounts.",
            )
    if target_role == "ADMIN" and actor_role != "SUPER_ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only SUPER_ADMIN can modify ADMIN accounts.",
        )
    return update_user_access(session, target, role=payload.role, is_active=payload.is_active)


@app.get("/api/v1/overview", response_model=OverviewResponse)
def overview(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)) -> dict:
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
    current_user: User = Depends(get_current_user),
) -> dict:
    items = search_candidates(session=session, query=query, limit=limit)
    return {"count": len(items), "items": items}


@app.get("/api/v1/candidates/{candidate_id}", response_model=CandidateResponse)
def candidate_detail(
    candidate_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> dict:
    candidate = get_candidate(session, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate


@app.get("/api/v1/candidates/{candidate_id}/preview")
def candidate_preview(
    candidate_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    try:
        preview_path = build_candidate_preview_image(session, candidate_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(preview_path, media_type="image/jpeg", filename=preview_path.name)


@app.post("/api/v1/candidates/search", response_model=CandidateSearchResponse)
def candidate_search(
    payload: CandidateSearchRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> dict:
    items = rank_candidates(session=session, query_text=payload.query_text, limit=payload.limit)
    return {"query_text": payload.query_text, "count": len(items), "items": items}


@app.post("/api/v1/candidates/track", response_model=CandidateTrackResponse)
def candidate_track(
    payload: CandidateTrackRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> dict:
    try:
        return build_tracking_video(
            session,
            selected_candidate_id=payload.selected_candidate_id,
            candidate_ids=payload.candidate_ids,
            query_text=payload.query_text,
            max_segments_per_candidate=payload.max_segments_per_candidate,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/api/v1/queue/videos", response_model=QueueVideoListResponse)
def queue_videos(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)) -> dict:
    items = list_queue_videos(session)
    return {"count": len(items), "items": items}


@app.get("/api/v1/queue/videos/{video_id}/metadata")
def queue_video_metadata(
    video_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> dict:
    try:
        return load_queue_video_metadata(session, video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/queue/videos/{video_id}/file")
def queue_video_file(
    video_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    try:
        video_path = load_queue_video_file_path(session, video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    media_type = "video/mp4" if video_path.suffix.lower() == ".mp4" else "application/octet-stream"
    return FileResponse(video_path, media_type=media_type, filename=video_path.name)


@app.post("/api/v1/queue/process-storage", response_model=QueueProcessResponse)
def process_storage(
    session: Session = Depends(get_session),
    current_user: User = Depends(require_admin),
) -> dict:
    try:
        return QueueSyncService().process_storage_queue(session)
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/tracking-artifacts/{artifact_id}")
def tracking_artifact_video(artifact_id: str, current_user: User = Depends(get_current_user)) -> FileResponse:
    video_path, _manifest_path = resolve_tracking_artifact_paths(artifact_id)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Tracking artifact not found")
    return FileResponse(video_path, media_type="video/mp4", filename=video_path.name)


@app.get("/api/v1/tracking-artifacts/{artifact_id}/manifest")
def tracking_artifact_manifest(artifact_id: str, current_user: User = Depends(get_current_user)) -> JSONResponse:
    _video_path, manifest_path = resolve_tracking_artifact_paths(artifact_id)
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Tracking manifest not found")
    return JSONResponse(content=json.loads(manifest_path.read_text(encoding="utf-8")))
