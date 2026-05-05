from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..core.dependencies import get_current_user
from ..core.models import User
from ..core.schemas import (
    CandidateListResponse,
    CandidateResponse,
    CandidateSearchRequest,
    CandidateSearchResponse,
    CandidateTrackRequest,
    CandidateTrackResponse,
)
from ..services.candidate_service import get_candidate, rank_candidates, search_candidates

router = APIRouter(prefix="/candidates", tags=["candidates"])


@router.get("", response_model=CandidateListResponse)
def list_candidates(
    query: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_current_user.__self__.__class__),
    current_user: User = Depends(get_current_user),
) -> dict:
    from ..database import get_session
    session_gen = get_session()
    session = next(session_gen)
    try:
        items = search_candidates(session=session, query=query, limit=limit)
        return {"count": len(items), "items": items}
    finally:
        session.close()


@router.get("/{candidate_id}", response_model=CandidateResponse)
def candidate_detail(
    candidate_id: str,
    session: Session = Depends(get_current_user.__self__.__class__),
    current_user: User = Depends(get_current_user),
) -> dict:
    from ..database import get_session
    session_gen = get_session()
    session = next(session_gen)
    try:
        candidate = get_candidate(session, candidate_id)
        if candidate is None:
            raise HTTPException(status_code=404, detail="Candidate not found")
        return candidate
    finally:
        session.close()


@router.get("/{candidate_id}/preview")
def candidate_preview(
    candidate_id: str,
    session: Session = Depends(get_current_user.__self__.__class__),
) -> Response:
    from ..database import get_session
    session_gen = get_session()
    session = next(session_gen)
    try:
        from ..services.candidate_service import build_candidate_preview_image
        candidate = get_candidate(session, candidate_id)
        if candidate is None:
            raise HTTPException(status_code=404, detail="Candidate not found")
        try:
            preview_path = build_candidate_preview_image(session, candidate_id)
            return FileResponse(preview_path, media_type="image/jpeg", filename=preview_path.name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        session.close()


@router.post("/search", response_model=CandidateSearchResponse)
def candidate_search(
    payload: CandidateSearchRequest,
    session: Session = Depends(get_current_user.__self__.__class__),
    current_user: User = Depends(get_current_user),
) -> dict:
    from ..database import get_session
    session_gen = get_session()
    session = next(session_gen)
    try:
        items = rank_candidates(
            session=session,
            query_text=payload.query_text,
            limit=payload.limit,
            camera_ids=payload.camera_ids,
            time_from=payload.time_from,
            time_to=payload.time_to,
        )
        return {"query_text": payload.query_text, "count": len(items), "items": items}
    finally:
        session.close()


@router.post("/track", response_model=CandidateTrackResponse)
def candidate_track(
    payload: CandidateTrackRequest,
    session: Session = Depends(get_current_user.__self__.__class__),
    current_user: User = Depends(get_current_user),
) -> dict:
    from ..database import get_session
    session_gen = get_session()
    session = next(session_gen)
    try:
        from ..services.candidate_service import build_tracking_video
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
    finally:
        session.close()
