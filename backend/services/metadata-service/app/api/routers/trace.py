from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..core.dependencies import get_current_user
from ..core.models import User
from ..services.trace_service import FeedbackPayload, apply_feedback, build_seed, trace_from_candidate_id

router = APIRouter(prefix="/trace", tags=["trace"])


class TraceRequest(BaseModel):
    candidate_id: str
    window_hours: float = Field(default=12.0, ge=1.0, le=72.0)
    min_similarity: float = Field(default=0.40, ge=0.0, le=1.0)


class TraceFeedbackRequest(BaseModel):
    candidate_id: str
    confirmed_segment_ids: list[str] = Field(default_factory=list)
    rejected_segment_ids: list[str] = Field(default_factory=list)
    window_hours: float = Field(default=12.0, ge=1.0, le=72.0)


@router.post("/run")
def trace_run(
    payload: TraceRequest,
    session: Session = Depends(get_current_user.__self__.__class__),
    current_user: User = Depends(get_current_user),
) -> dict:
    from ..database import get_session
    session_gen = get_session()
    session = next(session_gen)
    try:
        try:
            result = trace_from_candidate_id(
                session,
                candidate_id=payload.candidate_id,
                window_hours=payload.window_hours,
            )
            return {
                "seed_candidate_id": result.seed_candidate_id,
                "total_segments": result.total_segments,
                "cameras_visited": result.cameras_visited,
                "overall_score": result.overall_score,
                "window_from": result.window_from,
                "window_to": result.window_to,
                "trajectory": [
                    {
                        "camera_id": clip.camera_id,
                        "candidate_id": clip.candidate_id,
                        "start_time": clip.start_time,
                        "end_time": clip.end_time,
                        "duration_seconds": clip.duration_seconds,
                        "appearance_sim": clip.appearance_sim,
                        "segment_score": clip.segment_score,
                        "preview_url": clip.preview_url,
                        "video_url": clip.video_url,
                        "details": clip.payload,
                    }
                    for clip in result.trajectory
                ],
            }
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        session.close()


@router.post("/feedback")
def trace_feedback(
    payload: TraceFeedbackRequest,
    session: Session = Depends(get_current_user.__self__.__class__),
    current_user: User = Depends(get_current_user),
) -> dict:
    from ..database import get_session
    session_gen = get_session()
    session = next(session_gen)
    try:
        try:
            seed = build_seed(session, payload.candidate_id, window_hours=payload.window_hours)
            fb = FeedbackPayload(
                candidate_id=payload.candidate_id,
                confirmed_segment_ids=payload.confirmed_segment_ids,
                rejected_segment_ids=payload.rejected_segment_ids,
            )
            result = apply_feedback(session, seed, fb, window_hours=payload.window_hours)
            return {
                "seed_candidate_id": result.seed_candidate_id,
                "total_segments": result.total_segments,
                "cameras_visited": result.cameras_visited,
                "overall_score": result.overall_score,
                "window_from": result.window_from,
                "window_to": result.window_to,
                "refined": True,
                "trajectory": [
                    {
                        "camera_id": clip.camera_id,
                        "candidate_id": clip.candidate_id,
                        "start_time": clip.start_time,
                        "end_time": clip.end_time,
                        "duration_seconds": clip.duration_seconds,
                        "appearance_sim": clip.appearance_sim,
                        "segment_score": clip.segment_score,
                        "preview_url": clip.preview_url,
                        "video_url": clip.video_url,
                        "details": clip.payload,
                    }
                    for clip in result.trajectory
                ],
            }
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        session.close()
