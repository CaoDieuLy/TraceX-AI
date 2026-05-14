import os

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ...core.dependencies import get_current_user
from ...database import get_session
from shared.models import (
    EvidenceVideo,
    QueryCandidate,
    QueryCandidateTracklet,
    QueryHistory,
    Tracklet,
    User,
    Video,
)
from shared.tracklet_time import tracklet_time_window
from sqlalchemy.orm import contains_eager
from ...services.translation_display import translate_texts_to_vietnamese

router = APIRouter(tags=["history"])


def _positive_env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


_HISTORY_CANDIDATE_LIMIT = _positive_env_int("MAX_CANDIDATES", 50)


@router.get("")
def get_history(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> dict:
    rows = session.scalars(
        select(QueryHistory)
        .where(QueryHistory.user_id == current_user.id)
        .order_by(QueryHistory.created_at.desc())
        .limit(20)
    ).all()
    query_ids = [r.query_id for r in rows]

    candidate_counts: dict[str, int] = {}
    evidence_map: dict[str, int] = {}
    if query_ids:
        candidate_counts = {
            qid: int(count)
            for qid, count in session.execute(
                select(QueryCandidate.query_id, func.count(QueryCandidate.id))
                .where(
                    QueryCandidate.query_id.in_(query_ids),
                    QueryCandidate.rank_position <= _HISTORY_CANDIDATE_LIMIT,
                )
                .group_by(QueryCandidate.query_id)
            ).all()
        }
        # Latest evidence per query — max(id) is sufficient since id is monotonic.
        evidence_map = {
            qid: int(eid)
            for qid, eid in session.execute(
                select(EvidenceVideo.query_id, func.max(EvidenceVideo.id))
                .where(EvidenceVideo.query_id.in_(query_ids))
                .group_by(EvidenceVideo.query_id)
            ).all()
        }

    items = []
    for r in rows:
        evidence_id = evidence_map.get(r.query_id)
        items.append({
            "query_id": r.query_id,
            "query_text": r.query_text,
            "query_image_url": r.query_image_url,
            "status": r.status,
            "selected_candidate_id": r.selected_candidate_id,
            "candidate_count": candidate_counts.get(r.query_id, 0),
            "evidence_video_id": evidence_id,
            "has_evidence": evidence_id is not None,
            # Legacy fields, kept for backward compatibility with older clients.
            "video_id": r.video_id,
            "storage_path": None,
            "created_at": r.created_at.isoformat(),
            "updated_at": r.updated_at.isoformat(),
        })
    return {"count": len(items), "items": items}


def _verify_query_owner(
    session: Session, query_id: str, user_id: int
) -> QueryHistory:
    query = session.scalar(
        select(QueryHistory).where(
            QueryHistory.query_id == query_id,
            QueryHistory.user_id == user_id,
        )
    )
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")
    return query


@router.get("/{query_id}/candidates")
def get_history_candidates(
    query_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> dict:
    """Return the saved candidates for a past query, in the same shape that
    /search returns so the frontend can render them via VideoGrid without
    re-running the ranking pipeline."""
    query = _verify_query_owner(session, query_id, current_user.id)

    total_count = session.scalar(
        select(func.count())
        .select_from(QueryCandidate)
        .where(
            QueryCandidate.query_id == query_id,
            QueryCandidate.rank_position <= _HISTORY_CANDIDATE_LIMIT,
        )
    ) or 0

    candidates = session.scalars(
        select(QueryCandidate)
        .where(
            QueryCandidate.query_id == query_id,
            QueryCandidate.rank_position <= _HISTORY_CANDIDATE_LIMIT,
        )
        .order_by(QueryCandidate.rank_position.asc(), QueryCandidate.id.asc())
        .offset(offset)
        .limit(limit)
    ).all()
    selected_candidate_ids = session.scalars(
        select(QueryCandidate.candidate_id)
        .where(
            QueryCandidate.query_id == query_id,
            QueryCandidate.is_selected == True,  # noqa: E712
        )
        .order_by(QueryCandidate.rank_position.asc(), QueryCandidate.id.asc())
    ).all()

    # Pick one representative tracklet per candidate for the thumbnail URL and
    # collect tracklet summaries (camera + time window) so history cards match
    # fresh search result cards.
    rep_tracklets: dict[str, str] = {}
    tracklet_summaries: dict[str, list[dict]] = {}
    if candidates:
        cand_ids = [c.candidate_id for c in candidates]
        # First pick rep tracklet (min tracklet_id) per candidate for thumbnail.
        for cand_id, tracklet_id in session.execute(
            select(
                QueryCandidateTracklet.candidate_id,
                func.min(QueryCandidateTracklet.tracklet_id),
            )
            .where(QueryCandidateTracklet.candidate_id.in_(cand_ids))
            .group_by(QueryCandidateTracklet.candidate_id)
        ).all():
            rep_tracklets[cand_id] = tracklet_id

        # Then load all tracklets+video for these candidates to build summaries.
        rows = session.execute(
            select(QueryCandidateTracklet.candidate_id, Tracklet)
            .join(Tracklet, QueryCandidateTracklet.tracklet_id == Tracklet.tracklet_id)
            .join(Video, Tracklet.video_id == Video.video_id)
            .options(contains_eager(Tracklet.video))
            .where(QueryCandidateTracklet.candidate_id.in_(cand_ids))
        ).all()
        for cand_id, tracklet in rows:
            window = tracklet_time_window(tracklet)
            tracklet_summaries.setdefault(cand_id, []).append({
                "tracklet_id": tracklet.tracklet_id,
                "camera_id": tracklet.camera_id,
                "time_start": window[0].isoformat() if window else None,
                "time_end": window[1].isoformat() if window else None,
            })

    description_map = translate_texts_to_vietnamese([
        c.appearance_summary or "" for c in candidates
    ])

    results = []
    for c in candidates:
        thumbnail_url = c.preview_url or ""
        if not thumbnail_url:
            tid = rep_tracklets.get(c.candidate_id)
            if tid:
                thumbnail_url = f"/candidates/{tid}/preview"
        summaries = tracklet_summaries.get(c.candidate_id, [])
        tracklet_count = len(summaries)
        raw_description = c.appearance_summary or ""
        description = description_map.get(raw_description, raw_description)
        results.append({
            "id": c.candidate_id,
            "thumbnail_url": thumbnail_url,
            "description": description,
            "query_id": query_id,
            "is_selected": c.is_selected,
            "rank_position": c.rank_position,
            "fusion_score": c.fusion_score,
            "tracklet_count": tracklet_count,
            "tracklets": summaries,
        })

    return {
        "results": results,
        "query_id": query_id,
        "selected_candidate_id": query.selected_candidate_id,
        "selected_candidate_ids": selected_candidate_ids,
        "total_count": int(total_count),
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(results) < int(total_count),
    }


@router.get("/{query_id}/evidence")
def get_history_evidence(
    query_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> dict:
    """Return the latest evidence video metadata for a past query. The frontend
    chains this with /trace/timeline/{evidence_id} to load the segments."""
    _verify_query_owner(session, query_id, current_user.id)

    evidence = session.scalar(
        select(EvidenceVideo)
        .where(EvidenceVideo.query_id == query_id)
        .order_by(EvidenceVideo.created_at.desc())
        .limit(1)
    )
    if not evidence:
        raise HTTPException(status_code=404, detail="No evidence video for this query")

    return {
        "evidence_id": evidence.id,
        "query_id": query_id,
        "candidate_id": evidence.query_candidate_id,
        "trace_confidence": evidence.trace_confidence,
        "segment_count": evidence.segment_count,
        "total_duration": evidence.total_duration,
        "created_at": evidence.created_at.isoformat(),
    }


class SelectRequest(BaseModel):
    query: str = ""
    selectedIndex: int = 0


@router.post("/select", status_code=200)
def select_history(
    body: SelectRequest,
    _: User = Depends(get_current_user),
) -> dict:
    return {"ok": True}
