"""Candidates search router for query-service.

Flow:
1. metadata-service (/api/v1/search) forwards request here
2. Local text pre-filter → shortlist candidates
3. Forward shortlist + query to trace-service (/api/v1/candidates/search) for GPU re-ranking
4. Format and return results
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session


class SearchRequest(BaseModel):
    query: str = ""
    text: str | None = None  # alias
    top_k: int = 20
    offset: int = 0
    camera_ids: list[str] | None = None
    time_from: str | None = None
    time_to: str | None = None
    query_image_url: str | None = None  # URL of uploaded query image (for history display)

from shared.database import SessionLocal
from shared.models import PersonCandidate, QueryCandidate, QueryHistory, Tracklet, Video
from app.services.translation import detect_vietnamese, translate_to_english, warmup as warmup_translation
from sqlalchemy import func, or_, select, text
from sqlalchemy.orm import joinedload
import re

_CAM_NEIGHBOR_RADIUS = 10


def _extract_cam_num(cam_id: str) -> int | None:
    m = re.search(r'\d+', cam_id or "")
    return int(m.group()) if m else None


def _expand_camera_range(camera_ids: list[str], radius: int = _CAM_NEIGHBOR_RADIUS) -> list[str]:
    """Expand camera_ids to include all cams within ±radius of each cam number.

    cam_20 with radius=10 → cam_10 … cam_30.
    Cams with non-numeric IDs are kept as-is without expansion.
    """
    expanded: set[str] = set()
    for cam_id in camera_ids:
        num = _extract_cam_num(cam_id)
        if num is None:
            expanded.add(cam_id)
            continue
        # preserve prefix ("cam_") and zero-padding width ("01" → width=2)
        digits = re.search(r'\d+', cam_id).group()
        prefix = cam_id[: cam_id.index(digits)]
        width = len(digits)
        for i in range(max(1, num - radius), num + radius + 1):
            expanded.add(f"{prefix}{i:0{width}d}")
    return list(expanded)

_LOG_PATH = "/teamspace/studios/this_studio/TraceX-AI/.cursor/debug-a94b91.log"

def _debug_log(hypothesis_id: str, run_id: str, location: str, message: str, data: dict):
    try:
        with open(_LOG_PATH, "a") as f:
            f.write(json.dumps({
                "sessionId": "a94b91",
                "id": f"log_{int(datetime.now().timestamp() * 1000)}",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "location": location,
                "message": message,
                "data": data,
                "runId": run_id,
                "hypothesisId": hypothesis_id,
            }) + "\n")
    except Exception:
        pass

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])

_translation_warmed_up = False


def _ensure_translation_warmed_up():
    global _translation_warmed_up
    if not _translation_warmed_up:
        try:
            warmup_translation()
        except Exception:
            pass
        _translation_warmed_up = True


def _translate_query(query: str) -> str:
    """Translate Vietnamese query to English for better matching."""
    _ensure_translation_warmed_up()
    if detect_vietnamese(query):
        english = translate_to_english(query)
        if english != query:
            logger.info("Translated query: %r → %r", query, english)
            return english
    return query


def _post_to_trace_service(path: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    """Forward request to trace-service for GPU re-ranking."""
    import os
    trace_url = os.getenv(
        "TRACE_SERVICE_URL",
        "http://trace-service:8004"
    ).rstrip("/")
    url = f"{trace_url}{path}"
    max_attempts = 3
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt < max_attempts:
                import time
                time.sleep(0.5 * attempt)
        except Exception as exc:
            last_exc = exc
            break
        import time
        time.sleep(0.5)

    raise HTTPException(
        status_code=503,
        detail=f"Trace service unavailable: {last_exc}",
    )


def _parse_dt(value: str | None):
    """Parse ISO datetime string to timezone-aware datetime, or None."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            from datetime import timezone as _tz
            dt = dt.replace(tzinfo=_tz.utc)
        return dt
    except ValueError:
        return None


def _local_prefilter(
    session: Session,
    query_text: str,
    camera_ids: list[str] | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
    limit: int = 200,
) -> list[Tracklet]:
    """Pre-filter tracklets using camera, time range, and text matching."""
    cleaned_query = (query_text or "").strip().lower()

    # Join Video so we can filter by absolute recording timestamp.
    # Absolute tracklet time = video.created_at + start_time (seconds).
    statement = (
        select(Tracklet)
        .join(Video, Tracklet.video_id == Video.video_id)
        .options(joinedload(Tracklet.embedding))
        .order_by(Tracklet.created_at.desc(), Tracklet.id.desc())
    )

    if camera_ids:
        cam_lower = [c.lower().strip() for c in camera_ids if c.strip()]
        if cam_lower:
            statement = statement.where(func.lower(Tracklet.camera_id).in_(cam_lower))

    tf = _parse_dt(time_from)
    tt = _parse_dt(time_to)
    if tf:
        # Tracklet still active at time_from:
        # video.recorded_at + end_time seconds >= time_from
        statement = statement.where(
            text("videos.recorded_at + (tracklets.end_time * interval '1 second') >= :tf")
            .bindparams(tf=tf)
        )
    if tt:
        # Tracklet started before time_to:
        # video.recorded_at + start_time seconds <= time_to
        statement = statement.where(
            text("videos.recorded_at + (tracklets.start_time * interval '1 second') <= :tt")
            .bindparams(tt=tt)
        )

    rows = session.scalars(statement).all()
    if not rows:
        return []

    if not cleaned_query:
        return list(rows[:limit])

    # Token-based scoring against appearance_summary + color fields
    query_tokens = set(cleaned_query.split())
    scored: list[tuple[float, int, Tracklet]] = []

    for row in rows:
        search_text = " ".join([
            row.appearance_summary or "",
            row.gender or "",
            row.top_color or "",
            row.bottom_color or "",
            row.shoes_color or "",
            row.age_range or "",
        ]).lower()

        score = 0.0
        if cleaned_query in search_text:
            score += 0.5
        row_tokens = set(search_text.split())
        overlap = len(query_tokens & row_tokens)
        if overlap:
            score += min(overlap * 0.1, 0.5)

        scored.append((score, row.id, row))

    scored.sort(key=lambda x: (-x[0], -x[1]))
    positive = [row for s, _, row in scored if s > 0]
    return (positive if positive else list(rows))[:limit]


def _candidate_to_payload(row: Tracklet) -> dict:
    """Convert Tracklet DB row to candidate payload, including embedding vector."""
    appearance = " ".join(filter(None, [
        row.gender, row.top_color, "shirt" if row.top_color else "",
        row.bottom_color, "pants" if row.bottom_color else "",
    ])).strip()
    dinov2_emb: list[float] = []
    siglip_emb: list[float] = []
    if row.embedding:
        dinov2_emb = row.embedding.embedding_vector or []
        siglip_emb = list(row.embedding.siglip_embedding) if row.embedding.siglip_embedding is not None else []
    return {
        "candidate_id": row.tracklet_id,
        "camera_id": row.camera_id,
        "video_id": row.video_id,
        "track_id": row.track_id,
        "appearance_summary": row.appearance_summary or appearance,
        "gender": row.gender,
        "top_color": row.top_color,
        "bottom_color": row.bottom_color,
        "score": float(row.quality_score),
        "embedding_vector": siglip_emb or dinov2_emb,  # prefer SigLIP2 (same space as text queries)
    }


@router.post("")
def search_candidates(body: SearchRequest) -> dict[str, Any]:
    """Search candidates with GPU re-ranking. Accepts JSON body."""
    query = body.query or body.text or ""
    top_k = body.top_k
    offset = body.offset
    camera_ids = body.camera_ids
    time_from = body.time_from
    time_to = body.time_to

    db = SessionLocal()
    try:
        # Translate if Vietnamese
        search_query = _translate_query(query) if query else ""

        # ── Luồng 20.5: Create QueryHistory record ──────────────────────
        # Default user_id=1 (anonymous) — auth from metadata-service injects real user
        qid = str(uuid.uuid4())
        qh = QueryHistory(
            query_id=qid,
            user_id=1,
            query_text=query or "",
            status="searching",
            query_image_url=body.query_image_url or None,
        )
        db.add(qh)
        db.flush()  # FK constraint: query_candidates.query_id → query_history.query_id

        # Local pre-filter
        shortlist = _local_prefilter(
            db, search_query, camera_ids, time_from, time_to, limit=200
        )

        if not shortlist:
            qh.status = "candidates_found"
            qh.result_count = 0
            db.commit()
            return {"results": [], "query_id": qid}

        # Build shortlist payload for trace-service
        candidates_payload = [_candidate_to_payload(row) for row in shortlist]

        # GPU re-ranking via trace-service
        trace_payload = {
            "query_text": search_query or query,
            "candidates": candidates_payload,
            "limit": max(top_k, 5),
        }
        if camera_ids:
            trace_payload["camera_ids"] = camera_ids
        if time_from:
            trace_payload["time_from"] = time_from
        if time_to:
            trace_payload["time_to"] = time_to

        try:
            ranked = _post_to_trace_service("/api/v1/candidates/search", trace_payload)
        except HTTPException:
            # Trace-service unavailable: return local pre-filter results
            logger.warning("Trace service unavailable, returning local pre-filter results")
            ranked_items = candidates_payload[:top_k]
        else:
            ranked_items = ranked.get("items") or []

        # Save QueryCandidate records — ON CONFLICT DO NOTHING (same tracklet can appear in multiple queries)
        all_items = ranked_items if ranked_items else candidates_payload[:top_k]
        for rank_idx, item in enumerate(all_items):
            stmt = pg_insert(QueryCandidate).values(
                query_id=qid,
                candidate_id=item.get("candidate_id") or str(uuid.uuid4()),
                fusion_score=float(item.get("_fusion_score") or item.get("score") or 0.0),
                vector_score=float(item.get("_fusion_score") or 0.0) if item.get("_fusion_score") else None,
                rank_position=rank_idx + 1,
                primary_camera_id=item.get("camera_id") or "",
                appearance_summary=item.get("appearance_summary") or "",
                gender=item.get("gender") or "unknown",
                top_color=item.get("top_color") or "unknown",
                bottom_color=item.get("bottom_color") or "unknown",
            ).on_conflict_do_nothing(index_elements=["candidate_id"])
            db.execute(stmt)

        # Apply offset
        ranked_items = ranked_items[offset:offset + top_k]

        # Update query history
        qh.status = "candidates_found"
        qh.result_count = len(ranked_items)
        db.commit()

        # Format for frontend
        results = []
        for item in ranked_items:
            score = item.pop("_fusion_score", None)
            description = item.get("appearance_summary") or item.get("attribute_summary") or ""
            if score is not None:
                description = f"[{score:.3f}] {description}" if description else f"Score: {score:.3f}"

            thumbnail_url = "/candidates/{}/preview".format(item["candidate_id"])
            results.append({
                "id": item["candidate_id"],
                "thumbnail_url": thumbnail_url,
                "description": description.strip(),
                "_raw": item,
                "query_id": qid,
            })

        _debug_log("C", "pre-fix",
            "candidates.py:search",
            "search completed",
            {"query_id": qid, "query": query, "result_count": len(results),
             "has_query_id_in_results": any(r.get("query_id") for r in results)})
        return {"results": results, "query_id": qid}

    finally:
        db.close()
