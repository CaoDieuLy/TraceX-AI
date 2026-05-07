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
from sqlalchemy.orm import Session

from shared.database import SessionLocal
from shared.models import PersonCandidate, QueryCandidate, QueryHistory
from app.services.translation import detect_vietnamese, translate_to_english, warmup as warmup_translation
from sqlalchemy import func, or_, select

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


def _local_prefilter(
    session: Session,
    query_text: str,
    camera_ids: list[str] | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
    limit: int = 200,
) -> list[dict]:
    """Pre-filter candidates using text matching before GPU re-ranking."""
    cleaned_query = (query_text or "").strip()

    statement = select(PersonCandidate).order_by(
        PersonCandidate.updated_at.desc(), PersonCandidate.id.desc()
    )

    if camera_ids:
        cam_lower = [c.lower().strip() for c in camera_ids if c.strip()]
        if cam_lower:
            statement = statement.where(func.lower(PersonCandidate.camera_id).in_(cam_lower))

    rows = session.scalars(statement).all()
    if not rows:
        return []

    # Token-based text scoring
    query_tokens = set((cleaned_query or "").lower().split())
    scored: list[tuple[float, int, PersonCandidate]] = []

    for row in rows:
        score = 0.0
        search_text = str(row.search_text or "").lower()

        if cleaned_query:
            if cleaned_query.lower() in search_text:
                score += 0.5
            row_tokens = set(search_text.split())
            overlap = len(query_tokens & row_tokens)
            if overlap:
                score += min(overlap * 0.05, 0.3)
        else:
            score = 0.1  # No query = return all

        scored.append((score, row.id, row))

    # Sort: positive scores first, then by recency
    scored.sort(key=lambda x: (-x[0], -x[1]))

    positive = [row for score, _, row in scored if score > 0]
    if not positive:
        positive = list(rows)  # fallback: return all

    return positive[:limit]


def _candidate_to_payload(row: PersonCandidate) -> dict:
    """Convert DB row to candidate payload for trace-service."""
    raw = row.raw_metadata or {}
    return {
        "candidate_id": row.candidate_id,
        "camera_id": row.primary_camera_id,
        "video_id": row.video_id,
        "track_id": row.track_id,
        "human_key": row.human_key,
        "frame_idx": row.frame_idx,
        "search_text": row.search_text,
        "metadata_path": row.metadata_path,
        "attribute_summary": raw.get("attribute_summary"),
        "appearance_summary": raw.get("appearance_summary"),
        "semantic_attributes": raw.get("semantic_attributes") or [],
        "embedding_vector": raw.get("embedding_vector") or [],
        "attribute_embedding_vector": raw.get("attribute_embedding_vector") or [],
        "appearance_embedding_vector": raw.get("appearance_embedding_vector") or [],
        "visibility_scores": raw.get("visibility_scores") or {},
        "timeline": raw.get("timeline") or [],
        "matched_segments": raw.get("matched_segments") or [],
        "score": raw.get("score"),
    }


@router.post("")
def search_candidates(
    query: str = "",
    top_k: int = 20,
    offset: int = 0,
    camera_ids: list[str] | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
) -> dict[str, Any]:
    """
    Search candidates with GPU re-ranking.

    Args:
        query: Search text (Vietnamese or English)
        top_k: Number of results
        offset: Pagination offset
        camera_ids: Filter by cameras
        time_from: Time range start
        time_to: Time range end

    Returns:
        {"results": [{"id": ..., "thumbnail_url": ..., "description": ...}]}
    """
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
        )
        db.add(qh)

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

        # ── Luồng 20.5: Create QueryCandidate records for DB ──────────
        all_items = ranked_items if ranked_items else candidates_payload[:top_k]
        for rank_idx, item in enumerate(all_items):
            qc = QueryCandidate(
                query_id=qid,
                candidate_id=item.get("candidate_id") or item.get("human_key") or str(uuid.uuid4()),
                fusion_score=float(item.get("_fusion_score") or item.get("score") or 0.0),
                vector_score=float(item.get("_fusion_score") or 0.0) if item.get("_fusion_score") else None,
                rank_position=rank_idx + 1,
                primary_camera_id=item.get("camera_id") or "",
                appearance_summary=item.get("appearance_summary") or item.get("attribute_summary") or "",
                gender=item.get("gender") or "unknown",
                top_color=item.get("top_color") or "unknown",
                bottom_color=item.get("bottom_color") or "unknown",
            )
            db.add(qc)

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

            thumbnail_url = "/api/v1/candidates/{}/preview".format(item["candidate_id"])
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
