"""Candidates search router for query-service.

Flow:
1. metadata-service (/api/v1/search) forwards request here
2. Local text pre-filter → shortlist candidates (sorted by text relevance)
3. Union-find merge: group tracklets by cosine similarity + metadata + temporal/camera guards
4. Save QueryCandidate + QueryCandidateTracklet rows, format and return results
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

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
from shared.models import PersonCandidate, QueryCandidate, QueryCandidateTracklet, QueryHistory, Tracklet, Video
from app.services.translation import detect_vietnamese, translate_to_english, warmup as warmup_translation
from sqlalchemy import func, or_, select, text
from sqlalchemy.orm import contains_eager, joinedload
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
        .options(
            contains_eager(Tracklet.video),   # reuse joined rows — needed for _tracklet_abs_time
            joinedload(Tracklet.embedding),
        )
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
            getattr(row, "hat_color", "") or "",
            getattr(row, "bag_type", "") or "",
            getattr(row, "is_wearing_mask", "") or "",
            getattr(row, "hair_style", "") or "",
            getattr(row, "hair_color", "") or "",
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



# ── Identity merge: cosine similarity + temporal/camera guards + union-find ───

_MERGE_THRESHOLD = 0.85       # SigLIP2 cosine similarity to consider same identity
_MERGE_MAX_GAP_S = 7200.0    # max 2-hour gap between tracklets of the same person
_CONF_THRESHOLD = 0.70        # fallback: below this = uncertain → don't block merge

# Per-attribute confidence thresholds: both sides must exceed to block merge.
# Free-text fields (upper_clothing_type, *_desc) are NOT in this list — exact
# string match would incorrectly treat "blazer" vs "suit jacket" as a conflict.
_CONF_THRESHOLDS: dict[str, float] = {
    "gender":               0.70,
    "is_wearing_mask":      0.70,
    "bag_presence":         0.75,
    "hat_presence":         0.75,
    "age_range":            0.75,
    "upper_clothing_color": 0.82,
    "lower_clothing_color": 0.82,
    "shoes_color":          0.82,
    "hat_color":            0.82,
    "hair_color":           0.82,
    # backward compat (old SigLIP columns)
    "top_color":            0.82,
    "bottom_color":         0.82,
}

# "none" is a meaningful value (model confirmed absence) for these fields
_NONE_IS_VALID = frozenset({"hat_color", "bag_type", "is_wearing_mask"})
_UNKNOWN_VALUES = frozenset({"", "unknown", "null", "n/a", "not sure"})


def _norm_meta(attr: str, value: object) -> str | None:
    """Normalize a metadata value; return None if value is ambiguous/unknown."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in _UNKNOWN_VALUES:
        return None
    # "none" only counts as a real value for fields that explicitly track absence
    if s == "none" and attr not in _NONE_IS_VALID:
        return None
    return s


def _metadata_matches(t1: Tracklet, t2: Tracklet) -> bool:
    """Return False only when both tracklets have conflicting attribute values with sufficient confidence.

    Missing confidence (None) is treated as 0.0 (uncertain) — does not block merge.
    """
    checks = [
        # binary / presence fields — most reliable conflict signal
        ("gender",               "gender_conf"),
        ("is_wearing_mask",      "mask_conf"),
        ("bag_presence",         "bag_conf"),
        ("hat_presence",         "hat_conf"),
        # color fields — VLM self-reported confidence
        ("upper_clothing_color", "upper_clothing_conf"),
        ("lower_clothing_color", "lower_clothing_conf"),
        ("shoes_color",          "shoes_conf"),
        ("hat_color",            "hat_conf"),
        ("hair_color",           "hair_conf"),
        # backward compat (old SigLIP columns, populated for existing rows)
        ("top_color",            "top_color_conf"),
        ("bottom_color",         "bottom_color_conf"),
        ("age_range",            "age_range_conf"),
        # NOTE: upper_clothing_type, lower_clothing_type, *_desc intentionally
        # excluded — free-text from VLM; "blazer" ≠ "suit jacket" in string
        # comparison but may refer to the same garment.
    ]
    for attr, conf_field in checks:
        v1 = _norm_meta(attr, getattr(t1, attr, None))
        v2 = _norm_meta(attr, getattr(t2, attr, None))
        if v1 is None or v2 is None:
            continue  # one side unknown → not a conflict
        if v1 == v2:
            continue
        # Values differ → check confidence; missing conf → treat as uncertain
        c1 = getattr(t1, conf_field, None) or 0.0
        c2 = getattr(t2, conf_field, None) or 0.0
        threshold = _CONF_THRESHOLDS.get(attr, _CONF_THRESHOLD)
        if c1 >= threshold and c2 >= threshold:
            return False  # both sides confident about conflicting values
    return True


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


def _tracklet_embedding(t: Tracklet) -> list[float]:
    """SigLIP2 embedding preferred; fallback to DINOv2."""
    if not t.embedding:
        return []
    if t.embedding.siglip_embedding is not None:
        try:
            return list(t.embedding.siglip_embedding)
        except Exception:
            pass
    return list(t.embedding.embedding_vector or [])


def _tracklet_abs_window(t: Tracklet) -> tuple[float, float]:
    """Absolute (start_ts, end_ts) in Unix seconds. Uses video.recorded_at loaded via contains_eager."""
    try:
        base = t.video.recorded_at.timestamp() if (t.video and t.video.recorded_at) else 0.0
    except Exception:
        base = 0.0
    return base + float(t.start_time or 0.0), base + float(t.end_time or 0.0)


def _can_merge(t1: Tracklet, t2: Tracklet) -> bool:
    """Return True if t1 and t2 can belong to the same person identity."""
    # Check metadata first — cheapest way to reject obviously different people
    if not _metadata_matches(t1, t2):
        return False
    # Same video: tracker already separated them into distinct track_ids → different people
    if t1.video_id == t2.video_id:
        return False
    s1, e1 = _tracklet_abs_window(t1)
    s2, e2 = _tracklet_abs_window(t2)
    # Time gap between end of one and start of the other
    gap = max(s1 - e2, s2 - e1, 0.0)
    if gap > _MERGE_MAX_GAP_S:
        return False
    if t1.camera_id == t2.camera_id:
        # Same camera with overlapping time → physically impossible to be same person
        if min(e1, e2) > max(s1, s2):
            return False
    return True


def _merge_by_similarity(ranked_tracklets: list[Tracklet]) -> list[list[Tracklet]]:
    """Group tracklets by embedding cosine similarity using union-find.

    Returns groups ordered by best rank (lowest index in ranked_tracklets = highest score).
    The first element of each group is the representative (highest-ranked tracklet).
    Tracklets with no embedding are kept as singleton groups.
    """
    n = len(ranked_tracklets)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    embs = [_tracklet_embedding(t) for t in ranked_tracklets]

    for i in range(n):
        if not embs[i]:
            continue
        for j in range(i + 1, n):
            if find(i) == find(j) or not embs[j]:
                continue
            if _cosine_sim(embs[i], embs[j]) < _MERGE_THRESHOLD:
                continue
            if not _can_merge(ranked_tracklets[i], ranked_tracklets[j]):
                continue
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pi] = pj

    root_to_idxs: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        root_to_idxs[find(i)].append(i)

    # Each group is sorted by rank (ascending index = best score first)
    return [
        [ranked_tracklets[i] for i in sorted(idxs)]
        for idxs in sorted(root_to_idxs.values(), key=min)
    ]


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

        # shortlist is already sorted by text relevance from _local_prefilter
        # Merge tracklets that belong to the same person identity
        groups = _merge_by_similarity(shortlist)

        # Build intermediate list (avoids candidate_id scope leak between save + format loops)
        merged: list[dict] = []
        rep_embs: dict[str, list[float]] = {}

        for rank_idx, group in enumerate(groups):
            rep = group[0]
            candidate_id = rep.tracklet_id if len(group) == 1 else str(uuid.uuid4())
            fusion_score = float(rep.quality_score or 0.0)
            rep_emb = _tracklet_embedding(rep)
            rep_embs[candidate_id] = rep_emb

            db.execute(pg_insert(QueryCandidate).values(
                query_id=qid,
                candidate_id=candidate_id,
                fusion_score=fusion_score,
                rank_position=rank_idx + 1,
                primary_camera_id=rep.camera_id or "",
                appearance_summary=rep.appearance_summary or "",
                gender=rep.gender or "unknown",
                top_color=rep.top_color or "unknown",
                bottom_color=rep.bottom_color or "unknown",
            ).on_conflict_do_nothing(index_elements=["candidate_id"]))  # uq_query_candidates_candidate_id

            for member in group:
                if member.tracklet_id == rep.tracklet_id:
                    match_score = 1.0
                else:
                    member_emb = _tracklet_embedding(member)
                    match_score = (
                        _cosine_sim(rep_emb, member_emb)
                        if rep_emb and member_emb
                        else float(member.quality_score or 0.0)
                    )
                db.execute(pg_insert(QueryCandidateTracklet).values(
                    candidate_id=candidate_id,
                    tracklet_id=member.tracklet_id,
                    match_score=match_score,
                    match_type="vector",
                ).on_conflict_do_nothing(index_elements=["candidate_id", "tracklet_id"]))  # uq_query_candidate_tracklets_pair

            merged.append({"candidate_id": candidate_id, "group": group, "rep": rep, "fusion_score": fusion_score})

        paged = merged[offset:offset + top_k]
        qh.status = "candidates_found"
        qh.result_count = len(paged)
        db.commit()

        results = []
        for mc in paged:
            rep = mc["rep"]
            candidate_id = mc["candidate_id"]
            tracklet_count = len(mc["group"])
            description = rep.appearance_summary or ""
            if tracklet_count > 1:
                description = f"[{tracklet_count} tracklets] {description}".strip()
            results.append({
                "id": candidate_id,
                "thumbnail_url": f"/candidates/{candidate_id}/preview",
                "description": description,
                "_raw": {
                    "candidate_id": candidate_id,
                    "tracklet_count": tracklet_count,
                    "camera_id": rep.camera_id,
                    "appearance_summary": rep.appearance_summary,
                },
                "query_id": qid,
            })

        _debug_log("C", "post-merge",
            "candidates.py:search",
            "search completed with identity merge",
            {"query_id": qid, "query": query, "raw_tracklets": len(shortlist),
             "merged_candidates": len(merged), "result_count": len(results)})
        return {"results": results, "query_id": qid}

    finally:
        db.close()
