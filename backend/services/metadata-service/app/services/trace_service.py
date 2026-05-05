from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.models import PersonCandidate, QueueVideoAsset
from .candidate_service import candidate_to_ranking_payload, _queue_video_map

logger = logging.getLogger(__name__)

_TOPOLOGY_PATH = Path(__file__).parent.parent.parent.parent / "config" / "camera_topology.json"


@dataclass(frozen=True)
class TopologyEdge:
    from_camera: str
    to_camera: str
    min_seconds: float
    max_seconds: float


class CameraTopologyGraph:
    def __init__(self, topology_path: Path = _TOPOLOGY_PATH) -> None:
        raw: dict = {}
        if topology_path.exists():
            raw = json.loads(topology_path.read_text(encoding="utf-8"))

        self._camera_meta: dict[str, dict] = raw.get("cameras", {})
        self._defaults: dict = raw.get("defaults", {})
        self._adjacency: dict[str, list[TopologyEdge]] = defaultdict(list)
        use_bidirectional = bool(self._defaults.get("use_bidirectional", True))

        for edge_data in raw.get("edges", []):
            a = edge_data.get("from") or edge_data.get("from_camera_id", "")
            b = edge_data.get("to") or edge_data.get("to_camera_id", "")
            mn = float(edge_data.get("min_seconds", edge_data.get("min_sec", 5)))
            mx = float(edge_data.get("max_seconds", edge_data.get("max_sec", 300)))
            self._adjacency[a].append(TopologyEdge(a, b, mn, mx))
            if use_bidirectional:
                self._adjacency[b].append(TopologyEdge(b, a, mn, mx))

    def camera_info(self, camera_id: str) -> dict:
        return self._camera_meta.get(camera_id, {})

    def neighbors(self, camera_id: str) -> list[TopologyEdge]:
        return list(self._adjacency.get(camera_id, []))

    def reachable_cameras(
        self,
        from_camera: str,
        after_seconds: float,
        max_hop_seconds: float | None = None,
    ) -> list[tuple[str, float, float]]:
        limit = max_hop_seconds or float(self._defaults.get("max_hop_seconds", 300))
        result: list[tuple[str, float, float]] = []
        for edge in self.neighbors(from_camera):
            mn = after_seconds + edge.min_seconds
            mx = after_seconds + edge.max_seconds
            if mx <= limit + after_seconds:
                result.append((edge.to_camera, mn, mx))
        return result

    def travel_plausibility(
        self,
        from_camera: str,
        to_camera: str,
        elapsed_seconds: float,
    ) -> float:
        for edge in self.neighbors(from_camera):
            if edge.to_camera == to_camera:
                if elapsed_seconds < edge.min_seconds:
                    return 0.0
                if elapsed_seconds <= edge.max_seconds:
                    return 1.0
                decay = math.exp(-(elapsed_seconds - edge.max_seconds) / 120.0)
                return max(0.1, decay)
        unknown_max = float(self._defaults.get("unknown_camera_max_seconds", 120))
        if elapsed_seconds > unknown_max:
            return 0.0
        return 0.3


_TOPOLOGY = CameraTopologyGraph()


@dataclass
class SeedInfo:
    candidate_id: str
    camera_id: str
    embedding_vector: list[float]
    appearance_embedding: list[float]
    attribute_embedding: list[float]
    recorded_start: datetime | None
    tracklet_start_second: float
    tracklet_end_second: float
    window_from: datetime
    window_to: datetime


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        s = str(value).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None


def build_seed(
    session: Session,
    candidate_id: str,
    window_hours: float = 12.0,
) -> SeedInfo:
    row: PersonCandidate | None = session.scalar(
        select(PersonCandidate).where(PersonCandidate.candidate_id == candidate_id)
    )
    if row is None:
        raise ValueError(f"Candidate {candidate_id!r} not found")

    raw = row.raw_metadata or {}
    embed = raw.get("embedding_vector") or []
    app_embed = raw.get("appearance_embedding_vector") or embed
    attr_embed = raw.get("attribute_embedding_vector") or []

    qva: QueueVideoAsset | None = None
    if row.video_id:
        qva = session.scalar(
            select(QueueVideoAsset).where(QueueVideoAsset.video_id == row.video_id)
        )
    recorded_start: datetime | None = None
    if qva:
        vm = qva.raw_video_metadata or {}
        recorded_start = _parse_dt(vm.get("recorded_start"))

    timeline = raw.get("timeline") or []
    t_start = 0.0
    t_end = 0.0
    if isinstance(timeline, list) and timeline:
        starts = [float(seg.get("start_second", 0)) for seg in timeline if isinstance(seg, dict)]
        ends = [float(seg.get("end_second", 0)) for seg in timeline if isinstance(seg, dict)]
        t_start = min(starts) if starts else 0.0
        t_end = max(ends) if ends else 0.0

    if recorded_start:
        seed_abs = recorded_start + timedelta(seconds=(t_start + t_end) / 2)
    else:
        seed_abs = datetime.now(timezone.utc)

    window_delta = timedelta(hours=window_hours)
    return SeedInfo(
        candidate_id=candidate_id,
        camera_id=str(row.camera_id or ""),
        embedding_vector=embed,
        appearance_embedding=app_embed,
        attribute_embedding=attr_embed,
        recorded_start=recorded_start,
        tracklet_start_second=t_start,
        tracklet_end_second=t_end,
        window_from=seed_abs - window_delta,
        window_to=seed_abs + window_delta,
    )


def pruned_camera_set(
    seed: SeedInfo,
    all_cameras: list[str],
    max_hops: int = 3,
) -> set[str]:
    if not seed.camera_id:
        return set(all_cameras)

    reachable: set[str] = {seed.camera_id}
    frontier: set[str] = {seed.camera_id}

    for _ in range(max_hops):
        next_frontier: set[str] = set()
        for cam in frontier:
            for edge in _TOPOLOGY.neighbors(cam):
                if edge.to_camera not in reachable:
                    reachable.add(edge.to_camera)
                    next_frontier.add(edge.to_camera)
        if not next_frontier:
            break
        frontier = next_frontier

    unknown = [c for c in all_cameras if c not in _TOPOLOGY._adjacency]
    reachable.update(unknown)

    logger.info("Topology pruning: %d/%d cameras in trace scope", len(reachable), len(all_cameras))
    return reachable


@dataclass
class TraceCandidate:
    candidate_id: str
    camera_id: str
    video_id: str
    recorded_start: datetime | None
    tracklet_start_second: float
    tracklet_end_second: float
    abs_start: datetime | None
    abs_end: datetime | None
    embedding_vector: list[float]
    appearance_embedding: list[float]
    appearance_sim: float = 0.0
    payload: dict = field(default_factory=dict)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(va / na, vb / nb))


def spatiotemporal_retrieval(
    session: Session,
    seed: SeedInfo,
    camera_scope: set[str],
    min_similarity: float = 0.40,
) -> list[TraceCandidate]:
    rows: list[PersonCandidate] = list(session.scalars(
        select(PersonCandidate).where(
            PersonCandidate.camera_id.in_(list(camera_scope)),
            PersonCandidate.candidate_id != seed.candidate_id,
        )
    ).all())

    if not rows:
        return []

    video_ids = list({str(r.video_id or "") for r in rows})
    qva_map: dict[str, QueueVideoAsset] = {
        str(q.video_id): q
        for q in session.scalars(
            select(QueueVideoAsset).where(QueueVideoAsset.video_id.in_(video_ids))
        ).all()
    }

    results: list[TraceCandidate] = []
    for row in rows:
        raw = row.raw_metadata or {}
        embed = raw.get("embedding_vector") or []
        app_embed = raw.get("appearance_embedding_vector") or embed

        sim = _cosine_sim(seed.appearance_embedding or seed.embedding_vector, app_embed or embed)
        if sim < min_similarity:
            continue

        qva = qva_map.get(str(row.video_id or ""))
        vm = (qva.raw_video_metadata if qva else {}) or {}
        rec_start = _parse_dt(vm.get("recorded_start"))

        timeline = raw.get("timeline") or []
        t_start, t_end = 0.0, 0.0
        if isinstance(timeline, list) and timeline:
            starts = [float(s.get("start_second", 0)) for s in timeline if isinstance(s, dict)]
            ends = [float(s.get("end_second", 0)) for s in timeline if isinstance(s, dict)]
            t_start = min(starts) if starts else 0.0
            t_end = max(ends) if ends else 0.0

        abs_start = (rec_start + timedelta(seconds=t_start)) if rec_start else None
        abs_end = (rec_start + timedelta(seconds=t_end)) if rec_start else None

        if abs_start and (abs_start < seed.window_from or abs_start > seed.window_to):
            continue

        payload = candidate_to_ranking_payload(row, qva)
        cam_info = _TOPOLOGY.camera_info(str(row.camera_id or ""))
        payload["camera_name"] = cam_info.get("name", "")
        payload["camera_area"] = cam_info.get("area_name", "")
        payload["camera_floor"] = cam_info.get("floor")
        results.append(TraceCandidate(
            candidate_id=str(row.candidate_id),
            camera_id=str(row.camera_id or ""),
            video_id=str(row.video_id or ""),
            recorded_start=rec_start,
            tracklet_start_second=t_start,
            tracklet_end_second=t_end,
            abs_start=abs_start,
            abs_end=abs_end,
            embedding_vector=embed,
            appearance_embedding=app_embed,
            appearance_sim=round(sim, 6),
            payload=payload,
        ))

    results.sort(key=lambda c: (c.abs_start or datetime.min.replace(tzinfo=timezone.utc)))
    logger.info("Spatiotemporal retrieval: %d candidates after sim filter", len(results))
    return results


@dataclass
class TrajectorySegment:
    candidate_id: str
    camera_id: str
    abs_start: datetime | None
    abs_end: datetime | None
    appearance_sim: float
    topology_score: float
    velocity_score: float
    segment_score: float
    payload: dict


@dataclass
class TrajectoryPath:
    segments: list[TrajectorySegment]
    total_score: float
    is_seed: bool = False


def _segment_score(
    appearance_sim: float,
    topology_score: float,
    velocity_score: float,
) -> float:
    return round(
        appearance_sim * 0.55 +
        topology_score * 0.25 +
        velocity_score * 0.20,
        6,
    )


def build_trajectory(
    seed: SeedInfo,
    candidates: list[TraceCandidate],
    max_gap_seconds: float = 3600.0,
) -> list[TrajectorySegment]:
    if seed.recorded_start:
        seed_abs = seed.recorded_start + timedelta(
            seconds=(seed.tracklet_start_second + seed.tracklet_end_second) / 2
        )
    else:
        seed_abs = None

    before = sorted(
        [c for c in candidates if c.abs_start and seed_abs and c.abs_start < seed_abs],
        key=lambda c: c.abs_start,
        reverse=True,
    )
    after = sorted(
        [c for c in candidates if c.abs_start and seed_abs and c.abs_start >= seed_abs],
        key=lambda c: c.abs_start,
    )

    def chain(
        ordered: list[TraceCandidate],
        prev_camera: str,
        prev_time: datetime | None,
        forward: bool,
    ) -> list[TrajectorySegment]:
        segments: list[TrajectorySegment] = []
        used: set[str] = set()
        current_camera = prev_camera
        current_time = prev_time

        for cand in ordered:
            if cand.candidate_id in used:
                continue
            if not cand.abs_start:
                continue

            elapsed = abs((cand.abs_start - current_time).total_seconds()) if current_time else 0.0
            if elapsed > max_gap_seconds:
                break

            topo = _TOPOLOGY.travel_plausibility(current_camera, cand.camera_id, elapsed)
            vel_score = 1.0 if elapsed >= 2 else 0.1
            seg_score = _segment_score(cand.appearance_sim, topo, vel_score)

            if seg_score < 0.25:
                continue

            segments.append(TrajectorySegment(
                candidate_id=cand.candidate_id,
                camera_id=cand.camera_id,
                abs_start=cand.abs_start,
                abs_end=cand.abs_end,
                appearance_sim=cand.appearance_sim,
                topology_score=topo,
                velocity_score=vel_score,
                segment_score=seg_score,
                payload=cand.payload,
            ))
            used.add(cand.candidate_id)
            current_camera = cand.camera_id
            current_time = cand.abs_start

        return segments

    if seed_abs:
        before_segs = chain(before, seed.camera_id, seed_abs, forward=False)
        after_segs = chain(after, seed.camera_id, seed_abs, forward=True)
        before_segs.reverse()
    else:
        before_segs = []
        after_segs = []
        after_segs = chain(candidates, seed.camera_id, None, forward=True)

    return before_segs + after_segs


def rerank_trajectory(segments: list[TrajectorySegment]) -> list[TrajectorySegment]:
    if not segments:
        return []

    cleaned: list[TrajectorySegment] = []
    prev_end: datetime | None = None

    for seg in segments:
        if prev_end and seg.abs_start and seg.abs_start < prev_end:
            continue
        if seg.topology_score < 0.05:
            continue
        cleaned.append(seg)
        if seg.abs_end:
            prev_end = seg.abs_end
        elif seg.abs_start:
            prev_end = seg.abs_start

    return cleaned


@dataclass
class EvidenceClip:
    camera_id: str
    candidate_id: str
    start_time: str
    end_time: str
    duration_seconds: float
    appearance_sim: float
    segment_score: float
    preview_url: str
    video_url: str
    payload: dict


@dataclass
class TraceResult:
    seed_candidate_id: str
    total_segments: int
    cameras_visited: list[str]
    trajectory: list[EvidenceClip]
    overall_score: float
    window_from: str
    window_to: str


def build_evidence_clips(
    seed: SeedInfo,
    segments: list[TrajectorySegment],
) -> TraceResult:
    clips: list[EvidenceClip] = []

    for seg in segments:
        start_str = seg.abs_start.isoformat().replace("+00:00", "Z") if seg.abs_start else ""
        end_str = seg.abs_end.isoformat().replace("+00:00", "Z") if seg.abs_end else ""
        duration = (
            (seg.abs_end - seg.abs_start).total_seconds()
            if seg.abs_start and seg.abs_end
            else 0.0
        )
        video_url = seg.payload.get("available_link_video") or f"/api/v1/queue/videos/{seg.payload.get('video_id', '')}/file"
        preview_url = seg.payload.get("preview_image_url") or f"/api/v1/candidates/{seg.candidate_id}/preview"

        clips.append(EvidenceClip(
            camera_id=seg.camera_id,
            candidate_id=seg.candidate_id,
            start_time=start_str,
            end_time=end_str,
            duration_seconds=round(duration, 1),
            appearance_sim=seg.appearance_sim,
            segment_score=seg.segment_score,
            preview_url=preview_url,
            video_url=video_url,
            payload={k: v for k, v in seg.payload.items() if k not in ("raw_metadata", "embedding_vector", "appearance_embedding_vector")},
        ))

    overall = round(
        sum(c.segment_score for c in clips) / max(len(clips), 1), 6
    ) if clips else 0.0

    return TraceResult(
        seed_candidate_id=seed.candidate_id,
        total_segments=len(clips),
        cameras_visited=list(dict.fromkeys(c.camera_id for c in clips)),
        trajectory=[c for c in clips],
        overall_score=overall,
        window_from=seed.window_from.isoformat().replace("+00:00", "Z"),
        window_to=seed.window_to.isoformat().replace("+00:00", "Z"),
    )


@dataclass
class FeedbackPayload:
    candidate_id: str
    confirmed_segment_ids: list[str]
    rejected_segment_ids: list[str]


def apply_feedback(
    session: Session,
    seed: SeedInfo,
    feedback: FeedbackPayload,
    window_hours: float = 12.0,
) -> TraceResult:
    if not feedback.confirmed_segment_ids:
        raise ValueError("Need at least one confirmed segment for refinement")

    confirmed_rows = list(session.scalars(
        select(PersonCandidate).where(
            PersonCandidate.candidate_id.in_(feedback.confirmed_segment_ids)
        )
    ).all())

    all_embeds: list[list[float]] = []
    for row in confirmed_rows:
        raw = row.raw_metadata or {}
        e = raw.get("appearance_embedding_vector") or raw.get("embedding_vector") or []
        if e:
            all_embeds.append(e)

    if all_embeds and seed.appearance_embedding:
        all_embeds.append(seed.appearance_embedding)
        mat = np.array(all_embeds, dtype=np.float32)
        mean_embed = mat.mean(axis=0)
        norm = float(np.linalg.norm(mean_embed))
        refined_embed = (mean_embed / norm).tolist() if norm > 1e-8 else seed.appearance_embedding
    else:
        refined_embed = seed.appearance_embedding

    refined_seed = SeedInfo(
        candidate_id=seed.candidate_id,
        camera_id=seed.camera_id,
        embedding_vector=seed.embedding_vector,
        appearance_embedding=refined_embed,
        attribute_embedding=seed.attribute_embedding,
        recorded_start=seed.recorded_start,
        tracklet_start_second=seed.tracklet_start_second,
        tracklet_end_second=seed.tracklet_end_second,
        window_from=seed.window_from - timedelta(hours=window_hours * 0.2),
        window_to=seed.window_to + timedelta(hours=window_hours * 0.2),
    )

    return run_trace(session, refined_seed)


def run_trace(session: Session, seed: SeedInfo) -> TraceResult:
    all_cameras = [
        str(r) for r in session.scalars(
            select(PersonCandidate.camera_id).distinct()
        ).all() if r
    ]
    camera_scope = pruned_camera_set(seed, all_cameras)
    candidates = spatiotemporal_retrieval(session, seed, camera_scope)
    raw_segments = build_trajectory(seed, candidates)
    segments = rerank_trajectory(raw_segments)
    return build_evidence_clips(seed, segments)


def trace_from_candidate_id(
    session: Session,
    candidate_id: str,
    window_hours: float = 12.0,
) -> TraceResult:
    seed = build_seed(session, candidate_id, window_hours=window_hours)
    logger.info(
        "Trace started candidate=%s camera=%s window=[%s, %s]",
        candidate_id, seed.camera_id,
        seed.window_from.isoformat(), seed.window_to.isoformat(),
    )
    result = run_trace(session, seed)
    logger.info(
        "Trace complete candidate=%s segments=%d cameras=%s score=%.3f",
        candidate_id, result.total_segments,
        result.cameras_visited, result.overall_score,
    )
    return result
