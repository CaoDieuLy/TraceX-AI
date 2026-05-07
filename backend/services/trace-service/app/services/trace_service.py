"""Trace service - business logic for building traces."""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func
from sqlalchemy.orm import Session

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

from ..config import settings
from ..core.models import (
    Camera,
    CameraEdge,
    EvidenceTracklet,
    EvidenceVideo,
    QueryCandidate,
    QueryCandidateTracklet,
    Tracklet,
)


class TraceService:
    """Service for building traces from selected candidates."""

    def __init__(self, session: Session):
        self.session = session

    def deselect_other_candidates(self, query_id: UUID) -> None:
        """Deselect all other candidates for a query."""
        self.session.query(QueryCandidate).filter(
            QueryCandidate.query_id == query_id,
            QueryCandidate.is_selected == True,  # noqa: E712
        ).update({"is_selected": False, "selected_at": None})

    def get_candidate_tracklets(
        self,
        candidate_id: UUID,
        time_window_start: datetime,
        time_window_end: datetime,
    ) -> list[Tracklet]:
        """Get all tracklets for a candidate within time window.

        Args:
            candidate_id: The candidate UUID
            time_window_start: Start of time window
            time_window_end: End of time window

        Returns:
            List of tracklets sorted by video timestamp
        """
        # Get candidate to check for representative tracklet
        candidate = self.session.get(QueryCandidate, candidate_id)
        if not candidate:
            return []

        # Query tracklets through join table
        query = (
            self.session.query(Tracklet)
            .join(
                QueryCandidateTracklet,
                QueryCandidateTracklet.tracklet_id == Tracklet.tracklet_id,
            )
            .filter(QueryCandidateTracklet.candidate_id == candidate_id)
            .join(Tracklet.video)
            .filter(
                and_(
                    Tracklet.start_time >= 0,
                    Tracklet.end_time >= Tracklet.start_time,
                )
            )
            .order_by(Tracklet.start_time.asc())
        )

        tracklets = query.all()

        _debug_log("B", "pre-fix",
            "trace_service.py:95",
            "get_candidate_tracklets raw query result",
            {"candidate_id": str(candidate_id), "raw_count": len(tracklets),
             "first_tid": str(tracklets[0].tracklet_id) if tracklets else None})
        # Filter by time window (based on video created_at + start_time)
        filtered_tracklets = []
        for t in tracklets:
            video = t.video
            if video:
                tracklet_start = video.created_at.timestamp() + (t.start_time or 0)
                tracklet_end = video.created_at.timestamp() + (t.end_time or 0)

                # Check if tracklet overlaps with time window
                if tracklet_start <= time_window_end.timestamp() and tracklet_end >= time_window_start.timestamp():
                    filtered_tracklets.append(t)

        return filtered_tracklets

    def build_trace_segments(
        self,
        tracklets: list[Tracklet],
    ) -> list[dict[str, Any]]:
        """Build trace segments from tracklets.

        Each tracklet becomes a segment. Segments are ordered by time.
        Camera path is determined by spatiotemporal relationships.

        Args:
            tracklets: List of tracklets for the trace

        Returns:
            List of segment dictionaries
        """
        if not tracklets:
            return []

        segments = []
        for idx, tracklet in enumerate(tracklets):
            video = tracklet.video if tracklet.video_id else None

            duration = None
            if tracklet.start_time is not None and tracklet.end_time is not None:
                duration = tracklet.end_time - tracklet.start_time

            time_start = None
            time_end = None
            if video:
                time_start = video.created_at
                time_end = video.created_at

            segment = {
                "segment_order": idx + 1,
                "tracklet_id": tracklet.tracklet_id,
                "camera_id": tracklet.camera_id,
                "time_start": time_start,
                "time_end": time_end,
                "duration_seconds": duration,
                "thumbnail_url": tracklet.crop_url,
                "video_clip_url": self._get_video_clip_url(tracklet),
                "confidence": tracklet.quality_score,
            }
            segments.append(segment)

        # Sort by time
        segments.sort(key=lambda s: (s["time_start"] or datetime.min, s["segment_order"]))

        # Re-index segment order after sorting
        for idx, seg in enumerate(segments):
            seg["segment_order"] = idx + 1

        return segments

    def calculate_trace_confidence(
        self,
        segments: list[dict[str, Any]],
    ) -> float | None:
        """Calculate overall trace confidence.

        Based on:
        - Individual segment confidences
        - Camera path continuity
        - Number of segments

        Args:
            segments: List of trace segments

        Returns:
            Confidence score between 0 and 1
        """
        if not segments:
            return None

        confidences = [s.get("confidence") for s in segments if s.get("confidence") is not None]

        if not confidences:
            return 0.5  # Default confidence

        # Base confidence is average of segment confidences
        avg_confidence = sum(confidences) / len(confidences)

        # Penalize for gaps in camera path
        camera_ids = [s["camera_id"] for s in segments if s["camera_id"]]
        unique_cameras = len(set(camera_ids))

        # Bonus for multiple cameras (more complete trace)
        camera_bonus = min(0.1, unique_cameras * 0.02)

        # Bonus for more segments (more complete trace)
        segment_bonus = min(0.1, len(segments) * 0.01)

        final_confidence = min(1.0, avg_confidence + camera_bonus + segment_bonus)
        return round(final_confidence, 3)

    def create_evidence_video(
        self,
        query_id: UUID,
        candidate_id: UUID,
        segments: list[dict[str, Any]],
        trace_confidence: float | None,
        time_window_start: datetime,
        time_window_end: datetime,
    ) -> EvidenceVideo:
        """Create evidence video record.

        Args:
            query_id: Query ID
            candidate_id: Selected candidate ID
            segments: List of trace segments
            trace_confidence: Overall trace confidence
            time_window_start: Start of trace window
            time_window_end: End of trace window

        Returns:
            Created EvidenceVideo record
        """
        total_duration = sum(s["duration_seconds"] or 0 for s in segments if s["duration_seconds"])

        evidence = EvidenceVideo(
            query_id=query_id,
            query_candidate_id=candidate_id,
            video_url=self._generate_merged_video_url(query_id, candidate_id) if segments else None,
            total_duration=total_duration,
            segment_count=len(segments),
            time_window_start=time_window_start,
            time_window_end=time_window_end,
            trace_confidence=trace_confidence or 0.0,
        )
        self.session.add(evidence)
        self.session.flush()

        # Create evidence tracklets
        for seg in segments:
            evidence_tracklet = EvidenceTracklet(
                evidence_id=evidence.id,
                tracklet_id=seg["tracklet_id"],
                segment_order=seg["segment_order"],
                camera_id=seg["camera_id"],
                time_range={
                    "start": seg["time_start"].isoformat() if seg["time_start"] else None,
                    "end": seg["time_end"].isoformat() if seg["time_end"] else None,
                },
                thumbnail_url=seg.get("thumbnail_url"),
                confidence=seg.get("confidence"),
            )
            self.session.add(evidence_tracklet)

        _debug_log("A", "pre-fix",
            "trace_service.py:251",
            "create_evidence_video result",
            {"evidence_id": evidence.id, "query_id": str(query_id),
             "candidate_id": str(candidate_id), "segment_count": len(segments),
             "tracklet_count": sum(1 for s in segments if s.get("tracklet_id"))})
        return evidence

    def get_trace_segments(self, evidence_id: UUID) -> list[dict[str, Any]]:
        """Get trace segments for an evidence video.

        Args:
            evidence_id: Evidence video ID

        Returns:
            List of segment dictionaries
        """
        evidence_tracklets = (
            self.session.query(EvidenceTracklet)
            .filter(EvidenceTracklet.evidence_id == evidence_id)
            .order_by(EvidenceTracklet.segment_order.asc())
            .all()
        )

        segments = []
        for et in evidence_tracklets:
            tracklet = et.tracklet if et.tracklet_id else None

            time_start = None
            time_end = None
            if et.time_range:
                if et.time_range.get("start"):
                    time_start = datetime.fromisoformat(et.time_range["start"])
                if et.time_range.get("end"):
                    time_end = datetime.fromisoformat(et.time_range["end"])

            duration = None
            if time_start and time_end:
                duration = (time_end - time_start).total_seconds()

            segments.append({
                "segment_order": et.segment_order,
                "tracklet_id": et.tracklet_id,
                "camera_id": et.camera_id,
                "time_start": time_start,
                "time_end": time_end,
                "duration_seconds": duration,
                "thumbnail_url": et.thumbnail_url,
                "video_clip_url": self._get_video_clip_url(tracklet) if tracklet else None,
                "confidence": et.confidence,
            })

        return segments

    def delete_old_evidence(self, candidate_id: UUID) -> int:
        """Delete old evidence videos for a candidate.

        This implements cache overwrite behavior - when a new candidate is selected,
        old evidence is cleared.

        Args:
            candidate_id: Candidate ID

        Returns:
            Number of deleted records
        """
        count = (
            self.session.query(EvidenceVideo)
            .filter(EvidenceVideo.selected_candidate_id == candidate_id)
            .delete(synchronize_session=False)
        )
        return count

    def _get_video_clip_url(self, tracklet: Tracklet | None) -> str | None:
        """Get video clip URL for a tracklet.

        Args:
            tracklet: Tracklet instance

        Returns:
            Video clip URL or None
        """
        if not tracklet or not tracklet.video:
            return None

        video = tracklet.video
        if not video.storage_path:
            return None

        # Construct clip URL based on tracklet timing
        base_url = settings.storage_base_url.rstrip("/")
        clip_url = f"{base_url}/videos/{video.id}/clips/{tracklet.id}.mp4"
        return clip_url

    def _generate_merged_video_url(self, query_id: UUID, candidate_id: UUID) -> str:
        """Generate URL for merged trace video.

        Args:
            query_id: Query ID
            candidate_id: Candidate ID

        Returns:
            Merged video URL
        """
        base_url = settings.storage_base_url.rstrip("/")
        return f"{base_url}/traces/{query_id}/{candidate_id}/merged.mp4"

    def get_camera_path_from_segments(
        self,
        segments: list[dict[str, Any]],
    ) -> list[str]:
        """Extract camera path from trace segments.

        Args:
            segments: List of trace segments

        Returns:
            Ordered list of camera IDs in the path
        """
        camera_path = []
        for seg in segments:
            if seg.get("camera_id") and seg["camera_id"] not in camera_path:
                camera_path.append(seg["camera_id"])
        return camera_path
