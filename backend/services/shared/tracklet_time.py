"""Utilities for converting video-relative tracklet offsets to wall-clock time."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

CAMERA_VIDEO_PATTERN = re.compile(
    r"(?P<camera_id>cam_\d{2,})_"
    r"(?P<recorded_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<hour>\d{2})-(?P<minute>\d{2})"
    r"(?:-(?P<second>\d{2}))?",
    re.IGNORECASE,
)


def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_recorded_at_from_filename(value: str | None) -> datetime | None:
    """Parse camera recording time from names like cam_02_2026-04-28_11-00.mp4."""
    if not value:
        return None
    filename = Path(str(value)).name
    match = CAMERA_VIDEO_PATTERN.search(filename)
    if not match:
        return None
    try:
        return datetime(
            int(match.group("recorded_date")[:4]),
            int(match.group("recorded_date")[5:7]),
            int(match.group("recorded_date")[8:10]),
            int(match.group("hour")),
            int(match.group("minute")),
            int(match.group("second") or "0"),
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None


def video_recorded_at(video: Any, *, fallback_to_created_at: bool = False) -> datetime | None:
    """Return the video's recording timestamp, preferring DB metadata then filename.

    `created_at` is only a last-resort fallback when explicitly allowed. Search
    and trace grouping should normally use the real recording timestamp encoded
    in the video filename, not the DB insertion time.
    """
    if video is None:
        return None

    recorded = _as_utc(getattr(video, "recorded_at", None))
    if recorded is not None:
        return recorded

    for attr in ("source_filename", "video_id", "title", "storage_path"):
        parsed = parse_recorded_at_from_filename(getattr(video, attr, None))
        if parsed is not None:
            return parsed

    if fallback_to_created_at:
        return _as_utc(getattr(video, "created_at", None))
    return None


def tracklet_time_window(
    tracklet: Any,
    *,
    fallback_to_created_at: bool = False,
) -> tuple[datetime, datetime] | None:
    """Return absolute wall-clock start/end for a tracklet."""
    base = video_recorded_at(
        getattr(tracklet, "video", None),
        fallback_to_created_at=fallback_to_created_at,
    )
    if base is None:
        return None
    start_offset = float(getattr(tracklet, "start_time", 0.0) or 0.0)
    end_offset = float(getattr(tracklet, "end_time", 0.0) or 0.0)
    return base + timedelta(seconds=start_offset), base + timedelta(seconds=end_offset)


def tracklet_time_seconds(
    tracklet: Any,
    *,
    fallback_to_created_at: bool = False,
) -> tuple[float, float] | None:
    """Return absolute Unix-second start/end for a tracklet."""
    window = tracklet_time_window(
        tracklet,
        fallback_to_created_at=fallback_to_created_at,
    )
    if window is None:
        return None
    return window[0].timestamp(), window[1].timestamp()
