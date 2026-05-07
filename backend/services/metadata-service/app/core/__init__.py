# Core module - now re-exports from shared.models.
# Old local models (video_assets, video_queries, person_candidates)
# have been migrated to shared/models.py with updated table names.

from shared.models import (
    User,
    VideoQuery,
    QueryCandidate,
    QueryHistory,
    Video,
    Camera,
    Tracklet,
    TrackletEmbedding,
    TrackletAction,
    QueryJob,
    QueueVideoAsset,
)

__all__ = [
    "User", "VideoQuery", "QueryCandidate", "QueryHistory",
    "Video", "Camera", "Tracklet", "TrackletEmbedding",
    "TrackletAction", "QueryJob", "QueueVideoAsset",
]
