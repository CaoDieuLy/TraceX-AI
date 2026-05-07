# Core module - now re-exports from shared.models.
# Old local models (video_assets, video_queries, person_candidates)
# have been migrated to shared/models.py with updated table names.

from backend.services.shared.models import (
    PersonCandidate,
    QueueVideoAsset,
    User,
    VideoAsset,
    VideoQuery,
)

__all__ = ["User", "VideoAsset", "VideoQuery", "PersonCandidate", "QueueVideoAsset"]
