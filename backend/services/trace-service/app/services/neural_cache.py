"""Neural Cache — FIFO per-user evidence video cache for trace-service.

Implements the FIFO cache behavior:
  - When user selects a new candidate → DELETE old cache for that user
  - Store enhanced clips and merged video in /workspace/storage/cache/{user_id}/{query_id}/
  - Each user has isolated cache directory
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TRACE_CACHE_ROOT = Path("/workspace/storage/cache")


class NeuralCache:
    """
    FIFO cache per user for trace evidence.

    When a user selects a new candidate:
      1. DELETE /workspace/storage/cache/{user_id}/* (clear old evidence)
      2. CREATE folder for current query
      3. Store enhanced segments and merged video

    Cache structure:
      /workspace/storage/cache/
        {user_id}/
          {query_id}/
            merged_full_trace.mp4
            {tracklet_id}/
              crop_raw.mp4
              crop_sr.mp4
              crop_final.mp4
              thumb_000.jpg
            ...
    """

    def __init__(self, user_id: str, cache_root: Path | None = None):
        self.user_id = user_id
        self.cache_root = (cache_root or TRACE_CACHE_ROOT) / user_id
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def clear(self, keep_query_id: Optional[str] = None) -> None:
        """Clear all cache for this user, optionally keeping one query."""
        for item in self.cache_root.iterdir():
            if keep_query_id is not None and item.name == keep_query_id:
                continue
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                logger.debug("Cleared cache: %s", item)
            except Exception as exc:
                logger.warning("Failed to clear %s: %s", item, exc)

    def get_query_dir(self, query_id: str) -> Path:
        """Get cache directory for a specific query."""
        qdir = self.cache_root / query_id
        qdir.mkdir(parents=True, exist_ok=True)
        return qdir

    def get_merged_video(self, query_id: str) -> Path | None:
        """Get path to merged trace video if it exists."""
        merged = self.cache_root / query_id / "merged_full_trace.mp4"
        return merged if merged.exists() else None

    def get_segment_clip(self, query_id: str, tracklet_id: str) -> Path | None:
        """Get path to enhanced segment clip if it exists."""
        clip = self.cache_root / query_id / tracklet_id / "crop_final.mp4"
        if not clip.exists():
            clip = self.cache_root / query_id / tracklet_id / "crop_raw.mp4"
        return clip if clip.exists() else None

    def get_thumbnail(self, query_id: str, tracklet_id: str) -> Path | None:
        """Get path to segment thumbnail if it exists."""
        thumb = self.cache_root / query_id / tracklet_id / "thumb_000.jpg"
        return thumb if thumb.exists() else None

    def list_cached_queries(self) -> list[str]:
        """List all query IDs with cached evidence."""
        return [d.name for d in self.cache_root.iterdir() if d.is_dir()]

    def cache_size_mb(self, query_id: Optional[str] = None) -> float:
        """Get cache size in MB."""
        target = self.cache_root / query_id if query_id else self.cache_root
        if not target.exists():
            return 0.0
        total = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        return total / 1e6

    def evict_lru(self, max_queries: int = 3) -> None:
        """Evict least-recently-used queries to keep at most max_queries cached."""
        dirs = sorted(
            [(d, d.stat().st_mtime) for d in self.cache_root.iterdir() if d.is_dir()],
            key=lambda x: x[1],
        )
        for d, _ in dirs[:-max_queries]:
            try:
                shutil.rmtree(d)
                logger.info("Evicted LRU cache: %s", d)
            except Exception as exc:
                logger.warning("Failed to evict %s: %s", d, exc)
