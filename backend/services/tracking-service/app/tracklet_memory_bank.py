"""
tracklet_memory_bank.py — Tier 2 cross-camera Re-ID via vector similarity.

Architecture (2-tier):
  Tier 1 (OCMCTrack): short-term tracking → produces Tracklets (5-10s segments)
  Tier 2 (this module): when a tracklet ends, extract embedding vector from its
                        best frames via SigLIP2/TransReID, compare against the
                        MemoryBank, merge if cosine similarity > threshold.

Usage:
    bank = TrackletMemoryBank(similarity_threshold=0.88)
    bank.process_tracklet(tracklet, frames_by_idx)
    merged_id = bank.resolve(tracklet.track_id)   # returns canonical human_key
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

_COSINE_EPS = 1e-8


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < _COSINE_EPS or norm_b < _COSINE_EPS:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class TrackletEntry:
    track_id: str
    camera_id: str | None
    human_key: str              # canonical identity (may be merged)
    vector: np.ndarray          # mean embedding from best frames
    quality_score: float        # higher = more reliable embedding
    frame_count: int
    last_seen: float = field(default_factory=time.monotonic)


class TrackletMemoryBank:
    """
    In-memory vector store for cross-camera person Re-ID.

    Thread-safe. Vectors are quality-weighted means of SigLIP2/TransReID
    embeddings from the N best (highest-quality) frames of each tracklet.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.88,
        max_entries: int = 10_000,
        ttl_seconds: float = 3600.0,    # forget identities after 1 hour
        top_k_frames: int = 3,           # use N best frames per tracklet
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.top_k_frames = top_k_frames
        self._lock = threading.Lock()
        self._entries: dict[str, TrackletEntry] = {}   # human_key → entry
        self._track_to_human: dict[str, str] = {}      # track_id → human_key
        self._counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_tracklet(
        self,
        track_id: str,
        camera_id: str | None,
        candidate_frames: list[dict[str, Any]],
    ) -> str:
        """
        Extract embedding for tracklet, compare against bank, assign human_key.

        candidate_frames: list of dicts with keys:
            - "image": np.ndarray (BGR)
            - "quality": float (detection confidence or sharpness score)
            - "frame_idx": int

        Returns: human_key (str) — stable cross-camera identity
        """
        vector = self._embed_tracklet(candidate_frames)
        if vector is None:
            human_key = self._new_human_key()
            LOGGER.debug("[bank] No embedding for track %s → new identity %s", track_id, human_key)
            return human_key

        quality = float(np.mean([f.get("quality", 0.5) for f in candidate_frames[:self.top_k_frames]]))
        human_key = self._match_or_create(track_id, camera_id, vector, quality, len(candidate_frames))
        LOGGER.debug(
            "[bank] track=%s camera=%s → human_key=%s (quality=%.3f pool=%d)",
            track_id, camera_id, human_key, quality, len(self._entries),
        )
        return human_key

    def resolve(self, track_id: str) -> str | None:
        """Return the canonical human_key for a track_id, or None if unknown."""
        with self._lock:
            return self._track_to_human.get(track_id)

    def stats(self) -> dict:
        with self._lock:
            return {
                "identities": len(self._entries),
                "track_mappings": len(self._track_to_human),
                "threshold": self.similarity_threshold,
            }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._track_to_human.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _embed_tracklet(self, candidate_frames: list[dict]) -> np.ndarray | None:
        """Extract quality-weighted mean embedding from top-K frames."""
        if not candidate_frames:
            return None

        # Sort by quality descending, take top-K
        sorted_frames = sorted(candidate_frames, key=lambda f: float(f.get("quality", 0.5)), reverse=True)
        top_frames = sorted_frames[: self.top_k_frames]

        try:
            from .model_adapters import SigLIP2ModelHub
            hub = SigLIP2ModelHub()
            images = [f["image"] for f in top_frames if f.get("image") is not None]
            if not images:
                return None
            qualities = np.array([float(f.get("quality", 0.5)) for f in top_frames[: len(images)]], dtype=np.float32)
            qualities = qualities / (qualities.sum() + _COSINE_EPS)
            embeddings = hub.embed_images(images)   # (N, D) float32
            if embeddings is None or len(embeddings) == 0:
                return None
            # Quality-weighted mean
            weighted = np.average(embeddings, axis=0, weights=qualities[: len(embeddings)])
            return weighted.astype(np.float32)
        except Exception as exc:
            LOGGER.warning("[bank] Embedding failed: %s", exc)
            return None

    def _match_or_create(
        self,
        track_id: str,
        camera_id: str | None,
        vector: np.ndarray,
        quality: float,
        frame_count: int,
    ) -> str:
        with self._lock:
            self._evict_stale()

            # Find best match in bank
            best_key: str | None = None
            best_sim = self.similarity_threshold - _COSINE_EPS

            for key, entry in self._entries.items():
                sim = _cosine_similarity(vector, entry.vector)
                if sim > best_sim:
                    best_sim = sim
                    best_key = key

            if best_key is not None:
                # Merge: update entry vector with quality-weighted running mean
                entry = self._entries[best_key]
                w_new = quality / (entry.quality_score + quality + _COSINE_EPS)
                entry.vector = ((1 - w_new) * entry.vector + w_new * vector).astype(np.float32)
                entry.quality_score = max(entry.quality_score, quality)
                entry.frame_count += frame_count
                entry.last_seen = time.monotonic()
                self._track_to_human[track_id] = best_key
                LOGGER.info(
                    "[bank] MERGED track=%s → %s (sim=%.4f camera=%s)",
                    track_id, best_key, best_sim, camera_id,
                )
                return best_key

            # No match — create new identity
            human_key = self._new_human_key()
            self._entries[human_key] = TrackletEntry(
                track_id=track_id,
                camera_id=camera_id,
                human_key=human_key,
                vector=vector,
                quality_score=quality,
                frame_count=frame_count,
            )
            self._track_to_human[track_id] = human_key
            return human_key

    def _new_human_key(self) -> str:
        self._counter += 1
        return f"P{self._counter:06d}"

    def _evict_stale(self) -> None:
        """Remove entries older than TTL or when over capacity. Call under lock."""
        now = time.monotonic()
        stale = [k for k, e in self._entries.items() if now - e.last_seen > self.ttl_seconds]
        for k in stale:
            del self._entries[k]
        # Cap size
        if len(self._entries) > self.max_entries:
            oldest = sorted(self._entries.items(), key=lambda kv: kv[1].last_seen)
            for k, _ in oldest[: len(self._entries) - self.max_entries]:
                del self._entries[k]


# Module-level singleton shared across all ingestion jobs
_GLOBAL_BANK: TrackletMemoryBank | None = None
_BANK_LOCK = threading.Lock()


def get_global_bank(
    similarity_threshold: float = 0.88,
    ttl_seconds: float = 3600.0,
) -> TrackletMemoryBank:
    global _GLOBAL_BANK
    with _BANK_LOCK:
        if _GLOBAL_BANK is None:
            _GLOBAL_BANK = TrackletMemoryBank(
                similarity_threshold=similarity_threshold,
                ttl_seconds=ttl_seconds,
            )
    return _GLOBAL_BANK
