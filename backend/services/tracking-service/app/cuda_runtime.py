import logging
import os
from pathlib import Path

import numpy as np

from .types import BoundingBox

logger = logging.getLogger(__name__)

_HOMOGRAPHY_REGISTRY: dict[str, np.ndarray] = {}
_HOMOGRAPHY_REGISTRY_LOADED = False

_REGISTRY_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "homography_registry.json"


def _load_homography_registry() -> None:
    global _HOMOGRAPHY_REGISTRY, _HOMOGRAPHY_REGISTRY_LOADED
    if _HOMOGRAPHY_REGISTRY_LOADED:
        return
    _HOMOGRAPHY_REGISTRY_LOADED = True
    if not _REGISTRY_PATH.exists():
        logger.warning("Homography registry not found at %s", _REGISTRY_PATH)
        return
    try:
        import json
        raw = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        _HOMOGRAPHY_REGISTRY = {k: np.array(v, dtype=np.float64) for k, v in raw.items()}
        logger.info("Homography registry loaded: %d cameras", len(_HOMOGRAPHY_REGISTRY))
    except Exception as exc:
        logger.warning("Failed to load homography registry: %s", exc)


def get_homography_inv(camera_id: str | None) -> np.ndarray | None:
    _load_homography_registry()
    return _HOMOGRAPHY_REGISTRY.get(camera_id) if camera_id else None


def project_to_world(
    H_inv: np.ndarray,
    bbox: BoundingBox,
) -> tuple[float, float] | None:
    u = (bbox.x1 + bbox.x2) / 2.0
    v = float(bbox.y2)
    pt = H_inv @ np.array([u, v, 1.0], dtype=np.float64)
    if abs(pt[2]) < 1e-8:
        return None
    return float(pt[0] / pt[2]), float(pt[1] / pt[2])
