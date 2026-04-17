from functools import lru_cache
from pathlib import Path

from .config import settings
from .legacy_runtime import LEGACY_ROOT, legacy_workdir


@lru_cache(maxsize=1)
def get_pipeline_config() -> dict:
    try:
        with legacy_workdir():
            from src_vlm.hospital_pipeline import current_pipeline_config

            return {
                **current_pipeline_config(),
                "config_source": "legacy-module",
            }
    except Exception as exc:
        return {
            "text_query_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "caption_model": "Salesforce/blip-image-captioning-large",
            "bootstrap_detection_mode": "nvidia_ground_truth_per_person",
            "incremental_detection_mode": "opencv_hog_tracking_fallback",
            "queue_size": 32,
            "config_source": "tracking-service-fallback",
            "config_warning": f"legacy config import failed: {exc}",
        }


@lru_cache(maxsize=1)
def get_tracker():
    with legacy_workdir():
        from src_vlm.tracker import ReID_Tracker

        return ReID_Tracker(use_mock=settings.tracking_use_mock)


def run_tracking(candidate_info: dict) -> dict:
    tracker = get_tracker()
    with legacy_workdir():
        output_path = tracker.run_tracking_on_candidate(
            candidate_info,
            "data/videos/compressed",
            "data/videos/tracking_output",
        )

    resolved = Path(output_path).resolve() if output_path else None
    relative = None
    if resolved is not None:
        try:
            relative = str(resolved.relative_to(LEGACY_ROOT))
        except ValueError:
            relative = str(resolved)

    return {
        "output_path": str(resolved) if resolved else None,
        "relative_output_path": relative,
        "exists": bool(resolved and resolved.exists()),
        "tracking_use_mock": settings.tracking_use_mock,
    }
