from __future__ import annotations

from typing import Any


def build_tracking_runtime_contract() -> dict[str, Any]:
    """Return the single fixed pipeline contract exposed by ai_service.

    The deployed Lightning upstream currently exposes ingestion endpoints only.
    Runtime config therefore needs to come from the application contract rather
    than from a non-existent upstream `/api/v1/runtime-config` route.
    """

    return {
        "runtime_mode": "strict_single_pipeline",
        "deployment_mode": "vps_lightning_google_drive",
        "storage_flow": {
            "drive_root_layout": {
                "temp_dir": "temp/",
                "storage_dir": "storage/cam_xx/yyyy-mm-dd/*.mp4",
            },
            "video_container": "mp4",
            "video_codec": "h265_in_mp4",
            "storage_ingest_source_mode": "storage_ingest",
        },
        "ingestion_pipeline": {
            "decode_sampling_fps": 5,
            "detector": {
                "model_family": "RF-DETR",
                "model_name": "RFDETR2XLarge",
                "alias": "rfdetr-2xlarge",
                "nms_free": True,
            },
            "tracking": {
                "mode": "per_video_independent_tracklets",
                "tracker": "OCMCTrack-style corrective cascade",
                "unit_of_independence": "one_10_min_video",
            },
            "tracklet_quality_scoring": {
                "enabled": True,
                "minimum_confidence_score": 0.3,
                "minimum_tracklet_duration_seconds": 2.0,
                "filters": [
                    "low_confidence",
                    "short_tracklet",
                    "blurry_tracklet",
                    "insufficient_information",
                ],
            },
            "tracklet_output_fields": [
                "video_id",
                "object_id",
                "bboxes_per_frame",
            ],
        },
        "tracklet_feature_pipeline": {
            "frame_selection": {
                "mode": "hybrid_track_mean_plus_topk",
                "selected_frame_count": 5,
                "max_selected_frame_count": 10,
                "score_inputs": [
                    "laplacian_variance",
                    "bbox_area",
                    "detection_confidence",
                ],
                "pooling_methods": [
                    "mean_pooling",
                    "max_pooling",
                    "quality_weighted_mean",
                ],
            },
            "attribute_fields": ["gender", "age_group"],
            "attribute_embedding_fields": ["attribute_embedding_vector"],
            "appearance_fields": [
                "head_accessory",
                "hat",
                "hair_color",
                "skin_tone",
                "shirt",
                "pants",
                "shoes",
                "bag",
            ],
            "appearance_embedding_model": "SOLIDER + KPR",
            "appearance_embedding_fields": [
                "appearance_embedding_vector",
                "embedding_vector",
                "tracklet_vectors",
            ],
            "action_fields": [
                "action_clip",
                "behavior_analysis",
                "action_semantic_embedding",
            ],
            "action_clip_duration_seconds": 2.0,
            "action_clip_stride_seconds": 1.5,
            "aggregation": {
                "vectors_per_tracklet": "1-3",
                "output_key": "tracklet_feature_pipeline",
            },
        },
        "search_contract": {
            "frontend_path": "/search",
            "backend_path": "/internal/search",
            "semantic_stack": "ITSELF",
            "evaluation_stack": "TrackEval HOTA",
        },
    }
