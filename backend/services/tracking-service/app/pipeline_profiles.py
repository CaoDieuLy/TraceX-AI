from __future__ import annotations

from copy import deepcopy


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


PIPELINE_PROFILES: dict[str, dict] = {
    "accuracy_first": {
        "profile": "accuracy_first",
        "summary": "Prioritize identity consistency and dense-scene recall over raw throughput.",
        "validated_on": "2026-04-17",
        "components": {
            "detector": {
                "name": "RF-DETR 2x-large",
                "family": "detection_transformer",
                "nms_free": True,
                "deployment": "TensorRT FP16",
                "selection_reason": "Verified real-time DETR profile with strong COCO AP and no NMS bottleneck in crowded scenes.",
                "fallbacks": [
                    "YOLO26-X",
                    "MMPedestron",
                ],
                "sources": [
                    {
                        "label": "ICLR 2026 poster",
                        "url": "https://iclr.cc/virtual/2026/poster/10007257",
                    },
                    {
                        "label": "Ultralytics YOLO26 docs",
                        "url": "https://docs.ultralytics.com/",
                    },
                ],
            },
            "tracker": {
                "name": "OCMCTrack-style corrective cascade",
                "family": "online_mtmc",
                "geometry_aware": True,
                "occlusion_reasoning": True,
                "selection_reason": "Corrective matching cascade is directly aligned with multi-camera online correction and world-coordinate consistency.",
                "fallbacks": [
                    "ByteTrack with geometry gates",
                    "Legacy Deep SORT (fallback only)",
                ],
                "sources": [
                    {
                        "label": "OCMCTrack CVPRW 2024",
                        "url": "https://publica.fraunhofer.de/entities/publication/b8475da1-6fa2-4696-8d3d-78323b3d50cf",
                    },
                    {
                        "label": "ByteTrack paper",
                        "url": "https://huggingface.co/papers/2110.06864",
                    },
                ],
            },
            "reid": {
                "name": "SOLIDER + KPR",
                "family": "human_foundation_reid",
                "part_based": True,
                "occlusion_robust": True,
                "selection_reason": "SOLIDER is a strong verified MSMT17 baseline, and KPR directly addresses occlusion and multi-person ambiguity.",
                "fallbacks": [
                    "CLIP-ReID",
                ],
                "sources": [
                    {
                        "label": "SOLIDER repository benchmarks",
                        "url": "https://github.com/tinyvision/SOLIDER",
                    },
                    {
                        "label": "KPR ECCV 2024",
                        "url": "https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/10180_ECCV_2024_paper.php",
                    },
                    {
                        "label": "CLIP-ReID repository",
                        "url": "https://github.com/Syliz517/CLIP-ReID",
                    },
                ],
            },
            "semantic_search": {
                "name": "ITSELF with attention-guided ingest",
                "family": "vision_language_retrieval",
                "fine_grained_alignment": True,
                "reranking": "ranking_ensemble",
                "selection_reason": "Verified WACV 2026 text-based retrieval method for fine-grained person search.",
                "sources": [
                    {
                        "label": "ITSELF WACV 2026",
                        "url": "https://openaccess.thecvf.com/content/WACV2026/html/Nguyen_ITSELF_Attention_Guided_Fine-Grained_Alignment_for_Vision-Language_Retrieval_WACV_2026_paper.html",
                    },
                ],
            },
            "evaluation": {
                "name": "TrackEval HOTA",
                "family": "tracking_evaluation",
                "primary_metrics": [
                    "HOTA",
                    "DetA",
                    "AssA",
                ],
                "selection_reason": "HOTA is the recommended reference metric for balancing detection and association quality.",
                "sources": [
                    {
                        "label": "TrackEval",
                        "url": "https://github.com/JonathonLuiten/TrackEval",
                    },
                ],
            },
            "geometry": {
                "name": "Top-point world projection",
                "family": "global_position_estimation",
                "selection_reason": "Top-point projection is more robust than bottom-point projection when lower-body occlusions are common.",
                "sources": [
                    {
                        "label": "Top-point MTMC abstract",
                        "url": "https://www.researchgate.net/publication/384411542_OCMCTrack_Online_Multi-Target_Multi-Camera_Tracking_with_Corrective_Matching_Cascade",
                    },
                    {
                        "label": "NVIDIA Sparse4D docs",
                        "url": "https://docs.nvidia.com/vss/3.1.0/warehouse-docs/Sparse4D.html",
                    },
                ],
            },
        },
        "runtime_defaults": {
            "detector_precision": "fp16",
            "transformer_quantization": "avoid_int8_without_calibration",
            "association_mode": "geometry_then_reid_then_corrective_cascade",
            "search_mode": "cosine_plus_ranking_ensemble",
        },
        "hyperparameters": {
            "detector": {
                "confidence_threshold": 0.32,
                "person_class_only": True,
                "max_queries": 300,
                "temporal_smoothing": 0.15,
            },
            "tracker": {
                "high_confidence_threshold": 0.45,
                "low_confidence_threshold": 0.12,
                "new_track_threshold": 0.55,
                "iou_gate": 0.18,
                "appearance_gate": 0.22,
                "max_age_seconds": 8.0,
                "min_confirmed_frames": 4,
                "occlusion_overlap_threshold": 0.55,
                "occlusion_center_distance_ratio": 0.35,
                "search_region_expansion": 0.18,
                "motion_cone_angle_degrees": 42.0,
                "motion_damping": 0.35,
                "corrective_buffer_seconds": 14.0,
                "world_gate_max_speed_mps": 2.8,
            },
            "reid": {
                "global_embedding_weight": 0.55,
                "part_embedding_weight": 0.45,
                "visibility_floor": 0.2,
                "rerank_k1": 30,
                "rerank_k2": 6,
                "rerank_lambda": 0.35,
                "cross_camera_match_threshold": 0.27,
            },
            "semantic_search": {
                "fetch_multiplier": 10,
                "topk": 10,
                "score_weights": {
                    "embedding": 0.58,
                    "semantic_overlap": 0.22,
                    "visibility": 0.12,
                    "world_position": 0.08,
                },
                "minimum_semantic_overlap": 0.08,
            },
            "ingest": {
                "detection_fps": 4.0,
                "min_track_frames": 5,
                "min_person_area": 4500,
                "track_iou": 0.20,
                "sampled_content_frames": 12,
                "timeline_segments": 14,
            },
        },
    },
    "edge_accuracy": {
        "profile": "edge_accuracy",
        "summary": "Maintain accuracy-first principles while biasing toward edge-friendly deployment.",
        "validated_on": "2026-04-17",
        "components": {
            "detector": {
                "name": "YOLO26-X",
                "family": "end_to_end_detector",
                "nms_free": True,
                "deployment": "edge_fp16",
                "selection_reason": "Ultralytics positions YOLO26 for edge-first deployment with native end-to-end inference.",
                "sources": [
                    {
                        "label": "Ultralytics launch note",
                        "url": "https://www.ultralytics.com/news/ultralytics-redefines-state-of-the-art-vision-ai-with-yolo26",
                    },
                ],
            },
            "tracker": {
                "name": "ByteTrack with geometry gates",
                "family": "online_mot",
                "geometry_aware": True,
                "selection_reason": "ByteTrack remains a practical strong baseline when detector recall is high.",
                "sources": [
                    {
                        "label": "ByteTrack paper",
                        "url": "https://huggingface.co/papers/2110.06864",
                    },
                ],
            },
            "reid": {
                "name": "SOLIDER",
                "family": "human_foundation_reid",
                "part_based": False,
                "selection_reason": "SOLIDER offers strong verified MSMT17 performance without requiring promptable keypoints at inference.",
                "sources": [
                    {
                        "label": "SOLIDER repository benchmarks",
                        "url": "https://github.com/tinyvision/SOLIDER",
                    },
                ],
            },
            "semantic_search": {
                "name": "ITSELF ingest with lightweight rerank",
                "family": "vision_language_retrieval",
                "fine_grained_alignment": True,
                "reranking": "lightweight_ranking_ensemble",
                "selection_reason": "Retain fine-grained retrieval while reducing compute cost in post-processing.",
                "sources": [
                    {
                        "label": "ITSELF WACV 2026",
                        "url": "https://openaccess.thecvf.com/content/WACV2026/html/Nguyen_ITSELF_Attention_Guided_Fine-Grained_Alignment_for_Vision-Language_Retrieval_WACV_2026_paper.html",
                    },
                ],
            },
            "evaluation": {
                "name": "TrackEval HOTA",
                "family": "tracking_evaluation",
                "primary_metrics": [
                    "HOTA",
                    "DetA",
                    "AssA",
                ],
                "sources": [
                    {
                        "label": "TrackEval",
                        "url": "https://github.com/JonathonLuiten/TrackEval",
                    },
                ],
            },
            "geometry": {
                "name": "Top-point world projection",
                "family": "global_position_estimation",
                "sources": [
                    {
                        "label": "NVIDIA Sparse4D docs",
                        "url": "https://docs.nvidia.com/vss/3.1.0/warehouse-docs/Sparse4D.html",
                    },
                ],
            },
        },
        "runtime_defaults": {
            "detector_precision": "fp16",
            "association_mode": "geometry_then_reid",
            "search_mode": "cosine_plus_ranking_ensemble",
        },
        "hyperparameters": {
            "detector": {
                "confidence_threshold": 0.35,
                "person_class_only": True,
                "max_queries": 200,
                "temporal_smoothing": 0.12,
            },
            "tracker": {
                "high_confidence_threshold": 0.48,
                "low_confidence_threshold": 0.14,
                "new_track_threshold": 0.58,
                "iou_gate": 0.20,
                "appearance_gate": 0.24,
                "max_age_seconds": 6.0,
                "min_confirmed_frames": 4,
                "occlusion_overlap_threshold": 0.50,
                "occlusion_center_distance_ratio": 0.30,
                "search_region_expansion": 0.12,
                "motion_cone_angle_degrees": 36.0,
                "motion_damping": 0.30,
                "corrective_buffer_seconds": 10.0,
                "world_gate_max_speed_mps": 2.8,
            },
            "reid": {
                "global_embedding_weight": 0.72,
                "part_embedding_weight": 0.28,
                "visibility_floor": 0.25,
                "rerank_k1": 20,
                "rerank_k2": 4,
                "rerank_lambda": 0.45,
                "cross_camera_match_threshold": 0.30,
            },
            "semantic_search": {
                "fetch_multiplier": 8,
                "topk": 10,
                "score_weights": {
                    "embedding": 0.64,
                    "semantic_overlap": 0.20,
                    "visibility": 0.10,
                    "world_position": 0.06,
                },
                "minimum_semantic_overlap": 0.05,
            },
            "ingest": {
                "detection_fps": 3.0,
                "min_track_frames": 4,
                "min_person_area": 5000,
                "track_iou": 0.22,
                "sampled_content_frames": 10,
                "timeline_segments": 12,
            },
        },
    },
    "legacy_compat": {
        "profile": "legacy_compat",
        "summary": "Keep the legacy stack reachable for fallback workflows only.",
        "validated_on": "2026-04-17",
        "components": {
            "detector": {
                "name": "YOLOv4",
                "family": "anchor_based_detector",
                "nms_free": False,
            },
            "tracker": {
                "name": "Deep SORT",
                "family": "kalman_tracker",
                "geometry_aware": False,
            },
            "reid": {
                "name": "Torchreid / mars-small128",
                "family": "legacy_reid",
                "part_based": False,
            },
            "semantic_search": {
                "name": "SentenceTransformer baseline",
                "family": "text_embedding_baseline",
            },
            "evaluation": {
                "name": "MOTA / IDF1 legacy mix",
                "family": "legacy_tracking_evaluation",
            },
            "geometry": {
                "name": "Bottom-point projection",
                "family": "legacy_position_estimation",
            },
        },
        "runtime_defaults": {
            "detector_precision": "fp32",
            "association_mode": "appearance_then_motion",
            "search_mode": "cosine_only",
        },
        "hyperparameters": {
            "detector": {
                "confidence_threshold": 0.50,
                "person_class_only": True,
                "max_queries": 100,
                "temporal_smoothing": 0.05,
            },
            "tracker": {
                "high_confidence_threshold": 0.55,
                "low_confidence_threshold": 0.25,
                "new_track_threshold": 0.60,
                "iou_gate": 0.25,
                "appearance_gate": 0.30,
                "max_age_seconds": 5.0,
                "min_confirmed_frames": 5,
                "occlusion_overlap_threshold": 0.60,
                "occlusion_center_distance_ratio": 0.40,
                "search_region_expansion": 0.10,
                "motion_cone_angle_degrees": 30.0,
                "motion_damping": 0.20,
                "corrective_buffer_seconds": 0.0,
                "world_gate_max_speed_mps": 3.2,
            },
            "reid": {
                "global_embedding_weight": 1.0,
                "part_embedding_weight": 0.0,
                "visibility_floor": 0.0,
                "rerank_k1": 10,
                "rerank_k2": 3,
                "rerank_lambda": 0.60,
                "cross_camera_match_threshold": 0.35,
            },
            "semantic_search": {
                "fetch_multiplier": 6,
                "topk": 10,
                "score_weights": {
                    "embedding": 0.80,
                    "semantic_overlap": 0.15,
                    "visibility": 0.05,
                    "world_position": 0.0,
                },
                "minimum_semantic_overlap": 0.0,
            },
            "ingest": {
                "detection_fps": 2.0,
                "min_track_frames": 3,
                "min_person_area": 6000,
                "track_iou": 0.25,
                "sampled_content_frames": 8,
                "timeline_segments": 12,
            },
        },
    },
}


def resolve_pipeline_profile(profile_name: str, overrides: dict | None = None) -> dict:
    resolved_name = (profile_name or "accuracy_first").strip().lower()
    profile = PIPELINE_PROFILES.get(resolved_name) or PIPELINE_PROFILES["accuracy_first"]
    if overrides:
        return _deep_merge(profile, overrides)
    return deepcopy(profile)
